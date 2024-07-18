import torch

from spm.data.generate_data import AnnotatedTranscriptSampler, UpperBoundRejector
from spm.data.samplers import LogUniformGCDSampler, TranscriptSampler
from spm.data.str_repr import EncodedSamples
from spm.data.tensor_repr import NO_LOSS, TargetComponent, TensorRepr, block_size
from spm.gpt.trainer import Trainer, logging, time, wandb

# For now we keep it as a magic number because generate_data doesn't have a config.
UBOUND = 10000
REJECTOR_UBOUND = 210**3


@torch.no_grad()
def make_rlvf_batch(
    samples: EncodedSamples, model, device, temp, acceptance_mask: str | None, block_size: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get a batch of training samples for the RLVF gradient step. This is done by generating outputs from the model
    and keeping only the accepted ones. Then converting them to X and Y for computing the train loss.

    Args:
        samples: A list of samples from the sampler.
        model: The model to use for generation.
        device: The device to use for the tensors.
        temp: The temperature to use for generation.
        acceptance_mask: The mask to use for accepting generations (TRANSCRIPT or ANNOTATED_TRANSCRIPT).
        block_size: The block size to use for the model. If None, it will be taken from the model's config.
    Returns:
        (X, Y): The input and target tensors for the training step.
    """
    assert acceptance_mask is not None, "acceptance_mask must be provided."  # To satisfy mypy, mainly.
    if block_size is None:
        assert hasattr(model, "config"), "Model must have a config, or block_size must be provided."
        block_size = model.config.block_size
    # Generate samples and set up a TensorRepr.
    batch_size = len(samples)  # Just a simplifying choice. You can change this if needed.
    buf = TensorRepr.from_rlvf_samples(samples, block_size, acceptance_mask)
    Xs, Ys = [], []
    for input_length, inp, masks, masked in buf.val_batches(device, batch_size):
        inp_out = model.generate(
            inp, max_new_tokens=buf.m.generation_size, temperature=temp
        )  # Returns prompt + output.
        # keep only the output (inp_out will be useful soon for generating X and Y)
        output = inp_out[:, inp.shape[1] :]
        # keep only the accepted generations
        target = masked[acceptance_mask]
        target_hat = output * masks[acceptance_mask]
        target_eq = torch.eq(target_hat, target).all(dim=1)
        accepted_samples = inp_out[target_eq]
        if accepted_samples.shape[0] == 0:
            continue
        # We take the  mask because we want to generate Y from the annotated transcript.
        # To compute Y, we need to mask out the input and the padding. I.e., keep just the output (annotated trans.)
        # Fortunately, masks[ANNOTATED_TRANSCRIPT] does exactly this. We just have to pad it with zeros on the left, to
        # account for the input.
        # NOTE: This is a bit hacky, and the "correct" (but slower?) way would be to build X and Y as numpy arrays
        # and then convert them with np_to_torch, as in TensorRepr.get_train_batch().
        accepted_masks = masks[TargetComponent.ANNOTATED_TRANSCRIPT][target_eq]
        loss_mask = torch.cat(
            [torch.zeros((accepted_samples.shape[0], inp.shape[1]), device=device), accepted_masks], dim=1
        )
        # Trim or pad the accepted_samples and the loss_mask to block_size+1.
        if accepted_samples.shape[1] > block_size + 1:
            accepted_samples = accepted_samples[:, : block_size + 1]
            loss_mask = loss_mask[:, : block_size + 1]
        elif accepted_samples.shape[1] < block_size + 1:
            pad = torch.full(
                size=(accepted_samples.shape[0], block_size + 1 - accepted_samples.shape[1]),
                fill_value=buf.toks.pad,
                dtype=accepted_samples.dtype,
                device=accepted_samples.device,
            )
            accepted_samples = torch.cat([accepted_samples, pad], dim=1)
            loss_mask = torch.cat([loss_mask, torch.zeros_like(pad)], dim=1)
        assert accepted_samples.shape == loss_mask.shape, f"{accepted_samples.shape} != {loss_mask.shape}"
        assert accepted_samples.shape[1] == block_size + 1, f"{accepted_samples.shape[1]} != {block_size + 1}"
        accepted_samples = accepted_samples.long()  # long bc that's what TR.np_to_torch does (e.g. for CE loss).
        loss_mask = loss_mask.long()
        Xs.append(accepted_samples[:, :-1])
        Y = accepted_samples.clone()
        Y = Y * loss_mask + NO_LOSS * (1 - loss_mask)  # equiv to Y[~loss_mask] = NO_LOSS.
        Y = Y[:, 1:]  # remove the first token because Y is X shifted (and masked for the loss).
        Ys.append(Y)
    if len(Xs) == 0:
        return torch.tensor([]), torch.tensor([])
    return torch.cat(Xs), torch.cat(Ys)


class RLVFTrainer(Trainer):
    sampler: TranscriptSampler

    def __init__(self, cfg):
        super().__init__(cfg)
        # Get the string that follows TL and precedes the _ to determine the sampler
        data_type = cfg.data.split("TL")[1].split("_")[0]
        lu_sampler = LogUniformGCDSampler(UBOUND, base=self.meta.base)
        if data_type == "":  # It was TL
            logging.info("Using TranscriptSampler")
            self.sampler = TranscriptSampler(lu_sampler)
        else:
            annot_len = int(data_type[len("Plus") :])
            logging.info(f"Using AnnotatedTranscriptSampler with annot_len={data_type}")
            self.sampler = AnnotatedTranscriptSampler(lu_sampler, annot_len=annot_len)
        self.sampler = UpperBoundRejector(self.sampler, REJECTOR_UBOUND)  # type: ignore

    def run(self):
        t0 = time.time()
        while self.iter_num <= self.cfg.max_iters:
            # determine and set the learning rate for this iteration
            lr = self.get_lr() if self.cfg.decay_lr else self.cfg.learning_rate
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            # evaluate the loss and write checkpoints
            if self.cfg.eval_interval is not None:
                if self.iter_num % self.cfg.eval_interval == 0 or self.iter_num == self.cfg.max_iters:
                    self.eval_step()
            if self.cfg.save_iters is not None and self.iter_num in self.cfg.save_iters:
                self.save_checkpoint()

            # forward backward update, using the GradScaler if data type is float16
            samples = self.sampler(self.cfg.batch_size)
            encoded_samples = EncodedSamples(samples, self.meta.base)
            while block_size(encoded_samples) > self.cfg.block_size:
                logging.warning("Sample contains a transcript longer than model block size. Resampling.")
                samples = self.sampler(self.cfg.batch_size)
                encoded_samples = EncodedSamples(samples, self.meta.base)
            self.model.eval()
            X, Y = make_rlvf_batch(encoded_samples, self.model, self.cfg.device, self.cfg.temperature, self.cfg.rlvf)
            self.model.train()
            frac_accepted = X.shape[0] / self.cfg.batch_size
            if frac_accepted == 0:
                logging.warning("No accepted samples in this batch. Terminating.")
                break
            with self.ctx:
                logits, loss = self.model(X, Y)
                loss = loss * frac_accepted  # scale the gradient by the fraction of accepted samples
            # backward pass, with gradient scaling if training in fp16
            self.scaler.scale(loss).backward()
            # clip the gradient
            if self.cfg.grad_clip != 0.0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            if self.cfg.device != "cpu":
                # step the optimizer and scaler if training in fp16
                self.scaler.step(self.optimizer)  # errs on CPU
            self.scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            self.optimizer.zero_grad(set_to_none=True)

            # timing and logging
            t1 = time.time()
            ms_iter = (t1 - t0) * 1000
            t0 = t1
            if self.iter_num % self.cfg.log_interval == 0:
                # get loss as float. note: this is a CPU-GPU sync point (but so is self.get_batch()?)
                loss_float = loss.item()
                logging.info(
                    f"it {self.iter_num}: loss {loss_float:.4f}, dt {ms_iter:.2f}ms, "
                    f"lr {lr:.2e}, acc {frac_accepted:.2f}"
                )
                wandb.log(
                    {
                        "train/loss": loss_float,
                        "train/lr": lr,
                        "train/frac_accepted": frac_accepted,
                        "sys/ms_iter": ms_iter,
                    },
                    step=self.iter_num,
                )
            if frac_accepted <= self.cfg.early_stop:
                logging.info(f"Early stopping at iteration {self.iter_num}")
                break
            self.iter_num += 1
