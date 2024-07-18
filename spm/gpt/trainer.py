"""
Adapted from nanoGPT to expose the same API as miniGPT.
"""

import math
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict

import torch
import wandb

from spm import ANALYSIS_DIR, GIT_SHA, MODELS_DIR, logging
from spm.data.tensor_repr import TensorRepr
from spm.gpt.config import TrainerConfig
from spm.gpt.model import GPT


def average_scores(
    len_to_num_correct: dict[int, dict[str, int]], len_to_num_samples: dict[int, int]
) -> dict[str, dict[str, float]]:
    """Average the correct counts over the number of samples. Add a "total" key for all lengths."""
    len_to_avg_correct = {
        str(length): {col: num_correct / num_samples for col, num_correct in len_to_num_correct[length].items()}
        for length, num_samples in len_to_num_samples.items()
    }
    total_num_samples = sum(v for v in len_to_num_samples.values())
    eval_cols = next(iter(len_to_num_correct.values())).keys()
    len_to_avg_correct["total"] = {
        col: sum(len_to_num_correct[length][col] for length in len_to_num_correct) / total_num_samples
        for col in eval_cols
    }
    return len_to_avg_correct


@torch.no_grad()
def evaluate(model, tr: TensorRepr, device, batch_size) -> dict[str, dict[str, float]]:
    """
    Evaluate the model. Generation is batched by length of input. This is at least 100x faster than sequential.
    Returns:
        A dictionary mapping input length to metrics, where metrics is a dictionary mapping column to score.
    """
    len_to_num_samples: dict[int, int] = defaultdict(int)  # {len: num_samples}
    len_to_num_correct: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))  # {len: {col: num_correct}}
    for input_length, inp, masks, masked in tr.val_batches(device, batch_size):
        len_to_num_samples[input_length] += inp.shape[0]
        output = model.generate(inp, max_new_tokens=tr.m.generation_size)
        # trim input
        output = output[:, inp.shape[1] :]
        for col in masks:
            target = masked[col]
            target_hat = output * masks[col]
            target_eq = torch.eq(target_hat, target)
            # take row-wise 'and' and sum to get number of correct
            len_to_num_correct[input_length][col] += int(torch.all(target_eq, dim=1).sum().item())  # casting for mypy
    len_to_avg_scores = average_scores(len_to_num_correct, len_to_num_samples)
    return len_to_avg_scores


class Trainer:
    cfg: TrainerConfig
    model: GPT
    device_type: str
    ptdtype: torch.dtype
    ctx: nullcontext | torch.amp.autocast
    optimizer: torch.optim.Optimizer
    scaler: torch.cuda.amp.GradScaler

    def __init__(self, cfg: TrainerConfig):
        self.cfg = cfg
        # initialize datasets
        self.data = TensorRepr.from_name(self.cfg.data)
        self.meta = self.data.m  # for convenience
        cfg.vocab_size = self.meta.vocab_size
        cfg.block_size = self.meta.block_size
        # we only update config if wandb not disabled,
        # because this throws (fair) errors when Trainer is instantiated multiple times
        if not isinstance(wandb.run, wandb.sdk.lib.disabled.RunDisabled):
            wandb.config.update(cfg)
            wandb.config.update({f"data_{k}": v for k, v in asdict(self.meta).items()})
            wandb.config.update({"git_sha": GIT_SHA})

        self.iter_num = 0
        self.initialize_io()
        self.initialize_model()

    def initialize_io(self):
        torch.manual_seed(self.cfg.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.device_type = "cuda" if "cuda" in self.cfg.device else "cpu"
        self.ptdtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[self.cfg.dtype]
        self.ctx = (
            nullcontext()
            if self.device_type == "cpu"
            else torch.amp.autocast(device_type=self.device_type, dtype=self.ptdtype)
        )
        # self.ctx = nullcontext()  # DEBUG: disable autocast

    def initialize_model(self):
        if self.cfg.load_ckpt is not None:
            ckpt_path = (MODELS_DIR / self.cfg.load_ckpt).with_suffix(".pt")
            logging.info(f"loading model from {ckpt_path}")
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint {ckpt_path} not found.")
            ckpt = torch.load(ckpt_path, map_location=self.device_type)
            for k in ["n_layer", "n_head", "n_embd", "bias"]:
                if ckpt["config"][k] != self.cfg.__getattribute__(k):
                    logging.info(f"Overriding {k} with {ckpt['config'][k]}")
                    self.cfg.__setattr__(k, ckpt["config"][k])
            self.model = GPT(self.cfg)
            state_dict = ckpt["model"]
            # Fix keys prefixed due to compiled model. See https://github.com/karpathy/nanoGPT/issues/325
            unwanted_prefix = "_orig_mod."
            for k, v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
            self.model.load_state_dict(ckpt["model"])
            # self.iter_num = ckpt["iter_num"]  # Let's start from 0
        else:
            self.model = GPT(self.cfg)
        logging.info(f"initialized model with config {self.cfg}")
        self.model.to(self.cfg.device)
        # initialize gradscaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device_type == "cuda"))
        # initialize optimizer
        self.optimizer = self.model.configure_optimizers(config=self.cfg, device_type=self.device_type)
        if self.cfg.load_ckpt is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        # compile the model
        if self.device_type == "cuda":
            logging.info("Compiling model")
            self.model = torch.compile(self.model)

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(self):
        it = self.iter_num
        # 1) linear warmup for warmup_iters steps
        if it < self.cfg.warmup_iters:
            return self.cfg.learning_rate * it / self.cfg.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.cfg.lr_decay_iters:
            return self.cfg.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.cfg.warmup_iters) / (self.cfg.lr_decay_iters - self.cfg.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return self.cfg.min_lr + coeff * (self.cfg.learning_rate - self.cfg.min_lr)

    def eval_step(self):
        """Evaluate the model and log the results."""
        t0 = time.time()
        self.model.eval()
        inp_len_to_avg_scores = evaluate(
            self.model,
            self.data,
            device=self.cfg.device,
            batch_size=self.cfg.eval_batch_size,
        )
        self.model.train()
        ms_eval = (time.time() - t0) * 1000
        wandb_metrics = {
            f"{length}/{col}": score
            for length, scores in inp_len_to_avg_scores.items()
            for col, score in scores.items()
        }
        wandb_metrics["sys/ms_eval"] = ms_eval
        wandb.log(wandb_metrics, step=self.iter_num)
        total_scores = inp_len_to_avg_scores["total"]
        logging_metrics = {col: f"{score:.2f}" for col, score in total_scores.items()}
        logging_metrics["ms_eval"] = f"{ms_eval:.2f}"
        logging.info(logging_metrics)
        # Append to the log CSV
        logname = self.cfg.to_run_name() + f"_s{self.cfg.seed}.csv"
        if not (ANALYSIS_DIR / logname).exists():
            # create header row
            with open(ANALYSIS_DIR / logname, "w") as f:
                f.write("iter_num," + ",".join(total_scores.keys()) + "\n")
        with open(ANALYSIS_DIR / logname, "a") as f:
            f.write(f"{self.iter_num},{','.join(str(score) for score in total_scores.values())}\n")

    def save_checkpoint(self):
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "iter_num": self.iter_num,
            "config": asdict(self.cfg),
            "block_size": self.meta.block_size,
            "vocab_size": self.meta.vocab_size,
        }
        save_path = MODELS_DIR / f"{self.cfg.to_run_name()}_iter{self.iter_num}.pt"  # note: overwrites diff seeds.
        torch.save(checkpoint, save_path)
        logging.info(f"saved model to {save_path}")

    def run(self):
        X, Y = self.data.get_train_batch(self.cfg.batch_size, device=self.device_type)  # fetch the very first batch
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
            with self.ctx:
                logits, loss = self.model(X, Y)
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = self.data.get_train_batch(self.cfg.batch_size, device=self.device_type)
            # backward pass, with gradient scaling if training in fp16
            self.scaler.scale(loss).backward()
            # clip the gradient
            if self.cfg.grad_clip != 0.0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            # step the optimizer and scaler if training in fp16
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            self.optimizer.zero_grad(set_to_none=True)

            # timing and logging
            t1 = time.time()
            ms_iter = (t1 - t0) * 1000
            t0 = t1
            if self.iter_num % self.cfg.log_interval == 0:
                # get loss as float. note: this is a CPU-GPU sync point
                loss_float = loss.item()
                epoch = self.cfg.iter_to_epoch(self.iter_num)
                logging.info(
                    f"it {self.iter_num}: loss {loss_float:.4f}, dt {ms_iter:.2f}ms, ep {epoch:.2f}, lr {lr:.2e}"
                )
                wandb.log(
                    {
                        "train/loss": loss_float,
                        "train/lr": lr,
                        "sys/ms_iter": ms_iter,
                        "sys/epoch": epoch,
                    },
                    step=self.iter_num,
                )
            self.iter_num += 1
