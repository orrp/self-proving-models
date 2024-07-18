import argparse
from pathlib import Path

import torch
from pandas import DataFrame

from spm.data.samplers import CSVSampler, TranscriptSampler, AnnotatedTranscriptSampler
from spm.data.str_repr import EncodedSamples
from spm.data.tensor_repr import TensorRepr
from spm.gpt.config import TrainerConfig
from spm.gpt.rlvf_trainer import TargetComponent
from spm.gpt.trainer import Trainer


@torch.no_grad()
def evaluate(model, tr: TensorRepr, device, batch_size) -> tuple[
    dict[tuple[int, int], list[int]], dict[tuple[int, int], list[int]]
]:
    """
    Evaluate the model, storing prediction per-input (slower than normal evaluation).
    Args:
        model: The model to evaluate.
        tr: The TensorRepr object to use for evaluation.
        device: The device to use.
        batch_size: The batch size to use.
    Returns:
        Two dictionaries, the first mapping input to prediction and the second mapping input to (correct) target.
    """
    input_to_target = {}
    input_to_prediction = {}
    for _, inp, masks, masked in tr.val_batches(device, batch_size):
        # If we need to keep track of predictions per-input, we need to do this based off of the
        # "ANNOTATED_TRANSCRIPT" component; this is because this component gives all target columns (so, somewhat
        # confusingly, it's the right component even if our model doesn't have annotations or even proofs...)'
        assert TargetComponent.ANNOTATED_TRANSCRIPT in masks
        output = model.generate(inp, max_new_tokens=tr.m.generation_size)
        output = output[:, inp.shape[1]:]  # trim input
        target = masked[TargetComponent.ANNOTATED_TRANSCRIPT]
        target_hat = output * masks[TargetComponent.ANNOTATED_TRANSCRIPT]
        for inp_, target_, target_hat_ in zip(inp, target, target_hat):
            inp_, target_, target_hat_ = inp_.cpu().numpy(), target_.cpu().numpy(), target_hat_.cpu().numpy()
            input_decoded = tr.input_arr_to_ints(inp_)
            input_to_target[input_decoded] = tr.target_arr_to_ints(target_)
            input_to_prediction[input_decoded] = tr.target_arr_to_ints(target_hat_)
    return input_to_prediction, input_to_target


def make_eval_data(csv_path: Path, base: int, self_proving: bool, annot_len: int) -> TensorRepr:
    """Convert CSV file with inputs to the GCD problem to a TensorRepr object."""
    num_samples = sum(1 for _ in open(csv_path)) - 1  # subtract header
    sampler = CSVSampler(csv_path)
    val_cols = [TargetComponent.OUTPUT]
    if self_proving:
        val_cols.extend([TargetComponent.TRANSCRIPT, TargetComponent.ANNOTATED_TRANSCRIPT])
        sampler = TranscriptSampler(sampler) if annot_len == 0 else AnnotatedTranscriptSampler(sampler, annot_len)
    samples = sampler.sample(num_samples)
    encoded_samples = EncodedSamples(samples, base)
    tr = TensorRepr.from_samples(encoded_samples, encoded_samples, name=csv_path.stem)
    tr.save_val(val_cols)
    return tr

def main(args):
    config = {
        "load_ckpt": args.ckpt,
        "device": args.device,
        "eval_batch_size": args.batch_size,
    }
    cfg = TrainerConfig.from_defaults(config)
    trainer = Trainer(cfg)
    model = trainer.model
    csv_path = Path(args.csv)
    tr = make_eval_data(csv_path, args.base, args.self_proving, args.annot_len)
    # create a dataframe with columns "input" and "c, c*" for each c in tr.m.target_labels
    # and number of rows equal to the number of inputs in the CSV file
    df = DataFrame(columns=["input"] + [f"{c},{c}*" for c in tr.m.target_labels], index=range(len(tr)))
    model.eval()
    model.to(args.device)
    input_to_prediction, input_to_target = evaluate(model, tr, args.device, args.batch_size)
    for i, inp, target, prediction in enumerate(zip(
            input_to_target.keys(), input_to_target.values(), input_to_prediction.values()
    )):
        df.loc[i]["input"] = inp
        for c, x, x_star in zip(tr.m.target_labels, prediction, target):
            df.loc[i][c] = x
            df.loc[i][f"{c}*"] = x_star
    df.to_csv(f"{csv_path.with_suffix('_results.csv')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ckpt", type=str, help="Load model from checkpoint (name)")
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--csv", type=str, help="Path to CSV file with inputs to the GCD problem")
    parser.add_argument("--base", type=int, help="Base of representation", default=10)
    parser.add_argument("--batch_size", type=int, help="Batch size for evaluation", default=32)
    parser.add_argument("--self-proving", action="store_true",
                        help="Generate proofs of correctness (works only if model was trained to be Self-Proving)")
    parser.add_argument("--annot_len", type=int,
                        help="Length of the annotation for proofs (should match the length of annotations "
                             "used during training).")

    args = parser.parse_args()
    main(args)
