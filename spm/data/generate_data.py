import logging
import random
from argparse import ArgumentParser

import numpy as np

from spm.data.samplers import (
    AnnotatedTranscriptSampler,
    DisjointRejector,
    LogUniformGCDSampler,
    TranscriptSampler,
    UpperBoundRejector,
)
from spm.data.samples import AnnotatedTranscript
from spm.data.str_repr import EncodedSamples
from spm.data.tensor_repr import TargetComponent, TensorRepr


def save_samples(train_samples, val_samples, name_prefix, all_only=False):
    # encode samples
    train_encoded = EncodedSamples(train_samples, args.base)
    val_encoded = EncodedSamples(val_samples, args.base)
    # Compute name and val_cols
    n_samples_str = f"{len(train_samples):.0e}".replace("e+0", "e")
    ubound_str = f"{args.ubound:.0e}".replace("e+0", "e")
    name = f"{ubound_str}_m{n_samples_str}_b{args.base}"
    name = f"{name_prefix}_" + name if name_prefix else name
    # Save data
    tr = TensorRepr.from_samples(train_encoded, val_encoded, name)
    tr.save_train()
    val_cols = [TargetComponent.ANNOTATED_TRANSCRIPT]
    if not all_only:
        val_cols += [TargetComponent.TRANSCRIPT, TargetComponent.OUTPUT]
    tr.save_val(val_cols)


def set_seed(seed):
    print(f"Setting seed to {seed}")
    np.random.seed(seed)
    random.seed(seed)


def generate_baseline(args, lu_sampler):
    logging.info("Generating validation samples")
    set_seed(args.seed)
    val_samples = lu_sampler(args.n_val)
    logging.info("Generating training samples")
    set_seed(args.seed)
    train_samples = DisjointRejector(lu_sampler, val_samples)(args.n_train)
    save_samples(train_samples, val_samples, "Baseline", all_only=True)


def generate_transcripts(args, lu_sampler):
    sampler = UpperBoundRejector(TranscriptSampler(lu_sampler), args.rejector_ubound)
    logging.info("Generating validation samples")
    set_seed(args.seed)
    val_samples = sampler(args.n_val)
    logging.info("Generating training samples")
    set_seed(args.seed)
    train_samples = DisjointRejector(sampler, val_samples)(args.n_train)
    save_samples(train_samples, val_samples, "TL")


def generate_annotated_transcripts(args, lu_sampler):
    start_len, stop_len = args.annot_len_range
    annot_len = stop_len - 1
    sampler = UpperBoundRejector(AnnotatedTranscriptSampler(lu_sampler, annot_len=annot_len), args.rejector_ubound)
    logging.info("Generating validation samples")
    set_seed(args.seed)
    val_samples = sampler(args.n_val)
    logging.info("Generating training samples")
    set_seed(args.seed)
    train_samples: AnnotatedTranscript = DisjointRejector(sampler, val_samples)(args.n_train)
    while annot_len >= start_len - 1:
        name = f"ATL{annot_len}"
        save_samples(train_samples, val_samples, name)
        # trim the annotations by one
        train_samples.shorten_annot()
        val_samples.shorten_annot()
        annot_len -= 1


# if name is main
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--n_train", type=int, default=1024 * 100 * 100)
    parser.add_argument("--n_val", type=int, default=1000)
    parser.add_argument(
        "--ubound", type=int, default=10**4, help="Upper bound on the integers whose GCD is to be computed"
    )
    parser.add_argument(
        "--rejector_ubound", type=int, default=210**3, help="Upper bound on the any integer in the transcript"
    )
    parser.add_argument("--base", type=int, default=210, help="Base in which the integers are represented")
    parser.add_argument(
        "--annot_len_range",
        type=int,
        nargs=2,
        default=[3, 8],
        help="Two ints for the range (start and stop) of the length of the annotations",
    )
    parser.add_argument("--seed", type=int, default=10)
    args = parser.parse_args()
    lu_sampler = LogUniformGCDSampler(args.ubound, base=10)
    logging.info(f"Generating baseline with seed {args.seed}")
    generate_baseline(args, lu_sampler)
    logging.info(f"Generating transcripts with seed {args.seed}")
    generate_transcripts(args, lu_sampler)
    logging.info(f"Generating annotated transcripts with seed {args.seed}")
    generate_annotated_transcripts(args, lu_sampler)
