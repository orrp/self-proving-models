import argparse
import random

import numpy.random
import wandb
from sympy import primefactors

from spm import ANALYSIS_DIR, WANDB_DIR, logging
from spm.data.samplers import (
    DisjointRejector,
    LogUniformGCDSampler,
    Sampler,
    TranscriptSampler,
)
from spm.data.str_repr import EncodedSamples, StrRepr
from spm.data.tensor_repr import TargetComponent, TensorRepr
from spm.gpt.config import TrainerConfig
from spm.gpt.trainer import Trainer, evaluate

N_TRAIN = 1024000
N_VAL = 1000
UBOUND = 10000
VAL_COLS = [TargetComponent.ANNOTATED_TRANSCRIPT, TargetComponent.OUTPUT]

SCORES_DIR = ANALYSIS_DIR / "diffbases_scores"
CONFIG_ARGS = {
    "eval_interval": None,
    "log_interval": 10000,
    "save_best": None,
    "save_dir": None,
    "batch_size": 1024,
    "eval_batch_size": 256,
    "decay_lr": 1,
    "warmup_iters": 0,
    "grad_clip": 1.0,
    "epochs": 20,  # 100000 iters
    "learning_rate": 0.0003604137442311376,
    "weight_decay": 0.1,
    "beta1": 0.7502541703996637,
    "beta2": 0.95,
    "n_layer": 8,
    "n_head": 2,
    "n_embd": 256,
    "dropout": 0.0,
    "bias": False,
    "device": "cuda",
    "compile": True,
}

WANDB_PROJ = "self-proving-models_diffbases_ep20"


def save_score(base, seed, score, k_score):
    SCORES_DIR.mkdir(exist_ok=True)
    with open(SCORES_DIR / f"b{base}_s{seed}.txt", "w") as f:
        f.write(f"{score}\n{k_score}")


def make_data(base, seed, n_train) -> TensorRepr:
    base_sampler = LogUniformGCDSampler(ubound=UBOUND, base=base)
    sampler: Sampler = TranscriptSampler(base_sampler)
    val_samples = sampler(N_VAL)
    train_samples = DisjointRejector(sampler, val_samples)(n_train)
    # encode samples
    train_encoded = EncodedSamples(train_samples, base)
    val_encoded = EncodedSamples(val_samples, base)
    # Save data
    tr = TensorRepr.from_samples(train_encoded, val_encoded, to_tr_name(base, seed))
    tr.save_train()
    tr.save_val(VAL_COLS)
    return tr


def to_tr_name(base, seed):
    return f"diffbases_b{base}_s{seed}"


def sample_base(num_unique_primes, seed):
    """Sample a base with the given number of unique prime factors."""
    possible_bases = [
        base for base in range(2, StrRepr.MAX_REPR_BASE + 1) if len(primefactors(base)) == num_unique_primes
    ]
    random.seed(seed)
    return random.choice(possible_bases)


def main(args):
    seed = args.seed
    base = sample_base(args.num_unique_primes, seed)
    # set seed
    random.seed(seed)
    numpy.random.seed(seed)
    # generate data
    logging.info(f"Generating data for base {base} seed {seed}...")
    make_data(base, seed, N_TRAIN)
    # configure
    config_args = CONFIG_ARGS.copy()
    config_args["seed"] = seed
    config_args["data"] = to_tr_name(base, seed)
    set_args = {k: v for k, v in config_args.items() if v is not None}
    cfg = TrainerConfig.from_defaults(set_args)
    wandb.init(
        project=WANDB_PROJ,
        mode="online",
        dir=WANDB_DIR,
        name=f"diffbases_b{base}",
    )
    # train
    trainer = Trainer(cfg)
    trainer.run()
    # evaluate
    logging.info(f"Base {base} done training. Evaluating...")
    scores = evaluate(
        trainer.model,
        trainer.data,
        device=trainer.cfg.device,
        batch_size=trainer.cfg.eval_batch_size,
    )
    total_score = scores["total"][TargetComponent.ANNOTATED_TRANSCRIPT]
    k_score = scores["total"][TargetComponent.OUTPUT]
    wandb.log({"total/all": total_score, "total/k": k_score})
    save_score(base, config_args["seed"], total_score, k_score)
    logging.info(f"Base {base} scored {total_score},{k_score}. Cleaning up data...")
    trainer.data.m.delete()
    logging.info(f"Base {base} done.")


if __name__ == "__main__":
    # parse base and seed
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_unique_primes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        N_TRAIN = 10240
        WANDB_PROJ = "self-proving-models_diffbases_ep20"
        CONFIG_ARGS["n_layer"] = 1
        CONFIG_ARGS["n_head"] = 1
        CONFIG_ARGS["n_embd"] = 32
    main(args)
