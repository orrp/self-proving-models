import argparse
from argparse import _ArgumentGroup as ArgumentGroup

import wandb

from spm import WANDB_DIR
from spm.gpt.config import DEFAULTS, MODELS, TrainerConfig
from spm.gpt.trainer import Trainer


def main(config_args, wandb_args):
    set_args = {k: v for k, v in config_args.items() if v is not None}
    cfg = TrainerConfig.from_defaults(set_args)
    # configure wandb
    wandb_mode = "online" if wandb_args["wandb"] else "disabled"
    wandb.init(project=wandb_args["wandb_proj"], mode=wandb_mode, dir=WANDB_DIR, name=cfg.to_run_name())
    trainer = Trainer(cfg)
    trainer.run()


def set_defaults(group: ArgumentGroup):
    for action in group._group_actions:
        if not action.required:
            action.default = DEFAULTS[action.dest]


def make_parser() -> tuple[argparse.ArgumentParser, ArgumentGroup, ArgumentGroup]:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    config_group = parser.add_argument_group("TrainerConfig")
    # parse args for TrainerConfig
    config_group.add_argument("--load_ckpt", type=str, help="Load model from checkpoint (name)")
    config_group.add_argument("--device", type=str, required=True)
    config_group.add_argument("--epochs", type=int, required=True)
    config_group.add_argument("--data", "-d", type=str, help="Dataset name (ignored if --load_ckpt)")
    config_group.add_argument("--seed", type=int, help="Random seed", default=999)
    config_group.add_argument(
        "--eval_interval",
        type=int,
        help="Every how many iterations to evaluate. None to disable",
    )
    config_group.add_argument("--log_interval", type=int, help="Every how many iterations to log")
    config_group.add_argument(
        "--save_iters",
        type=int,
        nargs="+",
        help="Save model at these iterations. -1 For the last iteration. None to disable",
    )
    config_group.add_argument("--early_stop", type=float, help="Early stopping threshold", default=0.0)

    config_group.add_argument(
        "--decay_lr", type=int, help="Factor by which to shrink lr by the end of training. 0 to disable"
    )
    config_group.add_argument("--warmup_iters", type=int, help="How many steps to warm up for")
    config_group.add_argument(
        "--grad_clip",
        type=float,
        help="Clip gradients at this value, or disable if == 0.0",
    )
    config_group.add_argument("--batch_size", type=int, help="Training batch size")
    config_group.add_argument("--learning_rate", type=float, help="Max learning rate")
    config_group.add_argument("--weight_decay", type=float, help="Weight decay")
    config_group.add_argument("--beta1", type=float, help="Adam beta1")
    config_group.add_argument("--beta2", type=float, help="Adam beta2")
    config_group.add_argument("--dropout", type=float, help="Dropout probability")
    # Model architecture can be specified by name or by individual parameters
    config_group.add_argument("--model", type=str, choices=list(MODELS.keys()))
    config_group.add_argument("--n_layer", type=int, help="Number of layers")
    config_group.add_argument("--n_head", type=int, help="Number of heads")
    config_group.add_argument("--n_embd", type=int, help="Embedding dimension")

    # Eval settings
    config_group.add_argument(
        "--eval_batch_size",
        type=int,
        help="Evaluation batch size (larger for faster eval)",
    )
    config_group.add_argument("--temperature", type=float, help="Temperature for RLVF sampling")
    config_group.add_argument("--top_k", type=int, help="Top k for sampling")

    set_defaults(config_group)

    # wandb settings
    wandb_group = parser.add_argument_group("wandb")
    wandb_group.add_argument("--wandb", action="store_true", help="Enable tracking with WandB")
    wandb_group.add_argument("--wandb_proj", type=str, help="WandB project name", default="proj")
    return parser, config_group, wandb_group


def group_args(args: argparse.Namespace, group: ArgumentGroup) -> dict:
    return {a.dest: getattr(args, a.dest, None) for a in group._group_actions}


if __name__ == "__main__":
    parser, config_group, wandb_group = make_parser()
    args = parser.parse_args()
    config_args = group_args(args, config_group)
    wandb_args = group_args(args, wandb_group)
    main(config_args, wandb_args)
