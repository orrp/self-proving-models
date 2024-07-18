from dataclasses import dataclass, field

from spm import logging
from spm.data.tensor_repr import Meta

MODELS = {
    # original
    "openai-gpt": dict(n_layer=12, n_head=12, n_embd=768),  # 117M params
    "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
    "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
    "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
    "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
    "gopher": dict(n_layer=8, n_head=16, n_embd=512),  # 25.24M params somehow (originally called gopher 44m)
    "mini": dict(n_layer=6, n_head=6, n_embd=192),  # 2.68M params
    "micro": dict(n_layer=4, n_head=4, n_embd=128),  # 0.80M params
    "nano": dict(n_layer=3, n_head=3, n_embd=48),  # 0.09M params
    # custom
    "charton": dict(n_layer=4, n_head=8, n_embd=512),  # 12.63M params
}

DEFAULTS = {
    "seed": 0,
    "eval_interval": None,
    "log_interval": 1,
    "early_stop": 0.0,  # "None to disable"
    "save_iters": None,
    "batch_size": 64,
    "eval_batch_size": 256,
    "decay_lr": 0,
    "warmup_iters": 2000,
    "grad_clip": 1.0,
    "epochs": 1000,
    "compile": True,
    "learning_rate": 6e-4,
    "weight_decay": 0.1,
    "beta1": 0.9,
    "beta2": 0.95,
    "model": None,
    "dropout": 0.0,
    "bias": False,
    "n_layer": None,  # set from model name, or directly
    "n_head": None,
    "n_embd": None,
    "load_ckpt": None,
    "temperature": 1.0,
    "top_k": None,
    "rlvf": None,
    "data": None,
}


@dataclass
class TrainerConfig:
    seed: int
    device: str
    dtype = "float16"  # float16, float32, or bfloat16

    # eval and logging settings
    eval_interval: int | None  # None to disable evaluation
    log_interval: int  # 1
    save_iters: list[int] | None
    early_stop: float  # 0.0

    batch_size: int  # 64
    eval_batch_size: int  # 256

    epochs: int  # total number of training epochs
    max_iters: int = field(init=False)  # set from epochs post-init
    # learning rate decay, gradient clipping settings
    lr_decay_iters: int = field(init=False)  # set to be max_iters post-init
    decay_lr: int  # Factor by which to shrink lr by the end of training. 0 to disable
    min_lr: float = field(init=False)  # set to be learning_rate / decay_lr
    warmup_iters: int  # 2000  # how many steps to warm up for
    grad_clip: float  # 1.0  # clip gradients at this value, or disable if == 0.0

    compile: bool  # True  # use PyTorch 2.0 to compile the model to be faster

    # Dataset path
    data: str

    # Optimizer settings
    learning_rate: float  # 6e-4  # max learning rate
    weight_decay: float  # 1e-1
    beta1: float  # 0.9
    beta2: float  # 0.95

    # Model settings
    dropout: float  # 0.0
    bias: bool  # False  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    # architecture is set via model name, or directly.
    model: str | None
    n_layer: int | None
    n_head: int | None
    n_embd: int | None
    temperature: float  # 1.0
    top_k: int | None  # None

    # Checkpoint path
    load_ckpt: str | None

    rlvf: str | None

    # set from data meta
    block_size: int = field(init=False)
    vocab_size: int = field(init=False)

    def __post_init__(self):
        if self.load_ckpt is not None:
            self.data = self.load_ckpt.split("-")[1].split("_iter")[0]
            logging.info(f"Overriding dataset from checkpoint name {self.data}")
        elif self.data is None:
            raise ValueError("data or load_ckpt must be set")

        meta = Meta.load(self.data)
        self.block_size = meta.block_size
        self.vocab_size = meta.vocab_size
        self.num_samples = meta.num_samples
        self.max_iters = self.epoch_to_iter(self.epochs)
        self.lr_decay_iters = self.max_iters
        self.min_lr = self.learning_rate / self.decay_lr if self.decay_lr else self.learning_rate
        if self.model is not None:
            if not (self.n_layer is self.n_head is self.n_embd is None):
                raise ValueError("Set model or (n_layer, n_head, n_embd)")
            self.n_layer = MODELS[self.model]["n_layer"]
            self.n_head = MODELS[self.model]["n_head"]
            self.n_embd = MODELS[self.model]["n_embd"]
        elif self.n_layer is None or self.n_head is None or self.n_embd is None:
            raise ValueError("Set model or (n_layer, n_head, n_embd)")
        if self.save_iters is not None and -1 in self.save_iters:
            self.save_iters = [it for it in self.save_iters if it != -1] + [self.max_iters]

    @classmethod
    def from_defaults(cls, config: dict):
        # set device, data explicitly
        assert "device" in config, "device cannot be set from a default value"
        assert None not in config.values(), "You probably did not mean to set any values to None yourself."
        config = DEFAULTS | config
        return cls(**config)

    def to_run_name(self) -> str:
        model_name = self.model if self.model is not None else f"{self.n_layer}x{self.n_head}x{self.n_embd}"
        name = f"{model_name}-{self.data}"
        return f"{name}-rlvf" if self.rlvf else name

    def iter_to_epoch(self, it: int) -> float:
        return it * self.batch_size / self.num_samples

    def epoch_to_iter(self, epoch: float) -> int:
        return int(epoch * self.num_samples / self.batch_size)
