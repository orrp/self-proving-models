[![arXiv](https://img.shields.io/badge/arXiv-2405.15722-pink.svg)](https://arxiv.org/abs/2405.15722)
[![Models](https://img.shields.io/badge/Models-10.5281/zenodo.12752192-blue.svg)](https://zenodo.org/records/12752192)
[![Data](https://img.shields.io/badge/Data-10.5281/zenodo.12751514-blue.svg)](https://zenodo.org/records/12751514)
# Self-Proving Models

Self-Proving Models prove the correctness of their outputs to a verifier using an Interactive Proof System.
This repository includes tools to train these models, specifically for the Greatest Common Divisor (GCD) problem,
based on the theory described in our [paper](https://arxiv.org/abs/2405.15722).

This repository provides:
- A straightforward framework for training a Self-Proving GPT.
- Data generation scripts for the GCD problem.
- Scripts for reproducing experiments from the [paper](https://arxiv.org/abs/2405.15722).


## Setup

1. Create a new conda environment (recommended):
```bash
conda create -n spm python=3.12
conda activate spm
```
2. Clone the repository:
```bash
git clone https://github.com/orrp/self-proving-models.git 
```
3. Install the package:
```bash
cd self-proving-models
pip install -e .
```

## Data
You can download pre-generated [![Data](https://img.shields.io/badge/Data-blue.svg)](https://zenodo.org/records/12751514), and extract it to the `data/` directory.
To generate this data yourself, run
```bash
python self-proving-models/data/generate_data.py
```

This populates the `data/` directory with Transcripts and Annotated Transcripts
of interactions between an honest prover and the verifier.

Transcript datasets are named according to the following convention:
```
TL_{UPPER_BOUND}_m{NUM_TRAIN_SAMPLES}_b{BASE_OF_REPRESENTATION}
```
For Annotated Transcripts, `TL` is replaced with `ATL{ANNOTATION_LENGTH}`.

## Training
Once `data/` is populated, you can train a Self-Proving GPT via Transcript Learning:
```bash
python self-proving-models/train.py --data DATASET_NAME
```
where `DATASET_NAME` is the name of the dataset you want to use.
### Example
To train on about 10 million samples with an upper bound of 10,000 encoded in base 210:
```bash
python self-proving-models/train.py --data TL_1e4_m1e7_b210
```
### Useful arguments
- `--help`: Show all arguments.
- `--device DEVICE`: Specify the device to train on (`cpu` or `cuda`).
- `--epochs EPOCHS`: Number of epochs to train. Each epoch looks at a number of samples equal to the dataset size.
- `--data DATA, -d DATA`: Name of the dataset to use.
- `--seed SEED`: Set random seed
- `--save_iters SAVE_ITERS [SAVE_ITERS ...]`:
    Save model at these iterations. -1 For the last iteration. None to disable.
    Checkpoints are saved to `models/` as:
`{N_LAYERS}x{N_HEAD}x{DIM_EMBED}x{N_ITERATIONS}_{DATASET_NAME}_iter{ITERATION}.pt`
- `--load_ckpt LOAD_CKPT`: Load model from checkpoint (name).
 When you load a model, specify the checkpont name as described above (not the full path).
- `--wandb`: Enable tracking with [Weights & Biases](https://wandb.ai/).
Use `--wandb_proj WANDB_PROJ` to specify the project name.

## Reproducing results from the paper
Once you [obtain data](#data), you can train models on the datasets to reproduce the experimental
section of the [paper](https://arxiv.org/abs/2405.15722).

### Annotation length
To reproduce the ablation on the annotation length, run
```bash
./runs/annot_len.sh
```
Logs of these runs will be saved at `logs/`. The Figure 2 can be generated with
```bash
./figs/annotation.py        # Fig 2
```

### Base of representation
The [paper](https://arxiv.org/abs/2405.15722) shows that the number of unique primes in the base of representation
determines Verifiability of the model. This ablation requires generating many different datasets (one for each base).
For convenience, there is a script that first samples a random base with a given number of unique primes in its
factorization, then trains a model and deletes the dataset.
```bash
./self-proving-models/train_diff_bases.py --num_unique_primes NUM_UNIQUE_PRIMES --seed SEED
```

Run this script for twenty seeds. *Tip: You can use a [WandB sweep](https://docs.wandb.ai/guides/sweeps)
to easily  schedule runs from different machines, and then aggregate the results onto your local machine
for plotting.
Use `wandb sweep runs/diff_bases.yaml` to create the sweep.*

Each run will be logged to `logs/diff_bases/`.
You can then generate the  figure with
```bash
./figs/diff_bases.py  # Fig 3
```

## Citation
If you use Self-Proving Models or components of this codebase in your research, please cite the following paper:
```latex
@article{AGPR2024,
  author       = {Noga Amit and
                  Shafi Goldwasser and
                  Orr Paradise and
                  Guy N. Rothblum},
  title        = {Models That Prove Their Own Correctness},
  journal      = {CoRR},
  volume       = {abs/2405.15722},
  year         = {2024},
  url          = {https://doi.org/10.48550/arXiv.2405.15722},
  doi          = {10.48550/ARXIV.2405.15722},
  eprinttype    = {arXiv},
  eprint       = {2405.15722},
  timestamp    = {Wed, 19 Jun 2024 08:52:57 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2405-15722.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

## Acknowledgements
This codebase adapts Andrej Karpathy's [nanoGPT](https://www.github.com/karpathy/nanoGPT) as its GPT implementation.
The model can be found in `self-proving-models/gpt/`. Cassidy Laidlaw's
[boilerplate](https://github.com/cassidylaidlaw/python-boilerplate)
was used for repo structure and linting (`self-proving-models/lint.sh`).

## Contribution
Contributions are welcome! Please open an [issue](https://www.github.com/orrp/self-proving-models/issues)
or a [pull request](https://www.github.com/orrp/self-proving-models/pulls).

To install the package in development mode, run
```bash
pip install -e '.[dev]'
ln -s post-commit.sh .git/hooks/post-commit
```
The last command sets up a git hook to copy the git hash in `spm/__init__.py`. This hash is
then logged by WandB for reproducibility. This is useful when experiments are run from a server to which code is
deployed from a local machine (not via git).

### Package structure

**Root `spm/`**
- `train.py`: Main entry point for training models.
- `train_diff_bases.py`: Alternative entry point for training models on datasets with different bases of representation.
                         Generates and cleans up datasets automatically.              
- `utils.py`: Common utilities (e.g. implementation of the extended Euclidean algorithm).
- `__init__.py`: Common paths and constants.
- `systematic_models.py`: Systematic models for the GCD problem, useful for testing.

**Data `spm/data/`**
- `generate_data.py`: Generates datasets for training.
- `samples.py`: The `Samples` represents a dataset of input-output sequences to the GCD problem.
                `Transcripts` add a proof (Bézout coefficients) to the samples.
                `AnnotatedTranscripts` add a proof and its annotation.
- `samplers.py`: Samplers for generating `Samples`, `Transcripts`, and `AnnotatedTranscripts`.
- `str_repr.py`: A string representation of data samples (encoded in a given base).
- `tensor_repr.py`: A tensor representation of data samples. Uses `str_repr` to encode samples as strings, and
                    handles delimiters and padding. Contains utility methods for encoding, decoding, and saving.

**Model `spm/gpt/`**
- `model.py`: Defines the GPT model architecture. An object-oriented adaptation of [nanoGPT](https://www.github.com/karpathy/nanoGPT).
- `trainer.py`: Trainer for GPT models. Handles training, evaluation, and logging.
- `config.py`: Config for a trainer.
- `rlvf_trainer.py`: Specialized trainer for RLVF models. (Work in progress)
