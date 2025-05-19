# Self-Weighted Guidance (SWG)

Self-Weighted Guidance (SWG) paper repository.

Self-Weighted Guidance (SWG) is a novel guidance method in which both the data and its associated weights (see paper) are modeled by the same diffusion model. This unified modeling enables a self-guided sampling process, where the diffusion model effectively guides itself during generation. 

This repository is built using JAX/Flax, WandB for logging and Hydra for managing hyperparameters.
See ... for a PyTorch implementation with toy examples.

## Installation

We recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to manage dependencies.

Installation of MuJoCo is required. Follow this instructions https://gist.github.com/saratrajput/60b1310fe9d9df664f9983b38b50d5da

### 1. Clone the Repository

```bash
git clone https://github.com/atagle123/SWG-JAX.git
cd SWG-JAX
```

### 2. Create and Activate a Conda Environment

```bash
conda env create -f environment.yml
conda activate swg_jax
conda develop .
```

## Usability

To modify the main hyperparameters for experiments, edit the following file:

```bash
configs/D4RL/config.yaml
```

## Training

```bash
python scripts/train.py datasets={dataset_name} method=swg seed={your_seed}
```

## Evaluating

# change the evaluation parameters in scripts/evaluate.py and run

```bash
python scripts/evaluate.py
```

This repository builds upon jaxrl https://github.com/ikostrikov/jaxrl and IDQL https://github.com/philippe-eecs/IDQL codebase. 