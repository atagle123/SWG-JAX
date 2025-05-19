import os
import logging
from dataclasses import dataclass

import hydra
from omegaconf import DictConfig

from src.utils.training import Trainer


# ---------------------------- #
#     Constants & Defaults     #
# ---------------------------- #

@dataclass(frozen=True)
class Defaults:
    config_path: str = "../configs/D4RL"
    memory_fraction: str = "0.7"
    preallocate_memory: str = "true"
    wandb_log: bool = True
    val_dataset: bool = True
    log_freq: int = 10000
    save_freq: int = 100000
    eval_freq: int = 200000
    sample_params: dict[str, object] = None
    sample_variant: dict[str, list[int]] = None


DEFAULTS = Defaults(
    sample_params={
        "temperature": 0,
        "guidance_scale": 0,
        "clip_denoised": True,
        "batch_size": 1,
        "max_weight_clip": 1.0,
        "min_weight_clip": 0.0001,
    },
    sample_variant={"guidance_scale": [0, 1]}
)


# ---------------------------- #
#       Training Config        #
# ---------------------------- #

@dataclass
class TrainingConfig:
    log_freq: int
    save_freq: int
    eval_freq: int
    wandb_log: bool
    val_dataset: bool
    sample_params: dict[str, object]
    sample_variant: dict[str, list[int]]


# ---------------------------- #
#     Utility Functions        #
# ---------------------------- #

def configure_environment(memory_fraction: str, preallocate: str) -> None:
    """
    Set up memory configuration for JAX runtime.
    """
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = memory_fraction
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = preallocate
    logger.info("JAX environment configured: memory_fraction=%s, preallocate=%s", memory_fraction, preallocate)


def build_training_config(defaults: Defaults) -> TrainingConfig:
    """
    Build a training config object from the defaults.
    """
    return TrainingConfig(
        log_freq=defaults.log_freq,
        save_freq=defaults.save_freq,
        eval_freq=defaults.eval_freq,
        wandb_log=defaults.wandb_log,
        val_dataset=defaults.val_dataset,
        sample_params=defaults.sample_params,
        sample_variant=defaults.sample_variant,
    )


# ---------------------------- #
#        Main Script           #
# ---------------------------- #

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(config_path=DEFAULTS.config_path, config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for training execution.
    """
    logger.info("Starting experiment with configuration: %s", cfg)

    configure_environment(DEFAULTS.memory_fraction, DEFAULTS.preallocate_memory)
    training_config = build_training_config(DEFAULTS)

    trainer = Trainer(cfg, training_config)
    trainer.train()

    logger.info("Training completed successfully.")


if __name__ == "__main__":
    main()
