import os
import gym
import jax
import jax.numpy as jnp
from flax.core import frozen_dict
from omegaconf import OmegaConf
from tqdm import tqdm
import wandb

import d4rl.gym_mujoco
import d4rl.hand_manipulation_suite

from src.agent import DDPMSWGLearner
from src.jaxrl5.data import D4RLDataset, WeightedDataset
from src.jaxrl5.wrappers import wrap_gym
from src.utils.evaluation import evaluate_all_variants


@jax.jit
def merge_batch(batch1, batch2):
    return frozen_dict.freeze(
        {k: jnp.concatenate([batch1[k], batch2[k]], axis=0) for k in batch1}
    )


class Trainer:
    def __init__(self, cfg, training_cfg):
        self.cfg = cfg
        self.training_cfg = training_cfg
        self.num_episodes = 20

        if training_cfg.wandb_log:
            wandb.init(
                project="SWG_JAX",
                name=cfg.wandb.wandb_exp_name,
                config=OmegaConf.to_container(cfg, resolve=True),
            )

        os.makedirs(cfg.savepath.actor_savepath, exist_ok=True)
        os.makedirs(cfg.savepath.critic_savepath, exist_ok=True)

    def train(self):
        env = gym.make(self.cfg.dataset.env_entry)
        self.env = env
        dataset, dataset_val = self._load_dataset(env)
        self.envs_for_eval = [
            wrap_gym(gym.make(self.cfg.dataset.env_entry))
            for _ in range(self.num_episodes)
        ]

        model_cls = self.cfg.method.agent._target_
        agent = globals()[model_cls].create(
            self.cfg.seed, env.observation_space, env.action_space, self.cfg.agent
        )

        trained_agent = self._train_loop(agent, dataset, dataset_val)
        wandb.finish()
        return trained_agent

    def _load_dataset(self, env):
        dataset = D4RLDataset(env)

        if "antmaze" in self.cfg.dataset.env_entry:
            dataset.dataset_dict["rewards"] -= 1.0
        elif any(
            name in self.cfg.dataset.env_entry
            for name in ("halfcheetah", "walker2d", "hopper")
        ):
            dataset.normalize_returns()

        return dataset.split(0.95) if self.training_cfg.val_dataset else (dataset, None)

    def _log_info(self, info: dict[str, float], step: int, prefix: str) -> None:
        info_str = " | ".join([f"{prefix}/{k}: {v:.4f}" for k, v in info.items()])
        print(f"{info_str} | (step {step})")

        if self.training_cfg.wandb_log:
            wandb.log({f"{prefix}/{k}": v for k, v in info.items()}, step=step)

    def _train_loop(self, agent, dataset, dataset_val):
        keys = None
        critic_cfg = self.cfg.agent.train

        # Critic training
        for step in tqdm(range(1, critic_cfg.critic_steps + 1), smoothing=0.1):
            sample = dataset.sample_jax(critic_cfg.critic_batch_size, keys=keys)
            agent, info = agent.critic_update(sample)

            if step % self.training_cfg.log_freq == 0:
                self._log_info(info, step, prefix="train")

                if dataset_val:
                    val_sample = dataset_val.sample(
                        critic_cfg.critic_batch_size, keys=keys
                    )
                    _, val_info = agent.critic_update(val_sample)
                    self._log_info(val_info, step, prefix="val")

            if step % self.training_cfg.save_freq == 0 or step == 1:
                agent.save_critic(self.cfg.savepath.critic_savepath, step)

        # Build weights dataset
        weights_dataset = WeightedDataset(self.env)
        weights_dataset.build_weights(
            agent,
            critic_hyperparam=self.cfg.agent.train.critic_hyperparam,
            weights_function=self.cfg.agent.weight_build.weight_function,
            norm=self.cfg.agent.weight_build.norm,
        )

        weights_dataset, weights_val = (
            weights_dataset.split(0.95)
            if self.training_cfg.val_dataset
            else (weights_dataset, None)
        )

        # Actor training
        global_step = critic_cfg.critic_steps
        for step in tqdm(range(1, critic_cfg.actor_steps + 1), smoothing=0.1):
            global_step += 1
            sample = weights_dataset.sample_jax(critic_cfg.actor_batch_size, keys=keys)
            agent, info = agent.actor_update(sample)

            if step % self.training_cfg.log_freq == 0:
                self._log_info(info, global_step, prefix="train")

                if weights_val:
                    val_sample = weights_val.sample(
                        critic_cfg.actor_batch_size, keys=keys
                    )
                    _, val_info = agent.actor_update(val_sample)
                    self._log_info(val_info, global_step, prefix="val")

            if step % self.training_cfg.eval_freq == 0 and step > 1:
                eval_info = evaluate_all_variants(
                    agent,
                    self.envs_for_eval,
                    self.cfg.dataset.env_entry,
                    self.training_cfg.sample_params,
                    self.training_cfg.sample_variant,
                    num_episodes=self.num_episodes,
                )
                self._log_info(eval_info, global_step, prefix="eval")

            if step % self.training_cfg.save_freq == 0 or step == 1:
                agent.save_actor(self.cfg.savepath.actor_savepath, step)

        return agent
