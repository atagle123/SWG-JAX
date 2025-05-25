import os
import jax
import jax.numpy as jnp
import numpy as np
import flax.serialization
from flax import struct
from flax.training.train_state import TrainState
from functools import partial
from src.jaxrl5.types import PRNGKey


def count_params(params):
    return sum(jnp.prod(jnp.array(v.shape)) for v in jax.tree_util.tree_leaves(params))


@partial(jax.jit, static_argnames="apply_fn")
def _sample_actions(rng, apply_fn, params, observations: np.ndarray) -> np.ndarray:
    key, rng = jax.random.split(rng)
    dist = apply_fn({"params": params}, observations)
    return dist.sample(seed=key), rng


@partial(jax.jit, static_argnames="apply_fn")
def _eval_actions(apply_fn, params, observations: np.ndarray) -> np.ndarray:
    dist = apply_fn({"params": params}, observations)
    return dist.mode()


@partial(jax.jit, static_argnames=("critic_fn"))
def compute_q(critic_fn, critic_params, observations, actions):
    q_values = critic_fn({"params": critic_params}, observations, actions)
    q_values = q_values.min(axis=0)
    return q_values


@partial(jax.jit, static_argnames=("value_fn"))
def compute_v(value_fn, value_params, observations):
    v_values = value_fn({"params": value_params}, observations)
    return v_values


class Agent(struct.PyTreeNode):
    actor: TrainState
    rng: PRNGKey

    def eval_actions(self, observations: np.ndarray) -> np.ndarray:
        actions = _eval_actions(self.actor.apply_fn, self.actor.params, observations)
        return np.asarray(actions), self.replace(rng=self.rng)

    def sample_actions(self, observations: np.ndarray) -> np.ndarray:
        actions, new_rng = _sample_actions(
            self.rng, self.actor.apply_fn, self.actor.params, observations
        )
        return np.asarray(actions), self.replace(rng=new_rng)

    def save_critic(self, savepath, step):

        critic_path = os.path.join(savepath, f"critic_params_{step}.msgpack")
        with open(critic_path, "wb") as f:
            f.write(flax.serialization.to_bytes(self.target_critic.params))

            # Save value parameters
        value_path = os.path.join(savepath, f"value_params_{step}.msgpack")
        with open(value_path, "wb") as f:
            f.write(flax.serialization.to_bytes(self.value.params))

    def save_actor(self, savepath, step):

        actor_path = os.path.join(savepath, f"actor_params_{step}.msgpack")
        with open(actor_path, "wb") as f:
            f.write(flax.serialization.to_bytes(self.target_score_model.params))

    @classmethod
    def load(cls, cfg, env, actor_dir: str, critic_dir: str, actor_step, critic_step):
        """Load the parameters of the critic, value, and actor models."""
        # Create a new instance of the learner
        agent = cls.create(
            seed=cfg.seed,
            observation_space=env.observation_space,
            action_space=env.action_space,
            cfg=cfg.agent,
        )

        # Load critic parameters
        critic_path = os.path.join(critic_dir, f"critic_params_{critic_step}.msgpack")
        with open(critic_path, "rb") as f:
            critic_params = flax.serialization.from_bytes(
                agent.target_critic.params, f.read()
            )
        agent = agent.replace(
            target_critic=agent.target_critic.replace(params=critic_params)
        )

        # Load value parameters
        value_path = os.path.join(critic_dir, f"value_params_{critic_step}.msgpack")
        with open(value_path, "rb") as f:
            value_params = flax.serialization.from_bytes(agent.value.params, f.read())
        agent = agent.replace(value=agent.value.replace(params=value_params))

        # Load actor (score model) parameters
        actor_path = os.path.join(actor_dir, f"actor_params_{actor_step}.msgpack")
        with open(actor_path, "rb") as f:
            actor_params = flax.serialization.from_bytes(
                agent.target_score_model.params, f.read()
            )
        agent = agent.replace(
            target_score_model=agent.target_score_model.replace(params=actor_params)
        )

        print(f"Models loaded from {actor_dir} and {critic_dir}")
        return agent
