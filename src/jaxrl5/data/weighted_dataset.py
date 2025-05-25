import gym
import d4rl
import d4rl.gym_mujoco
import jax
import jax.numpy as jnp
import numpy as np

from functools import partial
from tqdm import tqdm

from src.jaxrl5.data.dataset import Dataset
from src.models.weights import build_weights_fn


@partial(jax.jit, static_argnames=("critic_fn",))
def compute_q(critic_fn, critic_params, observations, actions):
    q_values = critic_fn({"params": critic_params}, observations, actions)
    return q_values.min(axis=0)


@partial(jax.jit, static_argnames=("value_fn",))
def compute_v(value_fn, value_params, observations):
    return value_fn({"params": value_params}, observations)


class WeightedDataset(Dataset):
    def __init__(self, env: gym.Env, clip_to_eps: bool = True, eps: float = 1e-5):

        dataset_dict = d4rl.qlearning_dataset(env)

        if clip_to_eps:
            lim = 1 - eps
            dataset_dict["actions"] = np.clip(dataset_dict["actions"], -lim, lim)

        dones = np.full_like(dataset_dict["rewards"], False, dtype=bool)

        for i in range(len(dones) - 1):
            if (
                np.linalg.norm(
                    dataset_dict["observations"][i + 1]
                    - dataset_dict["next_observations"][i]
                )
                > 1e-6
                or dataset_dict["terminals"][i] == 1.0
            ):
                dones[i] = True

        dones[-1] = True

        dataset_dict["masks"] = 1.0 - dataset_dict["terminals"]
        del dataset_dict["terminals"]

        for k, v in dataset_dict.items():
            dataset_dict[k] = v.astype(np.float32)

        dataset_dict["dones"] = dones

        super().__init__(dataset_dict)

    def build_weights(
        self,
        agent,
        critic_hyperparam: float = 0.7,
        weights_function: str = "expectile",
        norm: bool = False,
    ):
        """
        Build weights for the dataset using JAX.
        """
        states = self.dataset_dict["observations"]
        actions = self.dataset_dict["actions"]

        # Split states and actions into batches
        batch_size = 256
        num_batches = states.shape[0] // batch_size + 1
        states_batches = jnp.array_split(states, num_batches)
        actions_batches = jnp.array_split(actions, num_batches)

        weights_list = []

        for states_batch, actions_batch in tqdm(
            zip(states_batches, actions_batches),
            total=num_batches,
            desc="Computing weights",
        ):

            states_batch = jax.device_put(states_batch)
            actions_batch = jax.device_put(actions_batch)

            qs = compute_q(
                agent.target_critic.apply_fn,
                agent.target_critic.params,
                states_batch,
                actions_batch,
            )
            vs = compute_v(agent.value.apply_fn, agent.value.params, states_batch)

            weights = build_weights_fn(
                qs, vs, weight_fn=weights_function, hyperparam=critic_hyperparam
            )

            weights_list.append(weights)

        weights_tensor = jnp.concatenate(weights_list, axis=0)

        min_weight = jnp.min(weights_tensor)
        max_weight = jnp.max(weights_tensor)
        mean_weight = jnp.mean(weights_tensor)
        std_weights = jnp.std(weights_tensor)

        print(
            f"min weight: {min_weight} | max weight: {max_weight} | mean weight: {mean_weight} | std: {std_weights}"
        )

        if norm:
            weights_tensor = (weights_tensor - min_weight) / (max_weight - min_weight)

        weights_tensor = jnp.nan_to_num(weights_tensor, nan=min_weight)

        weights_tensor_np = jax.device_get(
            weights_tensor
        )  # Convert from JAX array to NumPy
        weights_tensor_np = weights_tensor_np[..., None]

        assert weights_tensor_np.shape == (len(self.dataset_dict["actions"]), 1)

        self.dataset_dict = {
            "action_weight": np.append(
                self.dataset_dict["actions"], weights_tensor_np, axis=-1
            ),
            "observations": self.dataset_dict["observations"],
        }
