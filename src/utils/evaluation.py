import numpy as np
import gym
import itertools
import jax
import jax.numpy as jnp
from d4rl import get_normalized_score
from src.jaxrl5.wrappers import wrap_gym


def evaluate(agent, env: gym.Env, num_episodes: int, sample_params: dict) -> dict[str, float]:
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=num_episodes)

    for _ in range(num_episodes):
        observation, done = env.reset(), False
        while not done:
            action, new_rng = agent.eval_actions(observation, sample_params)
            agent.replace(rng=new_rng)
            observation, _, done, _ = env.step(action)

    return env.get_normalized_score(np.mean(env.return_queue)) * 100.0


def evaluate_parallel(agent, envs, env_entry: str, num_episodes: int, sample_params: dict, seed=42) -> dict[str, float]:

    observations = np.array([env.reset() for  env in envs])
    dones = np.zeros(num_episodes, dtype=bool)
    episode_returns = np.zeros(num_episodes)

    rngs = jax.random.split(agent.rng, num_episodes)
    # Iterate over environment steps
    while not jnp.all(dones):
        vectorized_eval_actions = jax.vmap(agent.eval_actions, in_axes=(0,0, None))
        actions, rngs = vectorized_eval_actions(observations, rngs, sample_params)
        actions = np.array(actions.squeeze())
        
        # Collect rewards and update states
        next_observations = []
        rewards = []
        next_dones = []

        for i, (env, done) in enumerate(zip(envs, dones)):
            observation = observations[i]
            if not done:
                action = actions[i]
                observation, reward, done, _ = env.step(action)
                next_observations.append(observation)
                rewards.append(reward)
                next_dones.append(done)

            else:
                # If the episode is done, we set the reward to 0 and continue with the final state
                next_observations.append(observation)
                rewards.append(0.0)
                next_dones.append(True)

        # Update the states for each environment
        observations = np.array(next_observations)
        dones = np.array(next_dones)
        episode_returns += np.array(rewards)

    scores = get_normalized_score(env_name=env_entry,score=episode_returns)*100 
    scores_mean = np.mean(scores)

    return scores_mean
    

def convert_sample_variants_format(sample_variant: dict) -> list:
    """
    Convert sample variants into a list of dictionaries with all combinations of parameters.
    """
    return [
        dict(zip(sample_variant.keys(), values))
        for values in itertools.product(*sample_variant.values())
    ]


def evaluate_all_variants(agent, envs, env_entry: str, base_sample_params: dict, sample_variant: dict, num_episodes: int) -> dict[str, float]:
    """
    Evaluate the agent for all parameter variants.
    """
    exp_result_dict={}
    params_variant_list = convert_sample_variants_format(sample_variant)
    # Evaluate for each parameter variant
    for param_variants in params_variant_list:
        sample_params = base_sample_params.copy()
        sample_params.update(param_variants)

        print(f"Evaluating with sample params: {sample_params}")
        eval_info = evaluate_parallel(
            agent, envs, env_entry, num_episodes=num_episodes, sample_params=sample_params
        )

        # Create a unique key for the parameter combination
        param_key = "_".join([f"param_{k}_{v}" for k, v in param_variants.items()])
        exp_result_dict[param_key] = eval_info

    return exp_result_dict