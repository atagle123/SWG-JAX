import os
import json
import logging
from pathlib import Path

import hydra
import gym
import d4rl.gym_mujoco  # registers D4RL environments

from src.jaxrl5.wrappers import wrap_gym
from src.utils.evaluation import evaluate_all_variants
from src.agent import DDPMSWGLearner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_dict(data: dict, file_path: str) -> None:
    """Save a dictionary to a JSON file."""
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f)

    except Exception as e:
        logging.error(f"Failed to save {file_path}: {e}")


def generate_results(config_path: str,
                    base_sample_params: dict,
                    sample_variant: list,
                    num_episodes: int,) -> dict:
    """
    Generate evaluation results for all parameter variants.
    """
    results=[]

    @hydra.main(config_path=config_path, config_name="config", version_base="1.3")
    def load_agent_and_evaluate(cfg):
        """Load the agent and run evaluations."""
        env = wrap_gym(gym.make(cfg.dataset.env_entry))

        agent_cls = globals()[cfg.method.agent._target_]
        agent = agent_cls.load(
            cfg,
            env=env,
            actor_dir=cfg.savepath.actor_savepath,
            critic_dir=cfg.savepath.critic_savepath,
            actor_step=base_sample_params["actor_step"],
            critic_step=base_sample_params["critic_step"],
        )

        envs_for_eval = [wrap_gym(gym.make(cfg.dataset.env_entry)) for _ in range(num_episodes)]
        result = evaluate_all_variants(
            agent,
            envs_for_eval,
            cfg.dataset.env_entry,
            base_sample_params,
            sample_variant,
            num_episodes=num_episodes
        )
        results.append(result)
    load_agent_and_evaluate()
    return results[0]


def evaluate_all_envs(
    seed: int,
    sample_params: dict,
    sample_variant: dict,
    envs: list[str],
    policies: list[str],
    experiment: str,
    version: str,
    method: str,
    num_episodes: int,
    folder_name: str
) -> None:
    
    """Main evaluation loop for all env-policy combinations."""
    results_dir = Path("results") / f"{folder_name}_eval" / str(seed)
    results_dir.mkdir(parents=True, exist_ok=True)

    save_dict(sample_params, results_dir / "params.json")

    for env in envs:
        for policy in policies:
            logger.info(f"Evaluating seed={seed}, env={env}, policy={policy}")

            dataset_path = Path("../logs") / method / "pretrained" / "D4RL" / env / f"{env}_{policy}-{version}"
            config_path = str(dataset_path / experiment / str(seed) / ".hydra")

            try:
                result = generate_results(config_path, sample_params, sample_variant, num_episodes)
                save_dict(result, results_dir / f"{env}_{policy}_{version}.json")

            except Exception as e:
                logger.error(f"Failed to evaluate {env} {policy} (seed={seed}): {e}")


if __name__=="__main__":
    SEEDS=[1234]
    SAMPLE_PARAMS={
        "actor_step":"1000000",
        "critic_step":"1000000",
        "temperature": 0,
        "guidance_scale": 1,
        "clip_denoised":True,
        "batch_size":1,
        "max_weight_clip":1.0,
        "min_weight_clip":0.0001}
    
    SAMPLE_VARIANT={"guidance_scale":[1,5,10,15,20,25,30]}


    envs=["HalfCheetah", "Walker2d", "Hopper"]
    policies=["medium_expert", "medium_replay", "medium"]
   # envs=["AntMaze"]
   # policies=["large_diverse","large_play","medium_play", "medium_diverse","umaze_diverse","umaze"]
   # envs=["Kitchen"]
   # policies=[ "partial", "complete","mixed"]
   # envs=["Adroit"]
   # policies=["pen_human","pen_cloned"]
    experiment="N_15_vp_expectile_paper"
    method="swg"
    version="v2" # "v0"
    num_episodes=50

    for seed in SEEDS: 
        evaluate_all_envs(seed,
            SAMPLE_PARAMS,
            SAMPLE_VARIANT,
            envs,
            policies,
            experiment,
            version,
            method, 
            num_episodes,
            folder_name = experiment)
    