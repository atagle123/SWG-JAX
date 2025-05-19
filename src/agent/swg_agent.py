from functools import partial
import jax
import jax.numpy as jnp
import gym
import optax
from flax import struct
from flax.training.train_state import TrainState
import flax.linen as nn
from src.jaxrl5.data.dataset import DatasetDict
from src.models.critic_model import Ensemble, Q_Model, ValueModel
from src.models.diffusion_model import DDPM, ddpm_sampler_swg
from src.models.models import MLP, MLPResNet, MLPResNet_mod
from src.models.helpers import cosine_beta_schedule, vp_beta_schedule, linear_beta_schedule, FourierFeatures
from src.agent.agent import Agent, compute_q, compute_v, count_params
from src.models.weights import expectile_loss, quantile_loss, exponential_loss
from src.jaxrl5.types import PRNGKey

class DDPMSWGLearner(Agent):
    score_model: TrainState
    target_score_model: TrainState
    critic: TrainState
    target_critic: TrainState
    value: TrainState
    discount: float
    tau: float
    actor_tau: float
    #value_objective_fn: callable
    #weight_fn: callable
    critic_hyperparam: float

    out_dim: int = struct.field(pytree_node=False)
    n_timesteps: int = struct.field(pytree_node=False)
    betas: jnp.ndarray
    alphas: jnp.ndarray
    alpha_hats: jnp.ndarray

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Box, 
        cfg):

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, value_key = jax.random.split(rng, 4)

        actions = action_space.sample()
        observations = observation_space.sample()
        action_dim = action_space.shape[0]

        out_dim = action_dim+1

        ### Diffusion model ###

        preprocess_time_cls = partial(FourierFeatures,
                                      output_size=cfg.diffusion.time_emb,
                                      learnable=True)

        cond_model_cls = partial(MLP,
                                hidden_dims=(cfg.diffusion.time_emb * 2, cfg.diffusion.time_emb * 2),
                                activations=nn.swish,
                                activate_final=False)

        base_model_cls = partial(MLPResNet_mod,
                                    use_layer_norm=cfg.diffusion.use_layer_norm,
                                    num_blocks=cfg.diffusion.num_blocks,
                                    dropout_rate=cfg.diffusion.dropout_rate,
                                    hidden_dim=cfg.diffusion.hidden_dim, 
                                    out_dim=out_dim,
                                    activations=nn.swish)
        
        actor_def = DDPM(time_preprocess_cls=preprocess_time_cls,
                            cond_encoder_cls=cond_model_cls,
                            reverse_encoder_cls=base_model_cls,)

        time = jnp.zeros((1, 1))
        observations = jnp.expand_dims(observations, axis = 0)
        actions = jnp.expand_dims(actions, axis = 0)

        new_dim = jnp.zeros((actions.shape[0], 1))           # Shape: (1, 1)
        actions_weight = jnp.concatenate([actions, new_dim], axis=-1) 

        actor_params = actor_def.init(actor_key, observations, actions_weight, time)['params']

        print(f"PARAMS: {count_params(actor_params)}")
        
        if cfg.train.cosine_decay:
            actor_lr = optax.cosine_decay_schedule(cfg.train.actor_lr, 3000000)

        score_model = TrainState.create(apply_fn=actor_def.apply,
                                        params=actor_params,
                                        tx=optax.adamw(learning_rate=actor_lr)) 
        
        target_score_model = TrainState.create(apply_fn=actor_def.apply,
                                               params=actor_params,
                                               tx=optax.GradientTransformation(
                                                    lambda _: None, lambda _: None))

        ### Critic and Value model ###
        hidden_dims = (cfg.critic.hidden_dim,)*cfg.critic.n_hidden

        critic_base_cls = partial(MLP, hidden_dims=hidden_dims, activate_final=True)
        critic_cls = partial(Q_Model, base_cls=critic_base_cls)
        critic_def = Ensemble(critic_cls, num=2)

        critic_params = critic_def.init(critic_key, observations, actions)["params"]

        critic_optimizer = optax.adam(learning_rate=cfg.train.critic_lr)
        critic = TrainState.create(apply_fn=critic_def.apply,
                                   params=critic_params, 
                                   tx=critic_optimizer)

        target_critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None))

        value_base_cls = partial(MLP, hidden_dims=hidden_dims,
                                 use_layer_norm=cfg.critic.value_layer_norm,
                                 activate_final=True)
        
        value_def = ValueModel(base_cls=value_base_cls)
        value_params = value_def.init(value_key, observations)["params"]
        
        value_optimizer = optax.adam(learning_rate=cfg.train.critic_lr)

        value = TrainState.create(apply_fn=value_def.apply,
                                  params=value_params,
                                  tx=value_optimizer)
        
        #critic_loss = {
        #    'expectile': expectile_loss,
        #    'quantile': quantile_loss,
        #    'exponential': exponential_loss
        #}
        #value_objective_fn = partial(critic_loss[cfg.train.critic_objective], hyperparam=cfg.train.critic_hyperparam)

        #weight_fn = partial(build_weights_fn, loss_type = cfg.train.critic_objective, hyperparam=cfg.train.critic_hyperparam)

        schedulers = {"vp": vp_beta_schedule,
                    "cosine": cosine_beta_schedule,
                    "linear": linear_beta_schedule}
        
        if cfg.diffusion.schedule in schedulers:
            betas = jnp.array(schedulers[cfg.diffusion.schedule](cfg.diffusion.n_timesteps))
        else:
            raise ValueError(f'Invalid beta schedule: {cfg.diffusion.schedule}')

        alphas = 1 - betas
        alpha_hat = jnp.array([jnp.prod(alphas[:i + 1]) for i in range(cfg.diffusion.n_timesteps)])

        return cls(
            actor=None,
            score_model=score_model,
            target_score_model=target_score_model,
            critic=critic,
            target_critic=target_critic,
            value=value,
            tau=cfg.train.target_update,
            discount=cfg.train.discount,
            rng=rng,
            betas=betas,
            alpha_hats=alpha_hat,
            out_dim=out_dim,
            n_timesteps=cfg.diffusion.n_timesteps,
            actor_tau=cfg.train.ema_update, 
            alphas=alphas,
            critic_hyperparam=cfg.train.critic_hyperparam,)

    def update_v(agent, batch: DatasetDict) -> tuple[Agent, dict[str, float]]:
        qs = agent.target_critic.apply_fn(
            {"params": agent.target_critic.params},
            batch["observations"],
            batch["actions"],
        )
        q = qs.min(axis=0)

        def value_loss_fn(value_params) -> tuple[jnp.ndarray, dict[str, float]]:
            v = agent.value.apply_fn({"params": value_params}, batch["observations"])

            value_loss = expectile_loss(q - v, agent.critic_hyperparam).mean() #agent.value_objective_fn(q - v).mean()

            return value_loss, {"value_loss": value_loss, "v": v.mean()}

        grads, info = jax.grad(value_loss_fn, has_aux=True)(agent.value.params)
        value = agent.value.apply_gradients(grads=grads)
        agent = agent.replace(value=value)
        return agent, info
    
    def update_q(agent, batch: DatasetDict) -> tuple[Agent, dict[str, float]]:
        next_v = agent.value.apply_fn(
            {"params": agent.value.params}, 
            batch["next_observations"]
        )

        target_q = batch["rewards"] + agent.discount * batch["masks"] * next_v

        def critic_loss_fn(critic_params) -> tuple[jnp.ndarray, dict[str, float]]:
            qs = agent.critic.apply_fn(
                {"params": critic_params}, batch["observations"], batch["actions"]
            )
            critic_loss = ((qs - target_q) ** 2).mean()

            return critic_loss, {
                "critic_loss": critic_loss,
                "q": qs.mean(),
            }

        grads, info = jax.grad(critic_loss_fn, has_aux=True)(agent.critic.params)
        critic = agent.critic.apply_gradients(grads=grads)
        agent = agent.replace(critic=critic)

        target_critic_params = optax.incremental_update(critic.params, agent.target_critic.params, agent.tau)
        
        target_critic = agent.target_critic.replace(params=target_critic_params)
        new_agent = agent.replace(critic=critic, target_critic=target_critic)
        return new_agent, info

    def update_actor(agent, batch: DatasetDict) -> tuple[Agent, dict[str, float]]:
        rng = agent.rng
        key, rng = jax.random.split(rng, 2)

        time = jax.random.randint(key, (batch['action_weight'].shape[0], ), 0, agent.n_timesteps)

        key, rng = jax.random.split(rng, 2)
        noise_sample = jax.random.normal(key, (batch['action_weight'].shape[0], agent.out_dim)) # B,A+1
        
        alpha_hats = agent.alpha_hats[time]
        time = jnp.expand_dims(time, axis=1)
        alpha_1 = jnp.expand_dims(jnp.sqrt(alpha_hats), axis=1) # TODO possible speed up here precalculating alpha 1 and 2
        alpha_2 = jnp.expand_dims(jnp.sqrt(1 - alpha_hats), axis=1)
        
        noisy_actions_weights = alpha_1 * batch['action_weight'] + alpha_2 * noise_sample 

        key, rng = jax.random.split(rng, 2)

        def actor_loss_fn(score_model_params) -> tuple[jnp.ndarray, dict[str, float]]:
            eps_pred = agent.score_model.apply_fn({'params': score_model_params},
                                       batch['observations'],
                                       noisy_actions_weights,
                                       time,
                                       rngs={'dropout': key},
                                       training=True)
            
            loss = ((eps_pred - noise_sample) ** 2) # B,A+1
            actor_loss = jnp.mean(jnp.sum(loss[:, :-1], axis=-1))
            weight_loss = jnp.mean(loss[:, -1])
            total_loss = jnp.mean(jnp.sum(loss, axis=-1))
            return total_loss, {'actor_loss': actor_loss, 'weight_loss': weight_loss}

        grads, info = jax.grad(actor_loss_fn, has_aux=True)(agent.score_model.params)
        score_model = agent.score_model.apply_gradients(grads=grads)

        agent = agent.replace(score_model=score_model)
        target_score_params = optax.incremental_update(score_model.params, agent.target_score_model.params, agent.actor_tau)

        target_score_model = agent.target_score_model.replace(params=target_score_params)
        new_agent = agent.replace(score_model=score_model, target_score_model=target_score_model, rng=rng)

        return new_agent, info

    @jax.jit
    def actor_update(self, batch: DatasetDict)-> tuple[Agent, dict[str, float]]:
        new_agent = self
        new_agent, actor_info = new_agent.update_actor(batch)
        return new_agent, actor_info
    
    @jax.jit
    def critic_update(self, batch: DatasetDict)-> tuple[Agent, dict[str, float]]:
        new_agent = self
        new_agent, critic_info = new_agent.update_v(batch)
        new_agent, value_info = new_agent.update_q(batch)

        return new_agent, {**critic_info, **value_info}

    def eval_actions(self, observations: jnp.ndarray, rng: PRNGKey, sample_params: dict)-> tuple[jnp.ndarray, PRNGKey]:

        assert len(observations.shape) == 1
        observations = jax.device_put(observations)
        observations = jnp.expand_dims(observations, axis = 0).repeat(sample_params["batch_size"], axis = 0)

        score_params = self.target_score_model.params
        actions_weights, rng = ddpm_sampler_swg(self.score_model.apply_fn, 
                                    score_params, 
                                    self.n_timesteps, 
                                    rng, 
                                    self.out_dim, 
                                    observations,
                                    self.alphas, 
                                    self.alpha_hats, 
                                    self.betas,
                                    temperature=sample_params["temperature"], 
                                    repeat_last_step=0, 
                                    clip_denoised=sample_params["clip_denoised"], 
                                    guidance_scale=sample_params["guidance_scale"], 
                                    max_weight_clip=sample_params["max_weight_clip"], 
                                    min_weight_clip=sample_params["min_weight_clip"])
        
        new_rng, _ = jax.random.split(rng, 2)
        actions = actions_weights[:, :-1]
        weights = actions_weights[:, -1]
        qs = compute_q(self.target_critic.apply_fn, self.target_critic.params, observations, actions) 
        idx = jnp.argmax(qs)
        action = actions[idx]
        return action, new_rng