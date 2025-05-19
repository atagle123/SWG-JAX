from functools import partial
from typing import Type
import flax.linen as nn
import jax.numpy as jnp
import jax

class DDPM(nn.Module):
    cond_encoder_cls: Type[nn.Module]
    reverse_encoder_cls: Type[nn.Module]
    time_preprocess_cls: Type[nn.Module]

    @nn.compact
    def __call__(self,
                 s: jnp.ndarray,
                 a: jnp.ndarray,
                 time: jnp.ndarray,
                 training: bool = False):

        t_ff = self.time_preprocess_cls()(time)
        cond = self.cond_encoder_cls()(t_ff, training=training)
        reverse_input = jnp.concatenate([a, s, cond], axis=-1)

        return self.reverse_encoder_cls()(reverse_input, training=training)
    

@partial(jax.jit, static_argnames=('actor_apply_fn', 'act_dim', 'T', 'repeat_last_step', 'clip_sampler', 'training'))
def ddpm_sampler(actor_apply_fn, actor_params, T, rng, act_dim, observations, alphas, alpha_hats, betas, temperature, repeat_last_step, clip_sampler, training = False):

    batch_size = observations.shape[0]
    
    def fn(input_tuple, time):
        current_x, rng = input_tuple
        
        input_time = jnp.expand_dims(jnp.array([time]).repeat(current_x.shape[0]), axis = 1)
        eps_pred = actor_apply_fn({"params": actor_params}, observations, current_x, input_time, training = training)

        alpha_1 = 1 / jnp.sqrt(alphas[time])
        alpha_2 = ((1 - alphas[time]) / (jnp.sqrt(1 - alpha_hats[time])))
        current_x = alpha_1 * (current_x - alpha_2 * eps_pred)

        rng, key = jax.random.split(rng, 2)
        z = jax.random.normal(key,
                            shape=(observations.shape[0], current_x.shape[1]),)
        z_scaled = temperature * z
        current_x = current_x + (time > 0) * (jnp.sqrt(betas[time]) * z_scaled)

        if clip_sampler:
            current_x = jnp.clip(current_x, -1, 1)

        return (current_x, rng), ()

    key, rng = jax.random.split(rng, 2)
    input_tuple, () = jax.lax.scan(fn, (jax.random.normal(key, (batch_size, act_dim)), rng) * temperature, jnp.arange(T-1, -1, -1))

    for _ in range(repeat_last_step):
        input_tuple, () = fn(input_tuple, 0)
    
    action_0, rng = input_tuple
    action_0 = jnp.clip(action_0, -1, 1)

    return action_0, rng




@partial(jax.jit, static_argnames=(
    'actor_apply_fn', 'act_dim', 'T', 'repeat_last_step', 
    'clip_denoised', 'training'
))
def ddpm_sampler_swg(
    actor_apply_fn, actor_params, T, rng, act_dim, observations,
    alphas, alpha_hats, betas, temperature, repeat_last_step,
    clip_denoised, guidance_scale, max_weight_clip, min_weight_clip,
    training=False
    ):

    batch_size = observations.shape[0]
    
    def sampling_fn(input_tuple, time):
        current_x, rng = input_tuple
        
        input_time = jnp.expand_dims(jnp.array([time]).repeat(current_x.shape[0]), axis = 1)
        alpha_1 = 1 / jnp.sqrt(alphas[time])
        alpha_2 = ((1 - alphas[time]) / (jnp.sqrt(1 - alpha_hats[time])))

        sigma=jnp.sqrt(1 - alpha_hats[time])

        ### SWG SAMPLING ###
        def swg_guidance(current_x):

            def grad_swg_fn(current_x):
                eps_pred = actor_apply_fn({"params": actor_params}, observations, current_x, input_time, training = training)
                eps_weight_pred = eps_pred[:, -1]
                pred_weight = (current_x[:,-1]-eps_weight_pred*sigma)*alpha_1
                pred_weight = jnp.clip(pred_weight, min_weight_clip, max_weight_clip)
                log_weight = jnp.log(pred_weight)
                return(jnp.sum(log_weight), eps_pred)

            grad_fn = jax.grad(grad_swg_fn, argnums=0, has_aux=True)

            grad_log_weight, eps_pred = grad_fn(current_x)
            grad_log_weight = jnp.nan_to_num(grad_log_weight, nan=0.0) # if any nan
            
            #grad_log_weight = jnp.clip(grad_log_weight, -1.0, 1)

            eps_pred = eps_pred - guidance_scale * grad_log_weight*sigma

            return eps_pred
        
        def no_guidance(current_x):
            eps_pred = actor_apply_fn({"params": actor_params}, observations, current_x, input_time, training = training)
            return eps_pred        

        eps_pred = jax.lax.cond(
                guidance_scale != 0,
                swg_guidance,
                no_guidance,
                current_x)
        
        current_x = alpha_1 * (current_x - alpha_2 * eps_pred)

        rng, key = jax.random.split(rng, 2)
        z = jax.random.normal(key, shape=(observations.shape[0], current_x.shape[1]),)

        z_scaled = temperature * z
        current_x = current_x + (time > 0) * (jnp.sqrt(betas[time]) * z_scaled)

        if clip_denoised:
            current_x = current_x.at[:, :-1].set(jnp.clip(current_x[:, :-1], -1, 1))
            current_x = current_x.at[:, -1].set(jnp.clip(current_x[:, -1], min_weight_clip, max_weight_clip))

        return (current_x, rng), ()

    key, rng = jax.random.split(rng, 2)
    input_tuple, () = jax.lax.scan(sampling_fn, (jax.random.normal(key, (batch_size, act_dim))* temperature, rng), jnp.arange(T-1, -1, -1))

    for _ in range(repeat_last_step):
        input_tuple, () = sampling_fn(input_tuple, 0)
    
    action_weight_0, rng = input_tuple

    return action_weight_0, rng