import jax
import jax.numpy as jnp
from functools import partial


### value function losses ###
def expectile_loss(diff, hyperparam=0.7):
    weight = jnp.where(diff > 0, hyperparam, (1 - hyperparam))
    return weight * (diff**2)

def quantile_loss(diff, hyperparam=0.6):
    weight = jnp.where(diff > 0, hyperparam, (1 - hyperparam))
    return weight * jnp.abs(diff)

def exp_w_clip(x, x0, mode='zero'):
  if mode == 'zero':
    return jnp.where(x < x0, jnp.exp(x), jnp.exp(x0))
  elif mode == 'first':
    return jnp.where(x < x0, jnp.exp(x), jnp.exp(x0) + jnp.exp(x0) * (x - x0))
  elif mode == 'second':
    return jnp.where(x < x0, jnp.exp(x), jnp.exp(x0) + jnp.exp(x0) * (x - x0) + (jnp.exp(x0) / 2) * ((x - x0)**2))
  else:
    raise ValueError()

def exponential_loss(diff, hyperparam, clip=jnp.log(6), mode='zero'):
    exp_diff = exp_w_clip(diff * hyperparam, clip, mode=mode)
    exp_diff = jax.lax.stop_gradient(exp_diff)
    return (exp_diff - 1) * (diff)


@partial(jax.jit, static_argnames=('weight_fn', 'hyperparam'))
def build_weights_fn(qs, vs, weight_fn, hyperparam):

    adv = qs - vs
    if weight_fn == 'expectile':
        weights = jnp.where(adv > 0, hyperparam, 1 - hyperparam)

    elif weight_fn == 'quantile':
        tau_weights = jnp.where(adv > 0, hyperparam, 1 - hyperparam)
        weights = tau_weights / adv

    elif weight_fn == 'exponential':
        weights = hyperparam * jnp.abs((jnp.exp(adv * hyperparam)-1))/jnp.abs(adv)

    elif weight_fn == "soft_adv":
        weights = jnp.where(adv > 0, hyperparam, 1 - hyperparam)

    elif weight_fn == "hard_adv":
        weights = jnp.where(adv >= (-0.01), 1, 0)

    elif weight_fn == "exp_adv":
        TEMPERATURE = 3
        CLIP = 80.0
        weights = jnp.exp(adv * TEMPERATURE)
        weights = jnp.clip(weights, a_min=0.001, a_max=CLIP) #/ CLIP

    elif weight_fn == "simple_adv": 
        CLIP = 80.0
        weights = adv
        weights = jnp.clip(weights, a_min=-CLIP, a_max=CLIP)

    elif weight_fn == "simple_q": 
        weights = qs

    elif weight_fn == "smooth_expectile":
        smooth_step = 1 / (1 + jnp.exp(-3.0 * adv)) 
        weights = smooth_step * hyperparam + (1 - smooth_step) * (1 - hyperparam)
        
    elif weight_fn == "dice":
        EXP_ADV_MAX = 100.
        EXP_ADV_MIN = 1e-40
        pi_residual = (adv) / 0.6
        weights = jnp.where(pi_residual >= 0, pi_residual / 2 + 1, jnp.exp(pi_residual))
        weights = jnp.clip(weights, a_min=EXP_ADV_MIN, a_max=EXP_ADV_MAX)

    else:
        raise ValueError(f'Invalid weight function: {weight_fn}')
        
    return weights