import jax
import flax
import optax
import flax.linen as nn
from jaxrl_m.typing import *
from flax.training.train_state import TrainState
from src.icvf_networks import LayerNormMLP





class RND(nn.Module):
    random_model: nn.Module
    predictive_model: nn.Module
    
    def __call__(self, state_action):
        pred = self.predictive_model(state_action)
        target = self.random_model(state_action)
        return pred, jax.lax.stop_gradient(target)



def rnd_bonus(
        rnd: TrainState,
        state: jax.Array,
        action: jax.Array
) -> jax.Array:
    state_action = jnp.concatenate([state, action], axis=1)
    pred, target = rnd.apply_fn(rnd.params, state_action)
    bonus = jnp.sum((pred - target) ** 2, axis=1)
    return bonus


def update_rnd(
        key: jax.random.PRNGKey,
        rnd: TrainState,
        states,
        actions,
) -> Tuple[jax.random.PRNGKey, TrainState, Dict]:
    def rnd_loss_fn(params):
        state_actions = jnp.concatenate([states, actions], axis=1)
        pred, target = rnd.apply_fn(params, state_actions)
        raw_loss = ((pred - target) ** 2).sum(axis=1)

        new_rms = rnd.rms.update(raw_loss)
        loss = raw_loss.mean(axis=0)
        return loss, new_rms
    
    loss, grads = jax.value_and_grad(rnd_loss_fn)(rnd.params)
    new_rnd = rnd.apply_gradients(grads=grads)
    info = {"rnd_loss": loss.mean()}
    return key, new_rnd, info

def create_rnd(
        state_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        learning_rate: float = 1e-3,
        seed: int = 0,
) -> TrainState:
    key = jax.random.PRNGKey(seed)
    key, rnd_key = jax.random.split(key)
    random_model = LayerNormMLP(hidden_dims=hidden_dims)
    predictive_model = LayerNormMLP(hidden_dims=hidden_dims)
    rnd_module = RND(random_model, predictive_model)
    params = rnd_module.init(rnd_key, jnp.ones((1, state_dim + action_dim)))
    return TrainState.create(
        apply_fn=rnd_module.apply,
        params=params,
        tx=optax.adam(learning_rate=learning_rate),
    )