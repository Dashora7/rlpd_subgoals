import jax
import flax
import optax
import flax.linen as nn
from jaxrl_m.typing import *
from flax.training.train_state import TrainState
from src.icvf_networks import LayerNormMLP
import chex
from antmaze_stats import *
from cog_stats import *


simple_mean_std = {'umaze': (oa_mean_umaze, oa_std_umaze),
            'medium': (oa_mean_med, oa_std_med),
            'large': (oa_mean_hard, oa_std_hard),
            'pickplace': (o_mean_pickplace, o_std_pickplace)}

mean_std = {}
for k, v in simple_mean_std.items():
    # insert the first two elements at the end of obs
    if k in ['umaze', 'medium', 'large']:
        mean_std[k] = ( np.insert(v[0], 29, v[0][:2]),  np.insert(v[1], 29, v[1][:2]) )

def normalize(arr: jax.Array, mean: jax.Array, std: jax.Array, eps: float = 1e-8) -> jax.Array:
    return (arr - mean) / (std + eps)

@chex.dataclass(frozen=True)
class RunningMeanStd:
    state: Dict[str, jax.Array]

    @staticmethod
    def create(eps: float = 1e-4) -> "RunningMeanStd":
        init_state = {
            "mean": jnp.array([0.0]),
            "var": jnp.array([0.0]),
            "count": jnp.array([eps])
        }
        return RunningMeanStd(state=init_state)

    def update(self, batch: jax.Array) -> "RunningMeanStd":
        batch_mean, batch_var, batch_count = batch.mean(), batch.var(), batch.shape[0]
        if batch_count == 1:
            batch_var = jnp.zeros_like(batch_mean)

        new_mean, new_var, new_count = self._update_mean_var_count_from_moments(
            self.state["mean"], self.state["var"], self.state["count"], batch_mean, batch_var, batch_count
        )
        return self.replace(state={"mean": new_mean, "var": new_var, "count": new_count})

    @staticmethod
    def _update_mean_var_count_from_moments(
        mean, var, count, batch_mean, batch_var, batch_count
    ):
        """Updates the mean, var and count using the previous mean, var, count and batch values."""
        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        return new_mean, new_var, new_count

    @property
    def std(self):
        return jnp.sqrt(self.state["var"])

    @property
    def mean(self):
        return self.state["mean"]

class RND(nn.Module):
    random_model: nn.Module
    predictive_model: nn.Module
    env: str
    simple: bool
    
    def __call__(self, state_action):
        oa_mean, oa_std = simple_mean_std[self.env] if self.simple else mean_std[self.env]
        state_action = normalize(state_action, oa_mean, oa_std)
        pred = self.predictive_model(state_action)
        target = self.random_model(state_action)
        return pred, jax.lax.stop_gradient(target)

class RNDTrainState(TrainState):
    rms: RunningMeanStd

@jax.jit
def rnd_bonus(
        rnd: RNDTrainState,
        state: jax.Array,
        action: jax.Array
) -> jax.Array:
    state_action = jnp.concatenate([state, action], axis=1)
    pred, target = rnd.apply_fn(rnd.params, state_action)
    bonus = jnp.mean((pred - target) ** 2, axis=1) / rnd.rms.std
    return bonus

@jax.jit
def update_rnd(
        key: jax.random.PRNGKey,
        rnd: RNDTrainState,
        states,
        actions,
) -> Tuple[jax.random.PRNGKey, RNDTrainState, Dict]:
    def rnd_loss_fn(params):
        state_actions = jnp.concatenate([states, actions], axis=1)
        pred, target = rnd.apply_fn(params, state_actions)
        raw_loss = ((pred - target) ** 2).sum(axis=1)
        new_rms = rnd.rms.update(raw_loss)
        loss = raw_loss.mean(axis=0)
        return loss, new_rms
    
    (loss, new_rms), grads = jax.value_and_grad(rnd_loss_fn, has_aux=True)(rnd.params)
    new_rnd = rnd.apply_gradients(grads=grads).replace(rms=new_rms)
    info = {"rnd_loss": loss.mean()}
    return key, new_rnd, info

def create_rnd(
        state_dim: int,
        action_dim: int,
        env: str,
        simple: bool,
        hidden_dims: Sequence[int] = (256, 256, 256),
        learning_rate: float = 3e-4,
        seed: int = 0,
) -> RNDTrainState:
    key = jax.random.PRNGKey(seed)
    key, rnd_key = jax.random.split(key)
    random_model = LayerNormMLP(hidden_dims=hidden_dims)
    predictive_model = LayerNormMLP(hidden_dims=hidden_dims)
    rnd_module = RND(random_model, predictive_model, env, simple)
    inp_dim = state_dim + action_dim if simple else state_dim + action_dim + 2
    params = rnd_module.init(rnd_key, jnp.ones((1, inp_dim)))
    return RNDTrainState.create(
        apply_fn=rnd_module.apply,
        params=params,
        tx=optax.adam(learning_rate=learning_rate),
        rms=RunningMeanStd.create()
    )
