import jax
import flax
import optax
import flax.linen as nn
from jaxrl_m.typing import *
from flax.training.train_state import TrainState
from src.icvf_networks import LayerNormMLP
import chex

oa_mean = np.array([3.9977181e+00, 7.0474000e+00, 4.9015650e-01, 6.3097394e-01,
 -2.0249341e-02,  1.2467855e-01, -1.7942959e-01,  3.6614746e-02,
  7.0847946e-01,  6.1714683e-02, -7.7518719e-01, -1.1250293e-01,
 -6.9801414e-01,  3.1167228e-02,  7.3759615e-01,  4.2154718e-02,
  1.1727795e-01, -6.3850585e-04, -5.2995738e-03,  1.9374553e-03,
 -1.3605792e-02, -2.4506841e-03,  1.6431838e-02, -2.4188210e-03,
 -1.5624404e-02,  1.6520927e-03, -1.6445911e-02,  6.4725121e-03,
  1.5356019e-02,  3.7892323e-02, -3.0319202e-01,  6.2268957e-02,
 -3.6210087e-01,  6.7616254e-02,  2.3050579e-01, -1.8155356e-01,
  4.0676609e-01])

oa_std =  np.array([2.7836988,  3.1514142,  0.1570694,  0.41083026, 0.33011514, 0.34351712,
 0.39814702, 0.45371208, 0.27702916, 0.4263024,  0.3027348,  0.42865035,
 0.26848754, 0.4343063,  0.28686404, 0.8156437,  0.74574715, 0.6920502,
 1.0524466,  1.0720621,  1.0845602,  2.4670615,  1.8190215,  2.7180765,
 1.9189895,  2.7241614,  1.7706113,  2.4668434,  1.6986042,  0.79764515,
 0.7433417,  0.84378815, 0.7249487,  0.7947736, 0.80010587, 0.812278,
 0.71354556])

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
    
    def __call__(self, state_action):
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
    bonus = jnp.sum((pred - target) ** 2, axis=1) / rnd.rms.std
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
        hidden_dims: Sequence[int] = (256, 256),
        learning_rate: float = 1e-3,
        seed: int = 0,
) -> RNDTrainState:
    key = jax.random.PRNGKey(seed)
    key, rnd_key = jax.random.split(key)
    random_model = LayerNormMLP(hidden_dims=hidden_dims)
    predictive_model = LayerNormMLP(hidden_dims=hidden_dims)
    rnd_module = RND(random_model, predictive_model)
    params = rnd_module.init(rnd_key, jnp.ones((1, state_dim + action_dim)))
    return RNDTrainState.create(
        apply_fn=rnd_module.apply,
        params=params,
        tx=optax.adam(learning_rate=learning_rate),
        rms=RunningMeanStd.create()
    )
