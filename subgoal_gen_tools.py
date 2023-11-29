import jax
import jax.numpy as jnp

def generate_subgoal(agent, obs, goal, t=1, n=1):
    if n > 1:
        if obs.ndim == 1:
            obs = jnp.expand_dims(obs, axis=0)
        if goal.ndim == 1:
            goal = jnp.expand_dims(goal, axis=0)
        obs = jnp.repeat(obs, n, axis=0)
        goal = jnp.repeat(goal, n, axis=0)
    return agent.sample_actions(obs, goal, seed=jax.random.PRNGKey(42), temperature=t)

def icvf_heuristic_old(icvf_fn, batch, usefulness=1, reachability=1):
    obs = batch["observations"]
    subgoal = batch["actions"]
    goal = batch["goals"]
    advantages = icvf_fn(subgoal, goal, goal) - icvf_fn(obs, goal, goal)
    value = icvf_fn(obs, subgoal, subgoal) - icvf_fn(obs, goal, goal)
    h = (usefulness * advantages) + (reachability * value)
    # h = jnp.where(advantages < 0, float("-inf"), h)
    return h

def icvf_heuristic(icvf_fn, batch, usefulness=1, reachability=1):
    obs = batch["observations"]
    subgoal = batch["actions"]
    goal = batch["goals"]
    advantages = icvf_fn(subgoal, goal, goal) - icvf_fn(obs, goal, goal)
    value = icvf_fn(obs, subgoal, subgoal) - icvf_fn(obs, goal, goal)
    h = jnp.where(advantages < 0, float("-inf"), usefulness * advantages) + (reachability * value)
    # h = jnp.where(advantages < 0, float("-inf"), value )
    return h


def select_subgoal(agent, icvf_fn, obs, goal, n=50, t=1):
    sg_samples = generate_subgoal(agent, obs, goal, t=t, n=n)
    batch = {} # repeat obses and goals, and use subgoals as actions
    batch["observations"] = jnp.repeat(obs.reshape(1, -1), n, axis=0)
    batch["actions"] = sg_samples
    batch["goals"] = jnp.repeat(goal.reshape(1, -1), n, axis=0)
    heuristic = icvf_heuristic(icvf_fn, batch, usefulness=1, reachability=1)    
    # select subgoal
    return sg_samples[jnp.argmax(heuristic)]