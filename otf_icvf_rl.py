#! /usr/bin/env python
import os
import pickle
from functools import partial
import d4rl
import d4rl.gym_mujoco
import d4rl.locomotion
import dmcgym
import gym
import numpy as np
import tqdm

from src import viz_utils
from rnd_tools import create_rnd, rnd_bonus, update_rnd
from src import icvf_learner as learner
from src.icvf_networks import create_icvf
import jax
from absl import app, flags
from flax.core.frozen_dict import unfreeze
import jax.numpy as jnp
from icvf_envs.antmaze import d4rl_utils, d4rl_ant, ant_diagnostics, d4rl_pm
try:
    from flax.training import checkpoints
except:
    print("Not loading checkpointing functionality.")
from ml_collections import config_flags

import wandb
from rlpd.agents import SACLearner
from rlpd.data import ReplayBuffer
from gc_buffer import GCSBuffer
import jaxrl_m.dataset as jrlm
from rlpd.data.d4rl_datasets import D4RLDataset
try:
    from rlpd.data.binary_datasets import BinaryDataset
except:
    print("not importing binary dataset")
from rlpd.evaluation import evaluate
from rlpd.wrappers import wrap_gym

FLAGS = flags.FLAGS

flags.DEFINE_string("project_name", "rlpd", "wandb project name.")
flags.DEFINE_string("env_name", "halfcheetah-expert-v2", "D4rl dataset name.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 10, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 5000, "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(1e6), "Number of training steps.")
flags.DEFINE_integer("pretrain_steps", 0, "Number of pre-training steps.")
flags.DEFINE_integer(
    "start_training", int(1e4), "Number of training steps to start training."
)
flags.DEFINE_float("offline_ratio", 0.0, "Offline ratio.")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("save_video", False, "Save videos during evaluation.")
flags.DEFINE_boolean("checkpoint_model", False, "Save agent checkpoint on evaluation.")
flags.DEFINE_boolean(
    "checkpoint_buffer", False, "Save agent replay buffer on evaluation."
)
flags.DEFINE_integer("utd_ratio", 1, "Update to data ratio.")
flags.DEFINE_boolean(
    "binary_include_bc", True, "Whether to include BC data in the binary datasets."
)
flags.DEFINE_boolean("use_rnd", False, "Use Random Network Distillation")
config_flags.DEFINE_config_file(
    "config",
    "configs/sac_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

def update_dict(d, additional):
    d.update(additional)
    return d

def combine(one_dict, other_dict):
    combined = {}

    for k, v in one_dict.items():
        if isinstance(v, dict):
            combined[k] = combine(v, other_dict[k])
        else:
            tmp = np.empty(
                (v.shape[0] + other_dict[k].shape[0], *v.shape[1:]), dtype=v.dtype
            )
            tmp[0::2] = v
            tmp[1::2] = other_dict[k]
            combined[k] = tmp

    return combined


def main(_):
    assert FLAGS.offline_ratio >= 0.0 and FLAGS.offline_ratio <= 1.0
    wandb.init(project=FLAGS.project_name, entity="dashora7")
    wandb.config.update(FLAGS)
    nitish_type = FLAGS.env_name.split('-')[1]
    ENVGOALS = {
        'medium': np.array([20.5, 21]),
        'large': np.array([33, 25]),
        'umaze':np.array([0, 9])}
    envgoal = ENVGOALS[nitish_type]
    icvf_config = update_dict(
        learner.get_default_config(),
        {
        'discount': 0.999, 
        'optim_kwargs': { # Standard Adam parameters for non-vision
                'learning_rate': 3e-4,
                'eps': 1e-8
            }
        } 
    )
    exp_prefix = f"s{FLAGS.seed}_{FLAGS.pretrain_steps}pretrain"
    if hasattr(FLAGS.config, "critic_layer_norm") and FLAGS.config.critic_layer_norm:
        exp_prefix += "_LN"

    log_dir = os.path.join(FLAGS.log_dir, exp_prefix)

    if FLAGS.checkpoint_model:
        chkpt_dir = os.path.join(log_dir, "checkpoints")
        os.makedirs(chkpt_dir, exist_ok=True)

    if FLAGS.checkpoint_buffer:
        buffer_dir = os.path.join(log_dir, "buffers")
        os.makedirs(buffer_dir, exist_ok=True)

    env = gym.make(FLAGS.env_name)
    
    env = wrap_gym(env, rescale_actions=True)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    env.seed(FLAGS.seed)
    # not ideal, but works for now:
    if "binary" in FLAGS.env_name:
        ds = BinaryDataset(env, include_bc_data=FLAGS.binary_include_bc)
    else:
        ds = D4RLDataset(env)

    eval_env = gym.make(FLAGS.env_name)
    
    
    eval_env = wrap_gym(eval_env, rescale_actions=True)
    eval_env.seed(FLAGS.seed + 42)

    kwargs = dict(FLAGS.config)
    model_cls = kwargs.pop("model_cls")
    agent = globals()[model_cls].create(
        FLAGS.seed, env.observation_space, env.action_space, **kwargs
    )

    replay_buffer = ReplayBuffer(
        env.observation_space, env.action_space, FLAGS.max_steps
    )
    replay_buffer.seed(FLAGS.seed)
    
    
    # Setup RND Parameters
    if FLAGS.use_rnd:
        rnd_update_freq = 1
        start_rnd = 5000
        rnd = create_rnd(29, 8, hidden_dims=[256, 256, 256], env=nitish_type, simple=True)
        rnd_key = jax.random.PRNGKey(42)
        rnd_ep_bonus = 0
        rnd_ep_loss = 0
        rnd_multiplier = 1.0 # float(1 / 10)

    start_icvf = 0
    # icvf_multiplier = 0.001 # for value
    icvf_multiplier = 0.1 # for potential
    icvf_ep_bonus = 0
    value_def = create_icvf('monolithic', hidden_dims=[512, 512, 512])
    icvf_agent = learner.create_learner(
        seed=FLAGS.seed, observations=np.ones((1, 29)),
        value_def=value_def, **icvf_config)
    def icvf_value_fn(obs, goal):
        return icvf_agent.value(obs, goal, goal).mean(0)
    value_fn = jax.jit(icvf_value_fn)
    # Bonus function
    def icvf_bonus(s, s_prime, goal, potential=True):
        val_to_sg = value_fn(s_prime, goal)
        if potential:
            last_val_to_sg = value_fn(s, goal)
            return val_to_sg - last_val_to_sg
        else:
            return val_to_sg
    
    # 1. Make an example transition.
    example_transition = {
        'observations': np.random.randn(29),
        'next_observations': np.random.randn(29),
        'dones_float': 1.0,
    }
    # 2. Make Buffer from example transition
    buffer = jrlm.ReplayBuffer.create(example_transition, size=FLAGS.max_steps)
    # 3. Construct GCS Buffer from Buffer
    icvf_buffer = GCSBuffer(buffer, **GCSBuffer.get_default_config())
    visualizer = None
    
    observation, done = env.reset(), False
    episode_lst = []
    for i in tqdm.tqdm(
        range(0, FLAGS.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm
    ):
        if i < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action, agent = agent.sample_actions(observation)
        next_observation, reward, done, info = env.step(action)

        if FLAGS.use_rnd and i % rnd_update_freq == 0:
            rnd_key, rnd, rnd_info = update_rnd(
                rnd_key, rnd, observation.reshape(1, -1), action.reshape(1, -1))
            loss = rnd_info['rnd_loss'].item()
            rnd_ep_loss += loss
        
        if not done or "TimeLimit.truncated" in info:
            mask = 1.0
        else:
            mask = 0.0

        replay_buffer.insert(
            dict(
                observations=observation,
                actions=action,
                rewards=reward,
                masks=mask,
                dones=done,
                next_observations=next_observation,
            )
        )
        episode_lst.append({
            "observations": observation,
            "dones_float": float(done),
            "next_observations": next_observation
        })
        observation = next_observation
        
        if done:
            # Reset and add to icvf buffer
            icvf_buffer.add_episode(episode_lst)
            observation, done = env.reset(), False
            episode_lst = []
            
            # Log episode info
            for k, v in info["episode"].items():
                decode = {"r": "return", "l": "length", "t": "time"}
                wandb.log({f"training/{decode[k]}": v}, step=i + FLAGS.pretrain_steps)
            if FLAGS.use_rnd:
                wandb.log(
                    {f"training/rnd_avg_bonus": rnd_ep_bonus / info["episode"]['l']},
                    step=i + FLAGS.pretrain_steps)
                rnd_ep_bonus = 0
                wandb.log(
                    {f"training/rnd_avg_loss": rnd_ep_loss / info["episode"]['l']},
                    step=i + FLAGS.pretrain_steps)
                rnd_ep_loss = 0
            
            wandb.log(
                {f"training/icvf_avg_bonus": icvf_ep_bonus / info["episode"]['l']},
                step=i + FLAGS.pretrain_steps)
            icvf_ep_bonus = 0

        if i >= FLAGS.start_training:
            
            online_batch = replay_buffer.sample(
                int(FLAGS.batch_size * FLAGS.utd_ratio * (1 - FLAGS.offline_ratio))
            )
            
            if FLAGS.offline_ratio > 0.0:  
                offline_batch = ds.sample(
                    int(FLAGS.batch_size * FLAGS.utd_ratio * FLAGS.offline_ratio)
                )
                batch = combine(offline_batch, online_batch)
            else:
                batch = online_batch
                batch = unfreeze(online_batch)
            
            if FLAGS.use_rnd and i > start_rnd:
                bonus = rnd_multiplier * rnd_bonus(
                    rnd, batch['observations'], batch['actions'])
                rnd_ep_bonus += bonus.mean().item()
                batch["rewards"] += np.array(bonus)
             
            if i > start_icvf:
                envgoals = envgoal.reshape(1, -1).repeat(
                    batch['observations'].shape[0], axis=0)
                goals = jnp.concatenate(
                    [envgoals, jnp.zeros((envgoals.shape[0], 27))], axis=-1)
                bonus_rew_icvf = icvf_multiplier * icvf_bonus(
                    batch['observations'], batch['next_observations'], goals)
                icvf_ep_bonus += bonus_rew_icvf.mean().item()
                batch["rewards"] += np.array(bonus_rew_icvf)
                batch["rewards"] *= 10
            
            if "antmaze" in FLAGS.env_name:
                batch["rewards"] -= 1

            agent, update_info = agent.update(batch, FLAGS.utd_ratio)
            icvf_batch = icvf_buffer.sample(FLAGS.batch_size)
            icvf_agent, icvf_update_info = icvf_agent.update(icvf_batch)

            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    wandb.log({f"training/{k}": v}, step=i + FLAGS.pretrain_steps)
                for k, v in icvf_update_info.items():
                    wandb.log({f"training/icvf_{k}": v}, step=i + FLAGS.pretrain_steps)

        if i % FLAGS.eval_interval == 0:
            eval_info = evaluate(
                agent,
                eval_env,
                num_episodes=FLAGS.eval_episodes,
                save_video=FLAGS.save_video,
                etype=nitish_type
            )
            """if visualizer is None and i > 1e4:
                visualizer = DebugPlotGenerator(FLAGS.env_name, icvf_buffer)
            elif i > 1e4:
                visualizations = visualizer.generate_debug_plots(agent)
                eval_metrics = {f'visualizations/{k}': v for k, v in visualizations.items()}
                wandb.log(eval_metrics, step=i + FLAGS.pretrain_steps)"""

            for k, v in eval_info.items():
                wandb.log({f"evaluation/{k}": v}, step=i + FLAGS.pretrain_steps)

            if FLAGS.checkpoint_model:
                try:
                    checkpoints.save_checkpoint(
                        chkpt_dir, agent, step=i, keep=20, overwrite=True
                    )
                except:
                    print("Could not save model checkpoint.")

            if FLAGS.checkpoint_buffer:
                try:
                    with open(os.path.join(buffer_dir, f"buffer"), "wb") as f:
                        pickle.dump(replay_buffer, f, pickle.HIGHEST_PROTOCOL)
                except:
                    print("Could not save agent buffer.")


###################################################################################################
#
# Creates wandb plots
#
###################################################################################################
class DebugPlotGenerator:
    def __init__(self, env_name, gc_dataset):
        if 'antmaze' in env_name:
            viz_env, viz_dataset = d4rl_ant.get_env_and_dataset(env_name)
            init_state = np.copy(viz_dataset['observations'][0])
            init_state[:2] = (12.5, 8)
            viz_library = d4rl_ant
            self.viz_things = (viz_env, viz_dataset, viz_library, init_state)

        elif 'maze' in env_name:
            viz_env, viz_dataset = d4rl_pm.get_gcenv_and_dataset(env_name)
            init_state = np.copy(viz_dataset['observations'][0])
            init_state[:2] = (3, 4)
            viz_library = d4rl_pm
            self.viz_things = (viz_env, viz_dataset, viz_library, init_state)
        else:
            raise NotImplementedError('Visualization not implemented for this environment')

        # intent_set_indx = np.random.default_rng(0).choice(dataset.size, FLAGS.config.n_intents, replace=False)
        # Chosen by hand for `antmaze-large-diverse-v2` to get a nice spread of goals, use the above line for random selection

        intent_set_indx = np.array([184588, 62200, 162996, 110214, 4086, 191369, 92549, 12946, 192021])
        self.intent_set_batch = gc_dataset.sample(9, indx=intent_set_indx)
        self.example_trajectory = gc_dataset.sample(50, indx=np.arange(1000, 1050))



    def generate_debug_plots(self, agent):
        example_trajectory = self.example_trajectory
        intents = self.intent_set_batch['observations']
        (viz_env, viz_dataset, viz_library, init_state) = self.viz_things

        visualizations = {}
        traj_metrics = get_traj_v(agent, example_trajectory)
        value_viz = viz_utils.make_visual_no_image(traj_metrics, 
            [
            partial(viz_utils.visualize_metric, metric_name=k) for k in traj_metrics.keys()
                ]
        )
        visualizations['value_traj_viz'] = wandb.Image(value_viz)

        if 'maze' in FLAGS.env_name:
            print('Visualizing intent policies and values')
            # Policy visualization
            methods = [
                partial(viz_library.plot_policy, policy_fn=partial(get_policy, agent, intent=intents[idx]))
                for idx in range(9)
            ]
            image = viz_library.make_visual(viz_env, viz_dataset, methods)
            visualizations['intent_policies'] = wandb.Image(image)

            # Value visualization
            methods = [
                partial(viz_library.plot_value, value_fn=partial(get_values, agent, intent=intents[idx]))
                for idx in range(9)
            ]
            image = viz_library.make_visual(viz_env, viz_dataset, methods)
            visualizations['intent_values'] = wandb.Image(image)

            for idx in range(3):
                methods = [
                    partial(viz_library.plot_policy, policy_fn=partial(get_policy, agent, intent=intents[idx])),
                    partial(viz_library.plot_value, value_fn=partial(get_values, agent, intent=intents[idx]))
                ]
                image = viz_library.make_visual(viz_env, viz_dataset, methods)
                visualizations[f'intent{idx}'] = wandb.Image(image)

            image_zz = viz_library.gcvalue_image(
                viz_env,
                viz_dataset,
                partial(get_v_zz, agent),
            )
            image_gz = viz_library.gcvalue_image(
                viz_env,
                viz_dataset,
                partial(get_v_gz, agent, init_state),
            )
            visualizations['v_zz'] = wandb.Image(image_zz)
            visualizations['v_gz'] = wandb.Image(image_gz)
        return visualizations

###################################################################################################
#
# Helper functions for visualization
#
###################################################################################################

@jax.jit
def get_values(agent, observations, intent):
    def get_v(observations, intent):
        intent = intent.reshape(1, -1)
        intent_tiled = jnp.tile(intent, (observations.shape[0], 1))
        v1, v2 = agent.value(observations, intent_tiled, intent_tiled)
        return (v1 + v2) / 2    
    return get_v(observations, intent)

@jax.jit
def get_policy(agent, observations, intent):
    def v(observations):
        def get_v(observations, intent):
            intent = intent.reshape(1, -1)
            intent_tiled = jnp.tile(intent, (observations.shape[0], 1))
            v1, v2 = agent.value(observations, intent_tiled, intent_tiled)
            return (v1 + v2) / 2    
            
        return get_v(observations, intent).mean()

    grads = jax.grad(v)(observations)
    policy = grads[:, :2]
    return policy / jnp.linalg.norm(policy, axis=-1, keepdims=True)

@jax.jit
def get_debug_statistics(agent, batch):
    def get_info(s, g, z):
        if agent.config['no_intent']:
            return agent.value(s, g, jnp.ones_like(z), method='get_info')
        else:
            return agent.value(s, g, z, method='get_info')

    s = batch['observations']
    g = batch['goals']
    z = batch['desired_goals']

    info_ssz = get_info(s, s, z)
    info_szz = get_info(s, z, z)
    info_sgz = get_info(s, g, z)
    info_sgg = get_info(s, g, g)
    info_szg = get_info(s, z, g)

    if 'phi' in info_sgz:
        stats = {
            'phi_norm': jnp.linalg.norm(info_sgz['phi'], axis=-1).mean(),
            'psi_norm': jnp.linalg.norm(info_sgz['psi'], axis=-1).mean(),
        }
    else:
        stats = {}

    stats.update({
        'v_ssz': info_ssz['v'].mean(),
        'v_szz': info_szz['v'].mean(),
        'v_sgz': info_sgz['v'].mean(),
        'v_sgg': info_sgg['v'].mean(),
        'v_szg': info_szg['v'].mean(),
        'diff_szz_szg': (info_szz['v'] - info_szg['v']).mean(),
        'diff_sgg_sgz': (info_sgg['v'] - info_sgz['v']).mean(),
    })
    return stats

@jax.jit
def get_gcvalue(agent, s, g, z):
    v_sgz_1, v_sgz_2 = agent.value(s, g, z)
    return (v_sgz_1 + v_sgz_2) / 2

def get_v_zz(agent, goal, observations):
    goal = jnp.tile(goal, (observations.shape[0], 1))
    return get_gcvalue(agent, observations, goal, goal)

def get_v_gz(agent, initial_state, target_goal, observations):
    initial_state = jnp.tile(initial_state, (observations.shape[0], 1))
    target_goal = jnp.tile(target_goal, (observations.shape[0], 1))
    return get_gcvalue(agent, initial_state, observations, target_goal)

@jax.jit
def get_traj_v(agent, trajectory):
    def get_v(s, g):
        return agent.value(s[None], g[None], g[None]).mean()
    observations = trajectory['observations']
    all_values = jax.vmap(jax.vmap(get_v, in_axes=(None, 0)), in_axes=(0, None))(observations, observations)
    return {
        'dist_to_beginning': all_values[:, 0],
        'dist_to_end': all_values[:, -1],
        'dist_to_middle': all_values[:, all_values.shape[1] // 2],
    }

####################





if __name__ == "__main__":
    app.run(main)