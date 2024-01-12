#! /usr/bin/env python
import os
import pickle
import jax
import d4rl
import d4rl.gym_mujoco
import d4rl.locomotion
import dmcgym
import gym
from nitish_env import NitishEnv, register
from nitish_env_simple import NitishEnvSimple, register_simple
from d4rl.locomotion import wrappers
import numpy as np
import tqdm
from absl import app, flags
from flax.core.frozen_dict import unfreeze
try:
    from flax.training import checkpoints
except:
    print("Not loading checkpointing functionality.")
from ml_collections import config_flags
from rnd_tools import create_rnd, rnd_bonus, update_rnd
import wandb
from rlpd.agents import SACLearner
from rlpd.data import ReplayBuffer
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
flags.DEFINE_float("offline_ratio", 0.5, "Offline ratio.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 10, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 5000, "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(1e6), "Number of training steps.")
flags.DEFINE_integer(
    "start_training", int(1e4), "Number of training steps to start training."
)
flags.DEFINE_integer("pretrain_steps", 0, "Number of offline updates.")
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
flags.DEFINE_string("icvf_path", None, "model path.")
flags.DEFINE_string("diffusion_path", None, "model path.")
flags.DEFINE_string("type", 'small', "antmaze type, if nitish's custom")
config_flags.DEFINE_config_file(
    "config",
    "configs/sac_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

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
    wandb.init(project=FLAGS.project_name, entity='dashora7')
    wandb.config.update(FLAGS)

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
    
    if FLAGS.env_name == 'nitish-custom-antmaze':
        register(FLAGS.type)
        env = wrappers.NormalizedBoxEnv(gym.make(
            'nitish-v0', subgoal_dense=True, icvf_path=FLAGS.icvf_path,
            diffusion_path=FLAGS.diffusion_path, etype=FLAGS.type)) # NitishEnv()
    elif FLAGS.env_name == 'nitish-custom-antmaze-simple':
        register_simple(FLAGS.type)
        env = wrappers.NormalizedBoxEnv(gym.make(
            'nitish-v0-simple', icvf_path=FLAGS.icvf_path,
            diffusion_path=FLAGS.diffusion_path, etype=FLAGS.type))
    else:
        env = gym.make(FLAGS.env_name)
    env = wrap_gym(env, rescale_actions=True)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    env.seed(FLAGS.seed)
    
    if FLAGS.env_name == 'nitish-custom-antmaze':
        register(FLAGS.type)
        eval_env = wrappers.NormalizedBoxEnv(gym.make(
            'nitish-v0', subgoal_dense=True, icvf_path=FLAGS.icvf_path,
            diffusion_path=FLAGS.diffusion_path, etype=FLAGS.type)) # NitishEnv()
    elif FLAGS.env_name == 'nitish-custom-antmaze-simple':
        register_simple(FLAGS.type)
        eval_env = wrappers.NormalizedBoxEnv(gym.make(
            'nitish-v0-simple', icvf_path=FLAGS.icvf_path,
            diffusion_path=FLAGS.diffusion_path, etype=FLAGS.type))
    else:
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
    
    use_rnd = True
    rnd_update_freq = 1
    start_rnd = 10000
    rnd = create_rnd(29, 8, hidden_dims=[256, 256, 256], env=FLAGS.type)
    rnd_key = jax.random.PRNGKey(42)
    rnd_ep_bonus = 0
    rnd_ep_loss = 0
    rnd_multiplier = float(1 / 2560)

    observation, done = env.reset(), False
    for i in tqdm.tqdm(
        range(0, FLAGS.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm
    ):
        if i < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action, agent = agent.sample_actions(observation)
        next_observation, reward, done, info = env.step(action)
        
        if use_rnd and i % rnd_update_freq == 0:
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
        observation = next_observation

        if done:
            wandb.log(
                {f"training/rnd_avg_bonus": rnd_ep_bonus / info["episode"]['l']}, step=i + FLAGS.pretrain_steps)
            rnd_ep_bonus = 0
            wandb.log(
                {f"training/rnd_avg_loss": rnd_ep_loss / info["episode"]['l']}, step=i + FLAGS.pretrain_steps)
            rnd_ep_loss = 0
            
            observation, done = env.reset(), False
            for k, v in info["episode"].items():
                decode = {"r": "return", "l": "length", "t": "time"}
                wandb.log({f"training/{decode[k]}": v}, step=i + FLAGS.pretrain_steps)
            

        if i >= FLAGS.start_training:
            online_batch = replay_buffer.sample(
                int(FLAGS.batch_size * FLAGS.utd_ratio * (1 - FLAGS.offline_ratio))
            )
            batch = unfreeze(online_batch)
            
            if use_rnd and i > start_rnd:
                bonus = rnd_multiplier * rnd_bonus(rnd, batch['observations'], batch['actions'])
                rnd_ep_bonus += bonus.mean().item()
                batch["rewards"] += np.array(bonus)

            if "antmaze" in FLAGS.env_name:
                batch["rewards"] *= 10
                batch["rewards"] -= 1

            agent, update_info = agent.update(batch, FLAGS.utd_ratio)
            
            """if use_rnd and i % rnd_update_freq == 0:
                rnd_key, rnd, rnd_info = update_rnd(
                    rnd_key, rnd,
                    batch['observations'],
                    batch['actions'])
                loss = rnd_info['rnd_loss'].item()
                rnd_ep_loss += loss"""

            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    wandb.log({f"training/{k}": v}, step=i + FLAGS.pretrain_steps)

        if i % FLAGS.eval_interval == 0:
            eval_info = evaluate(
                agent,
                eval_env,
                num_episodes=FLAGS.eval_episodes,
                save_video=FLAGS.save_video,
                training_env=env,
                etype=FLAGS.type
            )

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

if __name__ == "__main__":
    app.run(main)
