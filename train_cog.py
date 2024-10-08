#! /usr/bin/env python
import os
import numpy as np
import tqdm
from absl import app, flags
from flax.core import FrozenDict
from ml_collections import config_flags
from flax.training import checkpoints

import wandb
from flax.serialization import from_state_dict
from rlpd.data import MemoryEfficientReplayBuffer, ReplayBuffer
from rlpd.evaluation import evaluate
from rlpd.wrappers import wrap_pixels
from flax.core.frozen_dict import unfreeze, freeze
from pixel_rnd_tools import PixelRND
from rlpd.agents import DrQLearner
import matplotlib.pyplot as plt
import pickle
from jaxrl_m.networks import ensemblize
import tensorflow as tf
import time
from PIL import Image

### cog imports ###
import roboverse
from gym.wrappers import TimeLimit, FilterObservation, RecordEpisodeStatistics
import src.cog.utils as cog_utils
import types

### cog imports ###
import jax
import jax.numpy as jnp


FLAGS = flags.FLAGS

flags.DEFINE_string("project_name", "icvf-cog-rl", "wandb project name.")
flags.DEFINE_string("env_name", "Widow250PickTray-v0", "Environment name.")
flags.DEFINE_float("offline_ratio", 0.0, "Offline ratio.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 100, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 5000, "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", 1000000, "Number of training steps.")
flags.DEFINE_integer(
    "start_training", 5000, "Number of training steps to start training."
)
flags.DEFINE_integer("pretrain_steps", 0, "Number of offline updates.")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_string("save_dir", "exp_data_cog", "Directory to save checkpoints.")
flags.DEFINE_bool("checkpoint_model", False, "save model")
flags.DEFINE_bool("checkpoint_buffer", False, "save replay buffer")
flags.DEFINE_string("icvf_path", None, "model path.")
flags.DEFINE_boolean("use_rnd", False, "Use Random Network Distillation")
flags.DEFINE_boolean("save_video", True, "Save videos")
flags.DEFINE_integer("utd_ratio", 1, "Update to data ratio.")
flags.DEFINE_string("offline_rnd_path", None, "Path to offline RND model for data support constraint.")
config_flags.DEFINE_config_file(
    "config",
    "configs/rlpd_pixels_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)
config_flags.DEFINE_config_file(
    "rnd_config",
    "configs/pixel_rnd_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

def combine(one_dict, other_dict):
    combined = {}
    for k, v in one_dict.items():
        if isinstance(v, FrozenDict) or isinstance(v, dict):
            if len(v) == 0:
                combined[k] = v
            else:
                combined[k] = combine(v, other_dict[k])
        else:
            tmp = np.empty(
                (v.shape[0] + other_dict[k].shape[0], *v.shape[1:]), dtype=v.dtype
            )
            tmp[0::2] = v
            tmp[1::2] = other_dict[k]
            combined[k] = tmp
    return FrozenDict(combined)

def add_prefix(prefix, dict):
    return {prefix + k: v for k, v in dict.items()}

def main(_):
    assert FLAGS.offline_ratio >= 0.0 and FLAGS.offline_ratio <= 1.0
    use_icvf = FLAGS.icvf_path is not None
    wandb.init(project=FLAGS.project_name, entity="dashora7")
    wandb.config.update(FLAGS)
    use_offline_rnd = FLAGS.offline_rnd_path is not None
    
    envgoals = {
        # 'pickplace': np.load('/home/dashora7/cog_misc_data/pickplace_goal_img.npy'),
        'pickplace': np.array(Image.open('/global/home/users/dashora7/cog_misc_data/pickplace_goal.png')),
        # 'closeddrawer': np.load('/global/home/users/dashora7/cog_misc_data/drawer_goal_img.npy')
    }
    
    
    for k, v in envgoals.items():
        envgoals[k] = jnp.expand_dims(
            jax.image.resize(v, (128, 128, 3), 'bilinear'), axis=0)

    if FLAGS.save_dir is not None:
        log_dir = os.path.join(
            FLAGS.save_dir,
            f"{FLAGS.env_name}-s{FLAGS.seed}-icvf_{use_icvf}-rnd_{FLAGS.use_rnd}",
        )
        print("logging to", log_dir)
        if FLAGS.checkpoint_model:
            chkpt_dir = os.path.join(log_dir, "checkpoints")
            os.makedirs(chkpt_dir, exist_ok=True)
        if FLAGS.checkpoint_buffer:
            buffer_dir = os.path.join(log_dir, "buffers")
            os.makedirs(buffer_dir, exist_ok=True)

    def wrap(env):
        return wrap_pixels(
            env,
            action_repeat=1,
            num_stack=1,
            camera_id=0,
        )

    def render(env, *args, **kwargs):
        return env.render_obs()

    if FLAGS.env_name == "Widow250PickTray-v0":
        env_name_alt = "pickplace"
        cog_max_path_length = 40
    elif FLAGS.env_name == "Widow250DoubleDrawerOpenGraspNeutral-v0":
        env_name_alt = "closeddrawer"
        cog_max_path_length = 50
    elif FLAGS.env_name == "Widow250DoubleDrawerCloseOpenGraspNeutral-v0":
        env_name_alt = "blockeddrawer1_small"
        cog_max_path_length = 80

    env = roboverse.make(FLAGS.env_name, transpose_image=False)
    env.render = types.MethodType(render, env)
    env = FilterObservation(env, ["image"])
    env = TimeLimit(env, max_episode_steps=cog_max_path_length)  # TODO
    env, pixel_keys = wrap(env)
    env = RecordEpisodeStatistics(env, deque_size=1)
    env.seed(FLAGS.seed)

    eval_env = roboverse.make(FLAGS.env_name, transpose_image=False)
    eval_env.render = types.MethodType(render, eval_env)
    eval_env = FilterObservation(eval_env, ["image"])
    eval_env = TimeLimit(eval_env, max_episode_steps=cog_max_path_length)  # TODO
    eval_env, _ = wrap(eval_env)
    eval_env.seed(FLAGS.seed + 42)
    
    goal_img = envgoals[env_name_alt]
    replay_buffer = MemoryEfficientReplayBuffer(
        env.observation_space, env.action_space, FLAGS.max_steps
    )
    
    replay_buffer.seed(FLAGS.seed)

    ########### MODELS ###########

    # Crashes on some setups if agent is created before replay buffer.
    kwargs = dict(FLAGS.config)
    model_cls = kwargs.pop("model_cls")
    agent = globals()[model_cls].create(
        FLAGS.seed,
        env.observation_space,
        env.action_space,
        pixel_keys=pixel_keys,
        **kwargs,
    )
    
    offline_ds, _ = cog_utils.get_cog_dataset_rlpd(
        ["pickplace_prior"],
        #["pickplace_prior",
        #"DrawerOpenGrasp",
        #"drawer_task",
        #"closed_drawer_prior",
        #"blocked_drawer_1_prior",
        #"blocked_drawer_2_prior"],
        [1.0], v4=False, offline=True, brc=True
        # [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], v4=False, offline=True, brc=True
        #["pickplace_task", "pickplace_prior"],
        #[1.0, 1.0], v4=False, offline=True
    )
    example_batch = offline_ds.sample(2)
    
    
    # Setup RND Parameters
    if FLAGS.use_rnd:
        kwargs = dict(FLAGS.rnd_config)
        model_cls = kwargs.pop("model_cls")
        rnd = globals()[model_cls].create(
            FLAGS.seed + 123,
            env.observation_space,
            env.action_space,
            pixel_keys=pixel_keys,
            **kwargs,
        )
        rnd_update_freq = 1
        start_rnd = 5000
        rnd_ep_bonus = 0
        rnd_ep_loss = 0
        rnd_multiplier = 1.0 # float(1 / 10)
    
    icvf_multiplier = 0.001 # for value
    # icvf_multiplier = 0.1 # for potential
    if use_icvf:
        start_icvf = 0
        icvf_ep_bonus = 0
        # make ICVF shaper in this file. See if it's faster    
        from src import icvf_learner as learner
        from src.icvf_networks import create_icvf, ICVFViT, SqueezedLayerNormMLP, MonolithicVF, ICVFWithEncoder
        from jaxrl_m.vision import encoders
        
        
        with tf.io.gfile.GFile(FLAGS.icvf_path, 'rb') as f:
            icvf_params = pickle.load(f)
        params = icvf_params['agent']
        conf = icvf_params['config']
        hidden_dims = (256, 256)
        
        #icvf_def = ensemblize(SqueezedLayerNormMLP, 2)(hidden_dims + (1,))
        #encoder_def = encoders['ViT-B16']()
        #value_def = ICVFViT(encoder_def, icvf_def)
        
        vf_def = ensemblize(MonolithicVF, 2)(hidden_dims)
        encoder_def = encoders['resnetv2-26-1-128']()
        value_def = ICVFWithEncoder(encoder_def, vf_def)
        
        icvf_agent = learner.create_learner(
            seed=FLAGS.seed, observations=np.ones((1, 128, 128, 3)),
            value_def=value_def, **conf)
        icvf_agent = from_state_dict(icvf_agent, params)
        
        def icvf_value_fn(obs, goal):
            return icvf_agent.value(obs, goal, goal, train=False).mean(0)
        value_fn = jax.jit(icvf_value_fn)
        # Bonus function
        def icvf_bonus(s, s_prime, goal, potential=False):
            assert len(s.shape) == 5
            N = s.shape[0]
            s_prime = jax.image.resize(
                jnp.squeeze(s_prime, axis=-1), (N, 128, 128, 3), 'bilinear')
            val_to_sg = value_fn(s_prime, goal)
            if potential:
                last_val_to_sg = value_fn(s, goal)
                return val_to_sg - last_val_to_sg
            else:
                return val_to_sg
        icvf_bonus = jax.jit(icvf_bonus, static_argnums=(3,))
        curried_icvf = lambda s, s_prime: icvf_bonus(s, s_prime, goal_img)
    else:
        curried_icvf = None
    
    if use_offline_rnd:
        off_rnd_multiplier = 10.0
        off_rnd_ep_penalty = 0
        kwargs = dict(FLAGS.rnd_config)
        model_cls = kwargs.pop("model_cls")
        off_rnd = globals()[model_cls].create(
            FLAGS.seed + 123,
            env.observation_space,
            env.action_space,
            pixel_keys=pixel_keys,
            **kwargs,
        )
        with tf.io.gfile.GFile(FLAGS.offline_rnd_path, 'rb') as f:
            rnd_params = pickle.load(f)
        off_rnd = from_state_dict(off_rnd, rnd_params)
        
        def rnd_viz(s):
            assert len(s.shape) == 5
            N = s.shape[0]
            dic = freeze({"observations": {'pixels': s}})
            return -1 * off_rnd.get_reward(dic) * (off_rnd_multiplier / icvf_multiplier) # make same scale for viz
        
    else:
        rnd_viz = None
    
    # Training
    observation, done = env.reset(), False

    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm):
        if i < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action, agent = agent.sample_actions(observation)
        next_observation, reward, done, info = env.step(action)
        
        if not done or "TimeLimit.truncated" in info:
            mask = 1.0
        else:
            mask = 0.0
        # mask = 1.0
        
        
        if FLAGS.use_rnd and i % rnd_update_freq == 0:
            rnd, rnd_update_info = rnd.update(
                freeze({
                    "observations": {k: ob[None] for k, ob in observation.items()},
                    "actions": action[None],
                    "next_observations": {k: ob[None] for k, ob in next_observation.items()},
                    "rewards": np.array(reward)[None],
                    "masks": np.array(mask)[None],
                    "dones": np.array(done)[None],
                })
            )
            loss = rnd_update_info['rnd_loss'].item()
            rnd_ep_loss += loss
        
        if use_icvf and i > start_icvf:
            bonus_rew_icvf = icvf_multiplier * icvf_bonus(
                observation['pixels'][None],
                next_observation['pixels'][None], goal_img)
            reward += np.array(bonus_rew_icvf)
            icvf_ep_bonus += bonus_rew_icvf.item()
        
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
            observation, done = env.reset(), False
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
            if use_offline_rnd:
                wandb.log(
                    {f"training/off_rnd_avg_penalty": off_rnd_ep_penalty / info["episode"]['l']},
                    step=i + FLAGS.pretrain_steps)
                off_rnd_ep_penalty = 0
            if use_icvf:
                wandb.log(
                    {f"training/icvf_avg_bonus": icvf_ep_bonus / info["episode"]['l']},
                    step=i + FLAGS.pretrain_steps)
            icvf_ep_bonus = 0
        
        
        if i >= FLAGS.start_training:
            online_batch = replay_buffer.sample(
                int(FLAGS.batch_size * FLAGS.utd_ratio * (1 - FLAGS.offline_ratio))
            )
            batch = unfreeze(online_batch)
            batch['rewards'] -= 1
            
            if FLAGS.offline_ratio > 0:
                offline_batch = offline_ds.sample(
                    int(FLAGS.batch_size * FLAGS.offline_ratio * FLAGS.utd_ratio)
                )
                N = offline_batch['observations'].shape[0]
                offline_batch['observations'] = jax.image.resize(offline_batch['observations'], (N, 48, 48, 3), 'bilinear')
                offline_batch['next_observations'] = jax.image.resize(offline_batch['next_observations'], (N, 48, 48, 3), 'bilinear')
                offline_batch['observations'] = FrozenDict({'pixels': offline_batch['observations'][..., None]})
                offline_batch['next_observations'] = FrozenDict({'pixels': offline_batch['next_observations'][..., None]})
                
                batch = combine(offline_batch, batch)
            
            if FLAGS.use_rnd and i > start_rnd:
                bonus = rnd_multiplier * rnd.get_reward(freeze(online_batch))
                rnd_ep_bonus += bonus.mean().item()
                batch["rewards"] += np.array(bonus)
            
            if use_offline_rnd:
                bonus = off_rnd_multiplier * off_rnd.get_reward(freeze(online_batch))
                off_rnd_ep_penalty += -1 * bonus.mean().item()
                batch["rewards"] += -1 * np.array(bonus)
            
            agent, update_info = agent.update(batch, FLAGS.utd_ratio)
            
            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    wandb.log({f"training/{k}": v}, step=i + FLAGS.pretrain_steps)
            
        if i % FLAGS.eval_interval == 0:
            eval_info = evaluate(
                agent,
                eval_env,
                num_episodes=FLAGS.eval_episodes,
                save_video=FLAGS.save_video,
                vf=curried_icvf,
                rnd=rnd_viz,
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
