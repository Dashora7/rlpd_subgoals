#! /usr/bin/env python
import os
import numpy as np
import tqdm
from absl import app, flags
from flax.core import FrozenDict
from ml_collections import config_flags
from flax.training import checkpoints

from PIL import Image
import wandb
from rlpd.data import MemoryEfficientReplayBuffer, ReplayBuffer
from rlpd.data import franka_utils
from rlpd.evaluation import evaluate
from rlpd.wrappers import wrap_pixels
from flax.core.frozen_dict import unfreeze, freeze
from pixel_rnd_tools import PixelRND
from rlpd.agents import DrQLearner
import matplotlib.pyplot as plt
import pickle
from jaxrl_m.networks import ensemblize
from jaxrl_m.common import shard_batch
import tensorflow as tf
import time
import copy
### cog imports ###
from envs import KitchenEnv
from gym.wrappers import TimeLimit, FilterObservation, RecordEpisodeStatistics
import types
import jax
import jax.numpy as jnp

FLAGS = flags.FLAGS

flags.DEFINE_string("project_name", "online-franka", "wandb project name.")
flags.DEFINE_string("env_name", "KitchenMicrowaveV0", "Environment name.")
flags.DEFINE_float("offline_icvf_ratio", 0.5, "Offline ratio for ICVF training only.")
flags.DEFINE_integer("vf_update_step", 100, "Offline ratio for ICVF training only.")
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
flags.DEFINE_string("save_dir", "exp_data_franka", "Directory to save checkpoints.")
flags.DEFINE_bool("checkpoint_model", False, "save model")
flags.DEFINE_bool("checkpoint_buffer", False, "save replay buffer")
flags.DEFINE_string("vf_path", None, "model path.")
flags.DEFINE_boolean("use_rnd", False, "Use Random Network Distillation")
flags.DEFINE_boolean("save_video", True, "Save videos")
flags.DEFINE_integer("utd_ratio", 1, "Update to data ratio.")

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
    for k in ['observations', 'next_observations', 'rewards', 'masks']:
        v = one_dict[k]
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
    assert FLAGS.offline_icvf_ratio >= 0.0 and FLAGS.offline_icvf_ratio <= 1.0
    use_vf = True # FLAGS.vf_path is not None
    wandb.init(project=FLAGS.project_name, entity="dashora7")
    wandb.config.update(FLAGS)

    if FLAGS.save_dir is not None:
        log_dir = os.path.join(
            FLAGS.save_dir,
            f"{FLAGS.env_name}-s{FLAGS.seed}-vf_{use_vf}-rnd_{FLAGS.use_rnd}",
        )
        print("logging to", log_dir)
        if FLAGS.checkpoint_model:
            chkpt_dir = os.path.join(log_dir, "checkpoints")
            os.makedirs(chkpt_dir, exist_ok=True)
        if FLAGS.checkpoint_buffer:
            buffer_dir = os.path.join(log_dir, "buffers")
            os.makedirs(buffer_dir, exist_ok=True)
    
    if FLAGS.env_name == "KitchenMicrowaveV0":
        env_name_alt = "microwave"
        goalname = "microwave"
        # max_path_length = 50
    elif FLAGS.env_name == "KitchenSlideCabinetV0":
        env_name_alt = "slidecabinet"
        goalname = "slide cabinet"
        #max_path_length = 50
    elif FLAGS.env_name == "KitchenHingeCabinetV0":
        env_name_alt = "hingecabinet"
        goalname = "hinge cabinet"
        #max_path_length = 50
    
    import gym
    pixel_keys = ('image',)
    envname = "kitchen-" + env_name_alt + "-v0"
    env = gym.make(envname)
    env = RecordEpisodeStatistics(env, deque_size=1)
    env.seed(FLAGS.seed)
    
    eval_env = gym.make(envname)
    eval_env = TimeLimit(eval_env)
    eval_env.seed(FLAGS.seed + 42)
    
    
    ## FOR ICVF ##
    # Make a GCS conditioned Replay Buffer
    # Init it with the offline data and keep track of the end size
    # Continuously add trajectories to it during online RL training
    # At every step, sample 50/50 online offline data to train the ICVF
    # At every step, sample 100 online data to train the RL agent
    
    ## FOR SIMPLE VF ## 
    # Run it like a simple RLPD system
    online_replay_buffer = MemoryEfficientReplayBuffer(
        env.observation_space, env.action_space, FLAGS.max_steps,
        pixel_keys=pixel_keys
    )
    online_replay_buffer.seed(FLAGS.seed)
    
    
    
    # ["microwave_custom_reset"]
    
    offline_ds, _ = franka_utils.get_franka_dataset_simple(
        ["microwave_custom_reset"], [1.0], v4=True
    )
    example_batch = offline_ds.sample(2)

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
        rnd_multiplier = 20.0 # float(1 / 10), 10.0
    
    if use_vf:
        start_vf = 0
        vf_multiplier = 0.001 # for value
        # vf_multiplier = 0.1 # for potential
        vf_ep_bonus = 0
        # make ICVF shaper in this file. See if it's faster    
        from src import icvf_learner as learner
        from src.icvf_networks import VFWithImage, SqueezedLayerNormMLP, SimpleVF
        from jaxrl_m.vision import encoders
        from flax.serialization import from_state_dict
        
        hidden_dims = (256, 256)
        if FLAGS.vf_path is not None:
            with tf.io.gfile.GFile(FLAGS.vf_path, 'rb') as f:
                vf_params = pickle.load(f)
            params = vf_params['agent']
            conf = vf_params['config']
            # vf_def = ensemblize(SqueezedLayerNormMLP, 2)(hidden_dims + (1,))
            # encoder_def = encoders['ViT-B16']()
            vf_def = ensemblize(SimpleVF, 2)(hidden_dims)# + (1,))
            encoder_def = encoders['atari']()
            value_def = VFWithImage(encoder_def, vf_def)
            vf_agent = learner.create_learner(
                seed=FLAGS.seed, observations=np.ones((1, 128, 128, 3)),
                value_def=value_def, simple_vf=True, **conf)
            vf_agent = from_state_dict(vf_agent, params)
        else:
            vf_def = ensemblize(SimpleVF, 2)(hidden_dims)# + (1,))
            encoder_def = encoders['atari']()
            value_def = VFWithImage(encoder_def, vf_def)
            vf_agent = learner.create_learner(
                seed=FLAGS.seed, observations=np.ones((1, 128, 128, 3)),
                value_def=value_def, simple_vf=True)
        
        def vf_value_fn(obs):
            return vf_agent.value(obs, train=False).mean(0)
        value_fn = jax.jit(vf_value_fn)
        
        # Bonus function
        def vf_bonus(s, s_prime, potential=False):
            assert len(s.shape) == 5
            N = s.shape[0]
            s_prime = jax.image.resize(
                jnp.squeeze(s_prime, axis=-1), (N, 128, 128, 3), 'bilinear')
            val_to_sg = value_fn(s_prime)
            if potential:
                s = jax.image.resize(
                    jnp.squeeze(s, axis=-1), (N, 128, 128, 3), 'bilinear')
                last_val_to_sg = value_fn(s)
                return val_to_sg - last_val_to_sg
            else:
                return val_to_sg
        vf_bonus = jax.jit(vf_bonus, static_argnums=(2,))
        
    # Training
    observation, done = env.reset(), False
    print('Observation shape:', observation['image'].shape)
    
    if use_vf:
        curried_vf = lambda s, s_prime: vf_bonus(s, s_prime)
    
    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm):
        if i < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action, agent = agent.sample_actions(observation)
        next_observation, reward, done, info = env.step(action)
        
        if not done or "TimeLimit.truncated" in info:
            mask = 1.0
        else:
            print("THIS SHOULD NOT HAPPEN")
            import sys; sys.exit(0)
            mask = 0.0
        
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
        
        online_replay_buffer.insert(
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
            if use_vf:
                curried_vf = lambda s, s_prime: vf_bonus(s, s_prime)
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
            
            if use_vf:
                wandb.log(
                    {f"training/vf_avg_bonus": vf_ep_bonus / info["episode"]['l']},
                    step=i + FLAGS.pretrain_steps)
            vf_ep_bonus = 0
        
        
        if i >= FLAGS.start_training:
            
            ## UPDATE VF ##
            if i % FLAGS.vf_update_step == 0:
                vf_online = online_replay_buffer.sample(
                    int(FLAGS.batch_size * FLAGS.utd_ratio * (1 - FLAGS.offline_icvf_ratio))
                )
                vf_online = unfreeze(vf_online)
                offline_batch = offline_ds.sample(
                    int(FLAGS.batch_size * FLAGS.offline_icvf_ratio * FLAGS.utd_ratio)
                )
                vf_online['observations'] = vf_online['observations']['image'][..., 0]
                vf_online['next_observations'] = vf_online['next_observations']['image'][..., 0]
                vf_online['rewards'] -= 1
                
                vf_agent, update_info = vf_agent.update_single(vf_online)
                if i % FLAGS.log_interval == 0:
                    for k, v in update_info.items():
                        wandb.log({f"training_online_vf/{k}": v}, step=i + FLAGS.pretrain_steps)
                vf_agent, update_info = vf_agent.update_single(offline_batch)
                if i % FLAGS.log_interval == 0:
                    for k, v in update_info.items():
                        wandb.log({f"training_offline_vf/{k}": v}, step=i + FLAGS.pretrain_steps)

            ## UPDATE AGENT ##
            # every n steps, refresh RB. this should speed up training
            # add visualization of value across fails and success
            # Run RLPD experiments
            # TODO: Using less data for speed up! Ensure fair comparison!
            online_batch = online_replay_buffer.sample(
                int(FLAGS.batch_size * FLAGS.utd_ratio * (1 - FLAGS.offline_icvf_ratio))
            )
            batch = unfreeze(online_batch)
            
            if FLAGS.use_rnd and i > start_rnd:
                bonus = rnd_multiplier * rnd.get_reward(freeze(online_batch))
                rnd_ep_bonus += bonus.mean().item()
                batch["rewards"] += np.array(bonus)
            if use_vf and i > start_vf:
                bonus_rew_vf = vf_multiplier * vf_bonus(
                    batch['observations']['image'],
                    batch['next_observations']['image'])
                batch["rewards"] += np.array(bonus_rew_vf)
                vf_ep_bonus += bonus_rew_vf.mean().item()
            
            agent, update_info = agent.update(freeze(batch), FLAGS.utd_ratio)
            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    wandb.log({f"training/{k}": v}, step=i + FLAGS.pretrain_steps)
        
        if i % FLAGS.eval_interval == 0:
            eval_info = evaluate(
                agent,
                eval_env,
                num_episodes=FLAGS.eval_episodes,
                save_video=FLAGS.save_video,
                vf=curried_vf,
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
                        pickle.dump(online_replay_buffer, f, pickle.HIGHEST_PROTOCOL)
                except:
                    print("Could not save agent buffer.")


if __name__ == "__main__":
    app.run(main)