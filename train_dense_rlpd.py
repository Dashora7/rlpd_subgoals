#! /usr/bin/env python
import numpy as np
import tqdm
from absl import app, flags
from flax.core.frozen_dict import unfreeze, freeze
from flax.core import FrozenDict
from ml_collections import config_flags
from flax.training import checkpoints

import wandb
from rlpd.agents import DrQLearner
from rlpd.data import MemoryEfficientReplayBuffer, ReplayBuffer
from rlpd.evaluation import evaluate
from rlpd.wrappers import WANDBVideo, wrap_pixels
import os
import jax
from rlpd.data import franka_utils
from envs import KitchenEnv
import gym
from PIL import Image
from gym.wrappers import TimeLimit, FilterObservation, RecordEpisodeStatistics

FLAGS = flags.FLAGS

flags.DEFINE_string("project_name", "final-online-franka", "wandb project name.")
flags.DEFINE_string("env_name", "KitchenMicrowave-v0", "Environment name.")
flags.DEFINE_float("offline_ratio", 0.5, "Offline ratio.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 10, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 5000, "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(5e5), "Number of training steps.")
flags.DEFINE_integer(
    "start_training", int(5e3), "Number of training steps to start training."
)
flags.DEFINE_string('icvf_path', None, 'Path to the ICVF model to change reward.')
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("save_video", True, "Save videos during evaluation.")
flags.DEFINE_string("save_dir", None, "Directory to save checkpoints.")
flags.DEFINE_integer("utd_ratio", 1, "Update to data ratio.")
config_flags.DEFINE_config_file(
    "config",
    "configs/drq_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

def combine(one_dict, other_dict):
    combined = {}

    for k, v in one_dict.items():
        if isinstance(v, FrozenDict):
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


def main(_):
    icvf_relabel = False if FLAGS.icvf_path is None else True
    wandb.init(project=FLAGS.project_name, entity="dashora7")
    wandb.config.update(FLAGS)
    if FLAGS.save_dir is not None:
        log_dir = os.path.join(
            FLAGS.save_dir,
            f"{FLAGS.env_name}-s{FLAGS.seed}-rlpd",
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
        goalpath = "/global/scratch/users/dashora7/misc_data/grasp.png" 
        # max_path_length = 50
        # goalpath = "/global/scratch/users/dashora7/misc_data/dibya_custom_micro_goal.png" 
        # goalpath = "/home/dashora7/franka_misc_data/dibya_custom_micro_goal.png"
    elif FLAGS.env_name == "KitchenSlideCabinetV0":
        env_name_alt = "slidecabinet"
        goalname = "slide cabinet"
        goalpath = "/global/scratch/users/dashora7/misc_data/slide_opt.png" 
        #max_path_length = 50
    elif FLAGS.env_name == "KitchenHingeCabinetV0":
        env_name_alt = "hingecabinet"
        goalpath = "/global/scratch/users/dashora7/misc_data/hinge_retry.png" 
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

    online_replay_buffer = MemoryEfficientReplayBuffer(
        env.observation_space, env.action_space, FLAGS.max_steps,
        pixel_keys=pixel_keys
    )
    online_replay_buffer.seed(FLAGS.seed)
    
    offline_ds, _ = franka_utils.get_franka_dataset_rlpd(
        ["micro-failsonly-fs40-jvel"],
        [1.0], v4=False, offline=True, brc=True
        #"microwave-custom-failures-reset-jvel.npy"  
        #"hinge-failsonly-fs40-jvel-retry"
        #"slidedoor-failsonly-fs40-jvel",
        #"micro-failsonly-fs40-jvel"],
        #[1.0, 1.0, 1.0], v4=False, offline=True, brc=True
    )
    example_batch = offline_ds.sample(1)

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
    vf_obj = None 
    icvf_ep_bonus = 0 
    if icvf_relabel:
        start_icvf = 0
        icvf_multiplier = 0.001 # for value
        # icvf_multiplier = 0.1 # for potential
        # make ICVF shaper in this file. See if it's faster    
        from src import icvf_learner as learner
        from src.icvf_networks import create_icvf, ICVFViT, SqueezedLayerNormMLP, MonolithicVF, ICVFWithEncoder
        from jaxrl_m.vision import encoders
        from jaxrl_m.networks import ensemblize
        from flax.serialization import from_state_dict
        import tensorflow as tf
        import pickle
        import jax.numpy as jnp
        
        with tf.io.gfile.GFile(FLAGS.icvf_path, 'rb') as f:
            icvf_params = pickle.load(f)
        params = icvf_params['agent']
        conf = icvf_params['config']
        hidden_dims = (256, 256)
        # icvf_def = ensemblize(SqueezedLayerNormMLP, 2)(hidden_dims + (1,))
        # encoder_def = encoders['ViT-B16']()
        # value_def = ICVFViT(encoder_def, icvf_def)
        
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
                s = jax.image.resize(
                    jnp.squeeze(s, axis=-1), (N, 128, 128, 3), 'bilinear')
                last_val_to_sg = value_fn(s, goal)
                return val_to_sg - last_val_to_sg
            else:
                return val_to_sg
        icvf_bonus = jax.jit(icvf_bonus, static_argnums=(3,))
        vf_obj = lambda s, sp: icvf_bonus(s, sp, goal_img[None])
    
    # Training
    observation, done = env.reset(), False
    print('Observation shape:', observation['image'].shape)
    goal_img = np.array(Image.open(goalpath).resize((128, 128)))

    for i in tqdm.tqdm(range(1, FLAGS.max_steps), smoothing=0.1, disable=not FLAGS.tqdm):
        if i < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action, agent = agent.sample_actions(observation)
        next_observation, reward, done, info = env.step(action)

        #if not done or "TimeLimit.truncated" in info:
        if True:
            mask = 1.0
        else:
            mask = 0.0
        
        if icvf_relabel:
            bonus_rew_icvf = icvf_multiplier * icvf_bonus(
                observation['image'][None],
                next_observation['image'][None], goal_img[None])
            reward += np.array(bonus_rew_icvf)
            icvf_ep_bonus += bonus_rew_icvf.item()

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
            for k, v in info["episode"].items():
                decode = {"r": "return", "l": "length", "t": "time"}
                wandb.log({f"training/{decode[k]}": v}, step=i)
            wandb.log({"training/icvf_bonus": icvf_ep_bonus}, step=i)
            icvf_ep_bonus = 0

        if i >= FLAGS.start_training:
            # Get online batch
            online_batch = unfreeze(online_replay_buffer.sample(
                int(FLAGS.batch_size * FLAGS.utd_ratio * (1 - FLAGS.offline_ratio))
            ))
            online_batch['rewards'] -= 1
            # Get offline batch
            offline_batch = offline_ds.sample(
                    int(FLAGS.batch_size * FLAGS.offline_ratio * FLAGS.utd_ratio)
            )
            N = offline_batch['observations'].shape[0]
            offline_batch['observations'] = jax.image.resize(offline_batch['observations'], (N, 128, 128, 3), 'bilinear')
            offline_batch['next_observations'] = jax.image.resize(offline_batch['next_observations'], (N, 128, 128, 3), 'bilinear')
            if icvf_relabel:
                bonus_rew_icvf = icvf_multiplier * icvf_bonus(
                    offline_batch['observations'][..., None],
                    offline_batch['next_observations'][..., None],
                    jnp.repeat(goal_img[None], N, axis=0),
                )
                offline_batch["rewards"] += np.array(bonus_rew_icvf)
                
            offline_batch['observations'] = FrozenDict({'image': offline_batch['observations'][..., None]})
            offline_batch['next_observations'] = FrozenDict({'image': offline_batch['next_observations'][..., None]})
            
            batch = combine(offline_batch, online_batch)
            # batch = online_batch 
            agent, update_info = agent.update(batch, FLAGS.utd_ratio)

            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    wandb.log({f"training/{k}": v}, step=i)

        if i % FLAGS.eval_interval == 0:
            eval_info = evaluate(
                agent,
                eval_env,
                num_episodes=FLAGS.eval_episodes,
                save_video=FLAGS.save_video,
                vf=vf_obj
            )
            for k, v in eval_info.items():
                wandb.log({f"evaluation/{k}": v}, step=i)

            if FLAGS.save_dir is not None:
                checkpoints.save_checkpoint(
                    FLAGS.save_dir, target=agent, step=i, overwrite=True
                )

if __name__ == "__main__":
    app.run(main)
