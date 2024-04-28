#! /usr/bin/env python
import os
import numpy as np
import tqdm
from absl import app, flags
from flax.core import FrozenDict
from ml_collections import config_flags
from flax.training import checkpoints
import flax

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
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", 250000, "Number of training steps.")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_integer("utd_ratio", 1, "Update to data ratio.")
flags.DEFINE_integer("save_interval", 10000, "Interval to save model.")
flags.DEFINE_string("save_dir", "gs://rail-tpus-nitish-v4/rnd_franka", "Directory to save model.")

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

def main(_):
    wandb.init(project=FLAGS.project_name, entity="dashora7")
    wandb.config.update(FLAGS)
    
    if FLAGS.env_name == "KitchenMicrowaveV0":
        env_name_alt = "microwave"
    elif FLAGS.env_name == "KitchenSlideCabinetV0":
        env_name_alt = "slidecabinet"
    elif FLAGS.env_name == "KitchenHingeCabinetV0":
        env_name_alt = "hingecabinet"
    
    import gym
    pixel_keys = ('image',)
    envname = "kitchen-" + env_name_alt + "-v0"
    env = gym.make(envname)
    env = RecordEpisodeStatistics(env, deque_size=1)
    env.seed(FLAGS.seed)
    
    eval_env = gym.make(envname)
    eval_env = TimeLimit(eval_env)
    eval_env.seed(FLAGS.seed + 42)
    
    offline_ds, _ = franka_utils.get_franka_dataset_simple(
        ["franka_slidecabinet_ds"], [1.0], v4=True
    )
    example_batch = offline_ds.sample(2)
    
    # Setup RND Parameters
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
    rnd_ep_loss = 0
    
    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm):
        offline_batch = offline_ds.sample(int(FLAGS.batch_size))
        offline_batch['observations'] = {'image': offline_batch['observations'][..., None]}
        offline_batch['next_observations'] = {'image': offline_batch['next_observations'][..., None]}
        rnd, rnd_update_info = rnd.update(freeze(copy.deepcopy(offline_batch)))
        loss = rnd_update_info['rnd_loss'].item()
        wandb.log({f"training/rnd_avg_loss": loss}, step=i)
        
        if i % FLAGS.save_interval == 0:
            fname = os.path.join(FLAGS.save_dir, f'params-{i}.pkl')
            save_dict = flax.serialization.to_state_dict(rnd)
            pickle_byte_obj = pickle.dumps(save_dict)
            with tf.io.gfile.GFile(fname, 'w') as gf:
                gf.write(pickle_byte_obj)
                gf.close()
        
if __name__ == "__main__":
    app.run(main)