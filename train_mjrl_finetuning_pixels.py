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
import mj_envs.envs.relay_kitchen
import gym
from gym.wrappers import TimeLimit, FilterObservation, RecordEpisodeStatistics

FLAGS = flags.FLAGS

flags.DEFINE_string("project_name", "online-mjrl", "wandb project name.")
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
flags.DEFINE_string('icvf_path', None, 'Path to the ICVF Model.')
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

class MuJoCoPixelObs(gym.ObservationWrapper):
    def __init__(self, env, width, height, camera_name, device_id=-1, depth=False, *args, **kwargs):
        gym.ObservationWrapper.__init__(self, env)
        img_obs = gym.spaces.Box(low=0., high=255., shape=(width, height, 3, 1))
        self.observation_space = gym.spaces.Dict({'image': img_obs})
        self.width = width
        self.height = height
        self.camera_name = camera_name
        self.depth = depth
        self.device_id = device_id
    def get_image(self):
        img = self.sim.render(width=self.width, height=self.height, depth=self.depth,
                              camera_name=self.camera_name, device_id=self.device_id)
        img = img[::-1,:,:]
        return {'image': img[..., None]}
    def observation(self, observation):
        return self.get_image()
    
        
class SparseRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        self.env = env
    def reward(self, reward):
        return float(self.env.get_env_infos()['solved']) - 1


def main(_):
    wandb.init(project=FLAGS.project_name, entity="dashora7")
    wandb.config.update(FLAGS)
    if FLAGS.save_dir is not None:
        log_dir = os.path.join(
            FLAGS.save_dir,
            f"{FLAGS.env_name}-s{FLAGS.seed}-rlpd-rnd_{FLAGS.use_rnd}",
        )
        print("logging to", log_dir)
        if FLAGS.checkpoint_model:
            chkpt_dir = os.path.join(log_dir, "checkpoints")
            os.makedirs(chkpt_dir, exist_ok=True)
        if FLAGS.checkpoint_buffer:
            buffer_dir = os.path.join(log_dir, "buffers")
            os.makedirs(buffer_dir, exist_ok=True)
    
    if FLAGS.env_name == "KitchenMicrowaveV0":
        envname = 'kitchen_micro_open-v3'
    elif FLAGS.env_name == "KitchenSlideCabinetV0":
        envname = 'kitchen_sdoor_open-v3'
    elif FLAGS.env_name == "KitchenHingeCabinetV0":
        envname = 'kitchen_ldoor_open-v3'
    
    import gym
    pixel_keys = ('image',)
    env = gym.make(envname)
    env = MuJoCoPixelObs(env, 128, 128, 'left_cap2', -1, False)
    env = SparseRewardWrapper(env)
    env = RecordEpisodeStatistics(env, deque_size=1)
    env.seed(FLAGS.seed)
    
    eval_env = gym.make(envname)
    eval_env = MuJoCoPixelObs(env, 128, 128, 'left_cap2', -1, False)
    eval_env = SparseRewardWrapper(eval_env)
    eval_env = TimeLimit(eval_env)
    eval_env.seed(FLAGS.seed + 42)

    online_replay_buffer = MemoryEfficientReplayBuffer(
        env.observation_space, env.action_space, FLAGS.max_steps,
        pixel_keys=pixel_keys
    )
    online_replay_buffer.seed(FLAGS.seed)
    
    offline_ds, _ = franka_utils.get_franka_dataset_rlpd(
        ["dibya_micro_open"], [1.0], v4=False, offline=True
    )
    example_batch = offline_ds.sample(2)

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
    
    # Training
    observation, done = env.reset(), False
    print('Observation shape:', observation['image'].shape)
    
    for i in tqdm.tqdm(range(1, FLAGS.max_steps), smoothing=0.1, disable=not FLAGS.tqdm):
        if i < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action, agent = agent.sample_actions(observation)
        
        next_observation, reward, done, info = env.step(action)

        if not done or "TimeLimit.truncated" in info:
            mask = 1.0
        else:
            mask = 0.0

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
        
        if i >= FLAGS.start_training:
            
            online_batch = unfreeze(online_replay_buffer.sample(
                int(FLAGS.batch_size * FLAGS.utd_ratio * (1 - FLAGS.offline_ratio))
            ))
            
            offline_batch = offline_ds.sample(
                    int(FLAGS.batch_size * FLAGS.offline_ratio * FLAGS.utd_ratio)
            )
            N = offline_batch['observations'].shape[0]
            offline_batch['observations'] = jax.image.resize(offline_batch['observations'], (N, 128, 128, 3), 'bilinear')
            offline_batch['next_observations'] = jax.image.resize(offline_batch['next_observations'], (N, 128, 128, 3), 'bilinear')
            offline_batch['observations'] = FrozenDict({'image': offline_batch['observations'][..., None]})
            offline_batch['next_observations'] = FrozenDict({'image': offline_batch['next_observations'][..., None]})
            batch = combine(offline_batch, online_batch)
            agent, update_info = agent.update(batch, FLAGS.utd_ratio)

            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    wandb.log({f"training/{k}": v}, step=i)
        
        if i % FLAGS.eval_interval == 0:
            # print("evaluating at step", i, "Currently done?", done)
            eval_info = evaluate(
                agent,
                eval_env,
                num_episodes=FLAGS.eval_episodes,
                save_video=FLAGS.save_video,
            )
            for k, v in eval_info.items():
                wandb.log({f"evaluation/{k}": v}, step=i)
            
            observation, done = env.reset(), False
            
            if FLAGS.save_dir is not None:
                checkpoints.save_checkpoint(
                    FLAGS.save_dir, target=agent, step=i, overwrite=True
                )

if __name__ == "__main__":
    app.run(main)
