from typing import Dict

import gym
import numpy as np

from rlpd.wrappers.wandb_video import WANDBVideo
from rlpd.wrappers.cog_video_wandb import COGWANDBVideo

def evaluate(
    agent, env: gym.Env, num_episodes: int, save_video: bool = False, training_env=None, etype=None, cog=False
) -> Dict[str, float]:
    if save_video:
        env = COGWANDBVideo(env, name="eval_video", max_videos=1)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=num_episodes)
    for i in range(num_episodes):
        observation, done = env.reset(), False
        while not done:
            action = agent.eval_actions(observation)
            observation, _, done, info = env.step(action)
    return {"return": np.mean(env.return_queue), "length": np.mean(env.length_queue)}
