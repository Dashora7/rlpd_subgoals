from d4rl.locomotion.ant import AntMazeEnv
from d4rl.locomotion import maze_env
import numpy as np
import gym
"""[4.14765933 0.38331625]                                                                                                                                                                        [3/1880]
[8.44816448 0.790448  ]
[8.56858188 4.29370929]
[8.76939131 8.83480101]
[4.10515023 8.84229782]
[0.46984344 9.03254701]"""
# TODO: render subgoals
# TODO: bug check these rewards and subgoal positions, etc
# TODO: make your own subgoals hardcoded, one for eval, one for test
# TODO: add normalization to subgoal distance. 
class NitishEnv(AntMazeEnv):
    def __init__(self, subgoal_reward=0.2, subgoal_dense=False, **kwargs_dict):
        self.subgoal = np.array([-10, -10])
        self.subgoal_dist_factor = -1
        self.progress = True
        self.subgoal_dense = subgoal_dense
        self.state = None
        self.subgoal_reward = subgoal_reward
        super().__init__(max_episode_steps=700, **kwargs_dict)
        self.set_subgoal()

    def step(self, action):
        self.last_state = self.state
        obs, reward, done, info = super().step(action)
        self.state = obs
        if self.last_state is None:
            self.last_state = obs # account for first step
        if np.linalg.norm(self.get_xy() - self.subgoal) <= 0.5:
            # print("Reached subgoal!")
            reward += self.subgoal_reward
            self.set_subgoal()
            # print("New subgoal:", self.subgoal)
        elif self.subgoal_dense:
            if self.progress:
                # Hopefully helps hacking by only rewarding making progress towards a subgoal.
                norm_diff = np.linalg.norm(self.get_xy() - self.subgoal) - np.linalg.norm(obs_to_robot(self.last_state) - self.subgoal)
                reward -= (1 / self.subgoal_dist_factor) * self.subgoal_reward * norm_diff # progress based, normalized
            else:
                # Would probably cause hacking by standing very close to a subgoal, unless the reward of reaching the subgoal is high
                # Switching to next subgoal is supposed to be good, but this reward penalizes that!
                reward -= self.subgoal_reward * np.linalg.norm(self.get_xy() - self.subgoal)
                # print("Rew logger", self.get_xy(), self.subgoal, reward)
        return obs, reward, done, info

    # set subgoals
    def set_subgoal(self):
        robot_x, robot_y = obs_to_robot(self.state)
        robot_row, robot_col = self._xy_to_rowcol([robot_x, robot_y])
        target_x, target_y = self.target_goal
        target_row, target_col = self._xy_to_rowcol([target_x, target_y])
        waypoint_row, waypoint_col = self._get_best_next_rowcol(
          [robot_row, robot_col], [target_row, target_col])
      
        if waypoint_row == target_row and waypoint_col == target_col:
            waypoint_x = target_x
            waypoint_y = target_y
        else:
            waypoint_x, waypoint_y = self._rowcol_to_xy(
                [waypoint_row, waypoint_col],
                add_random_noise=True)

        goal = np.array([waypoint_x, waypoint_y])
        self.subgoal_dist_factor = np.linalg.norm(self.subgoal - goal) # dist between subgoals
        self.subgoal = goal

    def reset(self):
        obs = super().reset()
        self.state = obs
        self.last_state = obs
        self.set_subgoal()
        return obs


kwargs_dict = {
    'maze_map': maze_env.U_MAZE_TEST,
    'reward_type':'sparse', # don't use their dense, it sucks
    'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_v2/Ant_maze_u-maze_noisy_multistart_False_multigoal_False_sparse_fixed.hdf5',
    'non_zero_reset':False, 
    'eval':True,
    'maze_size_scaling': 4.0, # 4.0 default this makes row/col sizes for subgoal determination !
    'ref_min_score': 0.0,
    'ref_max_score': 1.0,
    'v2_resets': True
}
obs_to_robot = lambda obs: obs[:2]
gym.envs.register(
     id='nitish-v0',
     entry_point='nitish_env:NitishEnv',
     max_episode_steps=700,
     kwargs=kwargs_dict,
)