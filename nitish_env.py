from d4rl.locomotion.ant import AntMazeEnv
from d4rl.locomotion import maze_env
import numpy as np
import gym
import pickle
from src import icvf_learner as learner
from src.icvf_networks import icvfs, create_icvf, LayerNormMLP
from src.subgoal_diffuser import GCDDPMBCAgent
from flax.serialization import from_state_dict
ENV_TYPE = 'large' # medium, large, small
from subgoals import SUBGOALS
SUBGOALS = SUBGOALS[ENV_TYPE]
import jax
import jax.numpy as jnp
import time
from subgoal_gen_tools import select_subgoal
obs_to_robot = lambda obs: obs[:2]

class NitishEnv(AntMazeEnv):
    def __init__(self, subgoal_reward=0.25, value_sg_rew=False, value_sg_reach=False,
                 icvf_norm=False, icvf_path=None, eps=0.5, subgoal_bonus=0.0, normalize=False,
                 goal_sample_freq=20, reward_clip=100.0, only_forward=True, goal_caching=False,
                 subgoal_gen=True, diffusion_path=None, sg_cond=True, sample_when_reached=True,
                 **kwargs_dict):
        self.sg_cond = sg_cond
        self.subgoals = SUBGOALS.copy()
        self.sample_when_reached = sample_when_reached
        self.subgoal_dist_factor = -1
        self.subgoal_gen = subgoal_gen
        self.subgoal_bonus = subgoal_bonus
        self.sg_indx = 0
        self.reward_clipper = lambda x: np.clip(x, -reward_clip, reward_clip)
        self.icvf_norm = icvf_norm
        self.only_forward = only_forward
        self.normalize = normalize
        self.goal_sample_freq = goal_sample_freq
        self.value_sg_rew = value_sg_rew
        self.value_sg_reach = value_sg_reach
        self.subgoal_caching = goal_caching
        self.sg_cache = []
        self.stepnum = 0
        assert not self.icvf_norm or icvf_path is not None, "Need to provide path to ICVF model!" 
        assert not self.value_sg_rew or self.icvf_norm, "Need to use ICVF for value reward!"
        assert not self.value_sg_reach or self.icvf_norm, "Need to use ICVF for value reward!"
        assert not self.subgoal_caching or self.subgoal_gen, "Need to use subgoal generation for caching!"
        assert not self.subgoal_caching or not self.sg_cond, "Caching only makes sense for an unconditioned case."
        assert not self.subgoal_caching or self.sample_when_reached, "Caching only makes sense if sampling when reached."
        
        if icvf_path is not None:   
            with open(icvf_path, 'rb') as f:
                icvf_params = pickle.load(f) 
            params = icvf_params['agent']
            conf = icvf_params['config']
            value_def = create_icvf('multilinear', hidden_dims=[256, 256])
            agent = learner.create_learner(
                seed=42, observations=np.ones((1, 29)),
                value_def=value_def, **conf)
            agent = from_state_dict(agent, params)
            self.icvf_fn = jax.jit(lambda a, b, c: agent.value(a, b, c).sum(0))
            def icvf_repr_fn(obs):
                return agent.value(obs, method='get_phi')
            def icvf_value_fn(obs, goal):
                obs = obs.reshape(1, -1)
                goal = goal.reshape(1, -1)
                return -1 * agent.value(obs, goal, goal).mean()
            
            self.value_fn = jax.jit(icvf_value_fn)
        
        if self.icvf_norm:
            self.state_repr_func = jax.jit(icvf_repr_fn)
        else:
            self.state_repr_func = obs_to_robot # gets x/y position from state
            
        self.norm_func = lambda x, y: np.linalg.norm(x - y) # default L2 norm function
        self.repr_dist = lambda x, y: self.norm_func(self.state_repr_func(x), self.state_repr_func(y))
        
        self.eps = eps
        self.state = None
        self.subgoal_reward = subgoal_reward
        
        if self.subgoal_gen:
            assert icvf_path is not None, "Need to use path to ICVF model for diffusion selection!"
            assert diffusion_path is not None, "Need to provide path to DDPM model!" 
            with open(diffusion_path, 'rb') as f:
                diff_params = pickle.load(f)
            params = diff_params['agent']
            conf = diff_params['config']
            encoder_def = LayerNormMLP((256, 256))
            rng = jax.random.PRNGKey(42)
            rng, construct_rng = jax.random.split(rng)
            d_agent = GCDDPMBCAgent.create(
                rng=construct_rng,
                observations=jnp.ones((1, 29)),
                goals=jnp.ones((1, 29)),
                actions=jnp.ones((1, 29)),
                encoder_def=encoder_def,
                conditional=True
            )
            d_agent = from_state_dict(d_agent, params)
            self.sample_and_select_subgoal = jax.jit(
                lambda obs, goal: select_subgoal(d_agent, self.icvf_fn, obs, goal, n=100, t=1)
            )
            self.subgoal = self.sample_and_select_subgoal(self.subgoals[0], self.subgoals[-1])
        else:
            self.subgoal = self.subgoals[1] # start at 0 + 1
        
        super().__init__(max_episode_steps=timeouts[ENV_TYPE], **kwargs_dict)
        

    def step(self, action):
        self.last_state = self.state
        obs, reward, done, info = super().step(action)
        self.state = obs
        
        if self.last_state is None:
            self.last_state = obs # account for first step
        
        if self.subgoal_bonus > 0 and self.repr_dist(self.state, self.subgoal) <= self.eps:
            reward += self.subgoal_bonus
        
        ### ASSIGN PROGRESS REWARD
        if self.value_sg_rew:
            val_to_sg = self.value_fn(self.state, self.subgoal) # negative value
            last_val_to_sg = self.value_fn(self.last_state, self.subgoal) # negative value
            val_diff = np.array(val_to_sg - last_val_to_sg).item()
            factor = (1 / self.subgoal_dist_factor) if self.normalize else 1
            reward -= self.reward_clipper(self.subgoal_reward * val_diff) * factor
        else:
            norm_to_sg = self.repr_dist(self.state, self.subgoal)
            last_norm_to_sg = self.repr_dist(self.last_state, self.subgoal)
            norm_diff = np.array(norm_to_sg - last_norm_to_sg).item()
            factor = (1 / self.subgoal_dist_factor) if self.normalize else 1
            reward -= self.reward_clipper(self.subgoal_reward * norm_diff) * factor
        self.stepnum += 1
        
        ### ASSIGN SUBGOALS AND GIVE BONUS FOR REACHING
        if self.stepnum % self.goal_sample_freq == 0:
            self.goal_init()
        
        if self.sg_cond:
            obs = np.concatenate([obs, self.subgoal[:2]])
        
        return obs, reward, done, info

    def reset(self):
        obs = super().reset()
        self.sg_indx = 0
        self.state = obs
        self.stepnum = 0
        self.last_state = obs
        self.subgoals = SUBGOALS.copy()
        self.goal_init(True)
        if self.sg_cond:
            obs = np.concatenate([obs, self.subgoal[:2]])
        return obs
    
    def check_cache_contents(self, new_subgoal):
        for sg in self.sg_cache:
            if self.value_fn(sg, new_subgoal) <= self.eps:
                return True
        return False

    def goal_init(self, reset=False):
        if self.subgoal_gen:
            if not self.sample_when_reached or self.repr_dist(self.state, self.subgoal) <= self.eps or reset:
                if self.subgoal_caching and self.sg_indx < len(self.sg_cache):
                    self.subgoal = self.sg_cache[self.sg_indx]
                    self.sg_indx += 1
                else:
                    if self.subgoal_caching and self.check_cache_contents(self.subgoal):
                        self.sg_cache.append(self.subgoal)
                    self.subgoal = self.sample_and_select_subgoal(self.state, self.subgoals[-1])
            
            if self.value_sg_reach:
                self.subgoal_dist_factor = np.array(self.value_fn(
                    self.subgoal,
                    self.subgoals[-1])).item()
            else:
                self.subgoal_dist_factor = np.array(self.repr_dist(
                    self.subgoal,
                    self.subgoals[-1])).item()
        else:
            if self.value_sg_reach:
                idx = np.argmin([self.value_fn(self.state, sg) for sg in self.subgoals])
                self.subgoal_dist_factor = np.array(self.value_fn(
                    self.subgoals[idx],
                    self.subgoals[min(idx + 1, len(self.subgoals) - 1)])).item()
            else:
                idx = np.argmin([self.repr_dist(self.state, sg) for sg in self.subgoals])
                self.subgoal_dist_factor = np.array(self.repr_dist(
                    self.subgoals[idx],
                    self.subgoals[min(idx + 1, len(self.subgoals) - 1)])).item()
            if not self.sample_when_reached or self.value_fn(self.state, self.subgoal) <= self.eps:
                self.subgoal = self.subgoals[min(idx + 1, len(self.subgoals) - 1)]
            if self.only_forward:
                self.subgoals = self.subgoals[idx:]


mmaps = {'small': maze_env.U_MAZE_TEST, 'medium': maze_env.BIG_MAZE_TEST, 'large': maze_env.HARDEST_MAZE_TEST}
timeouts = {'small': 700, 'medium': 1000, 'large': 1000}
ds_dict = {
    'small': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_v2/Ant_maze_u-maze_noisy_multistart_False_multigoal_False_sparse_fixed.hdf5',
    'medium': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_v2/Ant_maze_big-maze_noisy_multistart_True_multigoal_True_sparse_fixed.hdf5', 
    'large': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_v2/Ant_maze_hardest-maze_noisy_multistart_True_multigoal_True_sparse_fixed.hdf5'
}

kwargs_dict = {
    'maze_map': mmaps[ENV_TYPE],
    'reward_type': 'sparse', # don't use their dense, it sucks
    'dataset_url':ds_dict[ENV_TYPE],
    'non_zero_reset':False, 
    'eval':True,
    'maze_size_scaling': 4.0, # 4.0 default this makes row/col sizes for subgoal determination !
    'ref_min_score': 0.0,
    'ref_max_score': 1.0,
    'v2_resets': True
}
gym.envs.register(
     id='nitish-v0',
     entry_point='nitish_env:NitishEnv',
     max_episode_steps=timeouts[ENV_TYPE],
     kwargs=kwargs_dict,
)