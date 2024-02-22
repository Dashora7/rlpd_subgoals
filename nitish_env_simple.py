from d4rl.locomotion.ant import AntMazeEnv
from d4rl.locomotion import maze_env
import numpy as np
import gym
import pickle
from src import icvf_learner as learner
from src.icvf_networks import icvfs, create_icvf, LayerNormMLP
from flax.serialization import from_state_dict
from subgoals import SUBGOALS
import jax
import jax.numpy as jnp
obs_to_robot = lambda obs: obs[:2]

class NitishEnvSimple(AntMazeEnv):
    def __init__(self, icvf_path=None, eps=0.5, reward_clip=100.0, etype='small',
                 prog_rew_factor=1.0, normalize=False, icvf_norm=True, multigoal=False, 
                 **kwargs_dict):
        self.reward_clipper = lambda x: np.clip(x, -reward_clip, reward_clip)
        self.stepnum = 0
        self.goal_sampler = lambda np_rand: maze_env.MazeEnv.goal_sampler(self, np_rand)
        self.normalize = normalize
        self.icvf_norm = icvf_norm
        sgs = SUBGOALS[etype]
        self.goal = sgs[-1]
        self.prog_rew_factor = prog_rew_factor
        self.multigoal = multigoal
        
        if self.icvf_norm:
            assert icvf_path is not None, "Need to provide path to ICVF model!"    
            with open(icvf_path, 'rb') as f:
                icvf_params = pickle.load(f)
            
            params = icvf_params['agent']
            conf = icvf_params['config']
            value_def = create_icvf('monolithic', hidden_dims=[512, 512, 512])
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
            
            self.state_repr_func = jax.jit(icvf_repr_fn)
            self.value_fn = jax.jit(icvf_value_fn)
        else:
            self.state_repr_func = obs_to_robot # gets x/y position from state
            
        self.norm_func = lambda x, y: np.linalg.norm(x - y) # default L2 norm function
        self.repr_dist = lambda x, y: self.norm_func(self.state_repr_func(x), self.state_repr_func(y))
        
        self.eps = eps
        self.s, self.s_prime = None, None
        self.start_val = 0
        self.goal_val = 1
        super().__init__(self, max_episode_steps=timeouts[etype], **kwargs_dict)
        

    def step(self, action):
        
        self.s = self.s_prime
        obs, reward, done, info = super().step(action)
        
        self.s_prime = obs
        if self.s is None:
            self.s = obs
        
        if reward > 0.0:
            done = True
        
        #if self.multigoal:
        #    obs = np.concatenate([obs, self.target_goal])
        
        
        
        """
        ### ASSIGN PROGRESS REWARD
        val_to_sg = self.value_fn(self.s_prime, self.goal) # negative value
        last_val_to_sg = self.value_fn(self.s, self.goal) # negative value
        val_diff = np.array(val_to_sg - last_val_to_sg).item()
        reward -= self.reward_clipper(self.prog_rew_factor * val_diff)
        ### ASSIGN VALUE REWARD
        if self.normalize:
            reward += self.rescale(np.array(self.value_fn(self.s_prime, self.goal)))
        else:
            val_to_sg = self.value_fn(self.s_prime, self.goal) # negative value
            reward -= self.reward_clipper(self.prog_rew_factor * np.array(val_to_sg))
        """
        self.stepnum += 1
        
        return obs, reward, done, info

    def rescale(self, reward):
        return (reward - self.start_val) / (self.goal_val - self.start_val)
    
    def reset(self):
        if self.multigoal:
            self.target_goal = None
        obs = super().reset()
        self.s = obs
        # self.goal_val = np.array(-1 * self.value_fn(self.goal, self.goal)).item()
        # self.start_val = np.array(-1 * self.value_fn(obs, self.goal)).item()
        self.s_prime = obs
        self.stepnum = 0
        return obs

# Changed to not test. Ensure that we use eval for the eval env. Need right map


mmaps = {'small': maze_env.U_MAZE_TEST, 'medium': maze_env.BIG_MAZE_TEST, 'large': maze_env.HARDEST_MAZE_TEST, 'large_multigoal': maze_env.HARDEST_MAZE_EVAL}
timeouts = {'small': 700, 'medium': 1000, 'large': 1000}
ds_dict = {
    'small': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_v2/Ant_maze_u-maze_noisy_multistart_False_multigoal_False_sparse_fixed.hdf5',
    'medium': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_v2/Ant_maze_big-maze_noisy_multistart_True_multigoal_True_sparse_fixed.hdf5', 
    'large': 'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_v2/Ant_maze_hardest-maze_noisy_multistart_True_multigoal_False_sparse_fixed.hdf5'
}

# We're using eval false
def register_simple(env_type, multigoal=False):
    kwargs_dict = {
        'maze_map': mmaps[env_type + '_multigoal' if multigoal else env_type],
        'reward_type': 'dense',
        'dataset_url': ds_dict[env_type],
        'non_zero_reset': False, 
        'eval': not multigoal,
        'maze_size_scaling': 4.0, # 4.0 default this makes row/col sizes for subgoal determination !
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
        'v2_resets': True
    }

    gym.envs.register(
        id='nitish-v0-simple',
        entry_point='nitish_env_simple:NitishEnvSimple',
        max_episode_steps=timeouts[env_type],
        kwargs=kwargs_dict,
    )