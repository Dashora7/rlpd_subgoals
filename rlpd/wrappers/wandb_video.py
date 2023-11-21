from typing import Optional

import gym
import numpy as np

import wandb


class WANDBVideo(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        name: str = "video",
        pixel_hw: int = 200, # 84
        render_kwargs={},
        max_videos: Optional[int] = None,
        nitish_env=False,
        nitish_type=None
    ):
        super().__init__(env)
        assert not nitish_env or nitish_type is not None, "Need to provide nitish_type if using nitish env!"
        assert nitish_type in ['small', 'medium', 'large'], "Nitish type must be small, medium, or large!"
        self._name = name
        self._nitish_env = nitish_env
        self._nitish_type = nitish_type
        self._pixel_hw = pixel_hw
        self._render_kwargs = render_kwargs
        self._max_videos = max_videos
        self._video = []
        

    def _add_frame(self, obs):
        if self._max_videos is not None and self._max_videos <= 0:
            return
        if isinstance(obs, dict) and "pixels" in obs:
            if obs["pixels"].ndim == 4:
                self._video.append(obs["pixels"][..., -1])
            else:
                self._video.append(obs["pixels"])
        else:
            if self._nitish_env:
                try:
                    if self._nitish_type == 'small':
                        self.env.viewer.cam.lookat[0] = 4
                        self.env.viewer.cam.lookat[1] = 4
                        self.env.viewer.cam.distance = 20
                    elif self._nitish_type == 'medium':
                        self.env.viewer.cam.lookat[0] = 10
                        self.env.viewer.cam.lookat[1] = 10
                        self.env.viewer.cam.distance = 40
                    elif self._nitish_type == 'large':
                        self.env.viewer.cam.lookat[0] = 20
                        self.env.viewer.cam.lookat[1] = 20
                        self.env.viewer.cam.distance = 70
                        
                        
                    self.env.viewer.add_marker(
                        pos=np.array([self.subgoal[0], self.subgoal[1], 0.5]),
                        type=2, size=np.array([0.75, 0.75, 0.75]), label="",
                        rgba=np.array([1.0, 0.0, 0.0, 1.0]))
                except:
                    pass
                
            self._video.append(
                self.render(
                    height=self._pixel_hw,
                    width=self._pixel_hw,
                    mode="rgb_array",
                    **self._render_kwargs
                )
            )
            if self._nitish_env:
                self.env.viewer._markers.clear()

    def reset(self, **kwargs):
        self._video.clear()
        obs = super().reset(**kwargs)
        self._add_frame(obs)
        return obs

    def step(self, action: np.ndarray):

        obs, reward, done, info = super().step(action)
        self._add_frame(obs)

        if done and len(self._video) > 0:
            if self._max_videos is not None:
                self._max_videos -= 1
            video = np.moveaxis(np.stack(self._video), -1, 1)
            if video.shape[1] == 1:
                video = np.repeat(video, 3, 1)
            video = wandb.Video(video, fps=20, format="mp4")
            wandb.log({self._name: video}, commit=False)

        return obs, reward, done, info
