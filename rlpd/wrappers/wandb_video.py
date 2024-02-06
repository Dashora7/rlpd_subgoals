from typing import Optional

import gym
import numpy as np
import matplotlib.pyplot as plt
import wandb
from matplotlib.backends.backend_agg import FigureCanvasAgg


class WANDBVideo(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        name: str = "video",
        pixel_hw: int = 200, # 84
        render_kwargs={},
        max_videos: Optional[int] = None,
        nitish_env=False,
        nitish_type=None,
        training_env=None
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
        self.training_env = training_env
        self._video = []
        self._plotdata = []
        self._plotvideo = []
        self._rewards = []
        self.lines = []

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
                    
                    if self.training_env is not None:
                        for cachegoal in self.training_env.sg_cache:
                            self.env.viewer.add_marker(
                                pos=np.array([cachegoal[0], cachegoal[1], 0.5]),
                                type=2, size=np.array([0.5, 0.5, 0.5]), label="",
                                rgba=np.array([0.0, 1.0, 0.0, 1.0]))
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

    def _add_plotdata(self, reward, info):
        val_to_sg = info.get("V(s, sg, sg)", None)
        if val_to_sg is not None:
            self._plotdata.append(val_to_sg)
        self._rewards.append(reward)
        self._update_plot()
        
    def _update_plot(self):
        if len(self.lines) == 0:
            _, (ax1, ax2)  = plt.subplots(1, 2, figsize=(12, 5))
            line = ax1.plot([], [])[0]
            ax1.set_title('Value to Subgoal')
            ax1.set_xlabel('Step')
            ax1.set_ylabel('V(s, sg, sg)')
            ax1.set_xlim(0, 1000)
            ax1.set_ylim(-20, 40)
            self.lines.append(line)
            line = ax2.plot([], [])[0]
            ax2.set_title('Reward')
            ax2.set_xlabel('Step')
            ax2.set_ylabel('R(s, s_prime)')
            ax2.set_xlim(0, 1000)
            ax2.set_ylim(-20, 20)
            self.lines.append(line)
            
        self.lines[0].set_data(np.arange(len(self._plotdata)), self._plotdata)
        self.lines[1].set_data(np.arange(len(self._rewards)), self._rewards)
        npimg = mplfig_to_npimage(plt.gcf())
        self._plotvideo.append(npimg)

    def reset(self, **kwargs):
        self._video.clear()
        self._plotdata.clear()
        self._plotvideo.clear()
        self._rewards.clear()
        self.lines = []
        plt.close()
        obs = super().reset(**kwargs)
        self._add_frame(obs)
        self._update_plot()
        return obs

    def step(self, action: np.ndarray):

        obs, reward, done, info = super().step(action)
        self._add_frame(obs)
        self._add_plotdata(reward, info)

        if done and len(self._video) > 0:
            if self._max_videos is not None:
                self._max_videos -= 1
            video = np.moveaxis(np.stack(self._video), -1, 1)
            pvid = np.moveaxis(np.stack(self._plotvideo), -1, 1)
            if video.shape[1] == 1:
                video = np.repeat(video, 3, 1)
            if pvid.shape[1] == 1:
                pvid = np.repeat(pvid, 3, 1)
            video = wandb.Video(video, fps=20, format="mp4")
            pvid = wandb.Video(pvid, fps=20, format="mp4")
            wandb.log({self._name: video, f"{self._name}_plot": pvid}, commit=False)
            
        return obs, reward, done, info


def mplfig_to_npimage(fig):
    """Converts a matplotlib figure to a RGB frame after updating the canvas."""
    #  only the Agg backend now supports the tostring_rgb function
    
    canvas = FigureCanvasAgg(fig)
    canvas.draw()  # update/draw the elements

    # get the width and the height to resize the matrix
    l, b, w, h = canvas.figure.bbox.bounds
    w, h = int(w), int(h)

    #  exports the canvas to a string buffer and then to a numpy nd.array
    buf = canvas.buffer_rgba()
    image = np.frombuffer(buf, dtype=np.uint8)
    return image.reshape(h, w, 4)[..., :3] # to RGB