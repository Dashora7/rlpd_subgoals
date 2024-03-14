from typing import Optional
from PIL import Image
import gym
import numpy as np
import matplotlib.pyplot as plt
import wandb
from matplotlib.backends.backend_agg import FigureCanvasAgg

class COGWANDBVideo(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        name: str = "video",
        pixel_hw: int = 200, # 84
        render_kwargs={},
        max_videos: Optional[int] = None,
        vf = None, # pass it in curried with goal
    ):
        super().__init__(env)
        self._name = name
        self._pixel_hw = pixel_hw
        self._render_kwargs = render_kwargs
        self._max_videos = max_videos
        self._video = []
        self._plotdata = []
        self._plotvideo = []
        self.lines = []
        self.vf = vf

    def _add_frame(self, obs):
        if self._max_videos is not None and self._max_videos <= 0:
            return
        if isinstance(obs, dict) and ("pixels" in obs):
            if obs["pixels"].ndim == 4:
                self._video.append(obs["pixels"][..., -1])
            else:
                self._video.append(obs["pixels"])
        elif isinstance(obs, dict) and ("image" in obs):
            if obs["image"].ndim == 4:
                self._video.append(obs["image"][..., -1])
            else:
                self._video.append(obs["image"])
        elif isinstance(obs, np.ndarray) and obs.ndim == 3:
            self._video.append(obs)
        else:    
            self._video.append(
                self.render(
                    height=self._pixel_hw,
                    width=self._pixel_hw,
                    mode="rgb_array",
                    **self._render_kwargs
                )
            )
        # resize the video to 48x48
        self._video[-1] = np.array(Image.fromarray(self._video[-1]).resize((300, 300)))
        _, ax1  = plt.subplots(1, 1, figsize=(3, 3))
        line = ax1.plot([], [])[0]
        ax1.set_title('Value to Goal')
        ax1.set_xlim(0, 50)
        ax1.set_ylim(-15, 5)
        s = self._video[-1][None, ..., None]
        self._plotdata.append(self.vf(s, s))
        line.set_data(np.arange(len(self._plotdata)), self._plotdata)
        npimg = mplfig_to_npimage(plt.gcf())
        plt.close()
        self._plotvideo.append(npimg)


    def reset(self, **kwargs):
        self._video.clear()
        self._plotdata.clear()
        self._plotvideo.clear()
        self.lines = []
        plt.close()
        obs = super().reset(**kwargs)
        self._add_frame(obs)
        return obs


    def step(self, action: np.ndarray):
        obs, reward, done, info = super().step(action)
        self._add_frame(obs)
        if done and len(self._video) > 0:
            if self._max_videos is not None:
                self._max_videos -= 1
            
            env_video = np.moveaxis(np.stack(self._video), -1, 1)
            plot_video = np.moveaxis(np.stack(self._plotvideo), -1, 1)
            video = np.concatenate([env_video, plot_video], axis=-1)
            
            if video.shape[1] == 1:
                video = np.repeat(video, 3, 1)
            video = wandb.Video(video, fps=20, format="mp4")
            wandb.log({self._name: video}, commit=False)
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
    
    # im_arr = image.reshape(h, w, 4)[..., :3] # to RGB
    # img = Image.fromarray(im_arr)
    # img.resize((48, 48), Image.ANTIALIAS)
    # return np.array(img)
    return image.reshape(h, w, 4)[..., :3] # to RGB