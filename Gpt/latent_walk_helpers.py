"""
Miscellaneous helper functions for generating latent walk interpolation videos.
"""
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
import numpy as np
import moviepy.editor
import os


def slerp(low, high, weight):
    """
    Spherical linear interpolation. Useful for navigating Gaussian latent spaces.
    https://discuss.pytorch.org/t/help-regarding-slerp-function-for-generative-model-sampling/32475
    """
    low_norm = low / torch.norm(low, dim=-1, keepdim=True)  # (N, nz)
    high_norm = high / torch.norm(high, dim=-1, keepdim=True)  # (N, nz)
    omega = torch.acos((low_norm * high_norm).sum(1, keepdim=True))  # (N, 1)
    so = torch.sin(omega)  # (N,)
    weight = weight.unsqueeze(0)
    low = low.unsqueeze(1)
    high = high.unsqueeze(1)
    res = (torch.sin((1.0 - weight) * omega) / so).unsqueeze(-1) * low + \
          (torch.sin(weight * omega) / so).unsqueeze(-1) * high
    return res


def slerpify(z, n_steps):
    """
    A helper function for generating slerp trajectories that perfectly loop.
    """
    assert z.dim() == 3
    v, s, dim = z.shape
    z_end = z.roll(shifts=-1, dims=1)
    t = torch.linspace(0.0, 1.0, n_steps, device="cuda")
    noise = slerp(z.view(-1, dim), z_end.view(-1, dim), t).view(v, s, n_steps, dim)
    return noise


def save_video(frames, fps, out_path):

    """
    Save an MP4 video from numpy frames.
    """

    duration = len(frames) / fps
    frames = frames[::-1]
    frames.append(frames[-1])
    frames.append(frames[-1])

    def make_frame(t):
        out = frames.pop()
        return out

    video = moviepy.editor.VideoClip(make_frame, duration=duration)
    video.write_videofile(out_path, fps=fps, codec='libx264', bitrate='50M')


def create_latent_walk_for_cnn(samples, filename="walk.mp4", fps=60, filters_per_row=8, filter_upscale=80):
    """
    Creates the latent walk video from G.pt samples.
    Note that this function is somewhat tailored for a specific CNN task model (first layer shape: (16, 3, 3, 3)).
    """
    #--------------------------
    conv2d_shape = (16, 3, 3, 3)
    outc, inc, kh, kw = conv2d_shape
    assert inc == 3, "conv2d must have RGB input channels"
    num_weights = np.prod(conv2d_shape)
    # --------------------------
    assert samples.dim() == 4
    n_samples, n_steps, n_videos, _ = samples.shape
    # conv2d weights have shape (out_channels, in_channels, kernel_size, kernel_size):
    filters = samples[..., :num_weights].reshape(n_samples * n_steps * n_videos * outc, *conv2d_shape[1:])
    # upscale filters to make visualization easier:
    filters = F.interpolate(filters, scale_factor=filter_upscale, mode='nearest')
    filters = filters.reshape(n_samples * n_steps * n_videos, outc, inc, kh * filter_upscale, kw * filter_upscale)
    n_frames = n_samples * n_steps
    frames = []
    for t in range(n_frames):
        input = filters[t * n_videos: (t + 1) * n_videos].view(-1, inc, kh * filter_upscale, kw * filter_upscale)
        frame = make_grid(input, normalize=True, scale_each=True, nrow=filters_per_row, padding=2 * filter_upscale,
                          pad_value=0)
        ndarr = frame.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        frames.append(ndarr)
    os.makedirs("latent_walks", exist_ok=True)
    save_video(frames, fps=fps, out_path=f"latent_walks/{filename}")
