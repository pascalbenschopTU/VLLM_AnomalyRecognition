from __future__ import annotations
import base64
import os
import time
from typing import List, Literal, Optional, Union, Dict, Any, Union

import cv2
import numpy as np
import PIL
import requests
import torch
import torch.utils.model_zoo as model_zoo
from PIL import Image
from termcolor import colored
from tqdm import tqdm
# from torchvision.models.resnet import Bottleneck, BasicBlock, ResNet
import base64
import logging
import math
import os
import sys
import time
import warnings
from functools import lru_cache
from io import BytesIO

import numpy as np
import requests
import torch
import torchvision
from packaging import version
from PIL import Image
from torchvision import io, transforms
from torchvision.transforms import InterpolationMode
from typing import Optional

import decord


logger = logging.getLogger(__name__)

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
FRAME_FACTOR = 30 #2  # FPS sampled = FPS / FRAME_FACTOR
FRAME_SELECTION = 1  # seconds
FPS = 30.0 # 2.0  # Original FPS = 30, setting this to 10 samples 1 in 3 frames
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 15000 #768

# Set the maximum number of video token inputs.
# Here, 128K represents the maximum number of input tokens for the VLLM model.
# Remember to adjust it according to your own configuration.
VIDEO_TOTAL_PIXELS = int(float(os.environ.get('VIDEO_MAX_PIXELS', 128000 * 28 * 28 * 0.9)))
logger.info(f"set VIDEO_TOTAL_PIXELS: {VIDEO_TOTAL_PIXELS}")


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def smart_nframes(
    ele: dict,
    total_frames: int,
    video_fps: int | float,
) -> int:
    """calculate the number of frames for video used for model inputs.

    Args:
        ele (dict): a dict contains the configuration of video.
            support either `fps` or `nframes`:
                - nframes: the number of frames to extract for model inputs.
                - fps: the fps to extract frames for model inputs.
                    - min_frames: the minimum number of frames of the video, only used when fps is provided.
                    - max_frames: the maximum number of frames of the video, only used when fps is provided.
        total_frames (int): the original total number of frames of the video.
        video_fps (int | float): the original fps of the video.

    Raises:
        ValueError: nframes should in interval [FRAME_FACTOR, total_frames].

    Returns:
        int: the number of frames for video used for model inputs.
    """
    assert not ("fps" in ele and "nframes" in ele), "Only accept either `fps` or `nframes`"
    if "nframes" in ele:
        print(f"Check total frames used: {total_frames}")
        nframes = total_frames
        # nframes = round_by_factor(ele["nframes"], FRAME_FACTOR)
    else:
        fps = ele.get("fps", FPS)
        min_frames = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
        max_frames = floor_by_factor(ele.get("max_frames", min(FPS_MAX_FRAMES, total_frames)), FRAME_FACTOR)
        nframes = total_frames / video_fps * fps
        if nframes > total_frames:
            logger.warning(f"smart_nframes: nframes[{nframes}] > total_frames[{total_frames}]")
        nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
        nframes = floor_by_factor(nframes, FRAME_FACTOR)
    if not (FRAME_FACTOR <= nframes and nframes <= total_frames):
        raise ValueError(f"nframes should in interval [{FRAME_FACTOR}, {total_frames}], but got {nframes}.")
    return nframes


def uniform_sample(frames, num=500, pad_to=8):
    """
    • Accepts  list[Tensor / ndarray / PIL]  **or**  Tensor (F,C,H,W)
    • Uniform-samples <= num frames.
    • Pads so len % pad_to == 0   (default multiple-of-8).
    • Returns   video_4d_tensor  (F,C,H,W)   and  numpy index array.
    """
    # ---- 0) normalise input ----------------------------------------
    if isinstance(frames, torch.Tensor):        # (F,C,H,W) tensor → list
        frames = list(frames)                   # each item is (C,H,W) tensor

    if len(frames) == 0:                        # safe now: frames is a list
        return torch.empty(0), np.array([])

    # ---- 1) uniform sampling ---------------------------------------
    num = min(num, len(frames))
    idx = np.linspace(0, len(frames) - 1, num).astype(int)
    sampled = [frames[i] for i in idx]

    # ---- 2) pad to multiple of pad_to ------------------------------
    rem = len(sampled) % pad_to
    if rem:
        pad_len = pad_to - rem
        sampled += [sampled[-1]] * pad_len
        idx = np.concatenate([idx, np.repeat(idx[-1], pad_len)])

    # ---- 3) stack into (F,C,H,W) tensor ----------------------------
    if isinstance(sampled[0], torch.Tensor):        # already tensors (C,H,W)
        video = torch.stack(sampled)
    else:                                           # NumPy or PIL
        arr = np.asarray(f).copy()        # now writable
        video = torch.stack([
            torch.from_numpy(arr).permute(2, 0, 1)
            for f in sampled
        ])

    return video, idx

# ---------- basic helpers ---------- #
def load_video_frames(vr, indices, max_wh = (480, 320)):
    """Return list[PIL.Image] for given decord.VideoReader and frame IDs."""
    frames = vr.get_batch([i for i in indices if i < len(vr)]).numpy()
    
    pil_frames = [PIL.Image.fromarray(f) for f in frames]

    # 2. if first frame already small, assume all small → early-exit
    max_w, max_h = max_wh
    print(f"size of images: w {pil_frames[0].width} h: {pil_frames[0].height}")
    if pil_frames[0].width <= max_w and pil_frames[0].height <= max_h:
        return pil_frames

    # 3. resize every frame, keeping aspect ratio
    resized = []
    for img in pil_frames:
        w, h = img.size
        if w <= max_w and h <= max_h:          # this one is fine already
            resized.append(img)
            continue

        scale = min(max_w / w, max_h / h)      # preserve aspect
        new_w, new_h = int(w * scale), int(h * scale)
        resized.append(img.resize((new_w, new_h), PIL.Image.BILINEAR))

    return resized

def _extract_image(img):
    print(f"img? {img}")
    if isinstance(img, PIL.Image.Image):
        return img.convert("RGB")
    if hasattr(img, "path"):
        return PIL.Image.open(img.path).convert("RGB")
    if isinstance(img, dict) and "image_url" in img:
        # handles {'image_url': {'url': '…'}}
        url = img["image_url"]
        if isinstance(url, dict) and "url" in url:
            return Image.open(url["url"]).convert("RGB")
        # or if it's just a string: {'image_url': "…"}
        return Image.open(url).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(img)}")

def _extract_video(video_path, max_frames=5000):
    decord.bridge.set_bridge("torch")
    vr = decord.VideoReader(video_path)
    n = len(vr)
    if n < 1:
        raise ValueError(f"No frames in {video_path}")
    
    num_loaded_frames = min(n, max_frames)
    indices = torch.arange(num_loaded_frames)    
    return load_video_frames(vr, indices)



def fetch_video(ele: dict, image_factor: int = IMAGE_FACTOR, return_video_sample_fps: bool = False) -> torch.Tensor | list[Image.Image]:
    if isinstance(ele["video"], str):
        try:
            video = _extract_video(ele["video"])
            sample_frames = 30
        except Exception as e:
            logger.warning(f"video_reader does not work, msg: {e}")

        # Sample frames
        video_name = ele["video"]

        if "Normal" in video_name:
            n = min(508, len(video))
            sampled_video = video[:n]
            indices = torch.arange(n)
        else:
            # sampled_video, indices = uniform_sample(video, len(video) // FRAME_SELECTION)
            sampled_video = video
            indices = torch.arange(len(video))
        
        return sampled_video, indices


def extract_vision_info(conversations: list[dict] | list[list[dict]]) -> list[dict]:
    vision_infos = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if (
                        "image" in ele
                        or "image_url" in ele
                        or "video" in ele
                        or ele["type"] in ("image", "image_url", "video")
                    ):
                        vision_infos.append(ele)
    return vision_infos


def process_vision_info(
    conversations: list[dict] | list[list[dict]],
    return_video_kwargs: bool = False,
) -> tuple[list[Image.Image] | None, list[torch.Tensor | list[Image.Image]] | None, Optional[dict]]:
    vision_infos = extract_vision_info(conversations)
    vision_infos[0]['nframes'] = -1
    ## Read images or videos
    image_inputs = []
    video_inputs = []
    video_sample_fps_list = []
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(_extract_image(vision_info))
        elif "video" in vision_info:
            video_input, indices = fetch_video(vision_info)
            video_sample_fps_list.append(FPS)
            video_inputs.append(video_input)
        else:
            print(f"Error, no image, image_url or video in content: {vision_info}", file=sys.stderr)
            continue
            # raise ValueError("image, image_url or video should in content.")
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None
    if return_video_kwargs:
        return image_inputs, video_inputs, {'fps': video_sample_fps_list, 'indices': indices}
    return image_inputs, video_inputs