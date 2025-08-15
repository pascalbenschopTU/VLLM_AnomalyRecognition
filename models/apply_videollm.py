#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified video‑anomaly evaluation driver supporting **Gemma‑3**, **Qwen‑2.5‑VL**
and **VideoLLaMA‑3**.

👉  The file is a *straight union* of the three original scripts
`full_predict_gemma.py`, `full_predict_qwen.py` and
`full_predict_videollama.py` – **no optimisations** were applied.  The only
additions are

* A `--model_type` CLI flag (`gemma`, `qwen`, or `videollama`).
* Tiny helper functions that decide which model / processor to load and which
  `process_request_*` routine to call.
* Per‑model default constants are set immediately after parsing `--model_type`
  so the remaining code is 1‑to‑1 identical to the Gemma baseline.

Everything else – prompts, helpers, evaluation loop, logging format, etc. – is
verbatim from the originals so downstream behaviour is unchanged.
"""
from __future__ import annotations
import os, argparse, json, time, glob, logging, sys
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any, Tuple
from copy import deepcopy

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoProcessor, pipeline, GenerationConfig

from vision_process import process_vision_info               # unchanged helper

# Optional – only needed for VideoLLaMA
try:
    import decord                                             # noqa: F401
except ImportError:                                           # pragma: no cover
    decord = None

# ────────────────────────────────────────────────────────────────── #
#                     MODEL–SPECIFIC CONSTANTS                      #
# ────────────────────────────────────────────────────────────────── #
GEMMA_MODEL_PATH      = "Gemma3-4B/"            # change if needed
GEMMA_3n_MODEL_PATH   = "Gemma-3n-e4b-it/"
QWEN_MODEL_PATH       = "Qwen2.5-VL-7B-Instruct/"
VIDEOLLAMA_MODEL_PATH = "VideoLLama3-7B"

MODEL_SPEC = {
    "gemma": {
        "default_path": GEMMA_MODEL_PATH,
        "NUM_FRAMES": 256,
        "FRAME_SAMPLE_RATE": 1,
        "DEFAULT_GENERATION_CONFIG": GenerationConfig(
            do_sample=True,
            temperature=0.1,
            max_new_tokens=128,
            min_new_tokens=10,
            repetition_penalty=1.5,
        ),
    },
    "gemma-3n": {
        "default_path": GEMMA_3n_MODEL_PATH,
        "NUM_FRAMES": 256,
        "FRAME_SAMPLE_RATE": 1,
        "DEFAULT_GENERATION_CONFIG": GenerationConfig(
            do_sample=True,
            temperature=0.1,
            max_new_tokens=64,
            min_new_tokens=10,
            repetition_penalty=1.5,
        ),
    },
    "qwen": {
        "default_path": QWEN_MODEL_PATH,
        "NUM_FRAMES": 256,
        "FRAME_SAMPLE_RATE": 1,
        "DEFAULT_GENERATION_CONFIG": {
            "temperature": 0.1,
            "max_new_tokens": 128,
            "min_new_tokens": 10,
            "repetition_penalty": 1.5,
        },
    },
    "videollama": {
        "default_path": VIDEOLLAMA_MODEL_PATH,
        "NUM_FRAMES": 256,
        "FRAME_SAMPLE_RATE": 1,
        "DEFAULT_GENERATION_CONFIG": {
            "temperature": 0.1,
            "max_new_tokens": 128,
            "min_new_tokens": 10,
            "repetition_penalty": 1.5,
        },
    },
}

# Constants common to all models (taken from Gemma baseline)
FPS              = 30
OVERLAP_FRAMES   = 4
qc = None  # placeholder for optional quantisation

# ────────────────────────────────────────────────────────────────── #
#                          Few Shot Prompting                        #
# ────────────────────────────────────────────────────────────────── #
DEFAULT_FEW_SHOT_EXAMPLES: List[Dict[str, Any]] = [
    {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "demo_images/few_shot/Shooting.png"}}]},
    {"role": "assistant", "content": "A person with raised arm firing a gun as seen from the muzzle flash. Label: Shooting."},

    {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "demo_images/few_shot/RoadAccidents.png"}}]},
    {"role": "assistant", "content": "A car crashes seen from the smoke on the right. Label: RoadAccidents."},

    {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "demo_images/few_shot/Fighting.png"}}]},
    {"role": "assistant", "content": "Two persons trying to hit people. Label: Fighting."},

    {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "demo_images/few_shot/Stealing.png"}}]},
    {"role": "assistant", "content": "A person breaking into a car. Label: Stealing."},
]

FEW_SHOT_EXAMPLES_VIDEOLLAMA = [
    {"role": "user", "content": [
        {"type": "image", "image": {"image_path": "demo_images/few_shot/Shooting.png"}}]},
    {"role": "assistant", "content": "A person with raised arm firing a gun as seen from the muzzle flash. Label: Shooting."},

    {"role": "user", "content": [
        {"type": "image", "image": {"image_path": "demo_images/few_shot/RoadAccidents.png"}}]},
    {"role": "assistant", "content": "A car crashes seen from the smoke on the right. Label: RoadAccidents."},

    {"role": "user", "content": [
        {"type": "image", "image": {"image_path": "demo_images/few_shot/Fighting.png"}}]},
    {"role": "assistant", "content": "Two persons trying to hit people. Label: Fighting."},

    {"role": "user", "content": [
        {"type": "image", "image": {"image_path": "demo_images/few_shot/Stealing.png"}}]},
    {"role": "assistant", "content": "A person breaking into a car. Label: Stealing."},
]


# ────────────────────────────────────────────────────────────────── #
#                       LABELS  +  CORRELATION                       #
# ────────────────────────────────────────────────────────────────── #
LABEL_UCF_CRIME = [
    "Abuse","Arrest","Arson","Assault","Burglary","Explosion",
    "Fighting","Normal","RoadAccidents","Robbery","Shooting",
    "Shoplifting","Stealing","Vandalism",
]
LABELS_RWF2000 = ["Fighting","Normal"]
LABELS_XD_Violence = [ "Abuse","Explosion","Fighting","Normal",
    "Riot","RoadAccidents","Shooting",
]

correlation_map = {
    "Stealing":    {"Shoplifting":0.8,"Burglary":0.5,"Robbery":0.3},
    "Shoplifting": {"Stealing":0.8,"Burglary":0.5},
    "Arson":       {"Explosion":0.7},
    "Explosion":   {"Arson":0.7},
    "Robbery":     {"Burglary":0.75,"Stealing":0.6},
    "Burglary":    {"Robbery":0.75},
    "Assault":     {"Fighting":0.7,"Abuse":0.6},
    "Abuse":       {"Assault":0.6},
    "Fighting":    {"Assault":0.8},
}

# ────────────────────────────────────────────────────────────────── #
#                          PROMPT SET‑UP                             #
# ────────────────────────────────────────────────────────────────── #
UNGUIDED_PROMPT_UCF = """
You are given a short video clip. Identify the primary action(s) you see and provide a concise description (≤40 words).
"""


GUIDED_PROMPT_UCF = """
You are given a short video clip. Analyze it and respond in the following format:

[Predicted Class]: Brief description of actions happening in the input frames (≤ 40 words).

Choose the most likely class from the options below.

1. Abuse: Person being abused or assaulted by another individual.
2. Arrest: Law enforcement detaining or arresting individuals.
3. Arson: Deliberate setting of fire causing a blaze.
4. Assault: Physical attack (punching, kicking, hitting).
5. Burglary: Unauthorized intrusion to commit theft.
6. Explosion: Sudden blast or large fireball.
7. Fighting: Close-quarters physical fight (wrestling, brawling).
8. Normal: Routine, non-violent, everyday activity.
9. RoadAccidents: Vehicle collision or traffic accident.
10. Robbery: Theft involving force or threat from a person.
11. Shooting: Discharge of a firearm (gun visible or muzzle flash).
12. Shoplifting: Theft from a store without force or threat.
13. Stealing: Theft of objects without direct confrontation.
14. Vandalism: Deliberate damage or destruction of property.
""".strip()

GUIDED_PROMPT_XD = """
You are given a short video clip. Analyze it and respond in the following format:

[Predicted Class]: Brief description of actions happening in the input frames (≤ 40 words).

Choose the most likely class from the options below.

1. Abuse: Person being abused or assaulted by another individual.
2. Explosion: Sudden blast or large fireball.
3. Fighting: Close-quarters physical fight (wrestling, brawling).
4. Normal: Routine, non-violent, everyday activity.
5. Riot: Large chaotic crowd, mass protest
6. RoadAccidents: Vehicle collision or traffic accident.
7. Shooting: Discharge of a firearm (gun visible or muzzle flash).
""".strip()

GUIDED_PROMPT_RWF2000 = """
You are given a short surveillance video clip. Analyze it and respond in the following format:

[Predicted Class]: Brief description of actions happening in the input frames (≤ 40 words).

Choose the most likely class from the options below.

1. Fighting: Physical altercation between individuals (e.g., punching, pushing, brawling).
2. Normal: Routine, peaceful activities with no signs of aggression or conflict.
""".strip()

GUIDED_PROMPT_RWF2000_DEPTH = """
You are given a short surveillance video clip converted to depth.
Ignore the colors and focus only on the motions.
Analyze it and respond in the following format:

[Predicted Class]: Brief description of actions happening in the input frames (≤ 40 words).

Choose the most likely class from the options below.

1. Fighting: Physical altercation between individuals (e.g., punching, pushing, brawling).
2. Normal: Routine, peaceful activities with no signs of aggression or conflict.
""".strip()

FEW_SHOT_EXAMPLES: List[Dict[str,Any]] = []   # identical to originals

prompt = GUIDED_PROMPT_UCF   # initialised later by set_prompt_and_labels()
labels = LABEL_UCF_CRIME

# ────────────────────────────────────────────────────────────────── #
#                       DATASET‑DRIVEN PROMPT                       #
# ────────────────────────────────────────────────────────────────── #

def set_prompt_and_labels(dataset_name: str):
    """Update global `prompt` and `labels` for chosen dataset."""
    global prompt, labels
    prompt = "Name any anomalies you see in the video."
    labels = ["Anomaly", "Normal"]
    # Basic settings
    if "RWF2000" in dataset_name:
        prompt = GUIDED_PROMPT_RWF2000
        labels = LABELS_RWF2000
    if "UCF_Crime" in dataset_name:
        prompt = GUIDED_PROMPT_UCF
        labels = LABEL_UCF_CRIME

    # Specialized settings
    if dataset_name == "UCF_Crime_Unguided":
        prompt = UNGUIDED_PROMPT_UCF
        labels = LABEL_UCF_CRIME
    if dataset_name == "RWF2000_Depth":
        prompt = GUIDED_PROMPT_RWF2000_DEPTH
        labels = LABELS_RWF2000
    if dataset_name == "XD_Violence":
        prompt = GUIDED_PROMPT_XD
        labels = LABELS_XD_Violence
        

# ────────────────────────────────────────────────────────────────── #
#                            HELPERS                                #
# ────────────────────────────────────────────────────────────────── #

def split_chunks(frames: torch.Tensor,
                 stamps: np.ndarray,
                 length: int,
                 overlap: int = OVERLAP_FRAMES):
    if len(frames) <= length:
        return [(frames, stamps)]
    step = max(length - overlap, 1)
    chunks = []
    for i in range(0, len(frames) - length + 1, step):
        chunks.append((frames[i:i+length], stamps[i:i+length]))
    if chunks[-1][0][-1] is not frames[-1]:
        chunks.append((frames[-length:], stamps[-length:]))
    return chunks


def pad_to_multiple_of_8(frames, timestamps):
    rem = len(frames) % 8
    frames = list(frames)
    timestamps = list(timestamps)
    if rem:
        pad_len = 8 - rem
        frames.extend([frames[-1]] * pad_len)
        timestamps.extend([timestamps[-1]] * pad_len)
    return frames, timestamps


def uniform_sample(frames, num=500):
    if len(frames) == 0:
        return [], np.array([])
    indices = np.linspace(0, len(frames) - 1, min(num, len(frames))).astype(int)
    sampled_frames = [frames[i] for i in indices]
    sampled_frames, indices = pad_to_multiple_of_8(sampled_frames, indices)
    return sampled_frames, indices


def build_conv(prompt: str, vid_path: str):
    return (
        [{"role": "system", "content": [{"type": "text", "text": prompt}]}]
        + FEW_SHOT_EXAMPLES
        + [{"role": "user", "content": [{"type": "video", "video": vid_path}]}]
    )

# ────────────────────────────────────────────────────────────────── #
#               MODEL‑SPECIFIC process_request FUNCTIONS            #
# ────────────────────────────────────────────────────────────────── #

# ——— Gemma‑3 ———————————————————————————————————————————————————————————

def strip_images_to_placeholders(conv):
    c = deepcopy(conv)
    for m in c:
        for it in m.get("content", []):
            if isinstance(it, str) or not isinstance(it, dict):
                continue
            if it.get("type") in ("image", "image_url"):
                it.clear(); 
                it["type"] = "image"
            if it.get("type") == "video":
                it.clear()
    return c

def process_request_gemma(model, processor, video_path: str,
                          num_frames: int, frame_sample_rate: int) -> List[Dict[str,str]]:
    # conv = build_conv(prompt, video_path)

    # img_inputs, vid_inputs, _ = process_vision_info(conv, return_video_kwargs=True)
    conv = build_conv(prompt, video_path)
    prompt_str = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)

    img_inputs, vid_inputs, _ = process_vision_info(conv, return_video_kwargs=True)
    if not vid_inputs:
        return [{"timestamp": "[0s]", "response": ""}]

    video_frames = vid_inputs[0]
    sampled_video, indices = uniform_sample(
        video_frames, len(video_frames) // frame_sample_rate)

    conv_base = strip_images_to_placeholders(conv)
    base_text = processor.apply_chat_template(
        conv_base, tokenize=False, add_generation_prompt=True
    )

    stamps = np.array(indices) / FPS
    chunks = split_chunks(sampled_video, stamps, length=num_frames)
    output = []

    for fr, ts in chunks:
        ts_tag = (f"[{ts[0]:.2f}s-{ts[-1]:.2f}s]" if len(ts) > 1 else f"[{ts[0]:.2f}s]")

        chat_text = base_text + " ".join(["<start_of_image>"] * len(fr))

        images_in_order = (list(img_inputs) if img_inputs else []) + list(fr)
   
        inputs = processor(
            text=[chat_text],
            images=[images_in_order],
            return_tensors="pt",
            padding="longest",
            pad_to_multiple_of=16,
        ).to(model.device)
        gen = model.generate(**inputs, generation_config=MODEL_SPEC["gemma"]["DEFAULT_GENERATION_CONFIG"])
        answer = processor.batch_decode(gen[:, inputs.input_ids.shape[1]:],
                                        skip_special_tokens=True,
                                        clean_up_tokenization_spaces=True)[0]
        output.append({"timestamp": ts_tag, "response": answer})
    return output

# ——— Qwen‑2.5‑VL ————————————————————————————————————————————————————

def process_request_qwen(model, processor, video_path: str,
                         num_frames: int, frame_sample_rate: int) -> List[Dict[str,str]]:
    conv = build_conv(prompt, video_path)
    prompt_str = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)

    img_inputs, vid_inputs, _ = process_vision_info(conv, return_video_kwargs=True)
    if not vid_inputs:
        return [{"timestamp": "[0s]", "response": ""}]
    video_frames = vid_inputs[0]
    sampled_video, indices = uniform_sample(video_frames, len(video_frames)//frame_sample_rate)

    stamps = np.array(indices) / FPS
    chunks = split_chunks(sampled_video, stamps, length=num_frames)
    out = []
    for fr, ts in chunks:
        ts_tag = (f"[{ts[0]:.2f}s-{ts[-1]:.2f}s]" if len(ts)>1 else f"[{ts[0]:.2f}s]")
        inputs = processor(
            text=[prompt_str],
            images=img_inputs,
            videos=[fr],
            fps=FPS,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        gen_ids = model.generate(**inputs, **MODEL_SPEC["qwen"]["DEFAULT_GENERATION_CONFIG"])
        ans = processor.batch_decode(gen_ids[:, inputs.input_ids.shape[1]:],
                                     skip_special_tokens=True,
                                     clean_up_tokenization_spaces=True)[0]
        out.append({"timestamp": ts_tag, "response": ans})
    return out

# ——— VideoLLaMA‑3 ——————————————————————————————————————————————————

def _probe_video_meta(video_path: str):
    # Prefer decord for simplicity; fall back to naive fps=30 if unavailable.
    fps = 30.0
    duration = None
    if decord is not None:
        vr = decord.VideoReader(video_path)
        fps = float(vr.get_avg_fps()) or fps
        duration = len(vr) / fps
    return fps, duration

def process_request_videollama(model, processor, video_path: str,
                               num_frames: int, frame_sample_rate: int) -> List[Dict[str,str]]:
    # Decord bridge is used inside VideoLLaMA's processor
    fps = FPS
    if decord is not None:
        decord.bridge.set_bridge("torch")

        fps, duration = _probe_video_meta(video_path)
        

    img_inputs, vid_inputs, _ = process_vision_info(build_conv(prompt, video_path), return_video_kwargs=True)
    if not vid_inputs:
        return [{"timestamp": "[0s]", "response": ""}]

    video_frames = vid_inputs[0]
    sampled_video, indices = uniform_sample(video_frames, len(video_frames)//frame_sample_rate)
    stamps = np.array(indices) / FPS
    chunks = split_chunks(sampled_video, stamps, length=num_frames)
    out = []

    # Revert decord bridge back to native for explicit slicing later
    if decord is not None:
        decord.bridge.set_bridge("native")

    for fr, ts in chunks:
        ts_tag = (f"[{ts[0]:.2f}s-{ts[-1]:.2f}s]" if len(ts)>1 else f"[{ts[0]:.2f}s]")
        temp_few_shot_examples = []
        if len(FEW_SHOT_EXAMPLES) > 0:
            temp_few_shot_examples = FEW_SHOT_EXAMPLES_VIDEOLLAMA
        conv = [
            {"role": "system", "content": [{"type": "text", "text": prompt}]},
            *temp_few_shot_examples,
            {"role": "user", "content": [
                {"type": "video", "video": {"video_path": video_path,
                                                "start_time": ts[0], "end_time": ts[-1],
                                                "max_frames": num_frames}},
                {"type": "text", "text": "Please classify the anomaly and describe it briefly."}
            ]},
        ]
        inputs = processor(
            conversation=conv,
            add_system_prompt=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        inputs = inputs.to(model.device, dtype=torch.bfloat16)

        gen_ids = model.generate(**inputs, **MODEL_SPEC["videollama"]["DEFAULT_GENERATION_CONFIG"])
        ans = processor.batch_decode(gen_ids, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=True)[0]
        out.append({"timestamp": ts_tag, "response": ans})
    return out

# ────────────────────────────────────────────────────────────────── #
#                MODEL LOADING DISPATCH (kept minimal)              #
# ────────────────────────────────────────────────────────────────── #

def load_model_and_processor(model_type: str, model_path: str):
    if model_type == "gemma":
        from transformers import Gemma3ForConditionalGeneration
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=qc,
            attn_implementation="flash_attention_2",
        )
        processor = AutoProcessor.from_pretrained(model_path, padding_side="left", use_fast=True)
        process_fn = process_request_gemma
    elif model_type == "gemma-3n":
        from transformers import Gemma3nForConditionalGeneration
        model = Gemma3nForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16,
            device_map="auto", 
            quantization_config=qc,
        ).eval()
        processor = AutoProcessor.from_pretrained(model_path, padding_side="left", use_fast=True)
        process_fn = process_request_gemma
    elif model_type == "qwen":
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
        process_fn = process_request_qwen
    elif model_type == "videollama":
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        process_fn = process_request_videollama
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    return model, processor, process_fn

# ────────────────────────────────────────────────────────────────── #
#                              LOGGER                               #
# ────────────────────────────────────────────────────────────────── #

def set_logger():
    global log
    log = logging.getLogger("eval")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = "%(asctime)s %(levelname)-8s %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(fmt, datefmt))
    log.addHandler(handler)

# ────────────────────────────────────────────────────────────────── #
#                           MAIN LOOP                               #
# ────────────────────────────────────────────────────────────────── #

def main():
    ap = argparse.ArgumentParser(description="Unified Gemma/Qwen/VideoLLaMA evaluator")
    ap.add_argument("--model_type", choices=["gemma", "gemma-3n", "qwen", "videollama"], default="gemma",
                    help="Select which model family to use")
    ap.add_argument("--model", help="Path or HuggingFace hub ID of the checkpoint; if omitted, uses built‑in default")
    ap.add_argument("--dataset_name", default="UCF_Crime")
    ap.add_argument("--video_root", required=True)
    ap.add_argument("--experiment", default="Default")
    ap.add_argument("--eval_json")
    ap.add_argument("--class_to_test")
    args = ap.parse_args()

    set_logger()
    set_prompt_and_labels(args.dataset_name)

    # Apply per‑model constants (NUM_FRAMES, FRAME_SAMPLE_RATE, generation cfg)
    selected_spec = MODEL_SPEC[args.model_type]
    global NUM_FRAMES, FRAME_SAMPLE_RATE, FEW_SHOT_EXAMPLES
    NUM_FRAMES = selected_spec["NUM_FRAMES"]
    FRAME_SAMPLE_RATE = selected_spec["FRAME_SAMPLE_RATE"]

    if "FEW_SHOT" in args.experiment.upper() and not "NO_FEW_SHOT" in args.experiment.upper():
        FEW_SHOT_EXAMPLES = DEFAULT_FEW_SHOT_EXAMPLES

    model_path = args.model or selected_spec["default_path"]
    model, processor, process_request = load_model_and_processor(args.model_type, model_path)
    log.info("Model loaded: %s (%s)", model_path, args.model_type)
    log.info("Few shot examples: %s", FEW_SHOT_EXAMPLES)
    log.info("Num frames: %d, frame sample rate: %d", NUM_FRAMES, FRAME_SAMPLE_RATE)

    # Zero‑shot classifier common to all variants
    classifier = pipeline("zero-shot-classification",
                          model="facebook/bart-large-mnli",
                          device=0 if torch.cuda.is_available() else -1)

    # Build video list (directory name is the ground‑truth class)
    video_list: List[Tuple[str,str]] = []
    for ext in ("*.mp4", "*.avi"):
        for path in glob.glob(os.path.join(args.video_root, "**", ext), recursive=True):
            rel = os.path.relpath(path, args.video_root).replace(os.sep, "/")
            current_cls = rel.split("/", 1)[0]
            video_list.append((path, current_cls))

    # Resume / create log file
    if args.eval_json and os.path.exists(args.eval_json):
        with open(args.eval_json) as f:
            eval_dict = json.load(f)
    else:
        logdir = Path(f"logs/{args.dataset_name}/{args.experiment}/{Path(model_path).name}")
        logdir.mkdir(parents=True, exist_ok=True)
        args.eval_json = str(logdir / f"eval_{time.strftime('%Y%m%d-%H%M%S')}.json")
        eval_dict = {}

    processed_idxs  = {int(k) for k in eval_dict.keys() if k.isdigit()}
    processed_names = {v["video_name"] for v in eval_dict.values()}

    # Counters for accuracy metrics
    correct = defaultdict(int)
    incorrect = defaultdict(int)
    adjusted_correct = defaultdict(float)
    top3_correct = defaultdict(int)

    # Main evaluation loop ---------------------------------------------------
    for vid_idx, (video_path, gt_cls) in enumerate(tqdm(video_list), 1):
        if args.class_to_test and args.class_to_test != gt_cls:
            continue
        if vid_idx in processed_idxs or Path(video_path).name in processed_names:
            log.info("Skipping already‑processed [%d] %s", vid_idx, video_path)
            continue

        answer_chunks = process_request(model, processor, video_path,
                                        num_frames=NUM_FRAMES,
                                        frame_sample_rate=FRAME_SAMPLE_RATE)

        top1 = adjusted_top1 = top3 = 0.0
        output_dict: Dict[str,Any] = {}

        for chunk in answer_chunks:
            text = chunk["response"]
            sample = ("\n\nDoes the text describe an anomaly? If yes, select the "
                      "most relevant label – otherwise 'Normal'.\n" + text)
            pred = classifier(sample, candidate_labels=labels)
            pl, ps = pred["labels"][:3], pred["scores"][:3]
            ts = chunk["timestamp"]
            output_dict[ts] = {
                "LLM_output": text,
                "LLM_classes": pl,
                "LLM_confidence": ps,
            }
            if pl[0] == gt_cls:
                top1 = adjusted_top1 = 1.0
            else:
                corr = correlation_map.get(gt_cls, {}).get(pl[0], 0.0)
                adjusted_top1 = max(adjusted_top1, corr)
            if gt_cls in pl:
                top3 = 1

        correct[gt_cls] += top1
        incorrect[gt_cls] += (1.0 - top1)
        adjusted_correct[gt_cls] += adjusted_top1
        top3_correct[gt_cls] += top3

        eval_dict[str(vid_idx)] = {
            "video_name": Path(video_path).name,
            "video_label": gt_cls,
            "video_prediction": output_dict,
            "top_1_score": top1,
        }
        with open(args.eval_json, "w") as f:
            json.dump(eval_dict, f, indent=2)

    # Final metrics -----------------------------------------------------------
    total = sum(correct.values()) + sum(incorrect.values())
    overall  = (sum(correct.values())/total)*100 if total else 0.0
    adj_acc  = (sum(adjusted_correct.values())/total)*100 if total else 0.0
    top3_acc = (sum(top3_correct.values())/total)*100 if total else 0.0

    log.info("Overall   accuracy: %.2f%%", overall)
    log.info("Adjusted  accuracy: %.2f%%", adj_acc)
    log.info("Top‑3     accuracy: %.2f%%", top3_acc)
    log.info("Results written to %s", args.eval_json)

if __name__ == "__main__":
    main()
