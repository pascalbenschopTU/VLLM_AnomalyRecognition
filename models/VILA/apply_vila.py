#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified evaluation script for VILA / NVILA video anomaly detection.

• Loads the VILA / NVILA model once.
• Feeds each video directly into the model – no FastAPI, no OpenAI client.
• Runs the same zero-shot-classification post-processing you already had.
"""
from __future__ import annotations
import os, argparse, time, json, glob, logging
from pathlib import Path
from copy import deepcopy
from collections import defaultdict
import numpy as np
import PIL
import torch
from tqdm import tqdm
from transformers import pipeline, GenerationConfig
import decord

# -------------------------------------------------------------------- #
#                    --------  MODEL UTILITIES --------                #
# -------------------------------------------------------------------- #
from llava import conversation as clib
from llava.model.builder import load_pretrained_model
from llava.media import Image, Video
from llava.mm_utils import process_images
from llava.utils import disable_torch_init, make_list, tokenizer as tok_utils
from llava.constants import MEDIA_TOKENS

DEFAULT_FEW_SHOT_EXAMPLES = [
    {"from": "human", "value": [Image("demo_images/few_shot/Shooting.png")]},
    {"from": "gpt", "value": "A person with raised arm firing a gun as seen from the muzzle flash. Label: Shooting."},
    {"from": "human", "value": [Image("demo_images/few_shot/RoadAccidents.png")]},
    {"from": "gpt", "value": "A car crashes seen from the smoke on the right. Label: RoadAccidents."},
    {"from": "human", "value": [Image("demo_images/few_shot/Fighting.png")]},
    {"from": "gpt", "value": "Two persons trying to hit people. Label: Fighting."},
    {"from": "human", "value": [Image("demo_images/few_shot/Stealing.png")]},
    {"from": "gpt", "value": "A person breaking into a car. Label: Stealing."},
]


FEW_SHOT_EXAMPLES = []
FRAME_SAMPLE_RATE = 1

DEFAULT_GEN_CFG = GenerationConfig(
    temperature=0.05,
    do_sample=True,
    max_new_tokens=128,
    min_new_tokens=20,
    repetition_penalty=1.5,
)

SHORT_GEN_CFG = GenerationConfig(
    temperature=0.05,
    do_sample=True,
    max_new_tokens=32,
    min_new_tokens=1,
    repetition_penalty=1.5,
)

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
#                          PROMPT SET‑UP                            #
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
7. Fighting: Close‑quarters physical fight (wrestling, brawling).
8. Normal: Routine, non‑violent, everyday activity.
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
    global prompt, labels, followup_prompt
    followup_prompt = None
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

# -------------------------------------------------------------------- #
#                      ----  basic helpers  ----                       #
# -------------------------------------------------------------------- #
def load_video_frames(vr, indices):
    """Return list[PIL.Image] for given decord.VideoReader and frame IDs."""
    frames = vr.get_batch([i for i in indices if i < len(vr)]).numpy()
    return [PIL.Image.fromarray(f) for f in frames]

def _extract_image(img):
    if isinstance(img, PIL.Image.Image):
        return img.convert("RGB")
    if hasattr(img, "path"):
        return PIL.Image.open(img.path).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(img)}")

def _extract_video(video: Video, max_frames=5000):
    decord.bridge.set_bridge("torch")
    vr = decord.VideoReader(video.path)
    n = len(vr)
    if n < 1:
        raise ValueError(f"No frames in {video.path}")
    
    num_loaded_frames = min(n, max_frames)
    indices = torch.arange(num_loaded_frames)    
    return load_video_frames(vr, indices)

def extract_media(messages):
    """Replace Image/Video objects in conversation with special tokens."""
    media = {"video": [], "image": []}
    for msg in messages:
        val = ""
        for part in make_list(msg["value"]):
            if isinstance(part, str):
                for tok in MEDIA_TOKENS.values():
                    part = part.replace(tok, "")
                val += part
            elif isinstance(part, (Image, PIL.Image.Image)):
                media["image"].append(_extract_image(part))
                val += MEDIA_TOKENS["image"]
            elif isinstance(part, Video):
                media["video"].append(_extract_video(part))
                val += MEDIA_TOKENS["video"]
        msg["value"] = val
    return media

def pad_to_multiple_of_8(frames, timestamps):
    frames = list(frames)
    timestamps = list(timestamps)
    rem = len(frames) % 8
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

def split_chunks(frames, timestamps, length, overlap=4):
    total = len(frames)
    if total<=length: return [ pad_to_multiple_of_8(frames, timestamps) ]
    step = max(length-overlap, (total-length)//(total//length or 1))
    chunks=[]
    for i in range(0, total-length+1, step):
        chunk_f = frames[i : i + length]
        chunk_t = timestamps[i : i + length]
        chunks.append(pad_to_multiple_of_8(chunk_f, chunk_t))
    if chunks and chunks[-1][0][-1]!=frames[-1]:
        chunk_f = frames[-length:]
        chunk_t = timestamps[-length:]
        chunks.append(pad_to_multiple_of_8(chunk_f, chunk_t))

    return chunks

@torch.inference_mode()
def generate_predictions(
        model, tokenizer, image_processor,
        shot_images, video_frames, conv
):
    video_enc = process_images(video_frames, image_processor, model.config).half()

    media = {"video": [video_enc]}
    if shot_images:
        shot_enc = process_images(shot_images, image_processor, model.config).half()
        media["image"] = [t for t in shot_enc]

    inputs = tok_utils.tokenize_conversation(
        conv, tokenizer, add_generation_prompt=True
    ).to(model.device).unsqueeze(0)

    output = model.generate(
        input_ids=inputs,
        media=media,
        media_config=defaultdict(dict),
        generation_config=DEFAULT_GEN_CFG,
    )

    answer_txt = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    if followup_prompt != None:
        follow_up_conv = conv + [
            {"from": "gpt", "value": answer_txt},
            {"from": "human",
             "value": followup_prompt}
        ]

        follow_up_inputs = tok_utils.tokenize_conversation(
            follow_up_conv, tokenizer, add_generation_prompt=True
        ).to(model.device).unsqueeze(0)

        follow_up_output = model.generate(
            input_ids=follow_up_inputs,
            media=media,
            media_config=defaultdict(dict),
            generation_config=SHORT_GEN_CFG,
        )

        followup_answer_txt = tokenizer.decode(follow_up_output[0], skip_special_tokens=True).strip()

        return f"Predicted class: {followup_answer_txt}, Description: {answer_txt}"

    return answer_txt

def process_request(
        model, tokenizer, image_processor,
        prompt: str,
        video_path: str,
        few_shot_examples: list,
        current_video=None
):
    # Build conversation ================================================
    conv = [{"from": "system", "value": prompt}]
    conv.extend(deepcopy(few_shot_examples))
    conv.append({"from": "human", "value": [Video(video_path)]})

    # Replace visual objects with tokens and get real media -------------
    conv_tokens = deepcopy(conv)
    media = extract_media(conv_tokens)

    video_frames = media["video"][0]
    shot_images  = media["image"]

    # Chunk the video if needed -----------------------------------------
    budget = model.config.num_video_frames
    
    if current_video and "Normal" in current_video:
        n = min(508, len(video_frames))
        video_frames = video_frames[:n]
        indices = torch.arange(n)

    sampled_video, indices = uniform_sample(video_frames, len(video_frames) // FRAME_SAMPLE_RATE)
    
    log.info(f"Length of sampled_video video: {len(sampled_video)}, original: {len(video_frames)}")
    stamps = np.array(indices) / 30  # default FPS
    chunks = split_chunks(sampled_video, stamps, budget)

    answers = []
    for v_chunk, ts in chunks:
        tag = f"{ts[0]:.2f}s–{ts[-1]:.2f}s" if len(ts) else "0s"
        resp = generate_predictions(
            model, tokenizer, image_processor,
            shot_images, v_chunk, conv_tokens
        )
        answers.append({"timestamp": tag, "response": resp})
    return answers

# -------------------------------------------------------------------- #
#                            ----  MAIN  ----                          #
# -------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_name", default="UCF_Crime")
    ap.add_argument("--video_root",  required=True,
                    help="Root folder that contains the videos")
    ap.add_argument("--experiment", default="Few_Shot")
    ap.add_argument("--model_path",  default="NVILA-8B-Video")
    ap.add_argument("--eval_json",   help="Resume / append to existing json")
    ap.add_argument("--class_to_test")
    args = ap.parse_args()

    torch.cuda.empty_cache()

    # -------------- logging / counters ----------------
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s - %(message)s")
    global log, FEW_SHOT_EXAMPLES
    log = logging.getLogger("eval")

    if "FEW_SHOT" in args.experiment.upper() and not "NO_FEW_SHOT" in args.experiment.upper():
        FEW_SHOT_EXAMPLES = DEFAULT_FEW_SHOT_EXAMPLES

    log.info(f"Few shot examples: {FEW_SHOT_EXAMPLES}")

    set_prompt_and_labels(args.dataset_name)

    correct, incorrect = defaultdict(int), defaultdict(int)
    adjusted_correct, top3_correct = defaultdict(float), defaultdict(int)

    # -------------- load model ONCE -------------------
    disable_torch_init()
    model_name = os.path.basename(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, model_name, None
    )
    log.info("Model loaded: %s", model_name)
    log.info("Model device: %s", model.device)

    # -------------- zero-shot classifier --------------
    classifier = pipeline("zero-shot-classification",
                          model="facebook/bart-large-mnli")

    # -------------- build video evaluation list -------------------
    video_list = []
    for ext in ("*.mp4", "*.avi"):
        pattern = os.path.join(args.video_root, "**", ext)
        for path in glob.glob(pattern, recursive=True):
            rel = os.path.relpath(path, args.video_root).replace(os.sep, "/")
            current_cls = rel.split("/", 1)[0]
            video_list.append((path, current_cls))

    video_list = sorted(video_list)

    # -------------- resume? ---------------------------
    if args.eval_json and os.path.exists(args.eval_json):
        with open(args.eval_json) as f:
            eval_dict = json.load(f)
    else:
        direc = f"../logs/{args.dataset_name}/{args.experiment}/{model_name}"
        os.makedirs(direc, exist_ok=True)
        args.eval_json = os.path.join(
            direc, f"eval_{time.strftime('%Y%m%d-%H%M%S')}.json")
        eval_dict = {}

    processed_idxs  = {int(k) for k in eval_dict.keys() if k.isdigit()}
    processed_names = {v["video_name"] for v in eval_dict.values()}

    # -------------- loop over videos ------------------
    for vid_idx, (video_path, gt_cls) in enumerate(tqdm(video_list)):
        if args.class_to_test and args.class_to_test != gt_cls:
            continue

        #  checkpoint: skip if we've already done this one
        if vid_idx in processed_idxs or Path(video_path).name in processed_names:
            log.info(f"Skipping already-processed [{vid_idx}] {video_path}")
            continue

        answer_chunks = process_request(
            model, tokenizer, image_processor,
            prompt, video_path, few_shot_examples=FEW_SHOT_EXAMPLES,
            current_video=video_path
        )

        # ---- post-process each chunk ------------------
        top1 = 0.0
        adjusted_top1 = 0.0
        top3 = 0.0
        
        output_dict = {}
        for chunk in answer_chunks:
            text = chunk["response"]

            q = ("\n\nDoes the text describe an anomaly? "
                 "If yes, select the most relevant label – otherwise 'Normal'.\n")
            sample = q + text
            pred = classifier(sample, candidate_labels=labels)
            predicted_labels = pred["labels"][:3]
            predicted_scores = pred["scores"][:3]

            timestamp = chunk["timestamp"]
            output_dict[timestamp] = {
                "LLM_output": text,
                "LLM_classes": predicted_labels,
                "LLM_confidence": predicted_scores,
            }

            # ---- scoring ----------------------------------
            if predicted_labels[0] == gt_cls:
                top1 = 1.0
                adjusted_top1 = 1.0
            else:
                corr = correlation_map.get(gt_cls, {}).get(predicted_labels[0], 0.0)
                adjusted_top1 = max(adjusted_top1, corr)

            if gt_cls in predicted_labels:
                top3 = 1
                

        correct[gt_cls] += top1
        incorrect[gt_cls] += (1.0 - top1)
        adjusted_correct[gt_cls] += adjusted_top1
        top3_correct[gt_cls] += top3

        # ---- save incremental -------------------------
        eval_dict[str(vid_idx)] = {
            "video_name": Path(video_path).name,
            "video_label": gt_cls,
            "video_prediction": output_dict,
            "top_1_score": top1,
        }
        
        with open(args.eval_json, "w") as f:
            json.dump(eval_dict, f, indent=2)

    # -------------- final metrics ---------------------
    total = sum(correct.values()) + sum(incorrect.values())
    overall_acc = (sum(correct.values()) / total) * 100 if total else 0
    adj_acc     = (sum(adjusted_correct.values()) / total) * 100 if total else 0
    top3_acc    = (sum(top3_correct.values()) / total) * 100 if total else 0

    log.info("Overall  accuracy: %.2f%%", overall_acc)
    log.info("Adjusted accuracy: %.2f%%", adj_acc)
    log.info("Top-3    accuracy: %.2f%%", top3_acc)
    log.info("Results written to %s", args.eval_json)


if __name__ == "__main__":
    main()
