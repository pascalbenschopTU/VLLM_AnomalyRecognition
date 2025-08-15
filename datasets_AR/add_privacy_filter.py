#!/usr/bin/env python3
"""
anonymize_ucf_crime_bbox.py
Apply a 21×21 median filter to the top 25 % of every person bounding box
in a UCF-Crime video dataset and write the processed videos to a
parallel directory tree.

Usage
-----
python anonymize_ucf_crime_bbox.py \
    --input_root  videos_temporal \
    --output_root videos_privacy \
    --kernel      21 \
    --score_thr   0.5 \
    --device      cuda        # or "cpu"
"""
from pathlib import Path
import argparse
import cv2
import torch
from ultralytics import YOLO

def load_model(weights_path: str, device: str):
    device = torch.device(device)
    model = YOLO(weights_path).to(device)
    return model, device

@torch.no_grad()
def person_bboxes(yolo_model, frame_bgr, score_thr=0.1):
    """
    Run YOLO inference on a BGR frame and return [[x1,y1,x2,y2], ...]
    only for class=person (0) above confidence threshold.
    """
    results = yolo_model.predict(
        source=frame_bgr,
        verbose=False,
        conf=score_thr,
        imgsz=640,
        classes=[0]
    )[0]
    return results.boxes.xyxy.int().tolist()

# ---------- Anonymization ---------- #
def anonymize_frame(frame, boxes, kernel):
    """
    Median-blur the top 25 % of every bbox in-place.
    """
    for x1, y1, x2, y2 in boxes:
        top = y1
        bottom = y1 + max(1, int(0.25 * (y2 - y1)))
        roi = frame[top:bottom, x1:x2]
        if roi.shape[0] >= kernel and roi.shape[1] >= kernel:
            frame[top:bottom, x1:x2] = cv2.medianBlur(roi, kernel)
    return frame

# ---------- Video processing ---------- #
def process_video(src_path: Path, dst_path: Path, yolo_model, device, kernel: int, score_thr: float):
    """Process a single video: detect, anonymize, and save."""
    # Skip if already exists
    if dst_path.exists():
        print(f"→ Skipping {src_path.name}, output exists")
        return

    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        print(f"⚠️  Cannot open {src_path}")
        return

    fps   = cap.get(cv2.CAP_PROP_FPS)
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc= cv2.VideoWriter_fourcc(*"mp4v")

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(dst_path), fourcc, fps, (w, h))

    max_frames = 5000
    frame_count = 0
    while frame_count < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        boxes = person_bboxes(yolo_model, frame, score_thr)
        out_frame = anonymize_frame(frame, boxes, kernel)
        writer.write(out_frame)
        frame_count += 1

    cap.release()
    writer.release()

# ---------- CLI ---------- #
def main():
    ap = argparse.ArgumentParser(
        description="Anonymize person faces (top 25 %) in UCF-Crime videos.")
    ap.add_argument("--input_root",  required=True)
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--kernel", type=int, default=21, help="odd int ≥ 3")
    ap.add_argument("--score_thr", type=float, default=0.5)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--yolo_weights", default="yolov8x.pt",
                    help="Path to YOLO weights file")
    args = ap.parse_args()

    in_root  = Path(args.input_root).expanduser().resolve()
    out_root = Path(args.output_root).expanduser().resolve()
    exts = {".mp4", ".avi", ".mov", ".mkv"}

    yolo_model, device = load_model(args.yolo_weights, args.device)

    for src in in_root.rglob("*"):
        if src.suffix.lower() not in exts:
            continue
        dst = out_root / src.relative_to(in_root)
        print(f"→ Processing: {src.relative_to(in_root)}")
        process_video(src, dst, yolo_model, device, args.kernel, args.score_thr)

if __name__ == "__main__":
    main()
