"""
Script to cut video segments based on temporal annotation JSON,
organize outputs by class folder (first path component), and ensure playable outputs.
Usage:
    python cut_dataset.py \
        --annotations test.json \
        --video-root /path/to/videos \
        --output-root /path/to/output \
        [--ffmpeg-bin ffmpeg] [--ffprobe-bin ffprobe]
"""
import os
import json
import argparse
import shutil
import subprocess


def load_annotations(json_files):
    annotations = {}
    for jf in json_files:
        with open(jf, 'r') as f:
            data = json.load(f)
        for rel_path, segments in data.items():
            annotations.setdefault(rel_path, []).extend(segments)
    return annotations


def is_valid_mp4(ffmpeg_bin, filepath):
    """Run ffmpeg to detect playback errors."""
    cmd = [ffmpeg_bin, '-v', 'error', '-i', filepath, '-f', 'null', '-']
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc.returncode == 0


def check_with_ffprobe(ffprobe_bin, filepath):
    """Use ffprobe to catch missing streams or metadata issues."""
    cmd = [ffprobe_bin, '-v', 'error', '-show_streams', '-show_format', filepath]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc.returncode == 0


def cut_segments(video_root, annotations, output_root, ffmpeg_bin, ffprobe_bin):
    for rel_path, segments in annotations.items():
        input_path = os.path.join(video_root, rel_path)
        class_name = rel_path.split(os.sep, 1)[0]
        out_dir = os.path.join(output_root, class_name)
        os.makedirs(out_dir, exist_ok=True)

        # if no temporal annotations, copy the whole file
        if not segments:
            # compute base_name here so it’s always correct for this rel_path
            base_name = os.path.splitext(os.path.basename(rel_path))[0]
            out_path = os.path.join(out_dir, f"{base_name}.mp4")
            print(f"[INFO] Copying {rel_path} → {out_path}")
            input_path = input_path.replace("Testing_Normal_Videos_Anomaly", "Normal_Videos_event")
            shutil.copy2(input_path, out_path)
            continue

        input_path = os.path.join(video_root, rel_path)
        if not os.path.isfile(input_path):
            print(f"[WARN] Video file {input_path} not found, skipping.")
            continue

        # derive class from first folder in path
        class_name = rel_path.split(os.sep)[0]
        out_dir = os.path.join(output_root, class_name)
        os.makedirs(out_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(rel_path))[0]
        for idx, seg in enumerate(segments, start=1):
            start, end = seg['start'], seg['end']
            duration = end - start
            out_name = f"{base_name}_{idx:02d}_{start:.3f}_{end:.3f}.mp4"
            out_path = os.path.join(out_dir, out_name)

            # copy cut with timestamp fixes and faststart
            cmd_copy = [
                ffmpeg_bin, '-y',
                '-ss', str(start),
                '-i', input_path,
                '-t', str(duration),
                '-c', 'copy',
                '-avoid_negative_ts', 'make_zero',
                '-fflags', '+genpts',
                '-movflags', '+faststart',
                out_path
            ]
            print(f"[INFO] Copy-cut {rel_path} segment {idx}: {start}-{end}s")
            try:
                subprocess.run(cmd_copy, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError:
                print(f"[ERROR] Copy cut failed for {out_path}, will re-encode.")
            else:
                # validate output
                if is_valid_mp4(ffmpeg_bin, out_path) and check_with_ffprobe(ffprobe_bin, out_path):
                    continue
                print(f"[WARN] Invalid after copy, retrying re-encode: {out_path}")

            # fallback: full re-encode with faststart
            cmd_re = [
                ffmpeg_bin, '-y',
                '-ss', str(start),
                '-i', input_path,
                '-t', str(duration),
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-movflags', '+faststart',
                out_path
            ]
            print(f"[INFO] Re-encoding {rel_path} segment {idx}")
            try:
                subprocess.run(cmd_re, check=True)
                if not (is_valid_mp4(ffmpeg_bin, out_path) and check_with_ffprobe(ffprobe_bin, out_path)):
                    print(f"[ERROR] Re-encoded invalid, removing: {out_path}")
                    os.remove(out_path)
                else:
                    print(f"[INFO] Re-encode successful: {out_path}")
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] Re-encode error: {e}, skipping {out_path}")
                if os.path.exists(out_path):
                    os.remove(out_path)


def main():
    parser = argparse.ArgumentParser(
        description="Cut annotated video segments into class folders with validation."
    )
    parser.add_argument('--annotations', nargs='+', required=True,
                        help="Temporal annotation JSON files.")
    parser.add_argument('--video-root', required=True,
                        help="Root directory containing source videos.")
    parser.add_argument('--output-root', required=True,
                        help="Directory to save cut segments organized by class.")
    parser.add_argument('--ffmpeg-bin', default='ffmpeg',
                        help="FFmpeg executable path.")
    parser.add_argument('--ffprobe-bin', default='ffprobe',
                        help="FFprobe executable path.")
    args = parser.parse_args()

    annotations = load_annotations(args.annotations)
    cut_segments(args.video_root, annotations,
                 args.output_root, args.ffmpeg_bin, args.ffprobe_bin)

if __name__ == '__main__':
    main()

# Usage:
# python temporal_dataset.py --annotations temporal_annotations/test.json --video-root videos/ --output-root videos_temporal