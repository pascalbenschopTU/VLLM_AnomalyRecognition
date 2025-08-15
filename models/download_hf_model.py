#!/usr/bin/env python3
# Save as get_gemma12.py and run once on a node that has internet access
# Usage:  python get_gemma12.py /scratch/$USER/models/gemma-3-12b-it
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

if len(sys.argv) != 2:
    print("Usage: python download_gemma.py <TARGET_DIR>")
    sys.exit(1)

# MODEL_NAME = "google/gemma-3-4b-it"
# MODEL_NAME = "DAMO-NLP-SG/VideoLLaMA3-7B"
# MODEL_NAME = "google/gemma-3n-e4b-it"
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

target_dir = Path(sys.argv[1]).expanduser().resolve()
target_dir.mkdir(parents=True, exist_ok=True)

print(f"Downloading {MODEL_NAME} into, {target_dir}")
snapshot_download(
    repo_id=MODEL_NAME,
    local_dir=target_dir,          # <-- goes here, **not** ~/.cache
    local_dir_use_symlinks=False,  # copy real files (no symlinks into cache)
    token=True,                    # use your cached token
    resume_download=True,          # pick up half-done shards
)
print("Done")
