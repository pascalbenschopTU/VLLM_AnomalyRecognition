# Video LLM Evaluation README

A comprehensive guide for running evaluation scripts on various VideoLLMs (Gemma3, Qwen2.5, VideoLLama3) and NVILA.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Dataset Structure](#dataset-structure)
4. [Configuration](#configuration)

   * [Custom Prompts & Labels](#custom-prompts--labels)
5. [Usage](#usage)

   * [Gemma3, Qwen2.5 & VideoLLama3](#gemma3-qwen25--videollama3)
   * [NVILA](#nvila)
6. [Examples](#examples)
7. [Troubleshooting](#troubleshooting)

---

## Overview

This repository provides scripts to evaluate VideoLLMs on video classification datasets. The following scripts are included:

* **apply\_videollm.py**: For Gemma3, Qwen2.5, and VideoLLama3.
* **apply\_vila.py**: For NVILA (place inside the `VILA` directory).

Supported datasets in experiments:

* RWF2000
* UCF-Crime
* Custom datasets (see [Dataset Structure](#dataset-structure))

---

## Requirements

* [Apptainer](../apptainer)

In this folder (models/) you need to place the weights of the models.

- Gemma3-4B/
- Qwen2.5-VL-7B-Instruct/
- VideoLLama3-7B/

These can be downloaded with the `download_hf_model.py` script.

Also clone the [VILA repository](https://github.com/NVlabs/VILA) and place the NVILA-8B-Video weights in this folder.

---

## Dataset Structure

Your dataset root should follow this layout:

```
dataset_root/
├── ClassA/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
├── ClassB/
│   ├── video3.mp4
│   ├── video4.mp4
│   └── ...
└── ...
```

* **Class folders**: Each subdirectory under `dataset_root` is a class label.
* **Video files**: Supported formats (e.g., `.mp4`, `.avi`).

---

## Configuration

### Custom Prompts & Labels

Before running evaluations, adapt the `set_prompt_and_labels` function in each script to match your dataset:

```python
# Inside apply_videollm.py or apply_vila.py

def set_prompt_and_labels(dataset_name):
    if dataset_name == "RWF2000":
        prompt = "Describe the action in the video."
        labels = ["punch", "kick", ...]
    elif dataset_name == "CustomDataset":
        prompt = "What event is happening?"
        labels = ["EventA", "EventB", ...]
    # Add your dataset here
    return prompt, labels
```

* **dataset\_name**: Match the folder name passed via the `--dataset_root` argument.
* **prompt**: The natural language prompt used by the VideoLLM.
* **labels**: List of class labels in the same order as your folders.

---

## Usage

### Gemma3, Qwen2.5 & VideoLLama3

Run the `apply_videollm.py` script with the following arguments:

```bash
python apply_videollm.py \
    --model_type gemma/videollama/qwen \
    --dataset_name dataset_name \
    --video_root path_to_dataset_root \
    --experiment experiment_name (optional) \
```


### NVILA

1. Place `apply_vila.py` inside the `VILA/` directory.
2. apply_vila.py expects model weights NVILA-8B-Video in the VILA/ directory
3. From the project root:

```bash
cd VILA
python apply_vila.py \
    --dataset_name dataset_name \
    --video_root path_to_dataset_root \
    --experiment experiment_name (optional) \
```

---

## Examples

```bash
# Evaluate RWF2000 with Gemma3
python apply_videollm.py --model_type gemma --dataset_name RWF2000 --video_root ../datasets_AR/RWF2000/ --experiment {experiment_name} 

# Evaluate UCF-Crime with NVILA
cd VILA
python apply_vila.py --dataset_name RWF2000 --video_root ../../datasets_AR/RWF2000/

# Use Few-Shot prompting on UCF-Crime
python apply_videollm.py --model_type videollama --dataset_name UCF_Crime --video_root ../datasets_AR/UCF_Crime/ --experiment Few_Shot"
```

---

## Troubleshooting

* **Missing dependencies**: Check the jobs for the correct apptainer container per model.
* **Dataset errors**: Verify your folder structure and update `labels` accordingly.

---

*For questions or issues, please open an issue on the repository.*
