#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combined aggregator + plotting for Video-LLM evaluation logs (matplotlib only).

What it does
------------
1) Recursively parses evaluation result text files ONLY from these experiments under --root/logs:
     Anomaly/, Following/, Spatial/, Spatial_Colored/
   Extracts:
   - Overall metrics (incl. total/correct/accuracy)
   - Per-class global accuracy (Top-1 only)
   - Per-scene accuracy (and per-scene per-class)
   Writes four CSVs next to the chosen --out base:
     <out>_overall.csv
     <out>_by_scene.csv
     <out>_by_scene_class.csv

   NOTE: Extra LLM stats and any Top-3 metrics are intentionally ignored.

2) Generates publication-ready PNGs (no seaborn, one figure per chart):
   - Overall accuracy across ALL experiments (grouped by model).
   - Per-experiment scene accuracy (grouped by model), y-scale 0..1.
   - Prompt-variant comparisons (short vs explain); if absent, falls back to first vs later.
   - First vs later accuracy per experiment, plus delta plots.

Usage
-----
python combined_video_llm_eval.py \
  --root . \
  --out out/aggregated_results \
  --plots   # include to generate charts

Notes
-----
- Colors: consistent per model using matplotlib tab colors.
- We never set custom styles; only pure matplotlib.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors


BASE_FONTSIZE = 13  # pick 12–14; tweak once here
plt.rcParams.update({
    "font.size": BASE_FONTSIZE,        # default for all text
    "axes.titlesize": BASE_FONTSIZE,
    "axes.labelsize": BASE_FONTSIZE,
    "xtick.labelsize": BASE_FONTSIZE,
    "ytick.labelsize": BASE_FONTSIZE,
    "legend.fontsize": BASE_FONTSIZE,
    "legend.title_fontsize": BASE_FONTSIZE,
    "figure.titlesize": BASE_FONTSIZE,
})

# ----------------------------- Regex helpers & constants -----------------------------

RE_FLOAT_PCT = re.compile(r'([-+]?\d*\.?\d+)\s*%')
RE_FLOAT = re.compile(r'([-+]?\d*\.?\d+)')
RE_COUNT_FRAC = re.compile(r'(\d+)\s*/\s*(\d+)')
RE_SCENE = re.compile(r'^\s*SCENE:\s*([A-Za-z0-9_\- ]+)\s*$', re.IGNORECASE)
RE_SOURCE_JSON = re.compile(r'^\s*Source JSON:\s*(.+)$')
RE_PER_CLASS_ACC_LINE = re.compile(r'^\s*-\s*([^:]+):\s*([-+]?\d*\.?\d+)')
RE_OVERALL_TOTAL = re.compile(r'^\s*total:\s*(\d+)', re.IGNORECASE)
RE_OVERALL_CORRECT = re.compile(r'^\s*correct:\s*(\d+)', re.IGNORECASE)
RE_OVERALL_ACCURACY = re.compile(r'^\s*accuracy:\s*([-+]?\d*\.?\d+)')
RE_RUN_ID = re.compile(r'eval_(\d{8}-\d{6})')

EXPERIMENTS = ["Anomaly", "Spatial", "Spatial_Colored", "Following"]

# ----------------------------- Data classes -----------------------------

@dataclass
class RunMeta:
    experiment: str
    model: str
    run_id: str
    file_path: str
    prompt_variant: str = "unknown"  # "explain" | "short" | "unknown"
    prompt_excerpt: Optional[str] = None
    source_json_path: Optional[str] = None

@dataclass
class OverallMetrics:
    macro_top1_acc_pct: Optional[float] = None
    micro_vad_auc: Optional[float] = None
    # Robustness (optional)
    normal_wrong_count: Optional[int] = None
    normal_total_count: Optional[int] = None
    normal_wrong_rate_pct: Optional[float] = None
    pct_snippets_non_normal_in_normal_videos: Optional[float] = None
    pct_snippets_non_normal_in_anomalous_videos: Optional[float] = None
    # OVERALL block (reliable)
    total: Optional[int] = None
    correct: Optional[int] = None
    accuracy: Optional[float] = None  # 0..1

    class_a_label: Optional[str] = None
    class_b_label: Optional[str] = None
    class_a_acc: Optional[float] = None  # 0..1
    class_b_acc: Optional[float] = None  # 0..1

@dataclass
class ClassMetrics:
    class_name: str
    per_class_acc: Optional[float] = None  # 0..1

@dataclass
class SceneMetrics:
    scene: str
    total: Optional[int] = None
    correct: Optional[int] = None
    accuracy: Optional[float] = None  # 0..1

@dataclass
class ParsedRun:
    meta: RunMeta
    overall: OverallMetrics = field(default_factory=OverallMetrics)
    per_class: List[ClassMetrics] = field(default_factory=list)
    scenes: List[SceneMetrics] = field(default_factory=list)
    per_scene_class: List[Tuple[str, ClassMetrics]] = field(default_factory=list)  # (scene, ClassMetrics)

# ----------------------------- Utilities -----------------------------

def _safe_float(x: Optional[str]) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None

def _parse_float_pct(s: str) -> Optional[float]:
    """Return percent value as float (e.g., '45.00%' -> 45.00)."""
    m = RE_FLOAT_PCT.search(s)
    if m:
        return float(m.group(1))
    m2 = RE_FLOAT.search(s)
    return float(m2.group(1)) if m2 else None

def _parse_fraction(s: str) -> Tuple[Optional[int], Optional[int]]:
    m = RE_COUNT_FRAC.search(s)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))

def _try_infer_meta_from_path(path: Path) -> Tuple[str, str]:
    """
    Expected: .../logs/<experiment>/merged/<model>/<file>
    Fallback: .../logs/<experiment>/<model>/<file>
    """
    parts = path.parts
    if "logs" in parts:
        i = parts.index("logs")
        # try logs/<experiment>/merged/<model>/file
        if i + 3 < len(parts) and parts[i+2] == "merged":
            experiment = parts[i+1]
            model = parts[i+3]
            return experiment, model
        # fallback logs/<experiment>/<model>/file
        if i + 2 < len(parts):
            experiment = parts[i+1]
            model = parts[i+2]
            return experiment, model
    return (path.parent.parent.name, path.parent.name)

def _infer_prompt_variant_from_json(json_path: str) -> Tuple[str, Optional[str]]:
    """
    Read JSON and detect if prompt includes 'explain' (case-insensitive) in any *prompt*/*instruction* fields.
    Returns (variant, excerpt) where variant in {'explain', 'short', 'unknown'}.
    """
    try:
        p = Path(json_path)
        if not p.exists():
            return "unknown", None
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return "unknown", None

    prompts = []

    def visit(obj, key_path=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                kp = f"{key_path}.{k}" if key_path else str(k)
                visit(v, kp)
        elif isinstance(obj, list):
            for idx, v in enumerate(obj):
                visit(v, f"{key_path}[{idx}]")
        else:
            if isinstance(obj, str) and ("prompt" in key_path.lower() or "instruction" in key_path.lower()):
                prompts.append(obj)

    visit(data)
    if not prompts:
        return "unknown", None

    combined = "\n---\n".join(prompts)
    excerpt = combined[:300].replace("\n", " ")
    if re.search(r'\bexplain\b', combined, re.IGNORECASE):
        return "explain", excerpt
    else:
        return "short", excerpt

# ----------------------------- Parser -----------------------------

def parse_eval_file(file_path: Path) -> ParsedRun:
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    # meta
    experiment, model = _try_infer_meta_from_path(file_path)
    run_id_match = RE_RUN_ID.search(file_path.name)
    run_id = run_id_match.group(1) if run_id_match else file_path.stem

    source_json_path: Optional[str] = None
    for ln in lines:
        m = RE_SOURCE_JSON.match(ln)
        if m:
            source_json_path = m.group(1).strip()
            break

    prompt_variant = "unknown"
    prompt_excerpt = None
    if source_json_path:
        prompt_variant, prompt_excerpt = _infer_prompt_variant_from_json(source_json_path)

    meta = RunMeta(
        experiment=experiment,
        model=model,
        run_id=run_id,
        file_path=str(file_path),
        prompt_variant=prompt_variant,
        prompt_excerpt=prompt_excerpt,
        source_json_path=source_json_path,
    )
    parsed = ParsedRun(meta=meta)
    overall = parsed.overall

    # State
    section = None
    in_per_class_accuracy = False
    in_overall_block = False
    in_scene_block = False
    current_scene: Optional[SceneMetrics] = None
    in_overall_per_class = False
    overall_per_class_pairs: List[Tuple[str, Optional[float]]] = []


    for ln in lines:
        s = ln.strip()

        # Section toggles
        if s.startswith('---') and 'Robustness' in s:
            section = 'robustness'
            in_per_class_accuracy = in_overall_block = False
            continue
        if s.startswith('Per-Class Accuracy'):
            in_per_class_accuracy = True
            section = 'per_class_accuracy'
            continue
        # Explicitly ignore any "Per-Class Top-3 Predictions" sections
        if s.startswith('Per-Class Top-3 Predictions'):
            in_per_class_accuracy = False
            section = 'ignore_top3'
            continue
        # Explicitly ignore any "Extra LLM-response statistics" section
        if s.startswith('--- Extra LLM-response statistics'):
            section = 'ignore_extra_llm'
            in_per_class_accuracy = False
            continue
        if s.startswith('Evaluation Results'):
            section = 'evaluation_results'
            in_overall_block = True
            in_per_class_accuracy = False
            continue

        # Scene header
        m_scene = RE_SCENE.match(ln)
        if m_scene:
            section = 'scene'
            in_scene_block = True
            current_scene = SceneMetrics(scene=m_scene.group(1).strip())
            parsed.scenes.append(current_scene)
            continue

        # Top metrics (header/summary area)
        if section is None or section == 'prelude':
            if ln.startswith('Macro Top-1 Accuracy'):
                overall.macro_top1_acc_pct = _parse_float_pct(ln)
                continue
            # Ignore Macro Top-3 entirely
            if ln.startswith('Micro VAD AUC Score'):
                m = RE_FLOAT.search(ln)
                overall.micro_vad_auc = float(m.group(1)) if m else None
                continue

        # Robustness
        if section == 'robustness':
            if 'Normal videos wrongly flagged (count)' in ln:
                a, b = _parse_fraction(ln)
                overall.normal_wrong_count, overall.normal_total_count = a, b
                continue
            if 'Normal videos wrongly flagged (rate)' in ln:
                overall.normal_wrong_rate_pct = _parse_float_pct(ln)
                continue
            if "Avg % of snippets predicted 'non-Normal' in Normal videos" in ln:
                overall.pct_snippets_non_normal_in_normal_videos = _parse_float_pct(ln)
                continue
            if "Avg % of snippets predicted 'non-Normal' in Anomalous videos" in ln:
                overall.pct_snippets_non_normal_in_anomalous_videos = _parse_float_pct(ln)
                continue

        # Per-Class Accuracy (global, Top-1 only)
        if in_per_class_accuracy and ln.strip().startswith('-'):
            m = RE_PER_CLASS_ACC_LINE.match(ln)
            if m:
                cls = m.group(1).strip()
                val = _safe_float(m.group(2))
                parsed.per_class.append(ClassMetrics(class_name=cls, per_class_acc=val))
            continue

        # Evaluation Results (OVERALL)
        if in_overall_block and section == 'evaluation_results':
            m = RE_OVERALL_TOTAL.match(ln)
            if m:
                overall.total = int(m.group(1))
                continue
            m = RE_OVERALL_CORRECT.match(ln)
            if m:
                overall.correct = int(m.group(1))
                continue
            m = RE_OVERALL_ACCURACY.match(ln)
            if m:
                overall.accuracy = float(m.group(1))
                continue

        # Start of OVERALL per_class_acc section
        if in_overall_block and ('per_class_acc' in s.lower()):
            in_overall_per_class = True
            continue

        # Collect the two "- <label>: <val>" lines
        if in_overall_per_class:
            if ln.strip().startswith('-'):
                m = RE_PER_CLASS_ACC_LINE.match(ln)
                if m:
                    lbl = m.group(1).strip()
                    val = _safe_float(m.group(2))
                    overall_per_class_pairs.append((lbl, val))
                    if len(overall_per_class_pairs) >= 2:
                        in_overall_per_class = False
                continue
            # end the block on blank/next section
            if s == "" or s.startswith('SCENE:') or s.startswith('---'):
                in_overall_per_class = False
                continue


        # Scene details
        if in_scene_block and current_scene is not None:
            if RE_OVERALL_TOTAL.match(ln):
                current_scene.total = int(RE_OVERALL_TOTAL.match(ln).group(1))
                continue
            if RE_OVERALL_CORRECT.match(ln):
                current_scene.correct = int(RE_OVERALL_CORRECT.match(ln).group(1))
                continue
            if RE_OVERALL_ACCURACY.match(ln):
                current_scene.accuracy = float(RE_OVERALL_ACCURACY.match(ln).group(1))
                continue
            if ln.strip().startswith('-'):
                m = RE_PER_CLASS_ACC_LINE.match(ln)
                if m:
                    cls = m.group(1).strip()
                    val = _safe_float(m.group(2))
                    parsed.per_scene_class.append((current_scene.scene, ClassMetrics(class_name=cls, per_class_acc=val)))
                continue

    # Commit the two overall per-class values (if found)
    if overall_per_class_pairs:
        if len(overall_per_class_pairs) >= 1:
            overall.class_a_label, overall.class_a_acc = overall_per_class_pairs[0]
        if len(overall_per_class_pairs) >= 2:
            overall.class_b_label, overall.class_b_acc = overall_per_class_pairs[1]


    return parsed

# ----------------------------- Aggregation -----------------------------

def aggregate(root: Path, out_base: Path) -> Dict[str, Path]:
    """
    Only include files from the four whitelisted experiment directories.
    Expected layouts:
      <root>/logs/<Experiment>/merged/<Model>/*_evaluation_results.txt
      <root>/logs/<Experiment>/<Model>/*_evaluation_results.txt
    """
    logs_dir = root / "logs"
    files = set()
    for exp in EXPERIMENTS:
        exp_dir = logs_dir / exp
        if not exp_dir.exists():
            continue
        # logs/<Experiment>/merged/<Model>/*_evaluation_results.txt
        for p in exp_dir.glob("merged/*/*_evaluation_results.txt"):
            if p.is_file():
                files.add(p.resolve())
        # logs/<Experiment>/<Model>/*_evaluation_results.txt
        for p in exp_dir.glob("*/*_evaluation_results.txt"):
            if p.is_file():
                files.add(p.resolve())
    files = sorted(files)

    if not files:
        print(f"[WARN] No evaluation files found under: {logs_dir} for experiments {EXPERIMENTS}")
        return {}

    overall_rows: List[Dict] = []
    class_rows: List[Dict] = []
    scene_rows: List[Dict] = []
    scene_class_rows: List[Dict] = []

    for fp in files:
        parsed = parse_eval_file(Path(fp))

        # Overall row
        o = {
            "experiment": parsed.meta.experiment,
            "model": parsed.meta.model,
            "run_id": parsed.meta.run_id,
            "file_path": parsed.meta.file_path,
            "prompt_variant": parsed.meta.prompt_variant,
            "source_json": parsed.meta.source_json_path,
            "prompt_excerpt": parsed.meta.prompt_excerpt,
            "macro_top1_acc_pct": parsed.overall.macro_top1_acc_pct,
            "micro_vad_auc": parsed.overall.micro_vad_auc,
            "overall_total": parsed.overall.total,
            "overall_correct": parsed.overall.correct,
            "overall_accuracy": parsed.overall.accuracy,
            # Per class acc
            "overall_class_a_label": parsed.overall.class_a_label,
            "overall_class_b_label": parsed.overall.class_b_label,
            "overall_class_a_acc": parsed.overall.class_a_acc,
            "overall_class_b_acc": parsed.overall.class_b_acc,
            # robustness (optional)
            "normal_wrong_count": parsed.overall.normal_wrong_count,
            "normal_total_count": parsed.overall.normal_total_count,
            "normal_wrong_rate_pct": parsed.overall.normal_wrong_rate_pct,
            "pct_snippets_non_normal_in_normal_videos": parsed.overall.pct_snippets_non_normal_in_normal_videos,
            "pct_snippets_non_normal_in_anomalous_videos": parsed.overall.pct_snippets_non_normal_in_anomalous_videos,
        }
        overall_rows.append(o)

        # Per-class (global, Top-1 only)
        for cm in parsed.per_class:
            class_rows.append({
                "experiment": parsed.meta.experiment,
                "model": parsed.meta.model,
                "run_id": parsed.meta.run_id,
                "file_path": parsed.meta.file_path,
                "prompt_variant": parsed.meta.prompt_variant,
                "class_name": cm.class_name,
                "per_class_acc": cm.per_class_acc,
            })

        # Scenes
        for sc in parsed.scenes:
            scene_rows.append({
                "experiment": parsed.meta.experiment,
                "model": parsed.meta.model,
                "run_id": parsed.meta.run_id,
                "file_path": parsed.meta.file_path,
                "prompt_variant": parsed.meta.prompt_variant,
                "scene": sc.scene,
                "scene_total": sc.total,
                "scene_correct": sc.correct,
                "scene_accuracy": sc.accuracy,
            })

        # Per-scene per-class
        for scene_name, cm in parsed.per_scene_class:
            scene_class_rows.append({
                "experiment": parsed.meta.experiment,
                "model": parsed.meta.model,
                "run_id": parsed.meta.run_id,
                "file_path": parsed.meta.file_path,
                "prompt_variant": parsed.meta.prompt_variant,
                "scene": scene_name,
                "class_name": cm.class_name,
                "scene_per_class_acc": cm.per_class_acc,
            })

    # To DataFrames
    df_overall = pd.DataFrame(overall_rows)
    df_by_scene = pd.DataFrame(scene_rows)
    df_by_scene_class = pd.DataFrame(scene_class_rows)

    out_base.parent.mkdir(parents=True, exist_ok=True)

    # CSVs
    csv_overall = out_base.with_name(out_base.name + "_overall.csv")
    csv_by_scene = out_base.with_name(out_base.name + "_by_scene.csv")
    csv_by_scene_class = out_base.with_name(out_base.name + "_by_scene_class.csv")

    df_overall.to_csv(csv_overall, index=False)
    df_by_scene.to_csv(csv_by_scene, index=False)
    df_by_scene_class.to_csv(csv_by_scene_class, index=False)

    print(f"[OK] wrote {csv_overall}")
    print(f"[OK] wrote {csv_by_scene}")
    print(f"[OK] wrote {csv_by_scene_class}")

    return {
        "overall": csv_overall,
        "by_scene": csv_by_scene,
        "by_scene_class": csv_by_scene_class,
    }

# ----------------------------- Plotting (matplotlib only, single-figure charts) -----------------------------

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _model_order(models: List[str]) -> List[str]:
    preferred = [
        "Gemma3-4B",
        "NVILA-8B-Video",
        "Qwen2.5-VL-7B-Instruct",
        "VideoLLama3-7B",
    ]
    out = [m for m in preferred if m in models]
    out += [m for m in sorted(models) if m not in out]
    return out

def _experiment_order(experiments: List[str]) -> List[str]:
    preferred = [
        "Anomaly",
        "Spatial",
        "Spatial_Colored",
        "Following",
    ]
    out = [e for e in preferred if e in experiments]
    out += [e for e in sorted(experiments) if e not in out]
    return out

def _scene_order(scenes: List[str]) -> List[str]:
    preferred = ["left", "middle", "right", "zoom", "hdri"]
    out = [s for s in preferred if s in scenes]
    out += [s for s in scenes if s not in out]
    return out

def _color_map(models: List[str]) -> Dict[str, str]:
    """
    Shades of blue from light to near-black using the 'Blues' colormap.
    This keeps figures readable when printed in grayscale.
    """
    cmap_blue = plt.get_cmap("Blues")
    n = max(2, len(models))
    # sample from light(≈0.30) to very dark(≈0.98)
    samples = np.linspace(0.30, 0.98, n)
    return {m: mcolors.to_hex(cmap_blue(samples[i])) for i, m in enumerate(models)}

def _grouped_bar(ax, group_labels: List[str], series_values: Dict[str, List[float]], series_colors: Dict[str, str], ylabel: str, title: str, y0: float = 0.0, y1: float = 1.0, annotate: bool = True):
    """Draw grouped bars: one group per x label, one bar per series inside the group."""
    n_groups = len(group_labels)
    series_names = list(series_values.keys())
    n_series = len(series_names)

    x = np.arange(n_groups, dtype=float)
    total_width = 0.82
    bar_w = total_width / max(n_series, 1)
    offsets = (np.arange(n_series) - (n_series - 1) / 2.0) * bar_w

    for si, name in enumerate(series_names):
        vals = np.array(series_values[name], dtype=float)
        ax.bar(x + offsets[si], vals, width=bar_w * 0.95, label=name, color=series_colors.get(name, None))

        if annotate:
            for xi, yi in zip(x + offsets[si], vals):
                if not np.isnan(yi):
                    # ax.text(xi, yi + (y1 - y0) * 0.01, f"{yi:.2f}", ha="center", va="bottom", fontsize=8)
                    ax.text(xi, yi + (y1 - y0) * 0.01, f"{yi:.2f}", ha="center", va="bottom", fontsize=10)


    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, rotation=25, ha="right")
    ax.set_ylim(y0, y1)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(ncols=min(4, len(series_names)))
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()


def _grouped_bar_with_hatches(
    ax,
    group_labels: List[str],
    models: List[str],
    variant_values: Dict[str, Dict[str, List[float]]],
    color_map: Dict[str, str],
    variants: List[str],
    variant_hatches: Dict[str, str],
    ylabel: str,
    title: str,
    y0: float = 0.0,
    y1: float = 1.0,
    annotate: bool = True,
):
    """
    Draw grouped bars where each group contains bars for (model x variant).
    variant_values[model][variant] -> list across groups
    """
    n_groups = len(group_labels)
    n_models = len(models)
    n_variants = len(variants)

    x = np.arange(n_groups, dtype=float)
    total_width = 0.86
    inner_pair_gap_frac = 0.25  # kept from your original (not used explicitly)
    group_width = total_width / max(n_models, 1)
    bar_w = group_width / max(n_variants, 1)

    # Offsets: center the model groups around each x, then inner offsets for variants
    model_offsets = (np.arange(n_models) - (n_models - 1) / 2.0) * group_width
    variant_offsets = (np.arange(n_variants) - (n_variants - 1) / 2.0) * bar_w

    # --- draw bars & collect per-variant arrays for later diff-annotation ---
    # pad for annotation above the higher of the two bars
    pad = (y1 - y0) * 0.03
    max_text_y = y1

    for mi, model in enumerate(models):
        per_variant_vals = {}
        per_variant_pos = {}

        for vi, variant in enumerate(variants):
            vals = np.array(
                variant_values.get(model, {}).get(variant, [np.nan] * n_groups),
                dtype=float,
            )
            positions = x + model_offsets[mi] + variant_offsets[vi]
            ax.bar(
                positions,
                vals,
                width=bar_w * 0.95,
                color=color_map.get(model, None),
                hatch=variant_hatches.get(variant, ""),
                edgecolor="black",
                label=f"{model} - {variant}",
            )

            per_variant_vals[variant] = vals
            per_variant_pos[variant] = positions

        # --- single annotation per pair: Δ = later - first ---
        if annotate and ("first" in per_variant_vals) and ("later" in per_variant_vals):
            f_vals = per_variant_vals["first"]
            l_vals = per_variant_vals["later"]
            l_pos  = per_variant_pos["later"]

            for gi in range(n_groups):
                f = f_vals[gi]
                l = l_vals[gi]
                if np.isnan(f) or np.isnan(l):
                    continue

                diff = l - f
                y_text = max(f, l) + pad
                x_text = l_pos[gi] - 0.06  # center above the second ("later") bar

                ax.text(
                    x_text,
                    y_text,
                    f"{diff:+.2f}",
                    ha="center",
                    va="bottom",
                    path_effects=[pe.withStroke(linewidth=2, foreground="white")],
                    fontsize=10.5,
                )
                max_text_y = max(max_text_y, y_text)

    # Axes/labels/legends
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Build clean 2-part legend: colors for models, hatches for variants
    model_handles = [Patch(facecolor=color_map[m], edgecolor="black", label=m) for m in models]
    labels = {"first": "Default", "later": "+ Explain your answer"}
    variant_handles = [
        Patch(facecolor="white", edgecolor="black", hatch=variant_hatches[v], label=labels.get(v, v))
        for v in variants
    ]
    leg1 = ax.legend(handles=model_handles, title="Models", loc="upper left", bbox_to_anchor=(1.005, 1.0))
    ax.add_artist(leg1)
    ax.legend(handles=variant_handles, title="Prompt variants", loc="lower left", bbox_to_anchor=(1.005, 0.0))

    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

    # Ensure headroom for the annotations
    ax.set_ylim(y0, max(y1, max_text_y + pad))

    plt.tight_layout()



def _read_csvs(base: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_overall = pd.read_csv(base.with_name(base.name + "_overall.csv"))
    df_by_scene = pd.read_csv(base.with_name(base.name + "_by_scene.csv"))
    # sanitize numeric columns
    for col in ["overall_accuracy", "overall_class_a_acc", "overall_class_b_acc"]:
        if col in df_overall.columns:
            df_overall[col] = pd.to_numeric(df_overall[col], errors="coerce")
    if "scene_accuracy" in df_by_scene.columns:
        df_by_scene["scene_accuracy"] = pd.to_numeric(df_by_scene["scene_accuracy"], errors="coerce")
    # parse run_id (eval_YYYYMMDD-HHMMSS)
    if "run_id" in df_overall.columns:
        def _to_ts(x: str):
            try:
                return pd.to_datetime(x, format="%Y%m%d-%H%M%S", errors="coerce")
            except Exception:
                return pd.NaT
        df_overall["run_ts"] = df_overall["run_id"].apply(_to_ts)
    return df_overall, df_by_scene

def _variant_columns(df_overall: pd.DataFrame) -> pd.Series:
    # Prefer prompt_variant if available; else infer first/later by run_ts within (experiment, model)
    if "prompt_variant" in df_overall.columns and df_overall["prompt_variant"].notna().any():
        v = df_overall["prompt_variant"].fillna("unknown").str.lower()
        v = v.replace({"explain": "explain", "short": "short"})
        v = v.where(v.isin(["short", "explain"]), other="unknown")
        return v
    # fallback: first vs later per (experiment, model)
    df = df_overall.copy()
    df["variant_fallback"] = "unknown"
    if "run_ts" in df.columns:
        df = df.sort_values("run_ts")
        first_idx = df.groupby(["experiment", "model"], dropna=False)["run_ts"].head(1).index
        last_idx = df.groupby(["experiment", "model"], dropna=False)["run_ts"].tail(1).index
        df.loc[first_idx, "variant_fallback"] = "first"
        df.loc[last_idx, "variant_fallback"] = "later"
    return df["variant_fallback"]

# ----------------------------- Plot routines -----------------------------

def plot_overall_across_experiments(base: Path, outdir: Path):
    df_overall, _ = _read_csvs(base)
    if df_overall.empty:
        print("[WARN] overall CSV empty; skipping overall plots.")
        return

    g = (df_overall
         .groupby(["experiment", "model"], dropna=False)["overall_accuracy"]
         .mean()
         .reset_index())

    if g.empty:
        print("[WARN] no overall_accuracy values; skipping.")
        return

    experiments = _experiment_order(sorted(g["experiment"].dropna().unique().tolist()))
    models = _model_order(sorted(g["model"].dropna().unique().tolist()))
    cmap = _color_map(models)

    series_vals = {}
    for m in models:
        vals = []
        for e in experiments:
            row = g[(g["experiment"] == e) & (g["model"] == m)]
            vals.append(float(row["overall_accuracy"].values[0]) if not row.empty else np.nan)
        series_vals[m] = vals

    fig, ax = plt.subplots(figsize=(10, 6))

    _grouped_bar(
        ax,
        group_labels=experiments,
        series_values=series_vals,
        series_colors=cmap,
        ylabel="Accuracy (0-1)",
        title="Accuracy of Vision-LLMs averaged — averaged over all scenes",
        y0=0.0, y1=1.0, annotate=True,
    )

    # Move legend outside and reserve right margin
    # ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, ncols=1)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0., title="Models")
    # ax.legend(loc="upper left")
    # plt.tight_layout(rect=[0.0, 0.0, 0.80, 1.0])
    plt.tight_layout()

    _ensure_dir(outdir)
    out_path = outdir / "overall_accuracy_by_model_across_experiments.pdf"
    plt.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"[OK] {out_path}")



def plot_scene_accuracy_all_experiments_combined(base: Path, outdir: Path):
    """
    Single figure: for each scene (x-axis), bars = models (mean scene_accuracy averaged across *all* experiments).
    Pure matplotlib, y-scale 0..1 with headroom for annotations.
    """
    _, df_by_scene = _read_csvs(base)
    if df_by_scene.empty:
        print("[WARN] by_scene CSV empty; skipping scene plots.")
        return

    required = {"experiment", "model", "scene", "scene_accuracy"}
    if not required.issubset(df_by_scene.columns):
        missing = required - set(df_by_scene.columns)
        print(f"[WARN] Missing columns in by_scene CSV: {missing}; skipping.")
        return

    # Mean across all experiments for each (model, scene)
    g = (
        df_by_scene
        .groupby(["model", "scene"], dropna=False)["scene_accuracy"]
        .mean()
        .reset_index()
    )
    if g.empty:
        print("[WARN] no scene_accuracy values; skipping.")
        return

    scenes = _scene_order(sorted(g["scene"].dropna().unique().tolist()))
    models = _model_order(sorted(g["model"].dropna().unique().tolist()))
    if not scenes or not models:
        print("[WARN] Not enough data to plot.")
        return

    cmap = _color_map(models)

    # Build series values: model -> [mean acc per scene]
    series_vals: Dict[str, List[float]] = {}
    for m in models:
        vals = []
        for s in scenes:
            row = g[(g["model"] == m) & (g["scene"] == s)]
            vals.append(float(row["scene_accuracy"].values[0]) if not row.empty else np.nan)
        series_vals[m] = vals

    n_exps = int(df_by_scene["experiment"].nunique())

    fig, ax = plt.subplots(figsize=(12, 6))
    _grouped_bar(
        ax,
        group_labels=scenes,
        series_values=series_vals,
        series_colors=cmap,
        ylabel="Accuracy (0-1)",
        title=f"How Scene Affects Model Accuracy — averaged over all experiments",
        y0=0.0, y1=1.0, annotate=True,
    )

    # Headroom for annotations + legend outside
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0., title="Models")

    plt.tight_layout()

    _ensure_dir(outdir)
    out_path = outdir / "scene_accuracy_by_model__combined.pdf"
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_path}")


def plot_first_vs_later_and_deltas(base: Path, outdir: Path):
    """
    Improved single figure:
      x-axis = [Anomaly, Following, Spatial, Spatial + Color]
      colored bars = models; hatched twin = 'prompt + Explain your answer'
      (falls back to first vs later if prompt variants absent).
    """
    df_overall, _ = _read_csvs(base)
    if df_overall.empty or "overall_accuracy" not in df_overall.columns:
        print("[WARN] overall CSV empty or missing overall_accuracy; skipping.")
        return

    # Normalize variants:
    # - Prefer explicit prompt variants (short/explain) if present
    # - Else fallback to first/later via run_ts
    df = df_overall.copy()
    df["variant_simple"] = _variant_columns(df)  # uses prompt_variant when present; else run_ts first/later

    has_prompt = {"short", "explain"} & set(df["variant_simple"].unique())
    if has_prompt:
        raw_variants = ["short", "explain"]
        disp_variants = ["first", "prompt + Explain your answer"]
        variant_map = dict(zip(raw_variants, disp_variants))
        hatches = {"first": "", "prompt + Explain your answer": "///"}
        title = "Accuracy by experiment — models × (first vs prompt + Explain your answer)"
        outfile = "first_vs_later__combined_improved.pdf"
    else:
        # robust earliest/latest using idxmin/idxmax on run_ts
        if "run_ts" in df.columns:
            df = df.sort_values("run_ts")
            idx_first = df.groupby(["experiment", "model"], dropna=False)["run_ts"].idxmin()
            idx_later = df.groupby(["experiment", "model"], dropna=False)["run_ts"].idxmax()
            df.loc[:, "variant_simple"] = "middle"  # temp
            df.loc[idx_first, "variant_simple"] = "first"
            df.loc[idx_later, "variant_simple"] = "later"
        raw_variants = ["first", "later"]
        disp_variants = ["first", "later"]
        variant_map = dict(zip(raw_variants, disp_variants))
        hatches = {"first": "", "later": "///"}
        title = "Effect of adding 'Explain your answer' to the prompt — averaged over all scenes"
        outfile = "first_vs_later__combined_improved.pdf"

    # Aggregate mean accuracies per (experiment, model, variant)
    g = (df.groupby(["experiment", "model", "variant_simple"], dropna=False)["overall_accuracy"]
           .mean()
           .reset_index())
    if g.empty:
        print("[WARN] no grouped data; skipping.")
        return

    # Exact experiment order requested; only keep those present
    desired = ["Anomaly", "Following", "Spatial", "Spatial_Colored"]
    experiments = [e for e in desired if e in g["experiment"].unique().tolist()]
    if not experiments:
        print("[WARN] none of the requested experiments are present; skipping.")
        return

    pretty_labels = [("Spatial + Color" if e == "Spatial_Colored" else e) for e in experiments]

    models = _model_order(sorted(g["model"].dropna().unique().tolist()))
    if not models:
        print("[WARN] no models found; skipping.")
        return
    cmap = _color_map(models)

    # Build values for grouped-bar-with-hatches:
    # variant_values[model][display_variant] -> list of values across experiments
    variant_values = {m: {dv: [] for dv in disp_variants} for m in models}
    for e in experiments:
        ge = g[g["experiment"] == e]
        for m in models:
            for rv, dv in variant_map.items():
                row = ge[(ge["model"] == m) & (ge["variant_simple"] == rv)]
                val = float(row["overall_accuracy"].values[0]) if not row.empty else np.nan
                variant_values[m][dv].append(val)

    # Figure size scales with experiments and #models
    # fig_width = max(16, 1.1 * len(experiments) * (1.0 + 0.40 * len(models)))
    # fig, ax = plt.subplots(figsize=(fig_width, 7.0))
    fig, ax = plt.subplots(figsize=(12, 6))

    _grouped_bar_with_hatches(
        ax,
        group_labels=pretty_labels,
        models=models,
        variant_values=variant_values,
        color_map=cmap,
        variants=disp_variants,
        variant_hatches=hatches,
        ylabel="Accuracy (0-1)",
        title=title,
        y0=0.0, y1=1.0, annotate=True,
    )

    # Reserve space for the two legends placed by the helper
    plt.tight_layout(rect=[0.0, 0.0, 0.90, 1.0])
    # plt.tight_layout()

    _ensure_dir(outdir)
    out_path = outdir / outfile
    plt.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"[OK] {out_path}")



def plot_anomaly_confusion_matrices(base: Path, outdir: Path):
    """
    Build 2x2 confusion matrices per model for the Anomaly experiment
    using **AVERAGES across all runs** from <out>_overall.csv.

    Rows=True class, Cols=Pred class. Canonical classes: ["Anomaly","Dancing"].
    """
    df_overall, _ = _read_csvs(base)
    if df_overall.empty:
        print("[WARN] overall CSV empty; skipping.")
        return

    need = {
        "experiment", "model",
        "overall_class_a_label", "overall_class_b_label",
        "overall_class_a_acc", "overall_class_b_acc"
    }
    if not need.issubset(df_overall.columns):
        print(f"[WARN] Missing columns: {need - set(df_overall.columns)}; skipping.")
        return

    dfe = df_overall[df_overall["experiment"] == "Anomaly"].copy()
    if dfe.empty:
        print("[WARN] No Anomaly rows; skipping.")
        return

    # numeric sanitize
    for col in ["overall_class_a_acc", "overall_class_b_acc"]:
        dfe[col] = pd.to_numeric(dfe[col], errors="coerce")

    # canonical label mapping (treat "Normal" as the negative class)
    def canon(lbl: str) -> Optional[str]:
        s = (str(lbl) if lbl is not None else "").strip().lower()
        if "anom" in s:
            return "Anomaly"
        if "danc" in s or "norm" in s:
            return "Dancing"
        return None

    # explode into long format: one row per (model, canon_class, acc)
    long_rows = []
    for _, r in dfe.iterrows():
        a_lbl = canon(r["overall_class_a_label"])
        b_lbl = canon(r["overall_class_b_label"])
        a_acc = r["overall_class_a_acc"]
        b_acc = r["overall_class_b_acc"]
        if a_lbl is not None and pd.notna(a_acc):
            long_rows.append({"model": r["model"], "canon": a_lbl, "acc": float(a_acc)})
        if b_lbl is not None and pd.notna(b_acc):
            long_rows.append({"model": r["model"], "canon": b_lbl, "acc": float(b_acc)})

    if not long_rows:
        print("[WARN] No usable per-class rows; skipping.")
        return

    g = (
        pd.DataFrame(long_rows)
        .groupby(["model", "canon"], dropna=False)["acc"]
        .mean()
        .unstack("canon")
        .reindex(columns=["Anomaly", "Dancing"])
        .rename_axis(index=None, columns=None)
    )

    models = _model_order([m for m in g.index.tolist() if isinstance(m, str)])
    if not models:
        print("[WARN] No models found; skipping.")
        return

    # plot grid
    n = len(models)
    ncols = min(4, max(1, n))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3.6 * ncols, 3.6 * nrows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.reshape(nrows, ncols)

    classes = ["Anomaly", "Dancing"]

    for idx, model in enumerate(models):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        rec_a = float(g.loc[model]["Anomaly"]) if (model in g.index and pd.notna(g.loc[model]["Anomaly"])) else np.nan
        rec_d = float(g.loc[model]["Dancing"]) if (model in g.index and pd.notna(g.loc[model]["Dancing"])) else np.nan

        def _clip(x):
            return float(np.clip(x, 0.0, 1.0)) if pd.notna(x) else np.nan

        rec_a = _clip(rec_a)
        rec_d = _clip(rec_d)

        conf = np.array([
            [rec_a, 1.0 - rec_a if pd.notna(rec_a) else np.nan],
            [1.0 - rec_d if pd.notna(rec_d) else np.nan, rec_d],
        ], dtype=float)

        ax.imshow(conf, vmin=0.0, vmax=1.0, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(classes); ax.set_yticklabels(classes)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(model)

        for i in range(2):
            for j in range(2):
                val = conf[i, j]
                if pd.notna(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=11)

    # hide any unused axes
    for k in range(n, nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r, c].axis("off")

    plt.tight_layout()
    _ensure_dir(outdir)
    out_path = outdir / "anomaly_confusion_matrices_by_model.pdf"
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_path}")


def plot_spatial_left_right_color_effect(base: Path, outdir: Path):
    """
    LEFT vs RIGHT using **AVERAGED** OVERALL per-class accuracies from <out>_overall.csv
    for Spatial (no color) vs Spatial_Colored. Bars are per side (left/right);
    for each model: no-color (solid) vs color-coded (hatched). Δ = color - no_color.
    """
    df_overall, _ = _read_csvs(base)
    if df_overall.empty:
        print("[WARN] overall CSV empty; skipping.")
        return

    need = {
        "experiment", "model",
        "overall_class_a_label", "overall_class_b_label",
        "overall_class_a_acc", "overall_class_b_acc"
    }
    if not need.issubset(df_overall.columns):
        print(f"[WARN] Missing columns: {need - set(df_overall.columns)}; skipping.")
        return

    dfe = df_overall[df_overall["experiment"].isin(["Spatial", "Spatial_Colored"])].copy()
    if dfe.empty:
        print("[WARN] No Spatial/Spatial_Colored rows; skipping.")
        return

    for col in ["overall_class_a_acc", "overall_class_b_acc"]:
        dfe[col] = pd.to_numeric(dfe[col], errors="coerce")

    # canonical label mapping to {'left','right'}
    def canon_lr(lbl: str) -> Optional[str]:
        s = (str(lbl) if lbl is not None else "").strip().lower()
        if "left" in s:
            return "left"
        if "right" in s:
            return "right"
        return None

    # explode to long: (experiment, model, side, acc), then average across runs
    long_rows = []
    for _, r in dfe.iterrows():
        exp, model = r["experiment"], r["model"]
        a_lbl = canon_lr(r["overall_class_a_label"]); a_acc = r["overall_class_a_acc"]
        b_lbl = canon_lr(r["overall_class_b_label"]); b_acc = r["overall_class_b_acc"]
        if a_lbl is not None and pd.notna(a_acc):
            long_rows.append({"experiment": exp, "model": model, "side": a_lbl, "acc": float(a_acc)})
        if b_lbl is not None and pd.notna(b_acc):
            long_rows.append({"experiment": exp, "model": model, "side": b_lbl, "acc": float(b_acc)})

    if not long_rows:
        print("[WARN] No usable left/right rows; skipping.")
        return

    g = (
        pd.DataFrame(long_rows)
        .groupby(["experiment", "model", "side"], dropna=False)["acc"]
        .mean()
        .reset_index()
    )

    scenes = ["left", "right"]
    models = _model_order(sorted(g["model"].dropna().unique().tolist()))
    if not models:
        print("[WARN] No models found; skipping.")
        return
    cmap = _color_map(models)

    # values: model -> {'no_color':[L,R], 'color':[L,R]}
    def _get(exp, model, side):
        row = g[(g["experiment"] == exp) & (g["model"] == model) & (g["side"] == side)]
        return float(row["acc"].values[0]) if not row.empty else np.nan

    vals = {
        m: {
            "no_color": [_get("Spatial", m, s) for s in scenes],
            "color":    [_get("Spatial_Colored", m, s) for s in scenes],
        }
        for m in models
    }

    # plot
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(scenes), dtype=float)
    n_models = len(models)
    total_width = 0.86
    group_width = total_width / max(n_models, 1)
    bar_w = group_width / 2.0

    model_offsets = (np.arange(n_models) - (n_models - 1) / 2.0) * group_width

    y0, y1 = 0.0, 1.0
    pad = (y1 - y0) * 0.03
    max_text_y = y1

    for mi, m in enumerate(models):
        base_pos = x + model_offsets[mi]
        no_color_pos = base_pos - bar_w / 2.0
        color_pos    = base_pos + bar_w / 2.0

        v0 = np.array(vals[m]["no_color"], dtype=float)
        v1 = np.array(vals[m]["color"],    dtype=float)

        ax.bar(no_color_pos, v0, width=bar_w * 0.95, color=cmap.get(m, None), label=m if mi == 0 else None)
        ax.bar(color_pos,    v1, width=bar_w * 0.95, color=cmap.get(m, None), hatch="///", edgecolor="black")

        # annotate Δ above color-coded bars
        for gi in range(len(scenes)):
            a, b = v0[gi], v1[gi]
            if np.isnan(a) or np.isnan(b):
                continue
            d = b - a
            y_text = max(a, b) + pad
            ax.text(color_pos[gi] - 0.06, y_text, f"{d:+.2f}", ha="center", va="bottom",
                    fontsize=10.5, path_effects=[pe.withStroke(linewidth=2, foreground="white")])
            max_text_y = max(max_text_y, y_text)

    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in scenes])
    ax.set_ylabel("Accuracy (0-1)")
    ax.set_title("Effect of color coding on predicting assailant — averaged over all scenes")

    # legends
    model_handles = [Patch(facecolor=cmap[m], edgecolor="black", label=m) for m in models]
    variant_handles = [
        Patch(facecolor="white", edgecolor="black", hatch="",    label="Spatial (no color)"),
        Patch(facecolor="white", edgecolor="black", hatch="///", label="Spatial + Color"),
    ]
    leg1 = ax.legend(handles=model_handles, title="Models", loc="upper left", bbox_to_anchor=(1.005, 1.0))
    ax.add_artist(leg1)
    ax.legend(handles=variant_handles, title="Experiments", loc="lower left", bbox_to_anchor=(1.005, 0.0))

    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_ylim(y0, max(y1, max_text_y + pad))
    plt.tight_layout(rect=[0.0, 0.0, 0.90, 1.0])

    _ensure_dir(outdir)
    out_path = outdir / "left_right__spatial_vs_color.pdf"
    plt.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"[OK] {out_path}")


# ----------------------------- CLI -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Aggregate Video-LLM eval results (Top-1 only) and generate matplotlib figures.")
    ap.add_argument("--root", type=str, default=".", help="Project root; expects <root>/logs/<Experiment>/...")
    ap.add_argument("--out", type=str, default="aggregated_results", help="Output base path (no extension)")
    ap.add_argument("--plots", action="store_true", help="Generate PNG figures from the aggregated CSVs")
    ap.add_argument("--plots_outdir", type=str, default=None, help="Directory to write PNGs (default: <out>_improved_plots)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_base = Path(args.out).resolve()

    # 1) Aggregate -> CSVs
    csv_paths = aggregate(root, out_base)

    # 2) Plots (optional)
    if args.plots and csv_paths:
        base = out_base
        outdir = Path(args.plots_outdir).resolve() if args.plots_outdir else base.with_name(base.name + "_improved_plots")
        _ensure_dir(outdir)

        try:
            plot_overall_across_experiments(base, outdir)
        except Exception as e:
            print(f"[ERR] plot_overall_across_experiments: {e}")

        try:
            # plot_scene_accuracy_per_experiment(base, outdir)
            plot_scene_accuracy_all_experiments_combined(base, outdir)
        except Exception as e:
            print(f"[ERR] plot_scene_accuracy_per_experiment: {e}")

        try:
            plot_first_vs_later_and_deltas(base, outdir)
        except Exception as e:
            print(f"[ERR] plot_first_vs_later_and_deltas: {e}")

        try:
            plot_spatial_left_right_color_effect(base, outdir)
        except Exception as e:
            print(f"[ERR] plot_spatial_left_right_color_effect: {e}")   

        try:
            plot_anomaly_confusion_matrices(base, outdir)
        except Exception as e:
            print(f"[ERR] plot_anomaly_confusion_matrices: {e}")        

if __name__ == "__main__":
    main()
