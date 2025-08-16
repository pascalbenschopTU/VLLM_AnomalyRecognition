# Re-run after session reset: generate the combined comparison plot again.
import pandas as pd
import io
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

exp1_csv = """
,Abuse,Arrest,Arson,Assault,Burglary,Explosion,Fighting,Normal,RoadAccidents,Robbery,Shooting,Shoplifting,Stealing,Vandalism
Gemma3-4B,0.0,0.4,0.0,0.25,0.13,0.36,1.0,0.96,0.17,0.2,0.12,0.08,0.0,0.0
VideoLLama3-7B,0.0,0.0,0.2,0.0,0.0,0.5,0.8,0.33,0.43,0.2,0.04,0.04,0.0,0.25
Qwen2.5-VL-7B-Instruct,0.0,0.4,0.1,0.25,0.2,0.36,0.8,0.98,0.13,0.2,0.0,0.12,0.0,0.0
NVILA-8B-Video,0.0,0.0,0.0,0.0,0.0,0.27,0.2,0.99,0.04,0.2,0.04,0.0,0.0,0.12
Mean,0.0,0.2,0.07500000000000001,0.125,0.0825,0.3725,0.7000000000000001,0.815,0.1925,0.2,0.05,0.06,0.0,0.0925
"""
exp2_csv = """
,Abuse,Arrest,Arson,Assault,Burglary,Explosion,Fighting,Normal,RoadAccidents,Robbery,Shooting,Shoplifting,Stealing,Vandalism
Gemma3-4B,0.0,1.0,0.1,0.25,0.0,0.18,0.8,0.9,0.35,0.6,0.04,0.52,0.0,0.0
Qwen2.5-VL-7B-Instruct,0.0,0.2,0.1,0.75,0.2,0.91,0.6,0.97,0.26,0.6,0.04,0.08,0.14,0.0
NVILA-8B-Video,0.5,0.8,0.1,0.75,0.0,0.55,0.0,0.99,0.13,0.2,0.0,0.0,0.0,0.12
VideoLLama3-7B,0.5,0.2,0.3,0.25,0.2,0.36,1.0,0.89,0.43,0.4,0.04,0.36,0.14,0.12
Mean,0.25,0.55,0.15000000000000002,0.5,0.1,0.5,0.6,0.9375000000000001,0.2925,0.44999999999999996,0.03,0.24,0.07,0.06
"""
exp3_csv = """
,Abuse,Arrest,Arson,Assault,Burglary,Explosion,Fighting,Normal,RoadAccidents,Robbery,Shooting,Shoplifting,Stealing,Vandalism
Qwen2.5-VL-7B-Instruct,0.0,0.4,0.1,1.0,0.13,0.91,1.0,0.97,0.26,0.4,0.04,0.08,0.14,0.0
NVILA-8B-Video,0.0,0.8,0.2,0.5,0.13,0.45,1.0,0.94,0.61,0.6,0.32,0.0,0.57,0.25
Gemma3-4B,0.0,0.4,0.1,0.0,0.13,0.14,1.0,0.49,0.87,0.4,0.32,0.04,0.29,0.0
VideoLLama3-7B,0.5,0.0,0.1,0.25,0.2,0.36,1.0,0.97,0.39,0.2,0.12,0.16,0.14,0.0
Mean,0.125,0.4,0.125,0.4375,0.14750000000000002,0.46499999999999997,1.0,0.8425,0.5325,0.39999999999999997,0.19999999999999998,0.07,0.28500000000000003,0.0625
"""

model_colors = {
    "Gemma3-4B": "tab:blue",
    "Qwen2.5-VL-7B-Instruct": "tab:orange",
    "VideoLLama3-7B": "tab:green",
    "NVILA-8B-Video": "tab:red"
}

def read_df(csv_text):
    return pd.read_csv(io.StringIO(csv_text), index_col=0)

df1 = read_df(exp1_csv).drop(index="Mean")
df2 = read_df(exp2_csv).drop(index="Mean")
df3 = read_df(exp3_csv).drop(index="Mean")

classes = ["Fighting", "RoadAccidents", "Shooting", "Stealing"]
models = sorted(set(df1.index).union(df2.index).union(df3.index))

# Build tidy table
records = []
for exp_name, df in [("Exp 1: unguided prompt", df1), ("Exp 2: guided prompt", df2), ("Exp 3: guided + few-shot prompt", df3)]:
    for m in models:
        if m in df.index:
            for c in classes:
                if c in df.columns:
                    records.append({"Experiment": exp_name, "Model": m, "Class": c, "Acc": float(df.loc[m, c])})
tidy = pd.DataFrame.from_records(records)

exp_order = ["Exp 1: unguided prompt", "Exp 2: guided prompt", "Exp 3: guided + few-shot prompt"]
markers = {"Exp 1: unguided prompt": "o", "Exp 2: guided prompt": "s", "Exp 3: guided + few-shot prompt": "^"}

gap_between_classes = 0.8
gap_between_models = 1.0
jitter = {"Exp 1: unguided prompt": -0.2, "Exp 2: guided prompt": 0.0, "Exp 3: guided + few-shot prompt": 0.2}

x_positions = {}
tick_positions = []
tick_labels = []

x = 0.0
for c in classes:
    for m in models:
        base = x
        x_positions[(c, m)] = base
        tick_positions.append(base)
        tick_labels.append(f"{m}")
        x += gap_between_models
    x += gap_between_classes

fig, ax = plt.subplots(figsize=(18, 6))

class_midpoints = []
for c in classes:
    start = x_positions[(c, models[0])]
    end   = x_positions[(c, models[-1])]
    class_midpoints.append((start + end) / 2.0)
    ax.axvline(x=end + gap_between_models*0.8, linestyle="--", linewidth=0.7)

for c in classes:
    for m in models:
        base = x_positions[(c, m)]
        xs, ys = [], []
        for e in exp_order:
            val = tidy[(tidy["Class"]==c) & (tidy["Model"]==m) & (tidy["Experiment"]==e)]["Acc"].values
            if len(val)==0: 
                continue
            xv = base + jitter[e]
            yv = val[0]
            xs.append(xv); ys.append(yv)
            ax.scatter(xv, yv, marker=markers[e], s=140, color=model_colors.get(m, "gray"), label=e if (c==classes[0] and m==models[0]) else None)

            # ax.vlines(xv, 0, yv, colors=model_colors.get(m, "gray"), linewidth=1.2, alpha=0.8)
        if len(xs) >= 2:
            xs_sorted, ys_sorted = zip(*sorted(zip(xs, ys)))
            ax.plot(xs_sorted, ys_sorted, linewidth=1.5, alpha=0.8, linestyle="--")

ax.set_xticks([])
ax.set_xticklabels([])
ax.set_ylabel("Accuracy", fontsize=17)
ax.tick_params(axis="y", labelsize=16)
ax.set_ylim(0, 1.05)
for mid, c in zip(class_midpoints, classes):
    ax.text(mid, 1.1, c, ha="center", va="bottom", fontsize=18)

ax.set_title("Few-Shot prompt impact per model", fontsize=22, pad=60)

# Double legends
prompt_handles = [Line2D([0],[0], marker=markers[e], color="black", linestyle="None", markersize=10, label=e) for e in exp_order]
model_handles = [Line2D([0],[0], marker="o", color=model_colors[m], linestyle="None", markersize=10, label=m) for m in models]

leg1 = ax.legend(handles=prompt_handles, title="Prompt setting", loc="upper right", fontsize=16, title_fontsize=17)
leg2 = ax.legend(handles=model_handles, title="Model", loc="upper center", bbox_to_anchor=(0.6, 1.0), fontsize=16, title_fontsize=17)

ax.grid(True, axis="both", linestyle="--", alpha=0.6)

ax.add_artist(leg1)  # keep both

plt.tight_layout()
fig.savefig("combined_prompt_impact_selected_classes.pdf", dpi=200, bbox_inches="tight")
plt.show()

