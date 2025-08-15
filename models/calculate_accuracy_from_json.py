import json
import os
import argparse
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import re

correlation_map = {
    "Stealing": {"Shoplifting": 0.8, "Burglary": 0.5, "Robbery": 0.3},
    "Shoplifting": {"Stealing": 0.8, "Burglary": 0.5, "Robbery": 0.3},
    "Arson": {"Explosion": 0.7},
    "Explosion": {"Arson": 0.7},
    "Robbery": {"Burglary": 0.75, "Stealing": 0.6},
    "Burglary": {"Robbery": 0.75, "Shoplifting": 0.5},
    "Assault": {"Fighting": 0.7, "Abuse": 0.6},
    "Abuse": {"Assault": 0.6, "Fighting": 0.1},
    "Fighting": {"Assault": 0.8},
    "Arrest": {"Fighting": 0.2},
    "RoadAccidents": {"Normal": 0.1},
}

LABEL_UCF_CRIME = [
    "Abuse","Arrest","Arson","Assault","Burglary","Explosion",
    "Fighting","Normal","RoadAccidents","Robbery","Shooting",
    "Shoplifting","Stealing","Vandalism",
]
LABELS_RWF2000 = ["Fighting","Normal"]
LABELS_XD_Violence = [ "Abuse","Explosion","Fighting","Normal",
    "Riot","RoadAccidents","Shooting",
]

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

def set_prompt_and_labels(dataset_name: str):
    """Update global `prompt` and `labels` for chosen dataset."""
    global prompt, labels
    if dataset_name == "UCF_Crime":
        prompt = GUIDED_PROMPT_UCF
        labels = LABEL_UCF_CRIME
    elif dataset_name == "UCF_Crime_Privacy":
        prompt = GUIDED_PROMPT_UCF
        labels = LABEL_UCF_CRIME
    elif dataset_name == "RWF2000":
        prompt = GUIDED_PROMPT_RWF2000
        labels = LABELS_RWF2000
    elif dataset_name == "RWF2000_Depth":
        prompt = GUIDED_PROMPT_RWF2000_DEPTH
        labels = LABELS_RWF2000
    elif dataset_name == "XD_Violence":
        prompt = GUIDED_PROMPT_XD
        labels = LABELS_XD_Violence
    else:
        prompt = "Name any anomalies you see in the video."
        labels = ["Anomaly", "Normal"]



def calculate_accuracy_from_json(args):
    # json_filepath = args.json_filepath
    json_filepath = args.json_filepath.replace('\\', '/')
    json_filepath = os.path.abspath(os.path.normpath(json_filepath))
    if not os.path.exists(json_filepath):
        print(f"File not found: {json_filepath}")
        return
    
    with open(json_filepath, 'r', encoding='utf-8') as f:
        eval_dict = json.load(f)

    labels = sorted({ entry["video_label"] for entry in eval_dict.values() })
    
    correct_per_class = {}
    incorrect_per_class = {}
    adjusted_correct_per_class = {}
    correct_per_class_top_3 = {}
    videos_per_class = {}
    for label in labels:
        videos_per_class[label] = 0

    y_true, y_scores = [], []

    normal_videos = 0
    normal_videos_flagged = 0
    normal_wrong_percentages = []
    anomalous_videos = 0
    anomalous_anomaly_percentages = []

    prompt_tokens  = set(re.findall(r"\w+", prompt.lower()))    # tokens in the prompt
    label_tokens   = [lbl.lower() for lbl in labels]           # assumes you pass labels list

    resp_cnt               = 0          # total number of LLM answers
    tot_len_tok            = 0          # sum of token counts (for avg length)
    prompt_overlap_sum     = 0.0        # accumulates prompt–answer token-overlap ratio
    no_label_cnt           = 0          # answers with zero class–label mention
    multi_label_cnt        = 0          # answers with >1 class label
    correct_label_cnt      = 0          # answers mentioning the ground-truth label
    incorrect_label_cnt    = 0          # answers mentioning any *other* label

    pbar = tqdm(eval_dict.values())
    
    for entry in pbar:
        actual_label = entry["video_label"]
        top_1_score = entry["top_1_score"]
        video_prediction = entry["video_prediction"]
        
        if actual_label not in correct_per_class:
            correct_per_class[actual_label] = 0
            incorrect_per_class[actual_label] = 0
            adjusted_correct_per_class[actual_label] = 0.0
            correct_per_class_top_3[actual_label] = 0
        
        videos_per_class[actual_label] += 1
        
        top_1_adjusted_score = top_1_score
        top_3_score = 0.0
        anomaly_scores = []
        for prediction in video_prediction.values():
            LLM_output = prediction["LLM_output"]
            contains_label = any(label.lower() in LLM_output.lower() for label in labels)
            resp_cnt += 1

            tokens        = re.findall(r"\w+", LLM_output)
            tot_len_tok  += len(tokens)

            # --- Prompt-reuse score (token overlap) -------------------------------------
            overlap_ratio  = len(set(t.lower() for t in tokens) & prompt_tokens) / \
                            max(1, len(prompt_tokens))
            prompt_overlap_sum += overlap_ratio    # later we’ll average over resp_cnt

            # --- Label analysis ---------------------------------------------------------
            found_labels = [lbl for lbl in label_tokens if re.search(rf"\b{re.escape(lbl)}\b",
                                                                    LLM_output, re.I)]

            if not found_labels:
                no_label_cnt += 1
            if len(found_labels) > 1:
                multi_label_cnt += 1
            # correct if it mentions the *true* class
            if actual_label.lower() in found_labels:
                correct_label_cnt += 1
            # incorrect if it mentions any other class (even when it also has the true class)
            if any(lbl != actual_label.lower() for lbl in found_labels):
                incorrect_label_cnt += 1    


            predicted_labels = prediction["LLM_classes"]
            top_1_pred = predicted_labels[0] if predicted_labels else None
            
            if actual_label in predicted_labels:
                top_3_score = 1.0

            if top_1_score != 1.0:
                if top_1_pred == actual_label:
                    top_1_score = 1.0
                    top_1_adjusted_score = 1.0
            
            if top_1_pred:
                correlation = correlation_map.get(actual_label, {}).get(top_1_pred, 0.0)
                top_1_adjusted_score = max(top_1_adjusted_score, correlation)

            # 1.0 for anomalous, 0.0 for Normal
            anomaly_scores.append(1.0 if top_1_pred != "Normal" else 0.0)

        correct_per_class[actual_label] += top_1_score
        incorrect_per_class[actual_label] += (1 - top_1_score)

        # y_score /= len(video_prediction.values())
        y_score = anomaly_scores
        
        adjusted_correct_per_class[actual_label] += top_1_adjusted_score
        correct_per_class_top_3[actual_label] += top_3_score

        # VAD AUC Calculation - Treat "Normal" as non-anomalous, others as anomalous
        # y_true.append(0 if actual_label == "Normal" else 1)
        anomaly_label = 0 if actual_label == "Normal" else 1
        y_true.extend([anomaly_label for _ in range(len(y_score))])
        # y_scores.append(y_score)  # Use 1 - top_1_score as an anomaly score
        y_scores.extend(y_score)

        pbar.set_description(f"y: {y_score}, 1: {top_1_score}, 1a: {top_1_adjusted_score}")

        total_snippets = len(anomaly_scores)
        anomaly_cnt    = sum(anomaly_scores)
        pct_anomaly    = (anomaly_cnt / total_snippets) if total_snippets else 0.0

        if actual_label == "Normal":
            normal_videos += 1
            if anomaly_cnt > 0:
                normal_videos_flagged += 1          # any false-positive at all
            normal_wrong_percentages.append(pct_anomaly)
        else:
            anomalous_videos += 1
            anomalous_anomaly_percentages.append(pct_anomaly)

    auc_score = roc_auc_score(y_true, y_scores) if len(set(y_true)) > 1 else 0.0
    auc_score *= 100

    # # ← ADD THIS BLOCK to plot+save ROC curve
    # fpr, tpr, _ = roc_curve(y_true, y_scores)
    # plt.figure()
    # plt.plot(fpr, tpr, label=f"AUC = {auc_score/100:.4f}")
    # plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.title("ROC Curve")
    # plt.legend(loc="lower right")

    # out_dir = os.path.dirname(json_filepath)
    # base_name = os.path.splitext(os.path.basename(json_filepath))[0]
    # save_path = os.path.join(out_dir, f"{base_name}_roc_curve.png")
    # plt.savefig(save_path, dpi=150)
    # plt.close()
    # print(f"ROC curve saved to: {save_path}")

    per_class_acc = {}
    per_class_adj = {}
    per_class_top3 = {}

    for cls in labels:
        print(f"Cls: {cls} {correct_per_class.get(cls, 0)}")
        per_class_acc[cls] = correct_per_class.get(cls, 0) / max(1, videos_per_class.get(cls, 0))
        per_class_adj[cls] = adjusted_correct_per_class.get(cls, 0) / max(1, videos_per_class.get(cls, 0))
        per_class_top3[cls] = correct_per_class_top_3.get(cls, 0) / max(1, videos_per_class.get(cls, 0))


    # Macro (balanced) accuracies
    macro_top1 = np.mean(list(per_class_acc.values())) * 100
    macro_top3 = np.mean(list(per_class_top3.values())) * 100
    macro_adj  = np.mean(list(per_class_adj.values())) * 100

    print(f"\nMacro Top-1 Accuracy (avg over classes): {macro_top1:.2f}%")
    print(f"Macro Top-3 Accuracy:                      {macro_top3:.2f}%")
    print(f"Macro Adjusted Accuracy:                   {macro_adj:.2f}%")
    print(f"Micro VAD AUC Score:                       {auc_score:.4f}\n")

    # ---------- 3)  PRINT ROBUSTNESS METRICS ---------------------------------
    normal_flagged_pct = (100 * normal_videos_flagged / normal_videos) \
                         if normal_videos else 0.0
    avg_wrong_normal   = (100 * np.mean(normal_wrong_percentages)) \
                         if normal_wrong_percentages else 0.0
    avg_detect_anom    = (100 * np.mean(anomalous_anomaly_percentages)) \
                         if anomalous_anomaly_percentages else 0.0

    print("\n--- Robustness ---")
    print(f"Normal videos wrongly flagged (count): {normal_videos_flagged}/{normal_videos}")
    print(f"Normal videos wrongly flagged (rate) : {normal_flagged_pct:.2f}%")
    print(f"Avg % of snippets predicted 'non-Normal' in Normal videos     : {avg_wrong_normal:.2f}%")
    print(f"Avg % of snippets predicted 'non-Normal' in Anomalous videos : {avg_detect_anom:.2f}%")
    # ------------------------------------------------------------------------

    # ---------- 4)  PRINT Text statistics ---------------------------------
    avg_resp_len        = tot_len_tok / max(1, resp_cnt)
    avg_prompt_overlap  = prompt_overlap_sum / max(1, resp_cnt) * 100     # %
    pct_no_label        = no_label_cnt    / max(1, resp_cnt) * 100
    pct_multi_label     = multi_label_cnt / max(1, resp_cnt) * 100
    pct_correct_label   = correct_label_cnt   / max(1, resp_cnt) * 100
    pct_incorrect_label = incorrect_label_cnt / max(1, resp_cnt) * 100

    print("\n--- Extra LLM-response statistics --------------------------------")
    print(f"Average response length (tokens):           {avg_resp_len:.1f}")
    print(f"Avg % prompt words repeated:                {avg_prompt_overlap:.2f}%")
    print(f"% responses with NO class label:            {pct_no_label:.2f}%")
    print(f"% responses with >1 class label:            {pct_multi_label:.2f}%")
    print(f"% responses containing CORRECT class label: {pct_correct_label:.2f}%")
    print(f"% responses containing INCORRECT label(s):  {pct_incorrect_label:.2f}%")


    # Convert dict_values to lists first
    correct_array = np.array(list(correct_per_class.values()))
    incorrect_array = np.array(list(incorrect_per_class.values()))

    # Compute accuracy per class
    accuracy = correct_array / (correct_array + incorrect_array)
    
    print("Per-Class Accuracy:")
    print(per_class_acc)
    print("\nPer-Class Top-3 Predictions:")
    print(correct_per_class_top_3)
    print("\nPer-Class Adjusted Correct Predictions:")
    print(per_class_adj)

    # --- Save output to file ---
    output_lines = []
    output_lines.append(f"\nMacro Top-1 Accuracy (avg over classes): {macro_top1:.2f}%")
    output_lines.append(f"Macro Top-3 Accuracy:                      {macro_top3:.2f}%")
    output_lines.append(f"Macro Adjusted Accuracy:                   {macro_adj:.2f}%")
    output_lines.append(f"Micro VAD AUC Score:                       {auc_score:.4f}\n")

    output_lines.append("\n--- Robustness ---")
    output_lines.append(f"Normal videos wrongly flagged (count): {normal_videos_flagged}/{normal_videos}")
    output_lines.append(f"Normal videos wrongly flagged (rate) : {normal_flagged_pct:.2f}%")
    output_lines.append(f"Avg % of snippets predicted 'non-Normal' in Normal videos     : {avg_wrong_normal:.2f}%")
    output_lines.append(f"Avg % of snippets predicted 'non-Normal' in Anomalous videos : {avg_detect_anom:.2f}%")

    output_lines.append("\nPer-Class Accuracy:")
    for cls in labels:
        output_lines.append(f"{cls:15}: {per_class_acc[cls]:.2f}")
    
    output_lines.append("\nPer-Class Top-3 Predictions:")
    for cls in labels:
        output_lines.append(f"{cls:15}: {per_class_top3[cls]:.2f}")
    
    output_lines.append("\nPer-Class Adjusted Correct Predictions:")
    for cls in labels:
        output_lines.append(f"{cls:15}: {per_class_adj[cls]:.2f}")

    # --- Append the extra LLM-response statistics to the report ---------------
    output_lines += [
        "\n--- Extra LLM-response statistics --------------------------------",
        f"Average response length (tokens):           {avg_resp_len:.1f}",
        f"Avg % prompt words repeated:                {avg_prompt_overlap:.2f}%",
        f"% responses with NO class label:            {pct_no_label:.2f}%",
        f"% responses with >1 class label:            {pct_multi_label:.2f}%",
        f"% responses containing CORRECT class label: {pct_correct_label:.2f}%",
        f"% responses containing INCORRECT label(s):  {pct_incorrect_label:.2f}%",
    ]


    out_dir = os.path.dirname(json_filepath)
    base_name = os.path.splitext(os.path.basename(json_filepath))[0]
    results_txt = os.path.join(out_dir, f"{base_name}_evaluation_results.txt")
    with open(results_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    print(f"\nResults saved to: {results_txt}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate accuracy from a JSON file.")
    parser.add_argument("-j", "--json_filepath", type=str, help="Path to the JSON file containing evaluation results.")
    parser.add_argument("-r", "--recalculate", action='store_true', help="recalculate the class predictions from LLM texts")
    args = parser.parse_args()

    if "RWF2000" in args.json_filepath:
        set_prompt_and_labels("RWF2000")
    elif "UCF_Crime" in args.json_filepath:
        set_prompt_and_labels("UCF_Crime")
    elif "XD_Violence" in args.json_filepath:
        set_prompt_and_labels("XD_Violence")

    calculate_accuracy_from_json(args)
