"""
Sentiment Analysis Pipeline
============================
Model : cardiffnlp/twitter-roberta-base-sentiment-latest (RoBERTa fine-tuned on ~124M tweets)
Dataset : tweet_eval -- sentiment subset (negative, neutral, positive)
Output : Strict JSON in two modes -> label_only | reasoning
"""

import os
import sys
import json
import random
from collections import Counter
from typing import List
from multiprocessing import freeze_support

import numpy as np
import torch
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
SAMPLE_SIZE = 200
RANDOM_SEED = 42
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
LABEL_MAP = ID2LABEL.copy()


# -----------------------------------------------------------------------
# Validator
# -----------------------------------------------------------------------
def validate_output(obj: dict, mode: str) -> None:
    """Raise ValueError if obj doesn't match the required schema."""
    required = {"label", "confidence", "reliability"}
    if mode == "reasoning":
        required.add("rationale")

    missing = required - set(obj.keys())
    if missing:
        raise ValueError(f"Missing keys: {missing}")

    if obj["label"] not in ("positive", "negative", "neutral"):
        raise ValueError(f"Invalid label: {obj['label']!r}")

    for field in ("confidence", "reliability"):
        v = obj[field]
        if not isinstance(v, (int, float)) or not (0.0 <= v <= 1.0):
            raise ValueError(f"{field} must be float in [0,1], got {v!r}")

    if mode == "reasoning":
        if not isinstance(obj["rationale"], str) or len(obj["rationale"]) < 5:
            raise ValueError("rationale must be a non-trivial string")

    json.loads(json.dumps(obj))  # round-trip check


# -----------------------------------------------------------------------
# Classifier
# -----------------------------------------------------------------------
def classify_text(text: str, mode: str = "label_only",
                  tokenizer=None, model_obj=None) -> dict:
    """
    Classify text into positive / negative / neutral.

    Parameters
    ----------
    text : str   - input text
    mode : str   - 'label_only' or 'reasoning'
    tokenizer    - HuggingFace tokenizer
    model_obj    - HuggingFace model

    Returns
    -------
    dict matching the required JSON schema.
    """
    if mode not in ("label_only", "reasoning"):
        raise ValueError(f"Invalid mode: {mode!r}")

    # Model inference
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model_obj(**inputs).logits[0].numpy()
    probs = softmax(logits)

    pred_id = int(np.argmax(probs))
    label = ID2LABEL[pred_id]
    confidence = round(float(probs[pred_id]), 2)

    # Reliability -- based on margin between top-2 probabilities
    sorted_probs = sorted(probs, reverse=True)
    margin = float(sorted_probs[0] - sorted_probs[1])
    reliability = round(min(0.5 + margin, 0.99), 2)
    if len(text.split()) < 3:
        reliability = round(reliability * 0.7, 2)

    # Build output
    result = {
        "label": label,
        "confidence": confidence,
        "reliability": reliability,
    }

    if mode == "reasoning":
        prob_str = ", ".join(f"{ID2LABEL[i]}: {probs[i]:.1%}" for i in range(3))
        parts = [f"Model probabilities -- {prob_str}."]
        if margin < 0.2:
            parts.append("Low margin between top classes indicates ambiguity.")
        if len(text.split()) < 3:
            parts.append("Very short text reduces reliability.")
        result["rationale"] = " ".join(parts[:3])

    validate_output(result, mode)
    return result


# -----------------------------------------------------------------------
# Dataset Runner
# -----------------------------------------------------------------------
def run_on_dataset(samples, mode: str, tokenizer=None, model_obj=None) -> List[dict]:
    """Run classify_text on each sample, return list of results."""
    results = []
    total = len(samples)
    for i, row in enumerate(samples):
        text = row["text"]
        gt = LABEL_MAP[row["label"]]
        pred = classify_text(text, mode=mode, tokenizer=tokenizer, model_obj=model_obj)
        results.append({"text": text, "ground_truth": gt, "prediction": pred})
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{total}] done")
    return results


# -----------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------
def compute_metrics(results: List[dict]) -> dict:
    """Compute accuracy, macro-F1, per-class precision/recall/F1."""
    labels = ["negative", "neutral", "positive"]
    total = len(results)
    correct = sum(1 for r in results if r["prediction"]["label"] == r["ground_truth"])

    per_class = {}
    for lb in labels:
        tp = sum(1 for r in results if r["prediction"]["label"] == lb and r["ground_truth"] == lb)
        fp = sum(1 for r in results if r["prediction"]["label"] == lb and r["ground_truth"] != lb)
        fn = sum(1 for r in results if r["prediction"]["label"] != lb and r["ground_truth"] == lb)
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r_ = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r_ / (p + r_) if (p + r_) else 0.0
        per_class[lb] = {"precision": round(p, 3), "recall": round(r_, 3), "f1": round(f1, 3)}

    macro_f1 = round(float(np.mean([v["f1"] for v in per_class.values()])), 3)

    return {
        "accuracy": round(correct / total, 3),
        "macro_f1": macro_f1,
        "correct": correct,
        "total": total,
        "per_class": per_class,
    }


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    # Fix Windows console encoding
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load model
    print(f"Loading model: {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model_obj = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model_obj.eval()
    print("Model loaded.\n")

    # Load dataset
    print("Loading TweetEval sentiment dataset ...")
    dataset = load_dataset("tweet_eval", "sentiment")
    print(f"  train : {len(dataset['train']):,}")
    print(f"  val   : {len(dataset['validation']):,}")
    print(f"  test  : {len(dataset['test']):,}")

    # Sample
    random.seed(RANDOM_SEED)
    idx = random.sample(range(len(dataset["test"])), SAMPLE_SIZE)
    samples = dataset["test"].select(idx)

    dist = Counter(LABEL_MAP[lb] for lb in samples["label"])
    print(f"\nSampled {SAMPLE_SIZE} from test split:")
    for lbl, cnt in sorted(dist.items()):
        print(f"  {lbl:<10}: {cnt}")

    # Run label_only mode
    print("\n== Running label_only mode ==")
    results_lo = run_on_dataset(samples, "label_only", tokenizer=tokenizer, model_obj=model_obj)

    # Run reasoning mode
    print("\n== Running reasoning mode ==")
    results_r = run_on_dataset(samples, "reasoning", tokenizer=tokenizer, model_obj=model_obj)

    # Export predictions
    export_lo = [{"text": r["text"], "prediction": r["prediction"]} for r in results_lo]
    export_r = [{"text": r["text"], "prediction": r["prediction"]} for r in results_r]

    path_lo = os.path.join(OUTPUT_DIR, "predictions_label_only.json")
    path_r = os.path.join(OUTPUT_DIR, "predictions_reasoning.json")

    with open(path_lo, "w", encoding="utf-8") as f:
        json.dump(export_lo, f, indent=2, ensure_ascii=False)
    with open(path_r, "w", encoding="utf-8") as f:
        json.dump(export_r, f, indent=2, ensure_ascii=False)

    print(f"\nExported {len(export_lo)} predictions:")
    print(f"  {path_lo}")
    print(f"  {path_r}")

    # Metrics
    metrics = compute_metrics(results_lo)

    print(f"\n== Evaluation ==")
    print(f"Overall Accuracy : {metrics['accuracy']:.1%}  ({metrics['correct']}/{metrics['total']})")
    print(f"Macro F1         : {metrics['macro_f1']:.3f}\n")
    print(f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 45)
    for cls, m in metrics["per_class"].items():
        print(f"{cls:<12} {m['precision']:>10.3f} {m['recall']:>10.3f} {m['f1']:>10.3f}")

    path_m = os.path.join(OUTPUT_DIR, "metrics.json")
    with open(path_m, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics exported: {path_m}")

    # Confusion matrix
    labels = ["negative", "neutral", "positive"]
    matrix = {gt: {pr: 0 for pr in labels} for gt in labels}
    for r in results_lo:
        matrix[r["ground_truth"]][r["prediction"]["label"]] += 1

    print(f"\n== Confusion Matrix (rows=true, cols=predicted) ==\n")
    print(f"{'':>12}", end="")
    for lb in labels:
        print(f"{lb:>12}", end="")
    print()
    print("-" * 48)
    for gt in labels:
        print(f"{gt:>12}", end="")
        for pr in labels:
            print(f"{matrix[gt][pr]:>12}", end="")
        print()

    errors = [r for r in results_lo if r["prediction"]["label"] != r["ground_truth"]]
    print(f"\nMisclassified: {len(errors)}/{len(results_lo)}")

    # Show sample outputs
    print("\n== Sample outputs (reasoning mode, first 3) ==")
    print(json.dumps(export_r[:3], indent=2, ensure_ascii=False))

    print("\nDone.")


if __name__ == "__main__":
    freeze_support()
    main()
