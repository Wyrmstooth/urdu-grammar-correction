# -*- coding: utf-8 -*-
"""
Script 3: Inference, Evaluation & Visualization
Loads trained model, runs comprehensive evaluation with BLEU scores,
generates per-category and per-error-type analysis.
"""
import json
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from collections import defaultdict, Counter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sacrebleu
import warnings
warnings.filterwarnings("ignore")

# Fix Windows console encoding for Urdu
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

print("=" * 60)
print("URDU GRAMMAR CORRECTION - EVALUATION")
print("=" * 60)

# ─── 1. Load Model ───
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

MODEL_PATH = "./urdu_gec_model/final"
if not __import__("os").path.exists(MODEL_PATH):
    # Try best checkpoint
    MODEL_PATH = "./urdu_gec_model/checkpoint-best"
    if not __import__("os").path.exists(MODEL_PATH):
        print("ERROR: No model found! Run 2_training.py first.")
        exit(1)

print(f"Loading model from: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(device)
model.eval()

# ─── 2. Correction Function ───
PREFIX = "correct grammar: "

def correct_urdu(text, num_beams=4):
    """Correct an Urdu sentence"""
    input_text = PREFIX + text if not text.startswith("correct:") else text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=256, truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_length=256, num_beams=num_beams,
            early_stopping=True, no_repeat_ngram_size=3
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ─── 3. Load Test Data ───
print("\n" + "=" * 60)
print("LOADING TEST DATA")
print("=" * 60)

with open("GenAI-Dataset.json", "r", encoding="utf-8-sig") as f:
    all_data = json.load(f)

# Use last 10% as test set
n_test = len(all_data) // 10
test_data = all_data[-n_test:]

# Also stratify - pick samples evenly across error types
error_samples = defaultdict(list)
for d in all_data[:-n_test]:
    error_samples[d.get("error_type", "unknown")].append(d)

stratified_test = []
for etype, samples in error_samples.items():
    n = max(5, min(25, len(samples) // 20))
    stratified_test.extend(samples[-n:])

print(f"Full test set: {len(test_data)} samples")
print(f"Stratified test: {len(stratified_test)} samples")

# ─── 4. Quick Inference Test ───
print("\n" + "=" * 60)
print("INFERENCE SAMPLES")
print("=" * 60)

test_cases = [
    ("میں کل بازار گیا تھا سبزیاں خریدنے کے لیے", "word_order"),
    ("وہ لڑکی بہت اچھا گاتی ہیں", "grammar"),
    ("اس نے مجے فون کیا", "spelling"),
    ("بچے باہر کھیل رہے ہے", "grammar"),
    ("وہ گھر گیا اور اور پھر واپس آیا", "extra_words"),
    ("میں نے بازار سبزی خریدی", "missing_words"),
    ("کل رات خوب بارس ہوئی", "spelling"),
    ("آج کا کھانا بہت لذیذ ہوئی", "grammar"),
]

for text, etype in test_cases:
    result = correct_urdu(text)
    print(f"\n[{etype}]")
    print(f"  IN:  {text}")
    print(f"  OUT: {result}")

# ─── 5. Evaluate on stratified test ───
print("\n" + "=" * 60)
print("BLEU EVALUATION (Stratified)")
print("=" * 60)

predictions = []
references = []
per_error_predictions = defaultdict(list)
per_error_references = defaultdict(list)
per_category_predictions = defaultdict(list)
per_category_references = defaultdict(list)

print(f"Evaluating {len(stratified_test)} samples...")
for i, d in enumerate(stratified_test):
    inp = d["input"].replace("correct: ", "")
    ref = d["output"]
    pred = correct_urdu(inp)

    predictions.append(pred)
    references.append(ref)

    etype = d.get("error_type", "unknown")
    cat = d.get("category", "unknown")
    per_error_predictions[etype].append(pred)
    per_error_references[etype].append(ref)
    per_category_predictions[cat].append(pred)
    per_category_references[cat].append(ref)

    if (i + 1) % 50 == 0:
        print(f"  Processed {i+1}/{len(stratified_test)}")

# ─── 6. Compute BLEU Scores ───
print("\n" + "=" * 60)
print("BLEU SCORES")
print("=" * 60)

bleu_overall = sacrebleu.corpus_bleu(predictions, [references])
print(f"\nOverall BLEU: {bleu_overall.score:.2f}")

print("\nPer Error Type:")
error_type_bleu = {}
for etype in sorted(per_error_predictions.keys()):
    if len(per_error_predictions[etype]) >= 3:
        bleu = sacrebleu.corpus_bleu(
            per_error_predictions[etype],
            [per_error_references[etype]]
        )
        error_type_bleu[etype] = bleu.score
        print(f"  {etype:20s}: {bleu.score:.2f} ({len(per_error_predictions[etype])} samples)")

print("\nPer Category:")
category_bleu = {}
for cat in sorted(per_category_predictions.keys()):
    if len(per_category_predictions[cat]) >= 3:
        bleu = sacrebleu.corpus_bleu(
            per_category_predictions[cat],
            [per_category_references[cat]]
        )
        category_bleu[cat] = bleu.score
        print(f"  {cat:25s}: {bleu.score:.2f} ({len(per_category_predictions[cat])} samples)")

# ─── 7. Evaluate on full test set ───
print("\n" + "=" * 60)
print("BLEU EVALUATION (Full Test Set)")
print("=" * 60)

full_predictions = []
full_references = []

# Take a subset for speed (up to 500)
eval_subset = test_data[:min(500, len(test_data))]
print(f"Evaluating full test subset: {len(eval_subset)} samples...")

for i, d in enumerate(eval_subset):
    inp = d["input"].replace("correct: ", "")
    ref = d["output"]
    pred = correct_urdu(inp)
    full_predictions.append(pred)
    full_references.append(ref)
    if (i + 1) % 100 == 0:
        print(f"  Processed {i+1}/{len(eval_subset)}")

bleu_full = sacrebleu.corpus_bleu(full_predictions, [full_references])
print(f"\nFull test BLEU: {bleu_full.score:.2f}")

# ─── 8. Visualization ───
print("\n" + "=" * 60)
print("GENERATING EVALUATION PLOTS")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Error type BLEU
ax = axes[0, 0]
types_sorted = sorted(error_type_bleu.items(), key=lambda x: x[1], reverse=True)
if types_sorted:
    bars = ax.bar([t[0] for t in types_sorted], [t[1] for t in types_sorted], color="coral")
    ax.set_title("BLEU Score by Error Type", fontsize=14, fontweight="bold")
    ax.set_ylabel("BLEU Score")
    ax.tick_params(axis="x", rotation=45)
    for bar, (_, val) in zip(bars, types_sorted):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f"{val:.1f}",
                ha="center", fontsize=9)
    ax.set_ylim(0, max(types_sorted[0][1] + 15, 100))

# Category BLEU
ax = axes[0, 1]
cats_sorted = sorted(category_bleu.items(), key=lambda x: x[1], reverse=True)
if cats_sorted:
    bars = ax.bar([t[0] for t in cats_sorted], [t[1] for t in cats_sorted], color="steelblue")
    ax.set_title("BLEU Score by Category", fontsize=14, fontweight="bold")
    ax.set_ylabel("BLEU Score")
    ax.tick_params(axis="x", rotation=45)
    for bar, (_, val) in zip(bars, cats_sorted):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f"{val:.1f}",
                ha="center", fontsize=9)
    ax.set_ylim(0, max(cats_sorted[0][1] + 15, 100))

# Summary bar chart - overall BLEU
ax = axes[1, 0]
metrics = {"Overall": bleu_overall.score, "Full Test": bleu_full.score}
bars = ax.bar(metrics.keys(), metrics.values(), color=["mediumseagreen", "skyblue"])
ax.set_title("Overall BLEU Scores", fontsize=14, fontweight="bold")
ax.set_ylabel("BLEU Score")
for bar, val in zip(bars, metrics.values()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{val:.1f}",
            ha="center", fontsize=12, fontweight="bold")
ax.set_ylim(0, max(metrics.values()) + 15)

# Sample counts
ax = axes[1, 1]
sample_counts = {k: len(v) for k, v in per_error_predictions.items()}
samples_sorted = sorted(sample_counts.items(), key=lambda x: x[1], reverse=True)
bars = ax.bar([s[0] for s in samples_sorted], [s[1] for s in samples_sorted], color="orchid")
ax.set_title("Evaluation Samples per Error Type", fontsize=14, fontweight="bold")
ax.set_ylabel("Count")
ax.tick_params(axis="x", rotation=45)
for bar, (_, val) in zip(bars, samples_sorted):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, str(val),
            ha="center", fontsize=9)

plt.tight_layout()
plt.savefig("evaluation_results.png", dpi=150, bbox_inches="tight")
print("Saved evaluation_results.png")

# ─── 9. Save Results ───
results = {
    "overall_bleu_stratified": round(bleu_overall.score, 2),
    "overall_bleu_full_test": round(bleu_full.score, 2),
    "per_error_type_bleu": {k: round(v, 2) for k, v in error_type_bleu.items()},
    "per_category_bleu": {k: round(v, 2) for k, v in category_bleu.items()},
    "num_params": sum(p.numel() for p in model.parameters()),
    "device": str(device),
}

with open("evaluation_metrics.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("Saved evaluation_metrics.json")

# ─── 10. Interactive Demo ───
print("\n" + "=" * 60)
print("DEMO - Try your own sentences")
print("=" * 60)

demo_sentences = [
    "میں نے کھانا کھایا اور پھر سو گیا جلدی",
    "کل رات بجلی چلی گئ",
    "بچوں نے شور مچایا بہت کمرے میں",
    "وہ ہر روز صبح جوگنگ کرتہ ہے",
    "اس نے مجھ کو کچھ نہیں بتایا",
]

for sent in demo_sentences:
    corrected = correct_urdu(sent)
    print(f"\n  IN:  {sent}")
    print(f"  OUT: {corrected}")

print("\n" + "=" * 60)
print("EVALUATION COMPLETE!")
print("=" * 60)
print(f"Files created: evaluation_results.png, evaluation_metrics.json")
