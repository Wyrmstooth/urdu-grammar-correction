# -*- coding: utf-8 -*-
"""
Script 1: Data Preparation & Analysis
Loads GenAI-Dataset.json, creates train/val/test splits,
generates dataset statistics and visualizations.
"""
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from datasets import Dataset, DatasetDict

random.seed(42)
np.random.seed(42)

# ─── 1. Load dataset ───
print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

with open("GenAI-Dataset.json", "r", encoding="utf-8-sig") as f:
    data = json.load(f)

print(f"Total entries: {len(data)}")

# ─── 2. Convert to DataFrame for analysis ───
df = pd.DataFrame(data)
# Strip 'correct: ' prefix from input
df["input_raw"] = df["input"].str.replace("correct: ", "", regex=False)

# ─── 3. Dataset Statistics ───
print("\n" + "=" * 60)
print("DATASET STATISTICS")
print("=" * 60)

cat_counts = Counter(df["category"])
err_counts = Counter(df["error_type"])

print("\nCategory distribution:")
for cat, cnt in sorted(cat_counts.items()):
    print(f"  {cat:30s}: {cnt:5d} ({cnt/len(df)*100:.1f}%)")

print("\nError type distribution:")
for err, cnt in sorted(err_counts.items()):
    print(f"  {err:30s}: {cnt:5d} ({cnt/len(df)*100:.1f}%)")

# Sequence lengths
input_lens = df["input_raw"].str.len()
output_lens = df["output"].str.len()
print(f"\nInput length:  min={input_lens.min()}, max={input_lens.max()}, mean={input_lens.mean():.1f}")
print(f"Output length: min={output_lens.min()}, max={output_lens.max()}, mean={output_lens.mean():.1f}")

# ─── 4. Plot distributions ───
print("\nGenerating plots...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Category distribution
cats_sorted = sorted(cat_counts.items(), key=lambda x: x[1], reverse=True)
ax = axes[0, 0]
bars = ax.bar([c[0] for c in cats_sorted], [c[1] for c in cats_sorted], color="steelblue")
ax.set_title("Category Distribution", fontsize=14, fontweight="bold")
ax.set_ylabel("Count")
ax.tick_params(axis="x", rotation=45)
for bar, (_, cnt) in zip(bars, cats_sorted):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, str(cnt),
            ha="center", fontsize=8)

# Error type distribution
errs_sorted = sorted(err_counts.items(), key=lambda x: x[1], reverse=True)
ax = axes[0, 1]
colors = ["coral", "gold", "mediumseagreen", "skyblue", "orchid", "orange", "teal", "crimson", "gray"]
bars = ax.bar([e[0] for e in errs_sorted], [e[1] for e in errs_sorted],
              color=colors[:len(errs_sorted)])
ax.set_title("Error Type Distribution", fontsize=14, fontweight="bold")
ax.set_ylabel("Count")
ax.tick_params(axis="x", rotation=45)
for bar, (_, cnt) in zip(bars, errs_sorted):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, str(cnt),
            ha="center", fontsize=8)

# Input length histogram
ax = axes[1, 0]
ax.hist(input_lens, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
ax.set_title("Input Sentence Length Distribution", fontsize=14, fontweight="bold")
ax.set_xlabel("Characters")
ax.set_ylabel("Frequency")
ax.axvline(input_lens.mean(), color="red", linestyle="--", label=f"Mean: {input_lens.mean():.0f}")
ax.legend()

# Output length histogram
ax = axes[1, 1]
ax.hist(output_lens, bins=50, color="mediumseagreen", edgecolor="white", alpha=0.8)
ax.set_title("Output Sentence Length Distribution", fontsize=14, fontweight="bold")
ax.set_xlabel("Characters")
ax.set_ylabel("Frequency")
ax.axvline(output_lens.mean(), color="red", linestyle="--", label=f"Mean: {output_lens.mean():.0f}")
ax.legend()

plt.tight_layout()
plt.savefig("dataset_analysis.png", dpi=150, bbox_inches="tight")
print("Saved dataset_analysis.png")

# Category vs Error type heatmap
fig2, ax2 = plt.subplots(figsize=(12, 6))
pivot = df.pivot_table(index="category", columns="error_type", aggfunc="size", fill_value=0)
im = ax2.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
ax2.set_xticks(range(len(pivot.columns)))
ax2.set_xticklabels(pivot.columns, rotation=45, ha="right")
ax2.set_yticks(range(len(pivot.index)))
ax2.set_yticklabels(pivot.index)
ax2.set_title("Category vs Error Type Distribution", fontsize=14, fontweight="bold")
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        val = pivot.values[i, j]
        ax2.text(j, i, str(val) if val > 0 else "", ha="center", va="center", fontsize=7)
plt.colorbar(im, ax=ax2, shrink=0.8)
plt.tight_layout()
plt.savefig("category_error_heatmap.png", dpi=150, bbox_inches="tight")
print("Saved category_error_heatmap.png")

# ─── 5. Create train/val/test splits ───
print("\n" + "=" * 60)
print("CREATING SPLITS")
print("=" * 60)

# Stratified split by both category and error_type
indices = list(range(len(df)))
random.shuffle(indices)

# 80/10/10 split
n = len(df)
train_idx = indices[:int(n * 0.8)]
val_idx = indices[int(n * 0.8):int(n * 0.9)]
test_idx = indices[int(n * 0.9):]

train_df = df.iloc[train_idx].reset_index(drop=True)
val_df = df.iloc[val_idx].reset_index(drop=True)
test_df = df.iloc[test_idx].reset_index(drop=True)

print(f"Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
print(f"Val:   {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
print(f"Test:  {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")

# ─── 6. Save processed splits as CSV and HuggingFace Dataset ───
train_df[["input", "output"]].to_csv("train.csv", index=False, encoding="utf-8-sig")
val_df[["input", "output"]].to_csv("val.csv", index=False, encoding="utf-8-sig")
test_df[["input", "output"]].to_csv("test.csv", index=False, encoding="utf-8-sig")
print("Saved train.csv, val.csv, test.csv")

# HF Dataset
hf_dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df[["input", "output"]]),
    "validation": Dataset.from_pandas(val_df[["input", "output"]]),
    "test": Dataset.from_pandas(test_df[["input", "output"]]),
})
hf_dataset.save_to_disk("urdu_gec_dataset_hf")
print("Saved HuggingFace dataset to urdu_gec_dataset_hf/")

# Save a test set without prefix for evaluation
test_df["input_raw"] = test_df["input"].str.replace("correct: ", "", regex=False)
test_df[["input_raw", "output", "category", "error_type"]].to_csv(
    "test_eval.csv", index=False, encoding="utf-8-sig"
)

print("\nData preparation complete!")
print(f"Files created: dataset_analysis.png, category_error_heatmap.png, train.csv, val.csv, test.csv, test_eval.csv")
