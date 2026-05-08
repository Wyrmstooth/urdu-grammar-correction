# -*- coding: utf-8 -*-
"""
Script 2: Fine-tune mT5-small on Urdu Grammar Correction Dataset
Optimized for NVIDIA RTX 5060 (8GB VRAM)
Uses fp16, gradient checkpointing, and efficient data loading.
"""
import json
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("URDU GRAMMAR CORRECTION - TRAINING")
print("=" * 60)

# ─── 1. GPU Setup ───
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    # Clear cache
    torch.cuda.empty_cache()

# ─── 2. Load Dataset ───
print("\n" + "=" * 60)
print("LOADING DATASET")
print("=" * 60)

with open("GenAI-Dataset.json", "r", encoding="utf-8-sig") as f:
    data = json.load(f)

# Convert to simple input/output format
records = [{"input": d["input"], "output": d["output"]} for d in data]
hf_dataset = Dataset.from_list(records)

# 80/10/10 split
split = hf_dataset.train_test_split(test_size=0.2, seed=42)
test_val = split["test"].train_test_split(test_size=0.5, seed=42)

dataset = DatasetDict({
    "train": split["train"],
    "validation": test_val["train"],
    "test": test_val["test"],
})
print(f"Train: {len(dataset['train'])}, Val: {len(dataset['validation'])}, Test: {len(dataset['test'])}")

# ─── 3. Load Model & Tokenizer ───
print("\n" + "=" * 60)
print("LOADING mT5-small")
print("=" * 60)

MODEL_NAME = "google/mt5-small"
PREFIX = "correct grammar: "
MAX_INPUT_LENGTH = 256
MAX_TARGET_LENGTH = 256

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# ─── 4. Tokenize Dataset ───
print("\n" + "=" * 60)
print("TOKENIZING")
print("=" * 60)

def preprocess(examples):
    # Strip "correct: " if present, then add our prefix
    inputs = [PREFIX + text.replace("correct: ", "", 1) for text in examples["input"]]
    targets = examples["output"]

    model_inputs = tokenizer(
        inputs, max_length=MAX_INPUT_LENGTH, truncation=True, padding=False
    )
    labels = tokenizer(
        targets, max_length=MAX_TARGET_LENGTH, truncation=True, padding=False
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized = dataset.map(preprocess, batched=True, remove_columns=["input", "output"])
print(f"Tokenized train size: {len(tokenized['train'])}")

# ─── 5. Training Arguments ───
print("\n" + "=" * 60)
print("TRAINING CONFIGURATION")
print("=" * 60)

output_dir = r"C:\Users\umer_\OneDrive\Desktop\genaiproj\checkpoint_dir"

training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=400,
    save_strategy="no",
    save_total_limit=0,
    load_best_model_at_end=False,
    predict_with_generate=True,
    generation_max_length=MAX_TARGET_LENGTH,
    bf16=True,
    fp16=False,
    gradient_checkpointing=False,
    max_grad_norm=1.0,
    report_to="none",
    dataloader_num_workers=0,
    dataloader_pin_memory=False,
)

print(f"Epochs: {training_args.num_train_epochs}")
print(f"Batch size (per device): {training_args.per_device_train_batch_size}")
print(f"Gradient accumulation: {training_args.gradient_accumulation_steps}")
print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"Learning rate: {training_args.learning_rate}")
print(f"FP16: {training_args.fp16}")
print(f"Total steps: ~{len(tokenized['train']) // (8 * 2) * training_args.num_train_epochs}")

# ─── 6. Data Collator ───
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    pad_to_multiple_of=8,
)

# ─── 7. Trainer ───
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    data_collator=data_collator,
)

# ─── 8. Train ───
print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)

start_time = time.time()
train_result = trainer.train()
train_time = time.time() - start_time

print(f"\nTraining completed in {train_time/60:.1f} minutes")
print(f"Best eval loss: {train_result.training_loss:.4f}")

# ─── 9. Save Model ───
print("\n" + "=" * 60)
print("SAVING MODEL")
print("=" * 60)

save_path = r"C:\Users\umer_\OneDrive\Desktop\genaiproj\urdu_gec_model\final"
os.makedirs(save_path, exist_ok=True)
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"Model saved to {save_path}")

# ─── 10. Training Plots ───
print("\n" + "=" * 60)
print("GENERATING TRAINING PLOTS")
print("=" * 60)

log_history = trainer.state.log_history

# Extract training and eval loss
train_losses = []
eval_losses = []
train_steps = []
eval_steps = []

for entry in log_history:
    if "loss" in entry and "eval_loss" not in entry:
        train_losses.append(entry["loss"])
        train_steps.append(entry["step"])
    if "eval_loss" in entry:
        eval_losses.append(entry["eval_loss"])
        eval_steps.append(entry["step"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Loss curve
ax1.plot(train_steps, train_losses, "b-", alpha=0.6, label="Train Loss", linewidth=1)
if eval_losses:
    ax1.plot(eval_steps, eval_losses, "r-o", label="Eval Loss", markersize=4, linewidth=2)
ax1.set_xlabel("Steps")
ax1.set_ylabel("Loss")
ax1.set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Smoothed loss
if len(train_losses) > 10:
    window = max(5, len(train_losses) // 20)
    smoothed = np.convolve(train_losses, np.ones(window)/window, mode="valid")
    ax2.plot(range(len(smoothed)), smoothed, "b-", linewidth=2)
ax2.set_xlabel("Steps")
ax2.set_ylabel("Smoothed Loss")
ax2.set_title(f"Smoothed Training Loss (window={window})", fontsize=14, fontweight="bold")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150, bbox_inches="tight")
print("Saved training_curves.png")

# ─── 11. Quick Evaluation ───
print("\n" + "=" * 60)
print("QUICK EVALUATION SAMPLE")
print("=" * 60)

test_samples = [
    "correct: \u0645\u06cc\u06ba \u06a9\u0644 \u0628\u0627\u0632\u0627\u0631 \u06af\u06cc\u0627 \u062a\u06be\u0627 \u0633\u0628\u0632\u06cc\u0627\u06ba \u062e\u0631\u06cc\u062f\u0646\u06d2 \u06a9\u06d2 \u0644\u06cc\u06d2",
    "correct: \u0648\u06c1 \u0644\u0691\u06a9\u06cc \u0628\u06c1\u062a \u0627\u0686\u06be\u0627 \u06af\u0627\u062a\u06cc \u06c1\u06cc\u06ba",
    "correct: \u0627\u0633 \u0646\u06d2 \u0645\u062c\u06d2 \u0641\u0648\u0646 \u06a9\u06cc\u0627",
    "correct: \u0628\u0686\u06d2 \u0628\u0627\u06c1\u0631 \u06a9\u06be\u06cc\u0644 \u0631\u06c1\u06d2 \u06c1\u06d2",
    "correct: \u0622\u062c \u06a9\u0627 \u06a9\u06be\u0627\u0646\u0627 \u0628\u06c1\u062a \u0644\u0630\u06cc\u0630 \u06c1\u0648\u0626\u06cc",
]

model.eval()
for i, sample in enumerate(test_samples):
    input_text = PREFIX + sample.replace("correct: ", "")
    inputs = tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=128, num_beams=4, early_stopping=True)
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nSample {i+1}:")
    inp_display = sample.replace("correct: ", "")
    enc = __import__('sys').stdout.encoding or 'utf-8'
    try:
        print(f"  Input:  {inp_display}")
        print(f"  Output: {prediction}")
    except UnicodeEncodeError:
        print(f"  Input:  {inp_display.encode('ascii','replace').decode()}")
        print(f"  Output: {prediction.encode('ascii','replace').decode()}")

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
