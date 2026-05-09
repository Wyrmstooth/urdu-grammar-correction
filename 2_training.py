# -*- coding: utf-8 -*-
"""
Urdu Grammar Correction - Optimized Training
mT5-small + LoRA (r=256), 50K samples, 5 epochs, ~3.5 hours
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
from peft import LoraConfig, get_peft_model, TaskType
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("URDU GRAMMAR CORRECTION - TRAINING")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    torch.cuda.empty_cache()

print("\n" + "=" * 60)
print("LOADING DATASET")
print("=" * 60)

with open("GenAI-Dataset.json", "r", encoding="utf-8-sig") as f:
    data = json.load(f)

records = [{"input": d["input"], "output": d["output"]} for d in data]
hf_dataset = Dataset.from_list(records)

# Shuffle then take 50K for training, 2K val, 2K test
hf_dataset = hf_dataset.shuffle(seed=42)
train_ds = hf_dataset.select(range(50000))
rest = hf_dataset.select(range(50000, len(hf_dataset)))
val_test = rest.train_test_split(test_size=0.5, seed=42)

dataset = DatasetDict({
    "train": train_ds,
    "validation": val_test["train"].select(range(2000)),
    "test": val_test["test"].select(range(2000)),
})
print(f"Train: {len(dataset['train'])}, Val: {len(dataset['validation'])}, Test: {len(dataset['test'])}")

print("\n" + "=" * 60)
print("LOADING mT5-small")
print("=" * 60)

MODEL_NAME = "google/mt5-small"
PREFIX = "correct grammar: "
MAX_LEN = 96

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=256,
    lora_alpha=512,
    lora_dropout=0.1,
    target_modules=["q", "v", "k", "o", "wi", "wo"],
    inference_mode=False,
    bias="none",
)

model = get_peft_model(model, lora_config)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nLoRA r=256, alpha=512")
print(f"Trainable: {trainable:,} ({100*trainable/total_params:.1f}% of total)")

print("\n" + "=" * 60)
print("TOKENIZING")
print("=" * 60)

def preprocess(examples):
    inputs = [PREFIX + t.replace("correct: ", "", 1) for t in examples["input"]]
    targets = examples["output"]
    model_inputs = tokenizer(inputs, max_length=MAX_LEN, truncation=True, padding=False)
    labels = tokenizer(targets, max_length=MAX_LEN, truncation=True, padding=False)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized = dataset.map(preprocess, batched=True, remove_columns=["input", "output"])
print(f"Tokenized train size: {len(tokenized['train'])}")

print("\n" + "=" * 60)
print("TRAINING CONFIGURATION")
print("=" * 60)

BATCH = 4
GRAD = 4
EFF = BATCH * GRAD
EPOCHS = 5
STEPS = len(tokenized["train"]) // EFF * EPOCHS

training_args = Seq2SeqTrainingArguments(
    output_dir="./checkpoint_dir",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH,
    per_device_eval_batch_size=BATCH,
    gradient_accumulation_steps=GRAD,
    learning_rate=5e-4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_steps=100,
    eval_strategy="no",
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=3,
    load_best_model_at_end=False,
    predict_with_generate=True,
    generation_max_length=MAX_LEN,
    bf16=True,
    fp16=False,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    max_grad_norm=1.0,
    report_to="none",
    dataloader_num_workers=0,
    ddp_find_unused_parameters=False,
    remove_unused_columns=False,
)

print(f"Epochs: {EPOCHS}")
print(f"Batch: {BATCH} x {GRAD} grad accum = effective {EFF}")
print(f"Learning rate: 5e-4")
print(f"Total steps: ~{STEPS}")
print(f"Estimated time: ~{STEPS * 0.75 / 3600:.1f} hours")

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer, model=model, padding=True, pad_to_multiple_of=8,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    data_collator=data_collator,
)

print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)

start_time = time.time()
train_result = trainer.train()
train_time = time.time() - start_time

print(f"\nTraining completed in {train_time/60:.1f} minutes ({train_time/3600:.1f} hours)")
print(f"Training loss: {train_result.training_loss:.4f}")

print("\n" + "=" * 60)
print("SAVING MODEL")
print("=" * 60)

save_path = "./urdu_gec_model/final"
os.makedirs(save_path, exist_ok=True)
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"Saved to {save_path}")

print("\n" + "=" * 60)
print("GENERATING PLOTS")
print("=" * 60)

log_history = trainer.state.log_history
train_losses = [e["loss"] for e in log_history if "loss" in e and "eval_loss" not in e]
train_steps = [e["step"] for e in log_history if "loss" in e and "eval_loss" not in e]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(train_steps, train_losses, "b-", alpha=0.6, linewidth=1)
ax1.set_xlabel("Steps"); ax1.set_ylabel("Loss")
ax1.set_title("Training Loss", fontsize=14, fontweight="bold")
ax1.grid(True, alpha=0.3)
if len(train_losses) > 10:
    w = max(5, len(train_losses) // 20)
    smoothed = np.convolve(train_losses, np.ones(w)/w, mode="valid")
    ax2.plot(range(len(smoothed)), smoothed, "b-", linewidth=2)
ax2.set_xlabel("Steps"); ax2.set_ylabel("Smoothed Loss")
ax2.set_title("Smoothed Training Loss", fontsize=14, fontweight="bold")
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("training_curves.png", dpi=150, bbox_inches="tight")
print("Saved training_curves.png")

print("\n" + "=" * 60)
print("QUICK EVALUATION")
print("=" * 60)

test_samples = [
    "correct: \u0645\u06cc\u06ba \u06a9\u0644 \u0628\u0627\u0632\u0627\u0631 \u06af\u06cc\u0627 \u062a\u06be\u0627 \u0633\u0628\u0632\u06cc\u0627\u06ba \u062e\u0631\u06cc\u062f\u0646\u06d2 \u06a9\u06d2 \u0644\u06cc\u06d2",
    "correct: \u0648\u06c1 \u0644\u0691\u06a9\u06cc \u0628\u06c1\u062a \u0627\u0686\u06be\u0627 \u06af\u0627\u062a\u06cc \u06c1\u06cc\u06ba",
    "correct: \u0627\u0633 \u0646\u06d2 \u0645\u062c\u06d2 \u0641\u0648\u0646 \u06a9\u06cc\u0627",
    "correct: \u0628\u0686\u06d2 \u0628\u0627\u06c1\u0631 \u06a9\u06be\u06cc\u0644 \u0631\u06c1\u06d2 \u06c1\u06d2",
    "correct: \u0622\u062c \u06a9\u0627 \u06a9\u06be\u0627\u0646\u0627 \u0628\u06c1\u062a \u0644\u0630\u06cc\u0630 \u06c1\u0648\u0626\u06cc",
]

model.eval()
for sample in test_samples:
    inp = tokenizer(PREFIX + sample.replace("correct: ", ""), return_tensors="pt", max_length=MAX_LEN, truncation=True).to(device)
    with torch.no_grad():
        out = model.generate(**inp, max_length=MAX_LEN, num_beams=4, early_stopping=True)
    pred = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"  Input:  {sample.replace('correct: ', '')}")
    print(f"  Output: {pred}\n")

print("=" * 60)
print("DONE!")
print("=" * 60)
