# -*- coding: utf-8 -*-
"""
Grammar-focused fine-tuning: oversamples grammar examples to ~30%
to fix gender/number agreement errors. Loads existing LoRA adapter,
trains 2-3 more epochs with lower LR on a balanced subset.
"""
import json
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import PeftModel, PeftConfig
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("GRAMMAR-FOCUSED FINE-TUNING")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    torch.cuda.empty_cache()

MODEL_PATH = "./urdu_gec_model/final"
BASE_MODEL = "google/mt5-small"
PREFIX = "correct grammar: "
MAX_LEN = 96

# ─── 1. Load existing model ───
print("\nLoading existing LoRA adapter...")
base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
model = PeftModel.from_pretrained(base_model, MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
print("Model loaded.")

# ─── 2. Load data and create balanced dataset ───
print("\n" + "=" * 60)
print("BUILDING BALANCED DATASET")
print("=" * 60)

with open("GenAI-Dataset.json", "r", encoding="utf-8-sig") as f:
    data = json.load(f)

# Group by error_type
by_type = {}
for d in data:
    etype = d.get("error_type", "unknown")
    if etype not in by_type:
        by_type[etype] = []
    by_type[etype].append(d)

print("Original distribution:")
for et in sorted(by_type.keys(), key=lambda x: len(by_type[x]), reverse=True):
    print(f"  {et:20s}: {len(by_type[et]):6d}")

# Build balanced training set:
# - Grammar: use all 1,370 entries, oversample to ~8,000 (repeat ~6x)
# - Non-grammar: sample 8,000 from typing, 1,000 each from other types
# Total: ~20,000 samples, grammar is ~40% of training

np.random.seed(42)
balanced_train = []

GRAMMAR_TARGET = 8000
OTHER_TARGET = 1000
TYPING_TARGET = 8000

# Add grammar (oversample)
grammar_entries = by_type.get("grammar", [])
for _ in range(GRAMMAR_TARGET // len(grammar_entries)):
    balanced_train.extend(grammar_entries)
remaining = GRAMMAR_TARGET - (GRAMMAR_TARGET // len(grammar_entries)) * len(grammar_entries)
if remaining > 0:
    balanced_train.extend(np.random.choice(grammar_entries, remaining, replace=False).tolist())

# Add other non-typing types
for etype in by_type:
    if etype in ("grammar", "typing"):
        continue
    entries = by_type[etype]
    for _ in range(max(1, OTHER_TARGET // len(entries))):
        balanced_train.extend(entries)
    rem = OTHER_TARGET - (max(1, OTHER_TARGET // len(entries))) * len(entries)
    if rem > 0:
        balanced_train.extend(np.random.choice(entries, rem, replace=False).tolist())

# Add typing (sample)
typing_entries = by_type.get("typing", [])
balanced_train.extend(np.random.choice(typing_entries, TYPING_TARGET, replace=False).tolist())

# Shuffle
np.random.shuffle(balanced_train)

# Count final distribution
final_dist = Counter(d.get("error_type", "unknown") for d in balanced_train)
print("\nBalanced training distribution:")
for et, cnt in final_dist.most_common():
    pct = 100 * cnt / len(balanced_train)
    print(f"  {et:20s}: {cnt:6d}  ({pct:5.1f}%)")
print(f"  {'TOTAL':20s}: {len(balanced_train):6d}")

# Convert to records
train_records = [{"input": d["input"], "output": d["output"]} for d in balanced_train]
train_ds = Dataset.from_list(train_records)

# Validation: use original test split (812 samples from 1_data_preparation)
# For speed, use 200 samples across types
val_records = []
# Take 30 from each non-typing type
for etype in sorted(by_type.keys()):
    if etype == "typing":
        continue
    entries = by_type[etype]
    n = min(30, len(entries))
    sampled = np.random.choice(entries, n, replace=False)
    for d in sampled:
        val_records.append({"input": d["input"], "output": d["output"]})

# Add 50 typing samples
typing_sampled = np.random.choice(by_type["typing"], 50, replace=False)
for d in typing_sampled:
    val_records.append({"input": d["input"], "output": d["output"]})

np.random.shuffle(val_records)
val_ds = Dataset.from_list(val_records)

print(f"\nValidation size: {len(val_ds)}")

# ─── 3. Tokenize ───
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

tokenized_train = train_ds.map(preprocess, batched=True, remove_columns=["input", "output"])
tokenized_val = val_ds.map(preprocess, batched=True, remove_columns=["input", "output"])

# ─── 4. Training config ───
print("\n" + "=" * 60)
print("TRAINING CONFIGURATION")
print("=" * 60)

BATCH = 4
GRAD = 4
EPOCHS = 3
LR = 2e-4  # Lower LR than original (5e-4)
STEPS = len(tokenized_train) // (BATCH * GRAD) * EPOCHS

print(f"Epochs: {EPOCHS}")
print(f"Batch: {BATCH} x {GRAD} grad = effective {BATCH * GRAD}")
print(f"Learning rate: {LR}")
print(f"Total steps: ~{STEPS}")
print(f"Estimated time: ~{STEPS * 0.75 / 60:.1f} min")

training_args = Seq2SeqTrainingArguments(
    output_dir="./grammar_checkpoint_dir",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH,
    per_device_eval_batch_size=BATCH,
    gradient_accumulation_steps=GRAD,
    learning_rate=LR,
    warmup_steps=100,
    weight_decay=0.01,
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=300,
    save_strategy="steps",
    save_steps=600,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    predict_with_generate=True,
    generation_max_length=256,
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

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer, model=model, padding=True, pad_to_multiple_of=8,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator,
)

# ─── 5. Train ───
print("\n" + "=" * 60)
print("STARTING GRAMMAR FINE-TUNING")
print("=" * 60)

train_result = trainer.train()

# ─── 6. Save ───
print("\n" + "=" * 60)
print("SAVING MODEL")
print("=" * 60)

save_path = "./urdu_gec_model/grammar_fixed"
os.makedirs(save_path, exist_ok=True)
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"Saved to {save_path}")

# ─── 7. Quick test on user sentences ───
print("\n" + "=" * 60)
print("TESTING ON USER SENTENCES")
print("=" * 60)

model.eval()

test_sentences = [
    ("ہم کل بازار گئے تھا", "ہم کل بازار گئے تھے"),
    ("استاد نے سبق سمجھایا تھے", "استاد نے سبق سمجھایا تھا"),
    ("میں نے ناشتہ نہیں کیا ہیں", "میں نے ناشتہ نہیں کیا ہے"),
    ("وہ اسکول دیر سے پہنچتی ہیں", "وہ اسکول دیر سے پہنچتی ہے"),
    ("پولیس نے رپورٹ درج کیا", "پولیس نے رپورٹ درج کی"),
    ("حکومت نے اعلان کئے", "حکومت نے اعلان کیا"),
    ("میری بہن لاہور گیا تھی", "میری بہن لاہور گئی تھی"),
    ("بچے پارک میں کھیل رہا تھے", "بچے پارک میں کھیل رہے تھے"),
    ("ڈاکٹر نے مشورہ دی", "ڈاکٹر نے مشورہ دی"),
    ("بارش سے سڑک بند ہو گیا", "بارش سے سڑک بند ہو گئی"),
    ("یہ کھانا بہت اچھے ہے", "یہ کھانا بہت اچھا ہے"),
    ("ہم رات تک کام کرتا رہے", "ہم رات تک کام کرتے رہے"),
    ("وہ دوست سے ملنے آئے تھی", "وہ دوست سے ملنے آئی تھی"),
    ("میں نے نوکری شروع کیا ہے", "میں نے نوکری شروع کی ہے"),
    ("یہ تصویریں خوبصورت تھا", "یہ تصویریں خوبصورت تھیں"),
    ("مری میں برف باری ہو رہے ہیں", "مری میں برف باری ہو رہی ہے"),
    ("بچے کلاس میں شور مچا رہا تھا", "بچے کلاس میں شور مچا رہے تھے"),
    ("میں نے اسائنمنٹ جمع نہیں کروائے", "میں نے اسائنمنٹ جمع نہیں کروائی"),
    ("حکومت نے سہولتیں شروع کیا ہیں", "حکومت نے سہولتیں شروع کی ہیں"),
    ("وہ موبائل پر گانے سن رہی تھا", "وہ موبائل پر گانے سن رہی تھی"),
]

correct_count = 0
improved_count = 0

for i, (text, expected) in enumerate(test_sentences, 1):
    input_text = PREFIX + text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=MAX_LEN, truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=MAX_LEN, num_beams=4,
                                 early_stopping=True, no_repeat_ngram_size=3)
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

    is_correct = prediction == expected
    if is_correct:
        correct_count += 1
        status = "OK"
    elif prediction != text:
        improved_count += 1
        status = "~"
    else:
        status = "X"

    print(f"{i:2d}. [{status}] IN:  {text}")
    print(f"       EXPECTED: {expected}")
    print(f"       GOT:      {prediction}")
    print()

print(f"\nCorrect: {correct_count}/{len(test_sentences)}")
print(f"Improved (changed but not perfect): {improved_count}/{len(test_sentences)}")
print(f"Missed: {len(test_sentences) - correct_count - improved_count}/{len(test_sentences)}")

print("\n" + "=" * 60)
print("DONE!")
print("=" * 60)
