# -*- coding: utf-8 -*-
"""Quick targeted grammar fine-tune: 1 epoch on generated grammar data"""
import json, os, torch
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq,
)
from peft import PeftModel
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("TARGETED GRAMMAR FINE-TUNE")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
torch.cuda.empty_cache()

MODEL_PATH = "./urdu_gec_model/grammar_fixed"
PREFIX = "correct grammar: "
MAX_LEN = 96

# Load base + already-fine-tuned LoRA
print("\nLoading model...")
base_model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")
model = PeftModel.from_pretrained(base_model, MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
print("Loaded.")

# Load generated grammar data
print("\nLoading grammar focus data...")
with open("grammar_focus_data.json", "r", encoding="utf-8") as f:
    grammar_data = json.load(f)
print(f"Base records: {len(grammar_data)}")

# Duplicate 4x for more exposure
grammar_data = grammar_data * 4
print(f"After 4x duplication: {len(grammar_data)}")

# Also take existing grammar data from main dataset for diversity
with open("GenAI-Dataset.json", "r", encoding="utf-8-sig") as f:
    full_data = json.load(f)

existing_grammar = [d for d in full_data if d.get("error_type") == "grammar"]
# Take all existing grammar + oversample
existing_train = existing_grammar * 2  # 1370 * 2 = 2740

# Add some typing examples so model doesn't forget typing
typing_samples = [d for d in full_data if d.get("error_type") == "typing"]
typing_train = list(np.random.choice(typing_samples, 2000, replace=False))

# Combine
all_train = grammar_data + existing_train + typing_train
np.random.shuffle(all_train)

# Convert
train_records = [{"input": d["input"], "output": d["output"]} for d in all_train]
train_ds = Dataset.from_list(train_records)

print(f"Final training size: {len(train_ds)}")

# Validation: small set of grammar sentences
val_samples = []
with open("grammar_focus_data.json", "r", encoding="utf-8") as f:
    orig_grammar = json.load(f)
val_samples = list(np.random.choice(orig_grammar, min(100, len(orig_grammar)), replace=False))
val_records = [{"input": d["input"], "output": d["output"]} for d in val_samples]
val_ds = Dataset.from_list(val_records)

print(f"Validation size: {len(val_ds)}")

# Tokenize
def preprocess(examples):
    inputs = [PREFIX + t.replace("correct: ", "", 1) for t in examples["input"]]
    targets = examples["output"]
    model_inputs = tokenizer(inputs, max_length=MAX_LEN, truncation=True, padding=False)
    labels = tokenizer(targets, max_length=MAX_LEN, truncation=True, padding=False)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train = train_ds.map(preprocess, batched=True, remove_columns=["input", "output"])
tokenized_val = val_ds.map(preprocess, batched=True, remove_columns=["input", "output"])

# Training config
BATCH = 8
GRAD = 2
EPOCHS = 1
LR = 3e-4  # higher LR for strong correction learning
STEPS = len(tokenized_train) // (BATCH * GRAD) * EPOCHS

print(f"\nTraining config:")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch: {BATCH} x {GRAD} = effective {BATCH * GRAD}")
print(f"  LR: {LR}")
print(f"  Steps: ~{STEPS}")
print(f"  Est time: ~{STEPS * 0.8 / 60:.1f} min")

training_args = Seq2SeqTrainingArguments(
    output_dir="./grammar_v2_checkpoint",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH,
    per_device_eval_batch_size=BATCH,
    gradient_accumulation_steps=GRAD,
    learning_rate=LR,
    warmup_steps=50,
    weight_decay=0.01,
    logging_steps=25,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=300,
    save_total_limit=1,
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

print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)

trainer.train()

# Save
save_path = "./urdu_gec_model/grammar_v2"
os.makedirs(save_path, exist_ok=True)
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"\nSaved to {save_path}")

# Quick test
print("\n" + "=" * 60)
print("TESTING USER SENTENCES")
print("=" * 60)

model.eval()

test_cases = [
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

results = []
for text, expected in test_cases:
    input_text = PREFIX + text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=256, truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=256, num_beams=4,
                                 early_stopping=True, no_repeat_ngram_size=3)
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    results.append({
        "input": text,
        "expected": expected,
        "prediction": prediction,
        "correct": prediction == expected,
    })

with open("_grammar_v2_results.json", "w", encoding="utf-8") as f:
    json.dump({
        "correct_count": sum(1 for r in results if r["correct"]),
        "results": results,
    }, f, ensure_ascii=False, indent=2)

print(f"Correct: {sum(1 for r in results if r['correct'])} / {len(results)}")
print("Results saved.")
print("\nDONE!")
