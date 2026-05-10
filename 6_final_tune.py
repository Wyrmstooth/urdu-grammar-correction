# -*- coding: utf-8 -*-
"""Final grammar fix: start from original model, train aggressively on targeted data"""
import json, os, torch, numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from peft import PeftModel
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda")
torch.cuda.empty_cache()
print("=" * 60)
print("FINAL GRAMMAR FIX - FRESH START")
print("=" * 60)

# Start from ORIGINAL model
MODEL_PATH = "./urdu_gec_model/final"
PREFIX = "correct grammar: "
MAX_LEN = 96

print("Loading original model...")
base = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")
model = PeftModel.from_pretrained(base, MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
print("Loaded.")

# Load targeted data
with open("grammar_focus_v2.json", "r", encoding="utf-8") as f:
    targeted = json.load(f)
print(f"Targeted records: {len(targeted)}")

# Duplicate 5x for heavy exposure
targeted = targeted * 5
np.random.shuffle(targeted)
print(f"After 5x: {len(targeted)}")

# Add some typing data so model doesn't forget
with open("GenAI-Dataset.json", "r", encoding="utf-8-sig") as f:
    full = json.load(f)
typing = [d for d in full if d.get("error_type") == "typing"]
typing_sample = list(np.random.choice(typing, 2000, replace=False))

all_train = targeted + typing_sample
np.random.shuffle(all_train)

train_records = [{"input": d["input"], "output": d["output"]} for d in all_train]
train_ds = Dataset.from_list(train_records)
print(f"Training size: {len(train_ds)}")

# Small val set
val_subset = list(np.random.choice(targeted, min(100, len(targeted)), replace=False))
val_records = [{"input": d["input"], "output": d["output"]} for d in val_subset]
val_ds = Dataset.from_list(val_records)

def preprocess(examples):
    inputs = [PREFIX + t.replace("correct: ", "", 1) for t in examples["input"]]
    targets = examples["output"]
    mi = tokenizer(inputs, max_length=MAX_LEN, truncation=True, padding=False)
    lab = tokenizer(targets, max_length=MAX_LEN, truncation=True, padding=False)
    mi["labels"] = lab["input_ids"]
    return mi

tok_train = train_ds.map(preprocess, batched=True, remove_columns=["input", "output"])
tok_val = val_ds.map(preprocess, batched=True, remove_columns=["input", "output"])

# Aggressive training
BATCH = 8
GRAD = 2
EPOCHS = 3
LR = 5e-4  # Higher LR for strong correction learning
STEPS = len(tok_train) // (BATCH * GRAD) * EPOCHS
print(f"\nConfig: EPOCHS={EPOCHS}, LR={LR}, steps~{STEPS}, est~{STEPS*0.8/60:.0f}min")

args = Seq2SeqTrainingArguments(
    output_dir="./grammar_final_ckpt",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH,
    per_device_eval_batch_size=BATCH,
    gradient_accumulation_steps=GRAD,
    learning_rate=LR,
    warmup_steps=50,
    weight_decay=0.01,
    logging_steps=50,
    eval_strategy="no",
    save_strategy="no",
    predict_with_generate=True,
    generation_max_length=256,
    bf16=True, fp16=False,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    max_grad_norm=1.0,
    report_to="none", dataloader_num_workers=0,
    ddp_find_unused_parameters=False, remove_unused_columns=False,
)

collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, pad_to_multiple_of=8)

trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=tok_train, data_collator=collator)

print("\nTraining...")
trainer.train()

# Save
save_path = "./urdu_gec_model/grammar_final"
os.makedirs(save_path, exist_ok=True)
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"Saved to {save_path}")

# Test
print("\n" + "=" * 60)
print("TESTING")
print("=" * 60)
model.eval()

tests = [
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
for t, e in tests:
    inp = tokenizer(PREFIX + t, return_tensors="pt", max_length=256, truncation=True).to(device)
    with torch.no_grad():
        out = model.generate(**inp, max_length=256, num_beams=4, early_stopping=True, no_repeat_ngram_size=3)
    pred = tokenizer.decode(out[0], skip_special_tokens=True)
    results.append({"i": t, "e": e, "p": pred, "ok": pred == e})

with open("_grammar_final_results.json", "w", encoding="utf-8") as f:
    json.dump({"correct": sum(1 for r in results if r["ok"]), "results": results}, f, ensure_ascii=False, indent=2)

print(f"Correct: {sum(1 for r in results if r['ok'])} / {len(results)}")
print("DONE!")
