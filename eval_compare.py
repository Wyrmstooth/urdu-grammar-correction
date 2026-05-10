# -*- coding: utf-8 -*-
"""Compare model-only vs hybrid on stratified test"""
import json, torch, warnings
warnings.filterwarnings("ignore")
from collections import defaultdict
import sacrebleu
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from grammar_rules_post import apply_rules

device = torch.device("cuda")
MODEL_PATH = r".\urdu_gec_model\final"
PREFIX = "correct grammar: "

print("Loading model...")
base = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")
model = PeftModel.from_pretrained(base, MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = model.to(device).eval()
print("Loaded.")

with open("GenAI-Dataset.json", "r", encoding="utf-8-sig") as f:
    all_data = json.load(f)

n_test = len(all_data) // 10
test_data = all_data[-n_test:]

error_samples = defaultdict(list)
for d in all_data[:-n_test]:
    error_samples[d.get("error_type", "unknown")].append(d)

stratified_test = []
for etype, samples in error_samples.items():
    n = max(3, min(20, len(samples) // 20))
    stratified_test.extend(samples[-n:])

print(f"Stratified test: {len(stratified_test)} samples")
eval_subset = test_data[:min(300, len(test_data))]
print(f"Full test subset: {len(eval_subset)} samples")

model_preds, hybrid_preds, references = [], [], []
per_error_model = defaultdict(list)
per_error_hybrid = defaultdict(list)
per_error_refs = defaultdict(list)

print("\nEvaluating stratified...")
for i, d in enumerate(stratified_test):
    inp = d["input"].replace("correct: ", "")
    ref = d["output"]
    input_text = PREFIX + inp
    inputs = tokenizer(input_text, return_tensors="pt", max_length=256, truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=256, num_beams=4, early_stopping=True, no_repeat_ngram_size=3)
    model_out = tokenizer.decode(outputs[0], skip_special_tokens=True)
    hybrid_out = apply_rules(model_out)
    model_preds.append(model_out)
    hybrid_preds.append(hybrid_out)
    references.append(ref)
    etype = d.get("error_type", "unknown")
    per_error_model[etype].append(model_out)
    per_error_hybrid[etype].append(hybrid_out)
    per_error_refs[etype].append(ref)
    if (i + 1) % 30 == 0:
        print(f"  {i + 1}/{len(stratified_test)}")

bleu_model = sacrebleu.corpus_bleu(model_preds, [references])
bleu_hybrid = sacrebleu.corpus_bleu(hybrid_preds, [references])

print("\n" + "=" * 60)
print("OVERALL BLEU (Stratified)")
print("=" * 60)
print(f"Model only:  {bleu_model.score:.2f}")
print(f"Hybrid:      {bleu_hybrid.score:.2f}")
print(f"Difference:  {bleu_hybrid.score - bleu_model.score:+.2f}")

print("\nPER ERROR TYPE BLEU:")
header = "{:25s}  {:>8s}  {:>8s}  {:>8s}  {:>8s}".format("Error Type", "Model", "Hybrid", "Delta", "Samples")
print(header)
print("-" * 70)
for etype in sorted(per_error_model.keys()):
    if len(per_error_model[etype]) >= 2:
        bm = sacrebleu.corpus_bleu(per_error_model[etype], [per_error_refs[etype]])
        bh = sacrebleu.corpus_bleu(per_error_hybrid[etype], [per_error_refs[etype]])
        delta = bh.score - bm.score
        row = "{:25s}  {:8.2f}  {:8.2f}  {:8.2f}  {:8d}".format(etype, bm.score, bh.score, delta, len(per_error_model[etype]))
        print(row)

print("\nFULL TEST SUBSET...")
full_model, full_hybrid, full_refs = [], [], []
for d in eval_subset:
    inp = d["input"].replace("correct: ", "")
    ref = d["output"]
    input_text = PREFIX + inp
    inputs = tokenizer(input_text, return_tensors="pt", max_length=256, truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=256, num_beams=4, early_stopping=True, no_repeat_ngram_size=3)
    model_out = tokenizer.decode(outputs[0], skip_special_tokens=True)
    hybrid_out = apply_rules(model_out)
    full_model.append(model_out)
    full_hybrid.append(hybrid_out)
    full_refs.append(ref)

bleu_full_model = sacrebleu.corpus_bleu(full_model, [full_refs])
bleu_full_hybrid = sacrebleu.corpus_bleu(full_hybrid, [full_refs])
print(f"Full test Model only: {bleu_full_model.score:.2f}")
print(f"Full test Hybrid:     {bleu_full_hybrid.score:.2f}")
print(f"Difference:           {bleu_full_hybrid.score - bleu_full_model.score:+.2f}")
