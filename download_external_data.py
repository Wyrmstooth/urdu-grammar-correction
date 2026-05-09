import json
import random
from datasets import load_dataset

random.seed(42)

print("Downloading mahwizzzz/urdu_error_correction...")
raw = load_dataset("mahwizzzz/urdu_error_correction", "default", split="train")
print(f"Total external samples: {len(raw)}")

print("\nFiltering by quality...")
filtered = []
for row in raw:
    inp = row.get("incorrect", row.get("input_text", "")).strip()
    out = row.get("correct", row.get("correct_text", "")).strip()
    if not inp or not out:
        continue
    if inp == out:
        continue
    if len(inp) < 10 or len(out) < 5:
        continue
    if len(inp) > 300:
        continue
    filtered.append((inp, out))

print(f"After quality filter: {len(filtered)}")

print("\nLoading existing dataset...")
with open("GenAI-Dataset.json", "r", encoding="utf-8-sig") as f:
    existing = json.load(f)
print(f"Existing: {len(existing)} samples")

print("\nMerging...")
existing_inputs = set()
for d in existing:
    clean = d["input"].replace("correct: ", "", 1).strip()
    existing_inputs.add(clean)

duplicates = 0
new_added = 0
max_new = 200000

for inp, out in filtered:
    if new_added >= max_new:
        break
    if inp in existing_inputs:
        duplicates += 1
        continue
    existing_inputs.add(inp)
    existing.append({
        "input": f"correct: {inp}",
        "output": out,
        "category": "news_web",
        "error_type": "typing"
    })
    new_added += 1

print(f"New added: {new_added}")
print(f"Duplicates skipped: {duplicates}")
print(f"Total: {len(existing)} samples")

print("\nShuffling...")
random.shuffle(existing)

with open("GenAI-Dataset.json", "w", encoding="utf-8") as f:
    json.dump(existing, f, ensure_ascii=False, indent=2)

print("Saved GenAI-Dataset.json")

print("\n--- Sample new entries ---")
for d in existing[-5:]:
    inp = d["input"].replace("correct: ", "")
    print(f"  IN:  {inp[:80]}")
    print(f"  OUT: {d['output'][:80]}")
    print()
