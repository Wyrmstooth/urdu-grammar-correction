# -*- coding: utf-8 -*-
import json
import random

random.seed(42)

with open("GenAI-Dataset.json", "r", encoding="utf-8-sig") as f:
    data = json.load(f)

print(f"Original: {len(data)}")

# 1. Remove identity pairs (input == output after stripping prefix)
clean = []
removed = 0
for d in data:
    inp = d["input"].replace("correct: ", "").strip()
    out = d["output"].strip()
    if inp == out:
        removed += 1
        continue
    clean.append(d)

print(f"Removed {removed} identity pairs")
print(f"After dedup: {len(clean)}")

# 2. Remove entries where input/output are too short
clean2 = []
removed_short = 0
for d in clean:
    inp = d["input"].replace("correct: ", "").strip()
    out = d["output"].strip()
    if len(inp) < 5 or len(out) < 5:
        removed_short += 1
        continue
    clean2.append(d)

print(f"Removed {removed_short} too-short entries")
print(f"Final: {len(clean2)}")

# 3. Shuffle
random.shuffle(clean2)

# Save
with open("GenAI-Dataset.json", "w", encoding="utf-8-sig") as f:
    json.dump(clean2, f, ensure_ascii=False, indent=2)

print(f"\nSaved cleaned dataset: {len(clean2)} entries")

# Category / error type summary
from collections import Counter
cats = Counter(d.get("category", "?") for d in clean2)
errs = Counter(d.get("error_type", "?") for d in clean2)
print("\nCategories:", dict(cats))
print("Error types:", dict(errs))
