# -*- coding: utf-8 -*-
import json
import random
from collections import Counter

random.seed(99)

with open("GenAI-Dataset.json", "r", encoding="utf-8-sig") as f:
    data = json.load(f)

print(f"Total: {len(data)}")

# 1. Check for identity pairs (input == output after removing prefix)
bad = []
for d in data:
    inp = d["input"].replace("correct: ", "")
    out = d["output"]
    if inp.strip() == out.strip():
        bad.append((inp, d.get("category", ""), d.get("error_type", "")))

print(f"\n=== IDENTITY PAIRS (input == output, no correction needed) ===")
print(f"Count: {len(bad)} ({len(bad)/len(data)*100:.1f}%)")

# Show by error type
bad_errs = Counter(b[2] for b in bad)
for e, c in bad_errs.most_common():
    print(f"  {e}: {c}")

# Show a few examples
print("\nExamples of bad pairs:")
for b in bad[:8]:
    print(f"  [{b[1]}/{b[2]}] {b[0][:80]}")

# 2. Check if the 'input' actually contains errors
print("\n=== CHECKING INPUT/OUTPUT DIFFERENCES ===")
same_or_similar = 0
for d in data:
    inp = d["input"].replace("correct: ", "")
    out = d["output"]
    # If lengths are very close and few words differ
    inp_words = set(inp.split())
    out_words = set(out.split())
    diff = inp_words.symmetric_difference(out_words)
    if len(diff) <= 2:
        same_or_similar += 1

print(f"Pairs with <= 2 word differences: {same_or_similar} ({same_or_similar/len(data)*100:.1f}%)")

# 3. Check the 'spelling' error type for quality
print("\n=== SPELLING ERROR SAMPLES ===")
spelling = [d for d in data if d.get("error_type") == "spelling"]
print(f"Spelling errors: {len(spelling)}")
for d in random.sample(spelling, 8):
    inp = d["input"].replace("correct: ", "")
    print(f"  IN:  {inp}")
    print(f"  OUT: {d['output']}")
    print()

# 4. Check generated data quality by looking at the NEW entries
print("\n=== CHECKING NEW ENTRIES (last 500) ===")
new_data = data[-500:]
bad_new = []
for d in new_data:
    inp = d["input"].replace("correct: ", "")
    out = d["output"]
    if inp.strip() == out.strip():
        bad_new.append(d)

print(f"Identity pairs in new data: {len(bad_new)}/{500}")
if bad_new:
    print("Examples:")
    for b in bad_new[:5]:
        print(f"  IN:  {b['input'][:80]}")
        print(f"  OUT: {b['output'][:80]}")
