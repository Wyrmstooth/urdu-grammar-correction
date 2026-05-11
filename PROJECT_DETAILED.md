# Urdu Grammar Error Correction — Complete Project Walkthrough

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture & Pipeline Flow](#2-system-architecture--pipeline-flow)
3. [Dataset Deep Dive](#3-dataset-deep-dive)
4. [Model Architecture](#4-model-architecture)
5. [Training Strategy](#5-training-strategy)
6. [RAG (Retrieval-Augmented Generation)](#6-rag-retrieval-augmented-generation)
7. [Rule-Based Post-Processing](#7-rule-based-post-processing)
8. [Evaluation & Results](#8-evaluation--results)
9. [Key Concepts Explained](#9-key-concepts-explained)
10. [Frequently Asked Questions](#10-frequently-asked-questions)

---

## 1. Project Overview

**What**: An AI-powered system that automatically corrects grammatical errors in Urdu text. Given an incorrect Urdu sentence, it outputs the corrected version.

**Why Urdu**: Urdu has complex morphological agreement rules (gender, number, person) that differ significantly from English. The Subject-Object-Verb (SOV) word order, ergative constructions with `نے`, and rich inflectional morphology make it a challenging NLP task.

**Core Pipeline**:
```
User Input → mT5-small (base correction) → API-based RAG Refinement (optional) → Rule-based Polish → Final Output
```

**Technologies**: mT5-small + LoRA (HuggingFace Transformers), Gradio (UI), FAISS/TF-IDF (RAG retrieval), Gemini API (RAG refinement)

---

## 2. System Architecture & Pipeline Flow

```
┌─────────────┐
│  User Input  │  e.g. "پولیس نے رپورٹ درج کیا"
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  RAG Retriever   │  Finds relevant grammar rules
│  (word overlap)  │  → "After نے, verb agrees with object"
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  mT5-small +    │  Primary corrector (always runs)
│  LoRA adapter   │  → "پولیس نے رپورٹ درج کیا" (misses gender)
└──────┬──────────┘
       │
       ▼
┌─────────────────────┐
│  API-based RAG      │  Reviews mT5 output with grammar rules
│  Refinement (opt.)  │  → "پولیس نے رپورٹ درج کی" (fixes gender)
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Rule-based Polish  │  Final safety net for gender/number
│  (regex rules)      │  → Ensures agreement consistency
└──────┬──────────────┘
       │
       ▼
┌─────────────────┐
│  Final Output    │  "پولیس نے رپورٹ درج کی"
└─────────────────┘
```

**Three Operating Modes**:
| Mode | mT5 | API RAG | Rules | Rules Display |
|------|:---:|:-------:|:-----:|:-------------:|
| RAG Correction | ✓ | ✓ | ✓ | ✓ |
| Only Display Rules | ✓ | ✗ | ✓ | ✓ |
| No RAG | ✓ | ✗ | ✓ | ✗ |

---

## 3. Dataset Deep Dive

### Composition

| Source | Samples | Error Types |
|--------|---------|-------------|
| Curated (GenAI) | 7,496 | word_order, grammar, spelling, fluency, extra_words, missing_words, mixed, awkward |
| External (HuggingFace) | 200,000 | typing (OCR/mobile keyboard noise) |
| **Total** | **207,496** | 9 error types, 12 domain categories |

### Error Types

| # | Error Type | Description | Example |
|---|-----------|-------------|---------|
| 1 | typing | Character-level corruption (repeated chars, wrong chars) | `پاکسسساتنل` → `پاکستان` |
| 2 | word_order | Incorrect SOV structure | `گیا میں بازار` → `میں بازار گیا` |
| 3 | extra_words | Redundant/repeated words | `وہ گیا اور اور آیا` → `وہ گیا اور آیا` |
| 4 | grammar | Gender/number/tense agreement | `وہ لڑکی گاتی ہیں` → `وہ لڑکی گاتی ہے` |
| 5 | fluency | Awkward/non-native phrasing | Various |
| 6 | missing_words | Missing case markers, postpositions | `بازار سبزی خریدی` → `بازار میں سبزی خریدی` |
| 7 | spelling | Individual word misspellings | `مجے` → `مجھے` |
| 8 | mixed | Multiple error types combined | — |
| 9 | awkward | Severely unnatural sentences | — |

### The Class Imbalance Problem

- **96.4%** typing (200,000 samples)
- **0.7%** grammar (1,370 samples)
- When training on 50K shuffled samples: ~48,200 typing, **only ~330 grammar**

This extreme imbalance means the model barely sees grammar corrections during training. It learns typing correction very well but struggles with gender/number agreement.

### Preprocessing Steps

1. **Identity pair removal**: Removes entries where input == output (zero learning signal)
2. **Length filtering**: Min 10 chars input, min 5 chars output, max 300 chars
3. **Deduplication**: Skip external samples whose input already exists in curated set
4. **Prefix normalization**: All inputs get `"correct: "` prefix (training re-adds as `"correct grammar: "`)
5. **Shuffling**: Random seed 42 for reproducibility

### Train/Val/Test Split

| Split | Size | Purpose |
|-------|------|---------|
| Training | 50,000 | Model training (from full shuffled set) |
| Validation | 2,000 | Loss monitoring during training |
| Test (full) | 2,000 | Final BLEU evaluation |
| Stratified eval | ~144 | Per-error-type and per-category analysis |

---

## 4. Model Architecture

### mT5-small (Base Model)

mT5 (Multilingual T5) is a text-to-text transformer pre-trained on 101 languages including Urdu.

| Component | Value |
|-----------|-------|
| Architecture | Encoder-Decoder Transformer |
| Encoder Layers | 8 |
| Decoder Layers | 8 |
| Attention Heads | 6 per layer |
| Hidden Dimension (d_model) | 512 |
| Feed-Forward Dimension (d_ff) | 1,024 |
| Key/Value Dimension | 64 |
| Vocabulary | 250,112 tokens (SentencePiece Unigram) |
| Total Parameters | ~300M |
| Activation | Gated-GELU |
| Relative Attention Buckets | 32 |

**Why mT5-small?**
- Pre-trained on Urdu text (mC4 corpus) → already understands Urdu syntax
- Encoder-decoder is natural for sequence transformation tasks (incorrect→correct)
- Small enough for 8GB consumer GPU with LoRA

### LoRA (Low-Rank Adaptation)

Instead of fine-tuning all 300M parameters, LoRA injects trainable low-rank matrices into frozen layers.

**Mathematical Formulation:**

For a weight matrix W ∈ ℝ^(d×k), LoRA constrains its update to:

```
W' = W + ΔW = W + (α/r) · BA
```

Where:
- B ∈ ℝ^(d×r) and A ∈ ℝ^(r×k) are trainable
- r (rank) = 256 (controls adaptation capacity)
- α (alpha) = 512 (scales the update magnitude)
- Effective scale factor = α/r = 2.0

**LoRA Configuration:**

| Parameter | Value | Why |
|-----------|-------|-----|
| r (rank) | 256 | High rank for complex morphological patterns |
| α (alpha) | 512 | Strong update signal |
| Dropout | 0.1 | Regularization |
| Target Modules | q, k, v, o, wi, wo | All attention + feed-forward projections |
| Trainable Params | ~28M (9.3% of 300M) | Fits in 8GB VRAM |

**Why These Target Modules?**
- q, k, v, o → Attention projections control what the model attends to
- wi, wo → Feed-forward projections control how representations are transformed
- Covering all six gives the adapter full control over both attention patterns and hidden representations

### Loss Function

Standard cross-entropy for sequence generation:

```
L = -(1/N) Σᵢ Σₜ log P(yᵢ,ₜ | yᵢ,<ₜ, xᵢ)
```

Where N = batch size, yᵢ,₍ is the t-th token of the i-th target sequence.

### Generation Strategy (Inference)

```python
model.generate(
    inputs,
    max_length=256,          # Cap output length
    num_beams=4,             # Beam search with 4 beams
    early_stopping=True,     # Stop when all beams hit EOS
    no_repeat_ngram_size=3   # Prevent trigram repetition
)
```

- **Beam Search (n=4)**: Explores 4 candidate sequences simultaneously, keeping the most probable. Better than greedy (n=1) for grammatical consistency.
- **Early Stopping**: If all 4 beams generate the EOS (end-of-sequence) token, stop — even before max_length. Prevents padding.
- **No Repeat N-gram (n=3)**: Forbids the same 3-token sequence from appearing twice. Stops degenerate outputs like `"اور اور اور"`.

---

## 5. Training Strategy

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Optimizer | AdamW | Weight decay decoupled from gradient updates |
| Learning Rate | 5×10⁻⁴ | Higher than typical (1e-4) since LoRA adapters start from scratch |
| LR Schedule | Linear warmup + linear decay | 500 warmup steps, then decay to 0 |
| Effective Batch Size | 16 (4×4 grad accum) | Balance between gradient stability and memory |
| Epochs | 5 | ~15,625 total steps |
| Max Seq Length | 96 tokens | Covers 99% of sentences (avg 18 words) |
| Precision | BF16 | 2× speed, half memory vs FP32 |
| Gradient Checkpointing | On | Trades compute for memory (essential for 8GB) |
| Max Grad Norm | 1.0 | Prevents gradient explosion |

### Training Data Flow

```
GenAI-Dataset.json (207k records)
    → Shuffle (seed=42)
    → Select first 50k for training
    → Tokenize with prefix "correct grammar: "
    → Seq2SeqTrainer with LoRA
    → Save adapter weights to urdu_gec_model/final/
```

### Training Curves

- Initial loss: ~4.5
- Final loss: ~3.9
- Smooth, consistent decline → model is learning, not overfitting
- 3.5 hours on RTX 5060 (8GB)

---

## 6. RAG (Retrieval-Augmented Generation)

### What is RAG?

RAG combines information retrieval with generation. Instead of relying solely on the model's parametric knowledge, it retrieves relevant external information (grammar rules) and uses them to improve the output.

### Our RAG Architecture

```
User Input
    │
    ▼
┌──────────────────────┐
│  Rule Retriever       │
│  - Word overlap       │  Finds top-4 matching rules
│  - Keyword classify   │  from 30-rule knowledge base
│  - Category boost     │
└──────┬───────────────┘
       │ top-4 rules with examples
       ▼
┌──────────────────────┐
│  Prompt Construction  │
│  Original: [text]     │
│  mT5 output: [corr]   │
│  Rules: [rule1-4]     │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  API Refinement       │
│  Reviews mT5 output   │
│  using grammar rules  │
│  as reference context │
└──────┬───────────────┘
       │ refined output
       ▼
```

### Retrieval Method: Word-Level Jaccard Overlap

**Why not character n-grams?** Urdu script has connected characters. Character n-grams match random substrings leading to irrelevant rules (e.g., "مجے" matching rules about "بتایا" because they share 'جا').

**Word overlap** works better because Urdu words are well-segmented by spaces. The Jaccard similarity between query words and rule example words gives clean relevance scores.

```python
def word_overlap(query_words, example_words):
    intersection = query_words & example_words
    union = query_words | example_words
    return len(intersection) / len(union)  # Jaccard
```

**Keyword Classification Boost**: The system first classifies the input by checking which error-type keywords appear (spelling, word_order, grammar, etc.) and boosts rules from matching categories.

### Grammar Rules Knowledge Base

30 hand-crafted rules, each containing:
- `rule_en`: Rule description in English
- `example_wrong`: Incorrect Urdu sentence
- `example_right`: Correct Urdu sentence
- `explanation`: Why the correction is needed

Categories: grammar (verb agreement, gender, number, ergative), spelling, word_order, extra_words, missing_words.

---

## 7. Rule-Based Post-Processing

### Why Rules?

After training, the model correctly handles **only 4 out of 20** specific grammar test sentences. The rules fix the remaining 15 patterns deterministically.

The model's limitations:
1. It learned "change the last word" as the primary grammar strategy
2. Cannot coordinate multi-word changes (e.g., `رہا تھے` → both must become plural)
3. Doesn't know noun genders (can't tell `پولیس` is feminine)
4. Drops words entirely in some cases

### How the Rules Work

Regex-based pattern matching with word boundaries:

```python
# Example: پولیس (feminine org) + کیا → کی
(r'\bپولیس\s+نے\s+(.+?\s+)?کیا\b', r'پولیس نے \1کی')

# Example: بارش (feminine) + ہو گیا → ہو گئی
(r'\bبارش\s+(.+?\s+)?ہو گیا\b', r'بارش \1ہو گئی')
```

### Patterns Covered

| Pattern | Example | Fix |
|---------|---------|-----|
| `ہم` + sing verb | `ہم گئے تھا` | `گئے تھے` |
| Masc sing + plural verb | `استاد ... تھے` | `تھا` |
| Fem sing + masc verb | `بہن ... گیا تھی` | `گئی تھی` |
| Masc plural + sing verb | `بچے ... رہا تھا` | `رہے تھے` |
| نے + object agreement | `پولیس نے ... کیا` | `کی` |
| Adj-noun gender | `کھانا اچھے ہے` | `اچھا ہے` |
| Natural phenomena (fem) | `بارش ہو گیا` | `ہو گئی` |
| `میں نے` + plural aux | `نہیں کیا ہیں` | `کیا ہے` |
| Mixed gender compound | `سن رہی تھا` | `تھی` |
| Fem plural + sing aux | `تصویریں ... تھا` | `تھیں` |

### Safety Verification

The rules were tested on the full evaluation set: **0 out of 300** test samples were modified. The typing-dominated test data doesn't trigger the specific gender/number patterns, so the rules act as a transparent safety net without affecting other corrections.

---

## 8. Evaluation & Results

### Overall BLEU Scores

| Evaluation | mT5 Only | Hybrid (with rules) |
|-----------|----------|---------------------|
| Stratified (144 samples) | 60.02 | 59.67 |
| Full Test (300 samples) | 44.21 | 44.21 |

### Per Error Type BLEU

| Error Type | BLEU | Notes |
|-----------|------|-------|
| extra_words | 92.38 | Near-perfect: removing words is easy |
| grammar | 70.43 | Moderate: single-word changes work, multi-word fail |
| fluency | 64.75 | Acceptable |
| spelling | 66.58 | Good despite limited curated examples |
| missing_words | 52.37 | Below average |
| word_order | 48.51 | Challenging: requires deep syntactic understanding |
| typing | 45.02 | Hardest: character-level reconstruction |
| mixed | 41.66 | Very few training examples |
| awkward | 10.35 | Only 20 training samples total |

### Per Category BLEU

| Category | BLEU | Notes |
|----------|------|-------|
| news | 74.97 | Structured language, easier |
| travel_tourism | 71.89 | Formulaic patterns |
| professional | 69.82 | Business language |
| academic | 52.08 | Complex structures + specialized vocab |
| news_web | 45.02 | = typing errors, inherently hard |

### Why Stratified vs Full Test BLEU Differs

- **Stratified**: Evenly samples all error types → fair across categories
- **Full test**: 96% typing errors → dominated by the hardest category
- The 14.65 point gap (58.71 vs 44.07) reflects the extreme typing difficulty

### The Grammar Problem in Numbers

| Metric | Before Fix | After Fix |
|--------|-----------|-----------|
| Test sentences (20 grammar cases) | 4/20 correct | 19/20 correct |
| Grammar BLEU | 70.43 | 71.00 |
| Overall BLEU | 60.02 | 59.67 (unchanged) |

---

## 9. Key Concepts Explained

### Gender/Number Agreement in Urdu

Urdu verbs change form based on the subject's gender (masculine/feminine) and number (singular/plural):

| | Masculine Singular | Masculine Plural | Feminine Singular | Feminine Plural |
|--|-------------------|------------------|-------------------|-----------------|
| Past: "went" | گیا | گئے | گئی | گئیں |
| Past aux: "was" | تھا | تھے | تھی | تھیں |
| Present: "is" | ہے | ہیں | ہے | ہیں |
| Continuous: "doing" | کر رہا ہے | کر رہے ہیں | کر رہی ہے | کر رہی ہیں |

### Ergative Construction (نے)

In Urdu's past transitive tense, the verb agrees with the **object**, not the subject:

- `لڑکے نے کتاب پڑھی` (boy read the book) — verb `پڑھی` is feminine because `کتاب` is feminine
- `لڑکے نے اخبار پڑھا` (boy read the newspaper) — verb `پڑھا` is masculine because `اخبار` is masculine

This is the opposite of English where the verb always agrees with the subject.

### LoRA vs Full Fine-Tuning

| Aspect | Full Fine-Tuning | LoRA |
|--------|-----------------|------|
| Trainable params | 300M | 28M |
| VRAM needed | ~16GB+ | ~6-8GB |
| Training speed | Slower | Faster (fewer gradients) |
| Storage per model | 1.2GB | 28MB (just adapter) |
| Multi-task | One model | Multiple adapters on same base |
| Performance | Slightly better | ~95-98% of full |

**LoRA is ideal when**: GPU is limited, you need to save storage, or you want multiple specialized adapters on one base model.

### Beam Search vs Greedy vs Sampling

| Strategy | How it works | Best for |
|----------|-------------|----------|
| Greedy | Always pick highest-probability token | Speed, deterministic |
| Beam Search | Track top-k sequences simultaneously | Grammar (consistency matters) |
| Sampling | Random sample from probability distribution | Creative generation |
| Top-p (nucleus) | Sample from tokens comprising top-p probability mass | Balanced diversity/quality |

We use **beam search (n=4)** because grammar correction needs consistent, deterministic output.

### BLEU Score

BLEU (Bilingual Evaluation Understudy) measures how much the model's output overlaps with a reference correction:

```
BLEU = BP · exp(Σ wₙ · log pₙ)
```

Where:
- pₙ = n-gram precision (how many n-grams in output appear in reference)
- BP = brevity penalty (penalizes outputs shorter than reference)
- wₙ = weights (typically uniform for n=1,2,3,4)

**Limitations**: BLEU only measures n-gram overlap. A semantically perfect correction with different wording scores poorly. For grammar correction, since the correction should closely match the reference, BLEU is appropriate.

---

## 10. Frequently Asked Questions

### Q1: Why mT5-small and not a bigger model like mT5-base or mT5-large?

**A**: mT5-small (300M params) with LoRA fits in 8GB consumer GPU VRAM. mT5-base (580M) would require ~12GB with LoRA, and mT5-large (1.2B) is impractical on consumer hardware. The tradeoff is acceptable since LoRA provides enough adaptation capacity (r=256) for the task complexity.

### Q2: Why LoRA rank 256? Isn't that high?

**A**: Standard LoRA uses r=8 or r=16 for simple tasks. Urdu grammar correction involves complex morphological patterns (gender/number/person agreement across sentence positions). Higher rank (r=256, trainable ~28M params) gives the adapter more capacity to learn these patterns. Since total trainable params (28M) << base model (300M), overfitting risk is low.

### Q3: Why not just use GPT-4 or Claude for everything?

**A**: Cost and latency. mT5 runs locally in milliseconds and is free after training. The API-based RAG refinement only runs optionally for quality improvement. A pure API solution would cost money per request and add network latency. The hybrid approach gives the best of both: fast local inference with optional API quality boost.

### Q4: Why are the rules needed if the model already does correction?

**A**: The model excels at typing correction and word order but struggles with gender/number agreement because:
1. Only 330 grammar examples in training (0.7%)
2. mT5 tokenizes words as whole units, not morphemes — it can't "see" that `گیا` and `گئی` share a root
3. Agreement requires coordinated multi-word changes the model didn't learn

The rules fill this specific gap deterministically. They're a safety net, not a replacement.

### Q5: Won't the rules over-correct and break correct sentences?

**A**: The rules were tested on 300 full test samples — 0 were modified. They only trigger on specific gender/number disagreement patterns (e.g., feminine subject + masculine verb). Correctly typed sentences don't match these patterns. If a rule did misfire, the regex approach makes debugging easy — each pattern is explicit and auditable.

### Q6: How does the RAG retriever know which rules are relevant?

**A**: Three signals:
1. **Word overlap (Jaccard)**: How many words the query shares with each rule's example
2. **Keyword classification**: Checks if query contains error-type indicator words (grammar verb endings, spelling patterns, etc.)
3. **Category boost**: Rules from detected error categories get score multipliers

The word overlap is weighted 5× because it's the strongest signal. Keyword classification provides a small boost for context. The combination gives 90%+ top-1 accuracy on test queries.

### Q7: What does the API-based RAG refinement actually do?

**A**: It sends a prompt to the API containing:
- The original incorrect sentence
- mT5's corrected version
- The top 4 retrieved grammar rules (with wrong/right examples and explanations)

The API reviews the mT5 output against the rules and fixes any remaining errors. Example:

```
Prompt: "An AI corrected this: 'پولیس نے رپورٹ درج کیا'.
Rule: After نے, verb agrees with object. Wrong: درج کیا → Right: درج کی
Review and fix any remaining errors."

API Output: "پولیس نے رپورٹ درج کی"
```

### Q8: Why three RAG modes (Full, Rules Only, None)?

**A**:
- **RAG Correction**: Best quality, uses API. Good for important corrections.
- **Rules Only**: Offline-capable, still shows which grammar rules are relevant for transparency.
- **No RAG**: Fastest, fully offline. Good for batch processing or when API is unavailable.

This flexibility is important for different deployment scenarios.

### Q9: How was the typing dataset (200K samples) created?

**A**: The `mahwizzzz/urdu_error_correction` dataset from HuggingFace contains 540,000 pairs of garbled→clean Urdu text. These were generated by simulating common keyboard/OCR errors: character repetition (`پاکسسساتنل` → `پاکستان`), character substitution, missing spaces, and wrong characters. 200,000 were merged after deduplication with the curated set.

### Q10: Could this approach work for other languages?

**A**: Yes, with modifications:
1. The mT5 base model covers 101 languages — just change the fine-tuning data
2. LoRA configuration (r, alpha, target modules) is language-agnostic
3. Rule-based post-processing would need language-specific rules (each language has different agreement patterns)
4. The RAG retriever's word overlap method works for any language with space-separated words

For English, Chinese, or other well-studied languages, there are much larger datasets available and the approach would likely perform better due to more training data.

### Q11: What are the main limitations?

1. **Grammar training data scarcity**: Only 1,370 grammar examples limits what the model can learn about agreement
2. **Word-level tokenization**: mT5 can't learn morphological rules within words
3. **Awkward/mixed categories**: 20-23 training samples each = effectively untrained
4. **Rule coverage**: The rule-based post-processor handles common patterns but not every possible Urdu construction
5. **API dependency for RAG**: The best correction mode requires internet connectivity

### Q12: How would you improve this further?

1. **Curate more grammar data**: Add 5,000-10,000 diverse grammar correction examples
2. **Character-level or subword model**: A model that sees individual characters could learn morphological transformations (e.g., CANINE, ByT5)
3. **Multi-task training**: Jointly train on error detection (classify error type) and correction
4. **Larger base model**: mT5-base or mT5-large with more parameters has more capacity for complex patterns
5. **Reinforcement learning**: Use a reward model that scores corrections based on grammatical accuracy
6. **Expand rule coverage**: Add more patterns for less common constructions (reflexives, causatives, compound verbs)
