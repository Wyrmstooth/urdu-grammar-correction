## 3.1 Dataset and Dataset Preparation

### Data Source

The dataset used in this study is the **GenAI Urdu Grammar Error Correction Dataset**, a collection of 7,496 carefully curated Urdu sentence pairs where each input contains one or more grammatical errors and the output provides the corrected version. The dataset was constructed by manually identifying common error patterns in Urdu text and creating corresponding correction pairs. An additional 200,000 sentence pairs were sourced from the `mahwizzzz/urdu_error_correction` HuggingFace dataset and merged into the training corpus, bringing the total to 207,496 samples. These external samples focus primarily on typing errors (garbled text from noisy input sources such as OCR or mobile keyboards).

### Data Format

Each entry in the dataset is a JSON object with the following fields:

```json
{
  "input": "correct: میں کل بازار گیا تھا سبزیاں خریدنے کے لیے",
  "output": "میں کل سبزیاں خریدنے کے لیے بازار گیا تھا",
  "category": "daily",
  "error_type": "word_order"
}
```

The `input` field always begins with the prefix `"correct: "` to signal the task to the model. The `output` field contains the grammatically correct sentence. Each sample is labeled with a domain `category` and an `error_type` indicating the kind of mistake present.

### Error Type Distribution

The dataset covers nine distinct error types:

| Error Type | Curated | External | Total | Description |
|-----------|---------|----------|-------|-------------|
| typing | 0 | 200,000 | 200,000 | Character-level errors (typos, OCR noise, garbled text) |
| word_order | 1,704 | 0 | 1,704 | Incorrect sentence structure (SOV violations) |
| extra_words | 1,701 | 0 | 1,701 | Redundant or repeated words |
| grammar | 1,370 | 0 | 1,370 | Verb gender/number agreement errors |
| fluency | 1,291 | 0 | 1,291 | Awkward phrasing, non-native constructions |
| missing_words | 969 | 0 | 969 | Missing case markers, prepositions, or required words |
| spelling | 418 | 0 | 418 | Typographical errors in individual words |
| mixed | 23 | 0 | 23 | Multiple error types combined |
| awkward | 20 | 0 | 20 | Severely unnatural sentences |
| **Total** | **7,496** | **200,000** | **207,496** | |

### Category Distribution (Curated Dataset)

The curated portion of the dataset spans 12 domain categories:

| Category | Samples | Description |
|----------|---------|-------------|
| daily | 1,479 | Everyday conversation |
| culture_entertainment | 741 | Movies, music, arts |
| professional | 711 | Workplace and business |
| formal | 664 | Formal writing |
| news | 652 | News articles |
| academic | 629 | Education and research |
| travel_tourism | 576 | Travel-related content |
| sports | 565 | Sports coverage |
| food | 519 | Food and cuisine |
| consumer_information | 508 | Product information |
| general | 387 | Miscellaneous topics |
| culture | 65 | Cultural content |

### Preprocessing

The dataset underwent the following preprocessing steps:

1. **Identity Pair Removal**: Entries where the input (after removing the `"correct: "` prefix) was identical to the output were removed, as they provide no learning signal.
2. **Length Filtering**: Sentences with fewer than 10 characters in the input or fewer than 5 characters in the output were excluded. Sentences exceeding 300 characters were also filtered out.
3. **Deduplication**: External samples whose input text already existed in the curated dataset were skipped during the merge process to avoid duplicates.
4. **Prefix Normalization**: All inputs were ensured to begin with the `"correct: "` task prefix. The training script automatically re-adds the prefix as `"correct grammar: "`.
5. **Shuffling**: The dataset was randomly shuffled (seed=42) to ensure uniform distribution of error types across splits.

### Train/Validation/Test Split

For training, the entire dataset was shuffled (seed=42) and split as follows:

| Split | Samples | Percentage |
|-------|---------|------------|
| Training | 50,000 | 78.1% |
| Validation | 2,000 | 3.1% |
| Test | 2,000 | 3.1% |
| Unused | 143,496 | 15.6% |

An additional stratified evaluation set was created by sampling evenly across all error types (up to 25 samples per type) from the non-test portion of the data, resulting in approximately 144 evaluation samples for comprehensive per-category and per-error-type analysis.

### Sentence Length Distribution

The average sentence length in the curated dataset is 18.4 words (min: 2, max: 29), with the external typing dataset showing similar length characteristics.

---

## 3.2 Methodology

### Model Architecture

The core model used is **mT5-small** (Multilingual T5-small), a sequence-to-sequence transformer architecture based on the T5 framework [Raffel et al., 2020]. mT5 was pre-trained on the mC4 corpus covering 101 languages, including Urdu, making it suitable for multilingual text-to-text generation tasks.

**Base Architecture (mT5-small):**

| Component | Specification |
|-----------|--------------|
| Architecture | Encoder-Decoder Transformer |
| Encoder Layers | 8 |
| Decoder Layers | 8 |
| Attention Heads | 6 per layer |
| Hidden Dimension (d_model) | 512 |
| Feed-Forward Dimension (d_ff) | 1,024 |
| Key/Value Dimension (d_kv) | 64 |
| Relative Attention Buckets | 32 |
| Activation Function | Gated-GELU |
| Vocabulary Size | 250,112 |
| Tokenizer | SentencePiece (Unigram) |
| Total Parameters | ~300 million |

The model takes an incorrect Urdu sentence as input and generates the corrected sentence as output. The task is framed as a text-to-text generation problem, consistent with the T5 paradigm.

### Fine-Tuning with LoRA

To efficiently adapt the large pre-trained model, **Low-Rank Adaptation (LoRA)** [Hu et al., 2022] was employed. LoRA injects trainable low-rank decomposition matrices into the attention and feed-forward layers of the transformer while keeping the original weights frozen. This dramatically reduces the number of trainable parameters and memory requirements.

**LoRA Configuration:**

| Parameter | Value |
|-----------|-------|
| Rank (r) | 256 |
| Alpha (α) | 512 |
| Dropout | 0.1 |
| Target Modules | q, k, v, o, wi, wo |
| Inference Mode | False |
| Trainable Parameters | ~28 million (9.3% of total) |

The effective scaling factor is α/r = 2, controlling the magnitude of the LoRA adaptation. All attention projection matrices (q, k, v, o) and feed-forward projections (wi, wo) were targeted to allow the adapter to modify both attention patterns and feed-forward representations.

### Mathematical Formulation

**mT5 Encoder-Decoder**: Given an input sequence x = (x₁, ..., xₙ), the encoder produces contextualized representations:

$$\mathbf{h} = \text{Encoder}(x)$$

The decoder then generates the output sequence y = (y₁, ..., yₘ) autoregressively:

$$P(y | x) = \prod_{t=1}^{m} P(y_t | y_{<t}, \mathbf{h})$$

**LoRA**: For each weight matrix W ∈ ℝ^(d×k) in the targeted modules, LoRA constrains its update to a low-rank decomposition:

$$W' = W + \Delta W = W + \frac{\alpha}{r} \cdot BA$$

where B ∈ ℝ^(d×r) and A ∈ ℝ^(r×k) are trainable matrices with rank r << min(d, k). The original weight matrix W is frozen and only B and A receive gradient updates.

**Loss Function**: The model is trained using the standard cross-entropy loss for sequence generation:

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{m_i} \log P(y_{i,t} | y_{i,<t}, x_i)$$

where N is the batch size and mᵢ is the length of the target sequence for the i-th example.

### Training Procedure

**Hyperparameters:**

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 5 × 10⁻⁴ |
| Learning Rate Schedule | Linear warmup + linear decay |
| Warmup Steps | 500 |
| Weight Decay | 0.01 |
| Batch Size (per device) | 4 |
| Gradient Accumulation Steps | 4 |
| Effective Batch Size | 16 |
| Epochs | 5 |
| Max Sequence Length | 96 tokens |
| Precision | BF16 (mixed precision) |
| Gradient Checkpointing | Enabled |
| Max Gradient Norm | 1.0 |
| Beam Search (inference) | 4 beams |
| Repetition Penalty | no_repeat_ngram_size = 3 |

**Training Configuration:**

The model was trained for 5 epochs on 50,000 samples. With an effective batch size of 16, each epoch consisted of 3,125 steps, totaling approximately 15,625 training steps. The learning rate increased linearly from 0 to 5 × 10⁻⁴ over the first 500 warmup steps, then decayed linearly to 0 over the remaining steps.

Gradient checkpointing was enabled to reduce memory consumption, and BF16 mixed precision was used for faster computation on compatible hardware (NVIDIA RTX 5060 Laptop GPU, 8GB VRAM). Training completed in approximately 3.5 hours.

### Generation Strategy

During inference, beam search with 4 beams is used for decoding:

$$\hat{y} = \arg\max_{y} \sum_{t=1}^{m} \log P(y_t | y_{<t}, x)$$

Early stopping is enabled when all beams either reach the end-of-sequence token or the maximum length of 256 tokens. A 3-gram repetition penalty prevents the model from generating the same trigram twice, reducing degenerate outputs.

---

## 4. Results and Analysis

### Overall Performance

The model was evaluated using **corpus-level BLEU (Bilingual Evaluation Understudy)** scores via the SacreBLEU library. Two evaluation strategies were employed:

1. **Stratified Evaluation**: 144 samples evenly distributed across error types
2. **Full Test Evaluation**: 500 samples from a held-out contiguous test portion

| Evaluation Type | BLEU Score |
|----------------|------------|
| Stratified (144 samples) | 58.71 |
| Full Test (500 samples) | 44.07 |

The gap between stratified and full-test BLEU reflects the dataset imbalance: the stratified set evenly samples all error types, while the full test set is dominated by typing errors (96.4% of data) which are inherently harder to correct perfectly due to character-level noise.

### Per Error Type Performance

| Error Type | BLEU Score | Samples |
|-----------|------------|---------|
| extra_words | 92.38 | 20 |
| grammar | 70.43 | 20 |
| fluency | 64.75 | 20 |
| spelling | 66.58 | 18 |
| missing_words | 52.37 | 20 |
| word_order | 48.51 | 20 |
| typing | 45.02 | 20 |
| mixed | 41.66 | 3 |
| awkward | 10.35 | 3 |

**Key Observations:**

- **extra_words (92.38 BLEU)**: Near-perfect performance. Removing redundant words is the simplest correction pattern and the model excels at it.
- **grammar (70.43 BLEU)**: Moderate performance on verb agreement errors. The model handles single-word auxilary corrections well but struggles with multi-word agreement patterns and cases requiring knowledge of noun gender.
- **spelling (66.58 BLEU)**: Good performance on individual word corrections despite limited training examples (418 curated samples, though typing errors also provide spelling correction signal).
- **word_order (48.51 BLEU)**: Below-average performance. Rearranging words requires deeper syntactic understanding, and the model sometimes confuses word order with other error types.
- **typing (45.02 BLEU)**: The most challenging category due to severe character-level corruption. The external dataset provided 200,000 examples, but the diversity of corruption patterns makes perfect reconstruction difficult. Many inputs are barely intelligible even to human readers.
- **awkward (10.35 BLEU)**: Very poor performance on severely unnatural sentences. With only 20 training examples (and 3 evaluation samples), the model lacks sufficient signal to learn these corrections.

### Per Category Performance

| Category | BLEU Score | Examples |
|----------|------------|----------|
| news | 74.97 | 25 |
| travel_tourism | 71.89 | 25 |
| professional | 69.82 | 25 |
| general | 67.68 | 20 |
| culture_entertainment | 66.11 | 25 |
| daily | 65.39 | 25 |
| consumer_information | 62.00 | 25 |
| food | 58.80 | 25 |
| sports | 56.03 | 25 |
| formal | 55.56 | 25 |
| academic | 52.08 | 25 |
| news_web | 45.02 | 20 |

The model performs best in **news** (74.97) and **travel_tourism** (71.89) categories, which have more structured and formulaic language. Performance drops in **academic** (52.08) and **formal** (55.56) categories, likely due to more complex sentence structures and specialized vocabulary. The **news_web** category (45.02) corresponds entirely to the external typing dataset and reflects the inherent difficulty of reconstructing severely corrupted text.

### Training Curves

The training loss curve shows a consistent downward trend across 15,625 steps. The initial loss of approximately 4.5 decreased to approximately 3.9 by the end of training, indicating steady learning. The smoothed loss curve reveals that the model continued to improve throughout training, suggesting that additional epochs or a larger training set could yield further gains.

### Inference Examples

| Error Type | Input | Model Output |
|-----------|-------|--------------|
| word_order | میں کل بازار گیا تھا سبزیاں خریدنے کے لیے | میں کل سبزیاں خریدنے کے لیے بازار گیا تھا |
| grammar | وہ لڑکی بہت اچھا گاتی ہیں | وہ لڑکی بہت اچھا گاتی ہے |
| spelling | اس نے مجے فون کیا | اس نے مجھے فون کیا |
| extra_words | وہ گھر گیا اور اور پھر واپس آیا | وہ گھر گیا اور پھر واپس آیا |
| missing_words | میں نے بازار سبزی خریدی | میں نے بازار میں سبزی خریدی |

### Strengths and Limitations

**Strengths:**
- Strong performance on extra word removal (92.38 BLEU) and moderate grammar correction (70.43 BLEU)
- Efficient training through LoRA, requiring only 28M trainable parameters out of 300M total
- Effective use of external typing data to improve spelling and character-level corrections
- Good generalization across 11 domain categories with consistent performance

**Limitations:**
- Class imbalance (96.4% typing errors) skews the model toward typing correction at the expense of other error types
- Limited grammar training data (1,370 examples, 0.7% of total) results in imperfect gender/number agreement correction
- Poor performance on severely unnatural sentences (awkward: 10.35 BLEU) and mixed errors (41.66 BLEU)
- The model sometimes over-edits, changing correct portions of the input, or under-edits, leaving errors uncorrected
- Word order correction (48.51 BLEU) remains challenging due to the complexity of Urdu syntax
