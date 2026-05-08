# GenAI Urdu Grammar Correction Project

## Project Overview
Fine-tuned mT5-small (300M parameters) on a custom 8,143-example Urdu grammar correction dataset to automatically detect and correct common Urdu grammatical errors.

## Files

| File | Purpose |
|------|---------|
| `GenAI-Dataset.json` | Extended dataset (8,143 entries, 12 categories, 9 error types) |
| `1_data_preparation.py` | Loads dataset, generates statistics & visualizations, creates train/val/test splits |
| `2_training.py` | Fine-tunes mT5-small with bf16 mixed precision on RTX 5060 GPU |
| `3_evaluation.py` | Evaluates model with BLEU scores per error type and category |
| `train.csv` / `val.csv` / `test.csv` | Train/validation/test splits |
| `test_eval.csv` | Test set with metadata for analysis |
| `urdu_gec_dataset_hf/` | HuggingFace Dataset format |
| `urdu_gec_model/final/` | Trained model and tokenizer |

## Dataset

- **Total**: 8,143 entries (3,004 new entries added)
- **Categories**: daily, professional, formal, academic, sports, news, culture_entertainment, travel_tourism, consumer_information, food, general, culture
- **Error Types**: word_order, grammar, spelling, extra_words, missing_words, fluency, awkward, mixed, correct
- **Format**: JSON with `input`, `output`, `category`, `error_type`
- **Input prefix**: `"correct: "` followed by incorrect Urdu sentence

## Model

- **Architecture**: google/mt5-small (~300M parameters)
- **Training**: 3 epochs, bf16 mixed precision, batch size 16 (8×2 gradient accumulation)
- **Learning Rate**: 2e-4 with linear decay
- **Training Time**: ~6 minutes on NVIDIA RTX 5060 (8GB)
- **Model Size**: ~1.2 GB

## Results

| Metric | Score |
|--------|-------|
| Overall BLEU (stratified) | 76.90 |
| Overall BLEU (full test) | 77.02 |

### BLEU by Error Type
| Error Type | BLEU |
|-----------|------|
| extra_words | 97.60 |
| grammar | 90.62 |
| spelling | 87.12 |
| correct | 83.99 |
| fluency | 79.91 |
| missing_words | 70.71 |
| word_order | 46.33 |
| mixed | 28.95 |
| awkward | 12.50 |

### BLEU by Category
| Category | BLEU |
|----------|------|
| formal | 89.97 |
| academic | 85.06 |
| travel_tourism | 83.60 |
| food | 80.80 |
| consumer_information | 80.28 |
| daily | 79.00 |
| culture_entertainment | 77.23 |
| professional | 74.12 |
| sports | 55.90 |
| news | 52.32 |

## Usage

### Data Preparation
```bash
python 1_data_preparation.py
```
Generates: `dataset_analysis.png`, `category_error_heatmap.png`, `train.csv`, `val.csv`, `test.csv`

### Training
```bash
python 2_training.py
```
Requirements: NVIDIA GPU with 8GB+ VRAM, PyTorch with CUDA, transformers, datasets, sentencepiece, accelerate

### Inference
```bash
python 3_evaluation.py
```

### Python API
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model = AutoModelForSeq2SeqLM.from_pretrained("./urdu_gec_model/final")
tokenizer = AutoTokenizer.from_pretrained("./urdu_gec_model/final")
model = model.to("cuda")
model.eval()

def correct_urdu(text):
    input_text = "correct grammar: " + text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True).to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=128, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example
print(correct_urdu("اس نے مجے فون کیا"))
# Output: اس نے مجھے فون کیا
```

## Generated Visualizations
- `dataset_analysis.png` - Category/error type distributions and sentence length histograms
- `category_error_heatmap.png` - Heatmap of category vs error type
- `training_curves.png` - Training and validation loss curves
- `evaluation_results.png` - BLEU scores by error type, category, and overall

## Requirements
- Python 3.10+
- PyTorch 2.11+ with CUDA
- transformers >= 5.0
- datasets, sentencepiece, sacrebleu, accelerate
- matplotlib, pandas, scikit-learn
- NVIDIA GPU with 8GB+ VRAM
