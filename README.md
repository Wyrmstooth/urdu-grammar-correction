# Urdu Grammar Error Correction (GEC)

Fine-tuned **google/mt5-small** for automatic grammar error correction in Urdu text.

## Model Details

- **Base Model**: mT5-small (300M parameters)
- **Task**: Seq2seq Urdu grammar correction
- **Dataset**: ~7,500 curated Urdu sentences with grammar errors
- **Accuracy**: ~81% BLEU on test set

## Error Types Handled

| Error Type | Description | Example |
|-----------|-------------|---------|
| Word Order | Incorrect sentence structure | میں کل بازار گیا تھا سبزیاں خریدنے کے لیے → میں کل سبزیاں خریدنے کے لیے بازار گیا تھا |
| Grammar | Verb gender/number agreement | وہ لڑکی بہت اچھا گاتی ہیں → وہ لڑکی بہت اچھا گاتی ہے |
| Spelling | Typographical errors | اس نے مجے فون کیا → اس نے مجھے فون کیا |
| Extra Words | Redundant words | وہ گھر گیا اور اور پھر واپس آیا → وہ گھر گیا اور واپس آیا |
| Missing Words | Missing case markers/articles | میں نے بازار سبزی خریدی → میں نے بازار میں سبزی خریدی |

## Project Structure

```
.
├── GenAI-Dataset.json       # Raw dataset (~7,500 entries)
├── 1_data_preparation.py    # Load, split, analyze dataset
├── 2_training.py           # Fine-tune mT5-small
├── 3_evaluation.py         # BLEU evaluation & inference demos
├── app.py                  # Gradio web UI
├── clean_dataset.py         # Dataset deduplication/cleaning
├── check_dataset.py         # Dataset validation
├── urdu_gec_model/         # Trained model checkpoints
│   └── final/              # Best model (mT5-small fine-tuned)
└── urdu_gec_dataset_hf/     # HuggingFace format dataset
```

## Setup

```bash
# Create conda environment
conda create -n urdu-gec python=3.10
conda activate urdu-gec

# Install dependencies
pip install transformers datasets torch sacrebleu matplotlib pandas numpy gradio
```

## Usage

### 1. Prepare Dataset
```bash
python 1_data_preparation.py
```
Outputs: `train.csv`, `val.csv`, `test.csv`, `urdu_gec_dataset_hf/`

### 2. Train Model
```bash
python 2_training.py
```
Trained model saved to: `urdu_gec_model/final/`

**Options:**
- `--epochs`: Number of training epochs (default: 5)
- `--batch-size`: Per-device batch size (default: 8)
- `--save-name`: Model save folder name (default: final)

### 3. Evaluate Model
```bash
python 3_evaluation.py
```
Runs BLEU evaluation, displays sample corrections, generates evaluation plots.

**Options:**
- `--model`: Path to model directory (default: urdu_gec_model/final)

### 4. Launch Web UI
```bash
python app.py
```
Opens at `http://127.0.0.1:7862`

**Options:**
- `--model`: Path to model (default: urdu_gec_model/final)
- `--port`: Server port (default: 7862)

## Sample Output

```
Input:  میں کل بازار گیا تھا سبزیاں خریدنے کے لیے
Output: میں کل سبزیاں خریدنے کے لیے بازار گیا تھا

Input:  اس نے مجے فون کیا
Output: اس نے مجھے فون کیا

Input:  کل رات بجلی چلی گئ
Output: کل رات بجلی چلی گئی
```

## Hardware Requirements

- **GPU**: NVIDIA RTX 5060 Laptop GPU (8.5 GB VRAM)
- **RAM**: 16 GB+
- **Storage**: ~2 GB for model + dataset

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Model | google/mt5-small |
| Max Input Length | 256 tokens |
| Max Target Length | 256 tokens |
| Batch Size | 8 per device |
| Gradient Accumulation | 2 steps |
| Effective Batch Size | 16 |
| Learning Rate | 2e-4 |
| Epochs | 5 |
| Warmup Steps | 500 |
| Precision | bf16 |

## BLEU Scores (Original Model)

| Metric | Score |
|---------|-------|
| Overall BLEU | 80.72 |
| Extra Words | 91.09 |
| Grammar | 84.96 |
| Missing Words | 75.76 |
| Spelling | 73.51 |
| Word Order | 67.34 |

## License

MIT License

## Author

Urdu Grammar Correction Project - Built with mT5-small and Transformers
