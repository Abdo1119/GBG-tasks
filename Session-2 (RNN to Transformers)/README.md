# Session 2 — Sentiment Analysis (RNN to Transformers)

A sentiment analysis pipeline using a pre-trained **RoBERTa** transformer model, evaluated on a real tweet dataset.

## Model

[`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) — RoBERTa fine-tuned on ~124M tweets for 3-class sentiment classification.

## Dataset

[`tweet_eval`](https://huggingface.co/datasets/tweet_eval) — sentiment subset with 3 labels:
- **negative** (0)
- **neutral** (1)
- **positive** (2)

| Split | Size |
|-------|------|
| Train | 45,615 |
| Val | 2,000 |
| Test | 12,284 |

The script samples **200 examples** from the test split for evaluation.

## Output Modes

### `label_only`
```json
{
  "label": "positive",
  "confidence": 0.94,
  "reliability": 0.99
}
```

### `reasoning`
```json
{
  "label": "positive",
  "confidence": 0.94,
  "reliability": 0.99,
  "rationale": "Model probabilities — negative: 0.5%, neutral: 5.0%, positive: 94.5%."
}
```

**Definitions:**
- `confidence` — probability of the chosen label given the text (from softmax)
- `reliability` — stability of the decision based on margin between top-2 classes; lower with ambiguity, mixed sentiment, or short text

## Results

| Metric | Value |
|--------|-------|
| **Accuracy** | 68.5% |
| **Macro F1** | 0.681 |

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| negative | 0.701 | 0.730 | 0.715 |
| neutral | 0.706 | 0.638 | 0.670 |
| positive | 0.605 | 0.719 | 0.657 |

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python sentiment_analysis.py
```

This will:
1. Download the model and dataset (first run only)
2. Classify 200 test samples in both modes
3. Export results to `output/`:
   - `predictions_label_only.json`
   - `predictions_reasoning.json`
   - `metrics.json`

## Project Structure

```
Session-2 (RNN to Transformers)/
├── sentiment_analysis.py      # Main script
├── requirements.txt           # Dependencies
├── README.md                  # This file
└── output/
    ├── predictions_label_only.json
    ├── predictions_reasoning.json
    └── metrics.json
```

## Colab Notebook

The original notebook (`sentiment_analysis.ipynb`) can be run directly on [Google Colab](https://colab.research.google.com) with no setup required.
