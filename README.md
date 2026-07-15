# 🛡️ Comment Toxicity Detection

A deep learning system that classifies online comments across six toxicity categories in real time, built with a Bidirectional LSTM and deployed as an interactive Streamlit application.

🔗 **Live App:** [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://commenttoxicityclassifier-etxsuwtmlvpfmsxbgjfvgj.streamlit.app/)

## Problem Statement

Online platforms face a persistent challenge in moderating toxic content — harassment, hate speech, and offensive language — at scale. This project builds an automated multi-label classifier that flags toxic comments across six categories, enabling platforms, forums, and content moderation services to filter, warn, or review content proactively rather than reactively.

## Business Use Cases

- **Social media platforms** — real-time filtering of toxic comments
- **Online forums & communities** — automated moderation support
- **Content moderation services** — augmenting human review pipelines
- **Brand safety** — keeping ads away from toxic surrounding content
- **E-learning platforms** — safer comment sections for students/educators
- **News & media outlets** — moderating article comment sections

## Dataset

Trained on the Jigsaw Toxic Comment Classification dataset (~159K labeled comments) with six binary labels per comment:

| Label | Positive examples (train) |
|---|---|
| toxic | 15,294 |
| obscene | 8,449 |
| insult | 7,877 |
| severe_toxic | 1,595 |
| identity_hate | 1,405 |
| threat | 478 |

This is a **multi-label** problem — a single comment can belong to multiple categories simultaneously.

## Approach

### 1. Data Exploration & Preprocessing
- Explored label distribution and comment length
- Cleaned text: lowercased, removed URLs, bracketed text, punctuation, and non-alphabetic characters
- Tokenized and padded sequences to a fixed length (150 tokens) using Keras `Tokenizer`

### 2. Model Architecture

```
Embedding(20000, 128)
  → Bidirectional LSTM(64, return_sequences=True)
  → GlobalMaxPooling1D
  → Dense(64, relu)
  → Dropout(0.3)
  → Dense(6, sigmoid)
```

- **Embedding** learns dense word representations from the training vocabulary
- **Bidirectional LSTM** captures context from both directions in a comment
- **Sigmoid output layer** (not softmax) supports multi-label prediction — a comment can be flagged for more than one category at once
- Trained with `binary_crossentropy` loss and `adam` optimizer, with early stopping on validation loss

### 3. Evaluation

## Results

| Label | Precision | Recall | F1-score | AUC |
|---|---|---|---|---|
| toxic | 0.85 | 0.74 | 0.79 | 0.975 |
| obscene | 0.86 | 0.74 | 0.79 | 0.987 |
| insult | 0.75 | 0.63 | 0.68 | 0.980 |
| severe_toxic | 0.78 | 0.04 | 0.08 | 0.988 |
| identity_hate | 0.00 | 0.00 | 0.00 | 0.959 |
| threat | 0.00 | 0.00 | 0.00 | 0.954 |

*(Metrics reported for the positive class at the default 0.5 decision threshold, on a held-out validation split.)*

### Interpreting these results

The model performs well on categories with sufficient training examples (`toxic`, `obscene`, `insult`), achieving F1-scores of 0.68–0.79 and AUC above 0.97 across the board.

For `severe_toxic`, `identity_hate`, and `threat`, F1-scores are near zero **despite AUC scores of 0.95–0.99** — this gap is the key finding. High AUC means the model *is* learning to rank toxic comments higher than non-toxic ones for these categories; it simply isn't confident enough to cross the default 0.5 classification threshold, because these categories have extremely few positive examples (as few as 478 out of ~159K rows for `threat`). This is a well-documented behavior of simple LSTM baselines on this dataset, not a bug in this implementation.

**Future improvements** (beyond this project's one-week scope):
- Lower the decision threshold specifically for rare classes, guided by precision-recall curves
- Class-weighted or focal loss to counteract label imbalance
- Oversampling rare-class examples during training
- A transformer-based model (e.g. BERT) for stronger few-shot generalization on rare categories

## Streamlit Application

The deployed app provides:
- **Single comment analysis** — real-time toxicity scoring across all six categories with visual breakdown
- **Bulk CSV upload** — batch predictions with downloadable results
- **Model info panel** — architecture details and sample test cases

## Tech Stack

`Python` · `TensorFlow/Keras` · `scikit-learn` · `pandas` · `Streamlit` · `NLP` · `Bidirectional LSTM`

## Project Structure

```
├── app.py                  # Streamlit application
├── toxicity_model.h5       # Trained Bi-LSTM model
├── tokenizer.pkl           # Fitted Keras tokenizer
├── config.pkl              # Label columns + max sequence length
├── requirements.txt        # Dependencies
└── README.md
```

## How to Run Locally

```bash
git clone <your-repo-url>
cd comment-toxicity-detection
pip install -r requirements.txt
streamlit run app.py
```

## Workflow Summary

1. Data loaded and explored in Google Colab (via Google Drive)
2. Text cleaned and tokenized, sequences padded to uniform length
3. Bidirectional LSTM trained with early stopping on validation loss
4. Model evaluated per-label with precision/recall/F1/AUC
5. Model, tokenizer, and config saved as artifacts
6. Streamlit app built for real-time and bulk inference
7. Deployed to Streamlit Community Cloud; code version-controlled on GitHub

## Author

Akshaya Vinod Kumar
