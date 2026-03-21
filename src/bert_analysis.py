"""
bert_analysis.py
─────────────────
BERT-based sentiment analysis using HuggingFace Transformers.
Model: cardiffnlp/twitter-roberta-base-sentiment
  → Labels: LABEL_0 = Negative, LABEL_1 = Neutral, LABEL_2 = Positive

First run downloads ~500MB model from HuggingFace Hub.

Run: python src/bert_analysis.py
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'reviews.csv')
OUT_CSV   = os.path.join(BASE_DIR, 'data', 'processed', 'bert_results.csv')
FIG_DIR   = os.path.join(BASE_DIR, 'outputs', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ── Model Config ─────────────────────────────────────────────────────────────
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
LABEL_MAP  = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}
MAX_LENGTH = 128   # max tokens (reviews are short)
BATCH_SIZE = 16    # adjust if CUDA OOM


def load_pipeline():
    """Load the HuggingFace sentiment pipeline."""
    device = 0 if torch.cuda.is_available() else -1
    device_name = "GPU (CUDA)" if device == 0 else "CPU"
    print(f"  Loading BERT model on {device_name}...")
    print(f"  Model: {MODEL_NAME}")
    print(f"  (First run will download ~500MB — please wait...)")

    sentiment_pipeline = pipeline(
        task="sentiment-analysis",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        device=device,
        max_length=MAX_LENGTH,
        truncation=True,
        batch_size=BATCH_SIZE,
    )
    print("  Model loaded successfully!")
    return sentiment_pipeline


def predict_batch(texts: list, pipe) -> list:
    """Run inference on a list of texts, return list of dicts."""
    # Replace None/NaN with empty string
    texts = [str(t) if t and str(t).strip() else "no review" for t in texts]
    results = pipe(texts)
    return results


def analyze_dataframe(df: pd.DataFrame, text_col: str = 'review_text') -> pd.DataFrame:
    """
    Run BERT sentiment on every row.
    Adds: bert_label, bert_score, bert_sentiment
    """
    pipe = load_pipeline()
    df   = df.copy()
    texts = df[text_col].tolist()

    print(f"\n  Analyzing {len(texts)} reviews in batches of {BATCH_SIZE}...")
    results = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        preds = predict_batch(batch, pipe)
        results.extend(preds)
        if (i // BATCH_SIZE + 1) % 10 == 0:
            print(f"    Processed {min(i + BATCH_SIZE, len(texts))}/{len(texts)} reviews...")

    df['bert_label']     = [r['label'] for r in results]
    df['bert_score']     = [round(r['score'], 4) for r in results]
    df['bert_sentiment'] = df['bert_label'].map(LABEL_MAP)
    print("  BERT inference complete!")
    return df


def compute_metrics(df: pd.DataFrame) -> dict:
    """Compare BERT predictions against ground truth labels."""
    from sklearn.metrics import (
        classification_report, confusion_matrix, accuracy_score
    )
    y_true = df['sentiment']
    y_pred = df['bert_sentiment']
    acc    = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    cm     = confusion_matrix(y_true, y_pred,
                              labels=['positive', 'negative', 'neutral'])
    return {'accuracy': acc, 'report': report, 'confusion_matrix': cm}


def plot_bert_vs_vader(df: pd.DataFrame, save: bool = True):
    """Side-by-side comparison: BERT vs VADER vs Ground Truth."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Sentiment Model Comparison', fontsize=14, fontweight='bold')
    colors = {'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#f39c12'}

    for ax, col, title in zip(
        axes,
        ['sentiment', 'vader_sentiment', 'bert_sentiment'],
        ['Ground Truth', 'VADER Predictions', 'BERT Predictions']
    ):
        if col not in df.columns:
            ax.set_title(f'{title}\n(not available)')
            continue
        counts = df[col].value_counts()
        ax.bar(counts.index, counts.values,
               color=[colors.get(x, '#95a5a6') for x in counts.index])
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Sentiment')
        ax.set_ylabel('Count')
        for i, (_, v) in enumerate(counts.items()):
            ax.text(i, v + 3, str(v), ha='center', fontweight='bold', fontsize=9)

    plt.tight_layout()
    if save:
        path = os.path.join(FIG_DIR, 'bert_vs_vader_comparison.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {path}")
    plt.close()


def plot_bert_confusion_matrix(cm: np.ndarray, labels: list, save: bool = True):
    """Heatmap for BERT confusion matrix."""
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title('BERT Confusion Matrix', fontsize=13, fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.tight_layout()
    if save:
        path = os.path.join(FIG_DIR, 'bert_confusion_matrix.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {path}")
    plt.close()


def plot_confidence_distribution(df: pd.DataFrame, save: bool = True):
    """Distribution of BERT confidence scores by predicted sentiment."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#f39c12'}
    for label in ['positive', 'negative', 'neutral']:
        subset = df[df['bert_sentiment'] == label]['bert_score']
        if not subset.empty:
            ax.hist(subset, bins=20, alpha=0.6, label=label.capitalize(),
                    color=colors[label], edgecolor='white')
    ax.set_title('BERT Confidence Score Distribution', fontsize=13, fontweight='bold')
    ax.set_xlabel('Confidence Score')
    ax.set_ylabel('Frequency')
    ax.legend()
    plt.tight_layout()
    if save:
        path = os.path.join(FIG_DIR, 'bert_confidence_dist.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {path}")
    plt.close()


def plot_rating_vs_sentiment(df: pd.DataFrame, save: bool = True):
    """Heatmap: star rating vs BERT predicted sentiment."""
    pivot = pd.crosstab(df['rating'], df['bert_sentiment'])
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt='d', cmap='YlOrRd', ax=ax)
    ax.set_title('Star Rating vs BERT Sentiment', fontsize=13, fontweight='bold')
    ax.set_xlabel('Predicted Sentiment')
    ax.set_ylabel('Star Rating')
    plt.tight_layout()
    if save:
        path = os.path.join(FIG_DIR, 'rating_vs_bert_sentiment.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {path}")
    plt.close()


if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        print("[ERR] Dataset not found. Run: python data/raw/generate_dataset.py")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    print(f"[OK] Loaded {len(df)} reviews")

    # Load VADER results if available (for comparison)
    vader_path = os.path.join(BASE_DIR, 'data', 'processed', 'vader_results.csv')
    if os.path.exists(vader_path):
        vader_df = pd.read_csv(vader_path)[['review_id', 'vader_sentiment', 'vader_compound']]
        df = df.merge(vader_df, on='review_id', how='left')
        print("  Merged VADER results for comparison")

    # ── Run BERT ──
    print("\n[BOT] Running BERT Sentiment Analysis...")
    df = analyze_dataframe(df)

    # ── Save ──
    df.to_csv(OUT_CSV, index=False)
    print(f"\n[OK] Results saved to: {OUT_CSV}")

    # ── Metrics ──
    print("\n[CHART] Computing Metrics...")
    metrics = compute_metrics(df)
    print(f"   BERT Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    from sklearn.metrics import classification_report
    print("\n   Classification Report:")
    print(classification_report(df['sentiment'], df['bert_sentiment']))

    # ── Plots ──
    print("\n[ART] Generating BERT plots...")
    plot_bert_vs_vader(df)
    plot_bert_confusion_matrix(metrics['confusion_matrix'],
                               labels=['positive', 'negative', 'neutral'])
    plot_confidence_distribution(df)
    plot_rating_vs_sentiment(df)
    print("\n[OK] All BERT analysis complete!")
