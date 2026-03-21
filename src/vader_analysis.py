"""
vader_analysis.py
──────────────────
VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment analysis.
Rule-based, no training needed. Great for short social media text.

VADER Compound Score interpretation:
  >= 0.05  → Positive
  <= -0.05 → Negative
  else     → Neutral

Run: python src/vader_analysis.py
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, 'data', 'raw', 'reviews.csv')
OUT_CSV    = os.path.join(BASE_DIR, 'data', 'processed', 'vader_results.csv')
FIG_DIR    = os.path.join(BASE_DIR, 'outputs', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ── Analyzer ─────────────────────────────────────────────────────────────────
analyzer = SentimentIntensityAnalyzer()


def get_vader_scores(text: str) -> dict:
    """Return VADER polarity scores for a single text."""
    if not isinstance(text, str) or len(text.strip()) == 0:
        return {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    return analyzer.polarity_scores(text)


def classify_sentiment(compound_score: float) -> str:
    """Classify sentiment based on VADER compound score."""
    if compound_score >= 0.05:
        return 'positive'
    elif compound_score <= -0.05:
        return 'negative'
    else:
        return 'neutral'


def analyze_dataframe(df: pd.DataFrame, text_col: str = 'review_text') -> pd.DataFrame:
    """
    Apply VADER sentiment analysis to an entire DataFrame.
    Adds: vader_neg, vader_neu, vader_pos, vader_compound, vader_sentiment
    """
    print(f"  Running VADER on {len(df)} reviews...")
    scores = df[text_col].apply(get_vader_scores)
    df = df.copy()
    df['vader_neg']       = scores.apply(lambda x: x['neg'])
    df['vader_neu']       = scores.apply(lambda x: x['neu'])
    df['vader_pos']       = scores.apply(lambda x: x['pos'])
    df['vader_compound']  = scores.apply(lambda x: x['compound'])
    df['vader_sentiment'] = df['vader_compound'].apply(classify_sentiment)
    print("  Done!")
    return df


def compute_metrics(df: pd.DataFrame) -> dict:
    """Compute accuracy and per-class metrics vs ground truth 'sentiment' column."""
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    y_true = df['sentiment']
    y_pred = df['vader_sentiment']
    acc    = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    cm     = confusion_matrix(y_true, y_pred, labels=['positive', 'negative', 'neutral'])
    return {'accuracy': acc, 'report': report, 'confusion_matrix': cm}


def plot_sentiment_distribution(df: pd.DataFrame, save: bool = True):
    """Bar chart of VADER sentiment distribution vs ground truth."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("VADER Sentiment Analysis Results", fontsize=15, fontweight='bold')

    colors = {'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#f39c12'}

    # Ground Truth
    gt_counts = df['sentiment'].value_counts()
    axes[0].bar(gt_counts.index, gt_counts.values,
                color=[colors.get(x, '#7f8c8d') for x in gt_counts.index])
    axes[0].set_title('Ground Truth Distribution', fontsize=12)
    axes[0].set_xlabel('Sentiment')
    axes[0].set_ylabel('Count')
    for i, (idx, val) in enumerate(gt_counts.items()):
        axes[0].text(i, val + 5, str(val), ha='center', fontweight='bold')

    # VADER Predictions
    vd_counts = df['vader_sentiment'].value_counts()
    axes[1].bar(vd_counts.index, vd_counts.values,
                color=[colors.get(x, '#7f8c8d') for x in vd_counts.index])
    axes[1].set_title('VADER Predicted Distribution', fontsize=12)
    axes[1].set_xlabel('Sentiment')
    axes[1].set_ylabel('Count')
    for i, (idx, val) in enumerate(vd_counts.items()):
        axes[1].text(i, val + 5, str(val), ha='center', fontweight='bold')

    plt.tight_layout()
    if save:
        path = os.path.join(FIG_DIR, 'vader_distribution.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {path}")
    plt.close()


def plot_compound_score_dist(df: pd.DataFrame, save: bool = True):
    """Distribution of VADER compound scores by true sentiment."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#f39c12'}
    for label in ['positive', 'negative', 'neutral']:
        subset = df[df['sentiment'] == label]['vader_compound']
        ax.hist(subset, bins=30, alpha=0.6, label=label.capitalize(),
                color=colors[label], edgecolor='white')
    ax.axvline(0.05, color='green', linestyle='--', linewidth=1.5, label='Positive threshold (0.05)')
    ax.axvline(-0.05, color='red', linestyle='--', linewidth=1.5, label='Negative threshold (-0.05)')
    ax.set_title('VADER Compound Score Distribution by True Sentiment', fontsize=13, fontweight='bold')
    ax.set_xlabel('Compound Score')
    ax.set_ylabel('Frequency')
    ax.legend()
    plt.tight_layout()
    if save:
        path = os.path.join(FIG_DIR, 'vader_compound_dist.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {path}")
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, labels: list, save: bool = True):
    """Heatmap of confusion matrix."""
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title('VADER Confusion Matrix', fontsize=13, fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.tight_layout()
    if save:
        path = os.path.join(FIG_DIR, 'vader_confusion_matrix.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {path}")
    plt.close()


def plot_platform_sentiment(df: pd.DataFrame, save: bool = True):
    """Stacked bar chart of VADER sentiment by platform."""
    pivot = df.groupby(['platform', 'vader_sentiment']).size().unstack(fill_value=0)
    colors = {'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#f39c12'}
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(9, 5))
    bottom = np.zeros(len(pivot_pct))
    for sentiment in ['positive', 'neutral', 'negative']:
        if sentiment in pivot_pct.columns:
            bars = ax.bar(pivot_pct.index, pivot_pct[sentiment],
                          bottom=bottom, color=colors[sentiment],
                          label=sentiment.capitalize(), alpha=0.85)
            for bar, val in zip(bars, pivot_pct[sentiment]):
                if val > 5:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_y() + bar.get_height() / 2,
                            f"{val:.1f}%", ha='center', va='center',
                            fontsize=8, color='white', fontweight='bold')
            bottom += pivot_pct[sentiment].values

    ax.set_title('VADER Sentiment Distribution by Platform', fontsize=13, fontweight='bold')
    ax.set_xlabel('Platform')
    ax.set_ylabel('Percentage (%)')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 115)
    plt.tight_layout()
    if save:
        path = os.path.join(FIG_DIR, 'vader_platform_sentiment.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {path}")
    plt.close()


if __name__ == "__main__":
    # ── Load data ──
    if not os.path.exists(DATA_PATH):
        print("[ERR] Dataset not found. Run: python data/raw/generate_dataset.py")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    print(f"[OK] Loaded {len(df)} reviews from {DATA_PATH}")

    # ── Analyze ──
    print("\n[?] Running VADER Sentiment Analysis...")
    df = analyze_dataframe(df)

    # ── Save results ──
    df.to_csv(OUT_CSV, index=False)
    print(f"\n[OK] Results saved to: {OUT_CSV}")

    # ── Metrics ──
    print("\n[CHART] Computing Metrics...")
    metrics = compute_metrics(df)
    print(f"\n   Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    from sklearn.metrics import classification_report
    print("\n   Classification Report:")
    print(classification_report(df['sentiment'], df['vader_sentiment']))

    # ── Plots ──
    print("\n[ART] Generating plots...")
    plot_sentiment_distribution(df)
    plot_compound_score_dist(df)
    plot_confusion_matrix(metrics['confusion_matrix'],
                          labels=['positive', 'negative', 'neutral'])
    plot_platform_sentiment(df)
    print("\n[OK] All VADER analysis complete!")
