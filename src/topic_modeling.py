"""
topic_modeling.py
──────────────────
LDA Topic Modeling with Gensim and interactive pyLDAvis visualization.
Discovers latent themes/topics in review text (e.g., food quality, delivery, pricing).

Run: python src/topic_modeling.py
Output: outputs/reports/lda_topics.html  ← open in browser
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import gensim
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE_DIR, 'data', 'raw', 'reviews.csv')
PROC_PATH   = os.path.join(BASE_DIR, 'data', 'processed', 'reviews_processed.csv')
FIG_DIR     = os.path.join(BASE_DIR, 'outputs', 'figures')
REPORT_DIR  = os.path.join(BASE_DIR, 'outputs', 'reports')
MODEL_DIR   = os.path.join(BASE_DIR, 'data', 'processed', 'lda_model')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ── LDA Config ────────────────────────────────────────────────────────────────
NUM_TOPICS  = 6       # number of topics to discover
PASSES      = 15      # training passes
RANDOM_SEED = 42


def load_and_preprocess(data_path: str) -> list:
    """
    Load reviews and tokenize them for LDA.
    Returns a list of token lists.
    """
    import sys
    sys.path.insert(0, BASE_DIR)
    from src.preprocess import preprocess_dataframe

    df = pd.read_csv(data_path)
    print(f"  Loaded {len(df)} reviews")

    # Try loading preprocessed data first
    if os.path.exists(PROC_PATH):
        print("  Loading pre-processed tokens from cache...")
        df_proc = pd.read_csv(PROC_PATH)
        # Convert string representation of list back to list
        import ast
        tokens_list = df_proc['tokens'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else []
        ).tolist()
    else:
        print("  Preprocessing text (this may take a moment)...")
        df_proc = preprocess_dataframe(df)
        tokens_list = df_proc['tokens'].tolist()

    # Filter very short token lists
    tokens_list = [t for t in tokens_list if len(t) >= 3]
    print(f"  Ready: {len(tokens_list)} documents for topic modeling")
    return tokens_list, df


def build_lda_model(tokens_list: list, num_topics: int = NUM_TOPICS):
    """
    Build LDA dictionary, corpus, and train the model.
    Returns: (dictionary, corpus, model)
    """
    print(f"\n  Building dictionary & corpus...")
    dictionary = corpora.Dictionary(tokens_list)

    # Filter rare/common tokens
    dictionary.filter_extremes(no_below=3, no_above=0.75)
    print(f"  Dictionary: {len(dictionary)} unique tokens")

    corpus = [dictionary.doc2bow(tokens) for tokens in tokens_list]
    print(f"  Corpus: {len(corpus)} documents")

    print(f"\n  Training LDA model ({num_topics} topics, {PASSES} passes)...")
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=PASSES,
        alpha='auto',
        eta='auto',
        random_state=RANDOM_SEED,
        update_every=1,
        chunksize=100,
        per_word_topics=True,
    )
    print("  LDA model trained!")
    return dictionary, corpus, lda_model


def compute_coherence(lda_model, tokens_list: list, dictionary) -> float:
    """Compute Cv coherence score for topic quality evaluation."""
    coherence_model = CoherenceModel(
        model=lda_model,
        texts=tokens_list,
        dictionary=dictionary,
        coherence='c_v'
    )
    score = coherence_model.get_coherence()
    return score


def find_optimal_topics(tokens_list: list, dictionary,
                        corpus, topic_range: range) -> pd.DataFrame:
    """
    Train multiple LDA models with different topic counts,
    return coherence scores for each.
    """
    print(f"\n  Finding optimal topic count ({list(topic_range)})...")
    results = []
    for k in topic_range:
        model = LdaModel(
            corpus=corpus, id2word=dictionary,
            num_topics=k, passes=10,
            random_state=RANDOM_SEED,
            alpha='auto', eta='auto',
        )
        score = compute_coherence(model, tokens_list, dictionary)
        results.append({'num_topics': k, 'coherence_score': score})
        print(f"    Topics={k}: Coherence={score:.4f}")

    return pd.DataFrame(results)


def get_topic_labels(lda_model, num_words: int = 8) -> dict:
    """
    Get top words for each topic and assign human-readable labels.
    """
    topics = {}
    for i in range(lda_model.num_topics):
        words = [word for word, _ in lda_model.show_topic(i, topn=num_words)]
        topics[i] = {
            'words': words,
            'label': f"Topic {i+1}",
        }

    # Auto-label based on top words (heuristic)
    LABEL_KEYWORDS = {
        'Food Quality':    ['taste', 'food', 'fresh', 'flavor', 'delicious', 'hot', 'spicy', 'cook'],
        'Delivery':        ['delivery', 'time', 'fast', 'late', 'arrive', 'hour', 'quick', 'delayed'],
        'Product Quality': ['quality', 'build', 'material', 'durable', 'break', 'cheap', 'premium', 'solid'],
        'Customer Service':['service', 'support', 'refund', 'return', 'response', 'complain', 'contact'],
        'Pricing & Value': ['price', 'value', 'cost', 'expensive', 'worth', 'money', 'cheap', 'affordable'],
        'Packaging':       ['pack', 'box', 'wrap', 'damage', 'seal', 'condition', 'broken', 'torn'],
    }

    for i, data in topics.items():
        words_set = set(data['words'])
        best_label, best_overlap = None, 0
        for label, keywords in LABEL_KEYWORDS.items():
            overlap = len(words_set & set(keywords))
            if overlap > best_overlap:
                best_overlap, best_label = overlap, label
        if best_label and best_overlap > 0:
            topics[i]['label'] = best_label

    return topics


def plot_top_words_per_topic(lda_model, topic_labels: dict, save: bool = True):
    """Bar chart of top words for each topic."""
    n_topics = lda_model.num_topics
    cols = 3
    rows = (n_topics + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 4))
    axes = axes.flatten()

    palette = plt.cm.Set2(np.linspace(0, 1, n_topics))

    for i in range(n_topics):
        top_words = lda_model.show_topic(i, topn=10)
        words, weights = zip(*top_words)
        ax = axes[i]
        bars = ax.barh(list(words)[::-1], list(weights)[::-1],
                       color=palette[i], alpha=0.85)
        ax.set_title(f"Topic {i+1}: {topic_labels[i]['label']}", fontsize=11, fontweight='bold')
        ax.set_xlabel('Weight')
        ax.tick_params(axis='y', labelsize=9)
        for bar, weight in zip(bars, list(weights)[::-1]):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                    f'{weight:.3f}', va='center', fontsize=7)

    # Hide unused axes
    for j in range(n_topics, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('LDA Topic Model — Top Words per Topic', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    if save:
        path = os.path.join(FIG_DIR, 'lda_top_words.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {path}")
    plt.close()


def plot_coherence_scores(coherence_df: pd.DataFrame, save: bool = True):
    """Line chart of coherence score vs number of topics."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(coherence_df['num_topics'], coherence_df['coherence_score'],
            'bo-', linewidth=2, markersize=7)
    best_idx = coherence_df['coherence_score'].idxmax()
    best_k   = coherence_df.loc[best_idx, 'num_topics']
    best_c   = coherence_df.loc[best_idx, 'coherence_score']
    ax.axvline(best_k, color='red', linestyle='--', alpha=0.7,
               label=f'Best k={best_k} (C={best_c:.4f})')
    ax.set_title('Coherence Score vs Number of Topics', fontsize=13, fontweight='bold')
    ax.set_xlabel('Number of Topics')
    ax.set_ylabel('Coherence Score (Cv)')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save:
        path = os.path.join(FIG_DIR, 'lda_coherence_scores.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {path}")
    plt.close()


def plot_topic_distribution(lda_model, corpus, topic_labels: dict, save: bool = True):
    """Pie chart of dominant topic distribution across corpus."""
    dominant_topics = []
    for doc_bow in corpus:
        topic_probs = lda_model.get_document_topics(doc_bow, minimum_probability=0.0)
        dominant = max(topic_probs, key=lambda x: x[1])[0]
        dominant_topics.append(dominant)

    counts = pd.Series(dominant_topics).value_counts().sort_index()
    labels = [topic_labels[i]['label'] for i in counts.index]

    fig, ax = plt.subplots(figsize=(9, 6))
    palette = plt.cm.Set2(np.linspace(0, 1, len(counts)))
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=labels, autopct='%1.1f%%',
        colors=palette, startangle=140, pctdistance=0.75,
        wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2)
    )
    for text in autotexts:
        text.set_fontsize(9)
    ax.set_title('Dominant Topic Distribution Across Reviews', fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save:
        path = os.path.join(FIG_DIR, 'lda_topic_distribution.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {path}")
    plt.close()


def export_pyldavis(lda_model, corpus, dictionary, save: bool = True) -> str:
    """Export interactive pyLDAvis HTML visualization."""
    print("  Preparing pyLDAvis visualization...")
    vis_data = gensimvis.prepare(lda_model, corpus, dictionary, sort_topics=False)
    html_path = os.path.join(REPORT_DIR, 'lda_topics.html')
    pyLDAvis.save_html(vis_data, html_path)
    print(f"  Saved interactive visualization: {html_path}")
    return html_path


def print_topics_table(lda_model, topic_labels: dict):
    """Pretty-print topics table in terminal."""
    print("\n" + "═" * 65)
    print(f"{'DISCOVERED TOPICS':^65}")
    print("═" * 65)
    for i in range(lda_model.num_topics):
        label = topic_labels[i]['label']
        words = ', '.join(topic_labels[i]['words'][:6])
        print(f"  Topic {i+1:2d} | {label:<22} | {words}")
    print("═" * 65)


if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        print("[ERR] Dataset not found. Run: python data/raw/generate_dataset.py")
        sys.exit(1)

    # ── Load & Preprocess ──
    print("\n🔤 Loading and preprocessing text...")
    tokens_list, df = load_and_preprocess(DATA_PATH)

    # ── Build Corpus ──
    dictionary = corpora.Dictionary(tokens_list)
    dictionary.filter_extremes(no_below=3, no_above=0.75)
    corpus = [dictionary.doc2bow(t) for t in tokens_list]

    # ── Optional: Find optimal topic count ──
    print("\n[?] Finding optimal number of topics (k=3 to 8)...")
    coherence_df = find_optimal_topics(tokens_list, dictionary, corpus, range(3, 9))
    plot_coherence_scores(coherence_df)

    # ── Train Final Model ──
    best_k = coherence_df.loc[coherence_df['coherence_score'].idxmax(), 'num_topics']
    print(f"\n[BEST] Best topic count: k={best_k}")

    _, _, lda_model = build_lda_model(tokens_list, num_topics=int(best_k))

    # ── Compute Final Coherence ──
    final_coherence = compute_coherence(lda_model, tokens_list, dictionary)
    print(f"\n  Final Coherence Score (Cv): {final_coherence:.4f}")

    # ── Get Topic Labels ──
    topic_labels = get_topic_labels(lda_model)
    print_topics_table(lda_model, topic_labels)

    # ── Save Model ──
    lda_model.save(os.path.join(MODEL_DIR, 'lda_model'))
    dictionary.save(os.path.join(MODEL_DIR, 'dictionary'))
    print(f"\n[OK] Model saved to: {MODEL_DIR}")

    # ── Plots ──
    print("\n[ART] Generating topic visualizations...")
    plot_top_words_per_topic(lda_model, topic_labels)
    plot_topic_distribution(lda_model, corpus, topic_labels)

    # ── pyLDAvis HTML ──
    print("\n[WEB] Exporting interactive pyLDAvis HTML...")
    html_path = export_pyldavis(lda_model, corpus, dictionary)

    print(f"\n[OK] Topic Modeling complete!")
    print(f"   Open: {html_path}")
