"""
preprocess.py
─────────────
Text cleaning and preprocessing utilities for sentiment analysis.
Includes: cleaning, tokenization, lemmatization, stopword removal,
and corpus building for LDA topic modeling.
"""

import re
import string
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data (only on first run)
def download_nltk_resources():
    resources = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'punkt_tab']
    for r in resources:
        try:
            nltk.download(r, quiet=True)
        except Exception:
            pass

download_nltk_resources()

STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

# Additional domain-specific stopwords to remove
EXTRA_STOPWORDS = {
    'product', 'item', 'order', 'delivery', 'would', 'also', 'got',
    'one', 'get', 'use', 'used', 'well', 'really', 'even', 'much',
    'good', 'great', 'bad', 'like', 'nice', 'buy', 'bought'
}
STOP_WORDS.update(EXTRA_STOPWORDS)


def clean_text(text: str) -> str:
    """
    Clean raw review text:
    - Lowercase
    - Remove URLs, mentions, hashtags
    - Remove emojis and special characters
    - Remove punctuation and extra spaces
    """
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove @mentions and #hashtags
    text = re.sub(r'@\w+|#\w+', '', text)

    # Remove emojis and non-ASCII
    text = text.encode('ascii', 'ignore').decode('ascii')

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove digits
    text = re.sub(r'\d+', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def tokenize_and_lemmatize(text: str, remove_stopwords: bool = True) -> list:
    """
    Tokenize text and lemmatize each token.
    Optionally removes stopwords.
    Returns list of clean tokens.
    """
    if not text:
        return []

    tokens = word_tokenize(text)

    # Filter tokens: keep only alphabetic tokens of length > 2
    tokens = [t for t in tokens if t.isalpha() and len(t) > 2]

    # Remove stopwords
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS]

    # Lemmatize
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]

    return tokens


def preprocess_dataframe(df: pd.DataFrame, text_col: str = 'review_text') -> pd.DataFrame:
    """
    Apply full preprocessing pipeline to a DataFrame.
    Adds columns: 'cleaned_text', 'tokens', 'token_count'
    """
    print(f"  Preprocessing {len(df)} reviews...")
    df = df.copy()
    df['cleaned_text'] = df[text_col].apply(clean_text)
    df['tokens'] = df['cleaned_text'].apply(tokenize_and_lemmatize)
    df['token_count'] = df['tokens'].apply(len)
    df['text_length'] = df[text_col].apply(lambda x: len(str(x)))
    print(f"  Done! Avg tokens per review: {df['token_count'].mean():.1f}")
    return df


def build_corpus(tokens_series):
    """
    Build a Gensim-compatible corpus for LDA topic modeling.
    Returns: (dictionary, corpus)
    """
    from gensim import corpora

    # Filter out very short token lists
    tokens_list = [t for t in tokens_series if len(t) >= 2]

    dictionary = corpora.Dictionary(tokens_list)

    # Filter extreme tokens (appear in <2 or >80% of docs)
    dictionary.filter_extremes(no_below=2, no_above=0.8)

    corpus = [dictionary.doc2bow(tokens) for tokens in tokens_list]
    print(f"  Corpus built: {len(dictionary)} unique tokens, {len(corpus)} documents")
    return dictionary, corpus


if __name__ == "__main__":
    import os
    import sys

    # Load dataset
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'reviews.csv')
    if not os.path.exists(data_path):
        print("[ERR] Dataset not found. Run: python data/raw/generate_dataset.py")
        sys.exit(1)

    df = pd.read_csv(data_path)
    print(f"[OK] Loaded {len(df)} reviews")

    # Preprocess
    df_processed = preprocess_dataframe(df)

    # Save
    out_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'reviews_processed.csv')
    df_processed.to_csv(out_path, index=False)
    print(f"[OK] Saved processed data to: {out_path}")

    # Sample output
    print("\nSample processed reviews:")
    for _, row in df_processed.head(3).iterrows():
        print(f"  Original : {row['review_text'][:80]}...")
        print(f"  Cleaned  : {row['cleaned_text'][:80]}")
        print(f"  Tokens   : {row['tokens'][:10]}")
        print()
