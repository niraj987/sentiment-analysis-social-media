# 🔍 Sentiment Analysis on Social Media / Reviews

> **Portfolio Project** | Data Science | NLP | Marketing & E-commerce

Real-time sentiment analysis system for brand feedback on **Amazon**, **Zomato**, and **Swiggy** reviews. Demonstrates end-to-end unstructured data mastery using rule-based (VADER), deep learning (BERT), and unsupervised (LDA) NLP techniques.

---

## 🚀 Features

| Feature | Description |
|---|---|
| 📊 **EDA** | Review length, rating distribution, top n-grams, word clouds |
| 😊 **VADER** | Rule-based sentiment, instant, no training needed |
| 🤖 **BERT** | HuggingFace `twitter-roberta-base-sentiment`, ~500MB download |
| 📌 **LDA** | Topic modeling (Gensim), coherence evaluation, pyLDAvis HTML |
| 🖥️ **Dashboard** | Streamlit app — live demo + full analytics |

---

## 📁 Project Structure

```
SentimentAnalysis/
├── data/
│   ├── raw/          ← reviews.csv (1200 synthetic reviews)
│   └── processed/    ← cleaned data, model outputs
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Sentiment_Analysis.ipynb
│   └── 03_Topic_Modeling.ipynb
├── src/
│   ├── preprocess.py         ← text cleaning pipeline
│   ├── vader_analysis.py     ← VADER sentiment
│   ├── bert_analysis.py      ← BERT deep learning
│   └── topic_modeling.py     ← LDA topic modeling
├── app/
│   └── dashboard.py          ← Streamlit dashboard
├── outputs/
│   ├── figures/              ← auto-generated charts
│   └── reports/              ← interactive LDA HTML
├── run_all.py                ← one-click quick start
└── requirements.txt
```

---

## ⚡ Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

> **Note:** BERT requires PyTorch. On CPU this works fine. For GPU, install CUDA-compatible torch from https://pytorch.org

### 2. One-click launch (recommended)

```bash
python run_all.py
```

This will:
- Generate the synthetic dataset (1200 reviews)
- Run VADER sentiment analysis
- Launch the Streamlit dashboard at `http://localhost:8501`

### 3. Manual step-by-step

```bash
# Generate dataset
python data/raw/generate_dataset.py

# Preprocess text
python src/preprocess.py

# VADER analysis (fast, no download)
python src/vader_analysis.py

# BERT analysis (downloads ~500MB model first time)
python src/bert_analysis.py

# LDA topic modeling
python src/topic_modeling.py

# Launch dashboard
streamlit run app/dashboard.py
```

### 4. Jupyter Notebooks

```bash
jupyter notebook
# Open notebooks/ and run them in order: 01 → 02 → 03
```

---

## 📊 Models Used

### VADER (Valence Aware Dictionary and sEntiment Reasoner)
- **Type:** Rule-based lexicon
- **Speed:** Instant (no model loading)
- **Best for:** Short social media text
- **Compound Score:** -1 (very negative) to +1 (very positive)
  - ≥ 0.05 → Positive
  - ≤ -0.05 → Negative
  - else → Neutral

### BERT (cardiffnlp/twitter-roberta-base-sentiment)
- **Type:** Fine-tuned transformer (RoBERTa)
- **Training data:** Twitter / Social media
- **Output:** Negative / Neutral / Positive + confidence score
- **Accuracy:** ~85-90% on social media text

### LDA (Latent Dirichlet Allocation)
- **Library:** Gensim
- **Topics:** Auto-optimized using coherence score
- **Topics discovered:** Food Quality, Delivery, Product Quality, Pricing, Customer Service, Packaging
- **Visualization:** Interactive pyLDAvis HTML (`outputs/reports/lda_topics.html`)

---

## 🎯 Skills Demonstrated

- ✅ **NLP Preprocessing** — tokenization, lemmatization, stopword removal
- ✅ **Rule-based NLP** — VADER lexicon sentiment analysis
- ✅ **Deep Learning NLP** — HuggingFace transformers pipeline
- ✅ **Unsupervised Learning** — LDA topic modeling, coherence evaluation
- ✅ **EDA** — visual exploration of unstructured text data
- ✅ **Model Evaluation** — accuracy, F1-score, confusion matrix
- ✅ **Dashboard** — production-style Streamlit app
- ✅ **Real-world Data** — Amazon, Zomato, Swiggy review simulation

---

## 📈 Sample Results

| Model | Accuracy | Speed | GPU Required |
|-------|----------|-------|--------------|
| VADER | ~72% | ⚡ Instant | ❌ No |
| BERT  | ~87% | 🐌 Slower | Recommended |

---

## 🛠️ Tech Stack

`Python` · `pandas` · `NLTK` · `spaCy` · `vaderSentiment` · `HuggingFace Transformers` · `PyTorch` · `Gensim` · `pyLDAvis` · `scikit-learn` · `Streamlit` · `Plotly` · `WordCloud` · `Matplotlib` · `Seaborn`

---

## 📬 Use Cases

This project applies directly to:
- **Amazon** — product review sentiment tracking
- **Zomato / Swiggy** — food quality & delivery experience monitoring  
- **Brand teams** — real-time NPS and customer feedback dashboards
- **Marketing analysts** — identifying pain points and positive drivers

---

*Built as a Data Science portfolio project demonstrating real-world NLP capabilities.*
