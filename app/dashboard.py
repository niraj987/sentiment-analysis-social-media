"""
dashboard.py
─────────────
Streamlit Dashboard for Sentiment Analysis on Social Media Reviews.
Features:
  - Real-time sentiment prediction (VADER + BERT)
  - Dataset analytics: sentiment distribution, platform comparison
  - Word cloud generator
  - LDA topic explorer

Run: streamlit run app/dashboard.py
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH    = os.path.join(BASE_DIR, 'data', 'raw', 'reviews.csv')
VADER_PATH   = os.path.join(BASE_DIR, 'data', 'processed', 'vader_results.csv')
BERT_PATH    = os.path.join(BASE_DIR, 'data', 'processed', 'bert_results.csv')
LDA_VIS_PATH = os.path.join(BASE_DIR, 'outputs', 'reports', 'lda_topics.html')
FIG_DIR      = os.path.join(BASE_DIR, 'outputs', 'figures')

sys.path.insert(0, BASE_DIR)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Main background */
.main { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); min-height: 100vh; }

/* Metric cards */
div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 12px;
    padding: 16px;
    backdrop-filter: blur(10px);
}

/* Header */
.hero-title {
    font-size: 2.6rem; font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    text-align: center; margin-bottom: 4px;
}
.hero-sub {
    text-align: center; color: #94a3b8; font-size: 1rem; margin-bottom: 28px;
}

/* Section headers */
.section-header {
    font-size: 1.3rem; font-weight: 700; color: #e2e8f0;
    border-left: 4px solid #a78bfa; padding-left: 12px; margin: 20px 0 12px;
}

/* Sentiment badges */
.badge-positive { background: #065f46; color: #34d399; border-radius: 20px;
                  padding: 4px 14px; font-weight: 700; display: inline-block; }
.badge-negative { background: #7f1d1d; color: #f87171; border-radius: 20px;
                  padding: 4px 14px; font-weight: 700; display: inline-block; }
.badge-neutral  { background: #78350f; color: #fbbf24; border-radius: 20px;
                  padding: 4px 14px; font-weight: 700; display: inline-block; }

/* ── Sidebar container ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1042 0%, #0f0c29 100%);
    border-right: 2px solid rgba(167, 139, 250, 0.35);
}

/* ── ALL sidebar text — bright so it's clearly readable ── */
[data-testid="stSidebar"],
[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}

/* Sidebar headings (### Navigation, ### Settings) */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #a78bfa !important;
    font-weight: 700 !important;
    letter-spacing: 0.03em;
}

/* Radio option text */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-baseweb="radio"] p,
[data-testid="stSidebar"] .stRadio label p {
    color: #f1f5f9 !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
}

/* Selected radio item highlight */
[data-testid="stSidebar"] [aria-checked="true"] ~ div p {
    color: #a78bfa !important;
    font-weight: 700 !important;
}

/* Checkbox */
[data-testid="stSidebar"] .stCheckbox label p,
[data-testid="stSidebar"] .stCheckbox span {
    color: #f1f5f9 !important;
    font-weight: 500 !important;
}

/* Bold text (project info section) */
[data-testid="stSidebar"] strong {
    color: #c4b5fd !important;
    font-weight: 700 !important;
}

/* Markdown bullet points and plain text */
[data-testid="stSidebar"] p {
    color: #cbd5e1 !important;
    font-size: 0.88rem !important;
    line-height: 1.6 !important;
}

/* Horizontal divider line */
[data-testid="stSidebar"] hr {
    border: none !important;
    border-top: 1px solid rgba(167, 139, 250, 0.4) !important;
    margin: 12px 0 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Cached Data Loading ───────────────────────────────────────────────────────
@st.cache_data
def load_raw_data():
    if os.path.exists(VADER_PATH):
        return pd.read_csv(VADER_PATH)
    elif os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None

@st.cache_data
def load_bert_data():
    if os.path.exists(BERT_PATH):
        return pd.read_csv(BERT_PATH)
    return None

@st.cache_resource
def get_vader_analyzer():
    return SentimentIntensityAnalyzer()

@st.cache_resource
def load_bert_pipeline():
    try:
        from transformers import pipeline
        import torch
        device = 0 if torch.cuda.is_available() else -1
        return pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment",
            device=device, max_length=128, truncation=True,
        )
    except Exception as e:
        return None


# ── VADER Predict ─────────────────────────────────────────────────────────────
def predict_vader(text: str, analyzer) -> dict:
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.05:
        label = 'positive'
    elif compound <= -0.05:
        label = 'negative'
    else:
        label = 'neutral'
    return {**scores, 'sentiment': label}


def predict_bert(text: str, pipe) -> dict:
    LABEL_MAP = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}
    try:
        result = pipe(text)[0]
        return {'sentiment': LABEL_MAP.get(result['label'], result['label']),
                'confidence': round(result['score'] * 100, 1)}
    except Exception as e:
        return {'sentiment': 'error', 'confidence': 0.0, 'error': str(e)}


def sentiment_badge(label: str) -> str:
    cls = f"badge-{label}"
    icons = {'positive': '😊 Positive', 'negative': '😠 Negative', 'neutral': '😐 Neutral'}
    return f'<span class="{cls}">{icons.get(label, label.title())}</span>'


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔍 Navigation")
    page = st.radio("", [
        "🏠 Home & Live Demo",
        "📊 Dataset Analytics",
        "☁️ Word Cloud Explorer",
        "📌 Topic Modeling",
        "🤖 Model Comparison",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    use_bert = st.checkbox("Enable BERT (requires download)", value=False)
    st.markdown("---")
    st.markdown("""
    **Project**  
    Sentiment Analysis on Social Media Reviews  
    
    **Models**  
    • VADER (rule-based)  
    • BERT (twitter-roberta)  
    • LDA Topic Modeling  
    
    **Platforms**  
    • Amazon  • Zomato  • Swiggy
    """)


# ── Load Data ─────────────────────────────────────────────────────────────────
df = load_raw_data()
analyzer = get_vader_analyzer()
bert_pipe = load_bert_pipeline() if use_bert else None


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: Home & Live Demo
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home & Live Demo":
    st.markdown('<div class="hero-title">🔍 Sentiment Analysis Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Real-time NLP sentiment analysis · Amazon · Zomato · Swiggy</div>', unsafe_allow_html=True)

    # KPIs
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📝 Total Reviews", f"{len(df):,}")
        with col2:
            pos_pct = (df.get('vader_sentiment', df.get('sentiment', '')).eq('positive').sum() / len(df) * 100)
            st.metric("😊 Positive", f"{pos_pct:.1f}%")
        with col3:
            neg_pct = (df.get('vader_sentiment', df.get('sentiment', '')).eq('negative').sum() / len(df) * 100)
            st.metric("😠 Negative", f"{neg_pct:.1f}%")
        with col4:
            platforms = df['platform'].nunique() if 'platform' in df.columns else 0
            st.metric("🏪 Platforms", platforms)

    st.markdown("---")
    st.markdown('<div class="section-header">💬 Try It Live — Paste Any Review</div>', unsafe_allow_html=True)

    example_reviews = [
        "Select an example...",
        "This product is absolutely amazing! Best purchase ever.",
        "Terrible quality. Broke after 2 days. Complete waste of money!",
        "The food was decent but delivery took way too long.",
        "Loved the biryani! Fresh, hot and perfectly spiced. Will order again!",
        "Received a damaged item and the seller refused to refund.",
        "It's okay, nothing special. Average product for the price.",
    ]
    selected = st.selectbox("🎯 Example Reviews", example_reviews)
    user_text = st.text_area(
        "Or type your own review:",
        value=selected if selected != "Select an example..." else "",
        height=120,
        placeholder="Type or paste a review here..."
    )

    if st.button("🚀 Analyze Sentiment", use_container_width=True) and user_text.strip():
        col_v, col_b = st.columns(2)

        with col_v:
            st.markdown("#### 📏 VADER Analysis")
            vader_result = predict_vader(user_text, analyzer)
            st.markdown(f"**Sentiment:** {sentiment_badge(vader_result['sentiment'])}", unsafe_allow_html=True)

            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=vader_result['compound'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Compound Score", 'font': {'size': 14}},
                gauge={
                    'axis': {'range': [-1, 1], 'tickwidth': 1},
                    'bar': {'color': "#a78bfa"},
                    'steps': [
                        {'range': [-1, -0.05], 'color': '#7f1d1d'},
                        {'range': [-0.05, 0.05], 'color': '#78350f'},
                        {'range': [0.05, 1], 'color': '#065f46'},
                    ],
                    'threshold': {'line': {'color': 'white', 'width': 4},
                                  'thickness': 0.75, 'value': vader_result['compound']}
                }
            ))
            gauge.update_layout(height=220, paper_bgcolor='rgba(0,0,0,0)', font_color='white', margin=dict(t=40,b=10))
            st.plotly_chart(gauge, use_container_width=True)

            sub_cols = st.columns(3)
            sub_cols[0].metric("😊 Pos", f"{vader_result['pos']:.2f}")
            sub_cols[1].metric("😐 Neu", f"{vader_result['neu']:.2f}")
            sub_cols[2].metric("😠 Neg", f"{vader_result['neg']:.2f}")

        with col_b:
            st.markdown("#### 🤖 BERT Analysis")
            if bert_pipe and use_bert:
                bert_result = predict_bert(user_text, bert_pipe)
                st.markdown(f"**Sentiment:** {sentiment_badge(bert_result['sentiment'])}", unsafe_allow_html=True)
                st.markdown(f"**Confidence:** {bert_result['confidence']}%")
                conf_fig = go.Figure(go.Bar(
                    x=[bert_result['confidence']],
                    orientation='h',
                    marker_color='#60a5fa',
                    text=[f"{bert_result['confidence']}%"],
                    textposition='inside',
                ))
                conf_fig.update_layout(
                    height=100, xaxis=dict(range=[0, 100], title='Confidence %'),
                    paper_bgcolor='rgba(0,0,0,0)', font_color='white',
                    margin=dict(t=10, b=10, l=10, r=10), showlegend=False,
                )
                st.plotly_chart(conf_fig, use_container_width=True)
            else:
                st.info("Enable **BERT** in the sidebar to see deep learning predictions.\n\n"
                        "*(First load downloads ~500MB model)*")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: Dataset Analytics
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Dataset Analytics":
    st.markdown('<div class="hero-title">📊 Dataset Analytics</div>', unsafe_allow_html=True)

    if df is None:
        st.error("Dataset not found. Run `python data/raw/generate_dataset.py` first.")
        st.stop()

    sentiment_col = 'vader_sentiment' if 'vader_sentiment' in df.columns else 'sentiment'

    # Filter controls
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        platforms = ['All'] + df['platform'].unique().tolist() if 'platform' in df.columns else ['All']
        sel_platform = st.selectbox("Filter by Platform", platforms)
    with col_f2:
        sentiments = ['All', 'positive', 'negative', 'neutral']
        sel_sent = st.selectbox("Filter by Sentiment", sentiments)

    filtered = df.copy()
    if sel_platform != 'All':
        filtered = filtered[filtered['platform'] == sel_platform]
    if sel_sent != 'All':
        filtered = filtered[filtered[sentiment_col] == sel_sent]

    st.markdown(f"*Showing **{len(filtered):,}** of {len(df):,} reviews*")
    st.markdown("---")

    # Row 1: Distributions
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Sentiment Distribution</div>', unsafe_allow_html=True)
        sent_counts = filtered[sentiment_col].value_counts().reset_index()
        sent_counts.columns = ['Sentiment', 'Count']
        fig = px.pie(sent_counts, names='Sentiment', values='Count',
                     color='Sentiment',
                     color_discrete_map={'positive': '#34d399', 'negative': '#f87171', 'neutral': '#fbbf24'},
                     hole=0.45)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white',
                          legend=dict(font=dict(color='white')), margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Rating Distribution</div>', unsafe_allow_html=True)
        if 'rating' in filtered.columns:
            rating_counts = filtered['rating'].value_counts().sort_index().reset_index()
            rating_counts.columns = ['Rating', 'Count']
            fig2 = px.bar(rating_counts, x='Rating', y='Count',
                          color='Rating', color_continuous_scale='viridis',
                          text='Count')
            fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white',
                               showlegend=False, margin=dict(t=10))
            fig2.update_traces(textposition='outside')
            st.plotly_chart(fig2, use_container_width=True)

    # Row 2: Platform comparison
    if 'platform' in df.columns:
        st.markdown('<div class="section-header">Platform Sentiment Comparison</div>', unsafe_allow_html=True)
        plat_sent = filtered.groupby(['platform', sentiment_col]).size().reset_index(name='count')
        fig3 = px.bar(plat_sent, x='platform', y='count', color=sentiment_col, barmode='group',
                      color_discrete_map={'positive': '#34d399', 'negative': '#f87171', 'neutral': '#fbbf24'},
                      text='count')
        fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white',
                           xaxis_title='Platform', yaxis_title='Review Count',
                           legend_title='Sentiment', margin=dict(t=10))
        fig3.update_traces(textposition='outside')
        st.plotly_chart(fig3, use_container_width=True)

    # Row 3: Sentiment over time
    if 'date' in df.columns:
        st.markdown('<div class="section-header">Sentiment Trend Over Time</div>', unsafe_allow_html=True)
        df_time = filtered.copy()
        df_time['date'] = pd.to_datetime(df_time['date'])
        df_time['month'] = df_time['date'].dt.to_period('M').astype(str)
        trend = df_time.groupby(['month', sentiment_col]).size().reset_index(name='count')
        fig4 = px.line(trend, x='month', y='count', color=sentiment_col,
                       color_discrete_map={'positive': '#34d399', 'negative': '#f87171', 'neutral': '#fbbf24'},
                       markers=True)
        fig4.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white',
                           xaxis_title='Month', yaxis_title='Reviews', margin=dict(t=10))
        st.plotly_chart(fig4, use_container_width=True)

    # Raw data
    with st.expander("📋 View Raw Data"):
        st.dataframe(filtered.head(100), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: Word Cloud Explorer
# ══════════════════════════════════════════════════════════════════════════════
elif page == "☁️ Word Cloud Explorer":
    st.markdown('<div class="hero-title">☁️ Word Cloud Explorer</div>', unsafe_allow_html=True)

    if df is None:
        st.error("Dataset not found.")
        st.stop()

    sentiment_col = 'vader_sentiment' if 'vader_sentiment' in df.columns else 'sentiment'

    col1, col2 = st.columns(2)
    with col1:
        selected_sentiment_wc = st.selectbox("Sentiment", ['All', 'positive', 'negative', 'neutral'])
    with col2:
        max_words = st.slider("Max Words", 50, 300, 150, 25)

    if selected_sentiment_wc == 'All':
        text_data = ' '.join(df['review_text'].dropna().astype(str))
        bg_color = '#0f0c29'
        wc_colormap = 'plasma'
        title = "All Sentiments"
    elif selected_sentiment_wc == 'positive':
        text_data = ' '.join(df[df[sentiment_col] == 'positive']['review_text'].dropna().astype(str))
        bg_color = '#052e16'
        wc_colormap = 'Greens'
        title = "Positive Reviews"
    elif selected_sentiment_wc == 'negative':
        text_data = ' '.join(df[df[sentiment_col] == 'negative']['review_text'].dropna().astype(str))
        bg_color = '#2d0000'
        wc_colormap = 'Reds'
        title = "Negative Reviews"
    else:
        text_data = ' '.join(df[df[sentiment_col] == 'neutral']['review_text'].dropna().astype(str))
        bg_color = '#1c1400'
        wc_colormap = 'YlOrBr'
        title = "Neutral Reviews"

    STOPWORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'of', 'to', 'was',
        'is', 'are', 'it', 'i', 'my', 'me', 'this', 'that', 'for', 'with',
        'very', 'so', 'be', 'has', 'have', 'they', 'their', 'we', 'you', 'your'
    }

    with st.spinner("Generating word cloud..."):
        wc = WordCloud(
            width=900, height=450,
            background_color=bg_color,
            colormap=wc_colormap,
            max_words=max_words,
            stopwords=STOPWORDS,
            collocations=False,
            prefer_horizontal=0.9,
        ).generate(text_data)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f"Word Cloud — {title}", color='white', fontsize=14, fontweight='bold', pad=10)
        fig.patch.set_facecolor(bg_color)
        plt.tight_layout(pad=0)
        st.pyplot(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: Topic Modeling
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📌 Topic Modeling":
    st.markdown('<div class="hero-title">📌 LDA Topic Modeling</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Discover hidden themes in reviews using Latent Dirichlet Allocation</div>',
                unsafe_allow_html=True)

    lda_img = os.path.join(FIG_DIR, 'lda_top_words.png')
    dist_img = os.path.join(FIG_DIR, 'lda_topic_distribution.png')
    coh_img  = os.path.join(FIG_DIR, 'lda_coherence_scores.png')

    if os.path.exists(lda_img):
        st.markdown('<div class="section-header">Top Words per Topic</div>', unsafe_allow_html=True)
        st.image(lda_img, use_column_width=True)
    else:
        st.info("Run `python src/topic_modeling.py` to generate topic visualizations.")

    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists(dist_img):
            st.markdown('<div class="section-header">Topic Distribution</div>', unsafe_allow_html=True)
            st.image(dist_img, use_column_width=True)
    with col2:
        if os.path.exists(coh_img):
            st.markdown('<div class="section-header">Coherence Scores</div>', unsafe_allow_html=True)
            st.image(coh_img, use_column_width=True)

    if os.path.exists(LDA_VIS_PATH):
        st.markdown('<div class="section-header">🌐 Interactive pyLDAvis</div>', unsafe_allow_html=True)
        with open(LDA_VIS_PATH, 'r', encoding='utf-8') as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=800, scrolling=True)
    else:
        st.info("Run `python src/topic_modeling.py` to generate the interactive visualization.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5: Model Comparison
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Comparison":
    st.markdown('<div class="hero-title">🤖 Model Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">VADER vs BERT — performance, accuracy, and agreement</div>',
                unsafe_allow_html=True)

    vader_img   = os.path.join(FIG_DIR, 'vader_distribution.png')
    vader_cm    = os.path.join(FIG_DIR, 'vader_confusion_matrix.png')
    vader_comp  = os.path.join(FIG_DIR, 'vader_compound_dist.png')
    bert_vs_img = os.path.join(FIG_DIR, 'bert_vs_vader_comparison.png')
    bert_cm     = os.path.join(FIG_DIR, 'bert_confusion_matrix.png')
    bert_conf   = os.path.join(FIG_DIR, 'bert_confidence_dist.png')
    rating_bert = os.path.join(FIG_DIR, 'rating_vs_bert_sentiment.png')

    if any(os.path.exists(p) for p in [vader_img, bert_vs_img]):
        tabs = st.tabs(["📏 VADER Results", "🤖 BERT Results", "⚖️ Combined Comparison"])

        with tabs[0]:
            col1, col2 = st.columns(2)
            with col1:
                if os.path.exists(vader_img):
                    st.image(vader_img, use_column_width=True)
            with col2:
                if os.path.exists(vader_cm):
                    st.image(vader_cm, use_column_width=True)
            if os.path.exists(vader_comp):
                st.image(vader_comp, use_column_width=True)

        with tabs[1]:
            col1, col2 = st.columns(2)
            with col1:
                if os.path.exists(bert_cm):
                    st.image(bert_cm, use_column_width=True)
            with col2:
                if os.path.exists(bert_conf):
                    st.image(bert_conf, use_column_width=True)
            if os.path.exists(rating_bert):
                st.image(rating_bert, use_column_width=True)

        with tabs[2]:
            if os.path.exists(bert_vs_img):
                st.image(bert_vs_img, use_column_width=True)
            else:
                st.info("Run `python src/bert_analysis.py` to generate BERT comparison plots.")

            # Model comparison table
            st.markdown("### 📋 Model Characteristics")
            comp_data = {
                "Feature": ["Type", "Speed", "Training Required", "Best For", "GPU Needed", "Accuracy (typical)"],
                "VADER": ["Rule-based", "⚡ Instant", "❌ No", "Short social text", "❌ No", "~70–75%"],
                "BERT (twitter-roberta)": ["Deep Learning", "🐌 Slower", "✅ Pre-trained", "Context-rich text", "Recommended", "~85–90%"],
            }
            st.table(pd.DataFrame(comp_data))
    else:
        st.info("""
        **No analysis results found yet.**  
        
        Run these scripts to generate results:
        ```
        python src/vader_analysis.py       # Fast, no download needed
        python src/bert_analysis.py        # Requires ~500MB download
        ```
        """)
