# ════════════════════════════════════════════════════════════════════
# Fake News Detector — Streamlit Web Application
# Author  : Adewale Samson Adeagbo
# Email   : adewalesamsonadeagbo@gmail.com
# GitHub  : github.com/cssadewale
# LinkedIn: linkedin.com/in/adewalesamsonadeagbo
# Project : 3MTT Capstone Project 2 — The TruthLens Institute
# Model   : Tuned XGBoost Classifier (AUC = 0.9393, Accuracy = 86.75%)
# ════════════════════════════════════════════════════════════════════

import streamlit as st
import joblib
import re
import string
import numpy as np
import scipy.sparse

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

# ── Download NLTK resources (runs once on first launch) ──────────────────────
nltk.download('stopwords',     quiet=True)
nltk.download('wordnet',       quiet=True)
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt',         quiet=True)
nltk.download('omw-1.4',       quiet=True)


# ════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title            = "Fake News Detector | TruthLens",
    page_icon             = "🔍",
    layout                = "wide",
    initial_sidebar_state = "expanded"
)


# ════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    /* Main header */
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    /* Verdict boxes */
    .verdict-fake {
        background: linear-gradient(135deg, #ff4757, #c0392b);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: 2px;
    }
    .verdict-real {
        background: linear-gradient(135deg, #2ecc71, #27ae60);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: 2px;
    }
    /* Metric card */
    .metric-card {
        background: #f8f9fa;
        border-left: 4px solid #3498db;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        margin: 0.4rem 0;
    }
    /* Footer */
    .footer {
        text-align: center;
        color: #888;
        font-size: 0.82em;
        padding: 1rem 0;
        border-top: 1px solid #eee;
        margin-top: 2rem;
    }
    /* Hide Streamlit default branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# LOAD MODELS  — cached so they load only ONCE per session
# ════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_models():
    """Load TF-IDF vectoriser, XGBoost model, and label encoder.
    @st.cache_resource ensures models are loaded only once per session,
    not on every user interaction."""
    tfidf = joblib.load('tfidf_vectorizer.joblib')
    model = joblib.load('tuned_xgboost_fake_news.joblib')
    le    = joblib.load('label_encoder.joblib')
    return tfidf, model, le


@st.cache_resource
def load_nlp_tools():
    """Initialise NLP tools — also cached once per session."""
    sw         = set(stopwords.words('english'))
    lem        = WordNetLemmatizer()
    sentiment  = SentimentIntensityAnalyzer()
    return sw, lem, sentiment


# Load everything at startup
try:
    tfidf, model, le = load_models()
    stop_words, lemmatizer, sia = load_nlp_tools()
    MODELS_LOADED = True
except Exception as e:
    MODELS_LOADED = False
    MODEL_ERROR   = str(e)


# ════════════════════════════════════════════════════════════════════
# TEXT PREPROCESSING — IDENTICAL TO TRAINING NOTEBOOK PIPELINE
# This function MUST exactly match clean_text() used during training
# ════════════════════════════════════════════════════════════════════
def clean_text(text: str) -> str:
    """Five-stage NLP normalisation pipeline.

    Stages:
        1. Remove special characters and digits (keep only letters)
        2. Lowercase conversion
        3. Punctuation removal
        4. Tokenisation
        5. Stopword removal + lemmatisation (WordNetLemmatizer)

    This function is identical to the clean_text() in the training
    notebook. Any deviation will cause feature mismatch errors.
    """
    if not isinstance(text, str):
        return ''
    # Stage 1: Remove special characters and digits
    sentence = re.sub('[^a-zA-Z]', ' ', text)
    # Stage 2: Lowercase
    sentence = sentence.lower()
    # Stage 3: Remove punctuation
    sentence = ''.join([c for c in sentence if c not in string.punctuation])
    # Stage 4: Tokenise
    tokens = sentence.split()
    # Stage 5: Remove stopwords and lemmatise
    clean_tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(clean_tokens)


def predict_article(title: str, body: str) -> tuple:
    """Run the complete prediction pipeline on a single article.

    Pipeline (mirrors training notebook exactly):
        1. Concatenate: title + ': ' + body
        2. Apply clean_text() — 5-stage NLP normalisation
        3. Extract word_count (token count after cleaning)
        4. Extract VADER compound sentiment score
        5. Transform with fitted TF-IDF vectoriser (20,000 features)
        6. scipy.sparse.hstack → (1, 20,002) feature matrix
        7. Tuned XGBoost predict + predict_proba

    Returns:
        label       (str)  : 'fake' or 'real'
        confidence  (float): probability of the predicted class (0.0–1.0)
        fake_prob   (float): raw probability of being fake
        real_prob   (float): raw probability of being real
        cleaned     (str)  : cleaned content for display
        word_count  (int)  : token count after cleaning
        sentiment   (float): VADER compound score
    """
    # ── Step 1: Combine title and body ────────────────────────────────────────
    raw_content = (title.strip() + ': ' + body.strip()).strip(': ')
    if not raw_content:
        return None, None, None, None, '', 0, 0.0

    # ── Step 2: Clean ─────────────────────────────────────────────────────────
    cleaned = clean_text(raw_content)

    # ── Step 3 & 4: Numerical features ───────────────────────────────────────
    word_count   = len(cleaned.split())
    sentiment    = sia.polarity_scores(cleaned)['compound']
    num_features = np.array([[word_count, sentiment]])

    # ── Step 5: TF-IDF vectorisation ──────────────────────────────────────────
    X_tfidf = tfidf.transform([cleaned])          # shape: (1, 20000)

    # ── Step 6: Stack into final feature matrix ───────────────────────────────
    X_final = scipy.sparse.hstack([X_tfidf, num_features])  # shape: (1, 20002)

    # ── Step 7: Predict ───────────────────────────────────────────────────────
    pred_int   = model.predict(X_final)[0]         # 0 = fake, 1 = real
    proba      = model.predict_proba(X_final)[0]   # [P(fake), P(real)]

    label      = le.inverse_transform([pred_int])[0]
    fake_prob  = float(proba[0])
    real_prob  = float(proba[1])
    confidence = max(fake_prob, real_prob)

    return label, confidence, fake_prob, real_prob, cleaned, word_count, sentiment


# ════════════════════════════════════════════════════════════════════
# SAMPLE ARTICLES
# ════════════════════════════════════════════════════════════════════
EXAMPLES = {
    "🔴 Fake Article": {
        "title": "SHOCKING: Government Secretly Controls Weather Machines",
        "body": (
            "Sources close to the deep state have confirmed that shadow elites are "
            "using HAARP technology to manipulate storms and droughts across the country. "
            "Citizens are being kept in the dark about this horrifying conspiracy. "
            "Share this before it gets deleted! The mainstream media won't tell you "
            "the truth about what is really happening behind closed doors. "
            "Wake up America — the elites are coming for your freedom and they will "
            "stop at nothing to silence the brave patriots who dare to speak out. "
            "This is the biggest cover-up in modern history."
        )
    },
    "🟢 Real Article": {
        "title": "Senate Passes Infrastructure Bill After Bipartisan Vote",
        "body": (
            "The United States Senate passed a $1.2 trillion infrastructure bill on "
            "Tuesday following weeks of bipartisan negotiations. The legislation "
            "allocates funding for road repairs, broadband expansion, and public transit "
            "upgrades across all 50 states. President Biden praised the vote as a "
            "historic achievement, calling it a generational investment in America's "
            "future. Republican Senator Rob Portman said the bill represented a "
            "practical, common-sense solution to the country's deteriorating "
            "infrastructure. The bill now moves to the House of Representatives "
            "for consideration, where its passage is expected but not guaranteed."
        )
    },
    "🟡 Ambiguous Article": {
        "title": "New Study Claims Daily Coffee Consumption May Extend Lifespan",
        "body": (
            "Researchers at a European university published findings suggesting that "
            "moderate coffee consumption of three to four cups per day is linked to a "
            "reduced mortality risk of up to 17 percent. The observational study tracked "
            "50,000 participants over 15 years across multiple countries. Critics note "
            "the research was funded by a coffee industry advocacy group and that "
            "correlation does not imply causation. The findings have not yet been "
            "independently peer-reviewed. Nutritionists caution against drawing firm "
            "conclusions until additional studies replicate the results."
        )
    }
}


# ════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════
with st.sidebar:

    # Logo
    try:
        st.image(
            "https://drive.google.com/uc?export=view&id=1BSdTj6PVZwEnSCqucDa5DVHUGcPV_6jK",
            use_column_width=True
        )
    except Exception:
        st.markdown("### 🔍 TruthLens Institute")

    st.markdown("---")

    # Model info
    st.markdown("### 📊 Model Performance")
    metrics = {
        "Model":        "Tuned XGBoost",
        "Accuracy":     "86.75%",
        "ROC-AUC":      "0.9393",
        "F1 (Fake)":    "0.87",
        "F1 (Real)":    "0.87",
        "Features":     "20,002",
        "Training set": "743 articles",
        "CV AUC":       "0.8987",
    }
    for k, v in metrics.items():
        st.markdown(
            f"<div class='metric-card'><b>{k}</b>: {v}</div>",
            unsafe_allow_html=True
        )

    st.markdown("---")

    # Pipeline
    st.markdown("### 🧰 NLP Pipeline")
    st.markdown("""
1. Title + Body → `content`
2. `clean_text()` — 5-stage NLP
   - Remove special chars & digits
   - Lowercase conversion
   - Punctuation removal
   - Tokenisation
   - Stopword removal + Lemmatisation
3. Word count extraction
4. VADER sentiment score
5. TF-IDF (20,000 features, unigrams + bigrams)
6. `scipy.sparse.hstack` → 20,002 dims
7. XGBoost prediction + probability
    """)

    st.markdown("---")

    # About
    st.markdown("### 👤 Author")
    st.markdown("""
**Adewale Samson Adeagbo**
Lead Data Scientist / NLP Engineer
Mathematics Teacher · 10+ Years

📧 adewalesamsonadeagbo@gmail.com
📱 08100866322
🔗 [LinkedIn](https://linkedin.com/in/adewalesamsonadeagbo)
💻 [GitHub](https://github.com/cssadewale)
🌐 [Portfolio](https://hmgconcepts.business.site)

*3MTT Capstone Project 2*
*The TruthLens Institute · 2025*
    """)


# ════════════════════════════════════════════════════════════════════
# MAIN PAGE — HEADER
# ════════════════════════════════════════════════════════════════════
st.markdown(
    "<div class='main-title'>🔍 Fake News Detector</div>",
    unsafe_allow_html=True
)
st.markdown(
    "#### Powered by Tuned XGBoost &nbsp;·&nbsp; Accuracy: 86.75% "
    "&nbsp;·&nbsp; ROC-AUC: 0.9393"
)
st.markdown("""
This tool classifies a news article as **FAKE** or **REAL** using a complete NLP
preprocessing pipeline and a Tuned XGBoost Classifier trained on 12,999 real-world
news articles across 8 misinformation categories.

> Built for **The TruthLens Institute** as part of the **3MTT Data Science Capstone
> Programme** — combating misinformation through machine learning.
""")

# Model loading error banner
if not MODELS_LOADED:
    st.error(
        f"⚠️ **Model files not found.** Please ensure `tfidf_vectorizer.joblib`, "
        f"`tuned_xgboost_fake_news.joblib`, and `label_encoder.joblib` are in the "
        f"same directory as `app.py`.\n\n**Error:** `{MODEL_ERROR}`"
    )
    st.stop()

st.markdown("---")


# ════════════════════════════════════════════════════════════════════
# INPUT SECTION
# ════════════════════════════════════════════════════════════════════
st.markdown("### 📝 Enter the News Article")

col_title, col_body = st.columns([1, 2])

with col_title:
    st.markdown("**Article Title** *(optional but recommended)*")
    title_input = st.text_input(
        label            = "title_field",
        placeholder      = "e.g. Breaking: New Government Policy Announced...",
        label_visibility = "collapsed",
        key              = "title_input"
    )

with col_body:
    st.markdown("**Article Body** *(paste the full text)*")
    body_input = st.text_area(
        label            = "body_field",
        placeholder      = "Paste the full article text here...",
        height           = 180,
        label_visibility = "collapsed",
        key              = "body_input"
    )

# ── Sample article buttons ────────────────────────────────────────────────────
st.markdown("##### 💡 Or try a sample article:")
btn_col1, btn_col2, btn_col3 = st.columns(3)

if btn_col1.button("🔴 Fake Example", use_container_width=True):
    st.session_state["title_input"] = EXAMPLES["🔴 Fake Article"]["title"]
    st.session_state["body_input"]  = EXAMPLES["🔴 Fake Article"]["body"]
    st.rerun()

if btn_col2.button("🟢 Real Example", use_container_width=True):
    st.session_state["title_input"] = EXAMPLES["🟢 Real Article"]["title"]
    st.session_state["body_input"]  = EXAMPLES["🟢 Real Article"]["body"]
    st.rerun()

if btn_col3.button("🟡 Ambiguous Example", use_container_width=True):
    st.session_state["title_input"] = EXAMPLES["🟡 Ambiguous Article"]["title"]
    st.session_state["body_input"]  = EXAMPLES["🟡 Ambiguous Article"]["body"]
    st.rerun()


# ════════════════════════════════════════════════════════════════════
# PREDICTION
# ════════════════════════════════════════════════════════════════════
st.markdown("---")
predict_clicked = st.button(
    "🚀  Analyse Article",
    type             = "primary",
    use_container_width = True
)

if predict_clicked:

    # ── Input validation ──────────────────────────────────────────────────────
    if not title_input.strip() and not body_input.strip():
        st.warning(
            "⚠️ Please enter an article title, body text, or both before clicking Analyse."
        )
        st.stop()

    # ── Run prediction ────────────────────────────────────────────────────────
    with st.spinner("Running NLP pipeline — cleaning text, extracting features, predicting..."):
        label, confidence, fake_prob, real_prob, cleaned, wc, sent = predict_article(
            title_input, body_input
        )

    if label is None:
        st.warning("⚠️ The article text was empty after cleaning. Please provide more content.")
        st.stop()

    # ════════════════════════════════════════════════════════════════
    # RESULT DISPLAY
    # ════════════════════════════════════════════════════════════════
    st.markdown("### 🎯 Prediction Result")

    # ── Verdict banner ────────────────────────────────────────────────────────
    if label == 'fake':
        st.markdown(
            "<div class='verdict-fake'>🚨 &nbsp; FAKE NEWS &nbsp; 🚨</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div class='verdict-real'>✅ &nbsp; REAL NEWS &nbsp; ✅</div>",
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Key metrics ───────────────────────────────────────────────────────────
    m1, m2, m3 = st.columns(3)
    m1.metric("Verdict",    label.upper())
    m2.metric("Confidence", f"{confidence * 100:.1f}%")
    m3.metric("VADER Sentiment", f"{sent:+.4f}")

    st.markdown("---")

    # ── Probability breakdown ─────────────────────────────────────────────────
    st.markdown("#### 📊 Probability Breakdown")
    p1, p2 = st.columns(2)

    with p1:
        st.markdown("**🔴 Probability: FAKE**")
        st.progress(fake_prob)
        st.markdown(f"**`{fake_prob * 100:.2f}%`**")

    with p2:
        st.markdown("**🟢 Probability: REAL**")
        st.progress(real_prob)
        st.markdown(f"**`{real_prob * 100:.2f}%`**")

    st.markdown("---")

    # ── Confidence interpretation ─────────────────────────────────────────────
    st.markdown("#### 🔎 Confidence Interpretation")

    if confidence >= 0.85:
        st.success(
            f"**High confidence ({confidence*100:.1f}%)** — The model is strongly "
            f"confident this article is **{label.upper()}**. The linguistic patterns "
            f"align clearly with {'fabricated' if label=='fake' else 'credible'} news content."
        )
    elif confidence >= 0.65:
        st.info(
            f"**Moderate confidence ({confidence*100:.1f}%)** — The model leans toward "
            f"**{label.upper()}** but detects mixed signals. The article may contain "
            f"some characteristics of both classes. Human review is advisable."
        )
    else:
        st.warning(
            f"**Low confidence ({confidence*100:.1f}%)** — This article is borderline. "
            f"The model cannot confidently classify it as fake or real. "
            f"**Human editorial review is strongly recommended** before acting on this prediction."
        )

    st.markdown("---")

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.markdown("#### ⚠️ Important Disclaimer")
    st.markdown("""
> This tool is a **machine learning classifier** trained on a specific dataset of
> US political news articles (2015–2016). It should be used as a **screening aid**,
> not as a definitive source of truth.
>
> - **False positives** (real news flagged as fake) and **false negatives**
>   (fake news missed) can occur.
> - Always verify important articles through reputable fact-checking organisations
>   such as Snopes, PolitiFact, or AFP Fact Check.
> - The model was trained on 826 balanced articles and may not generalise perfectly
>   to all news domains, regions, or time periods.
    """)

    # ── Pipeline details expander ─────────────────────────────────────────────
    with st.expander("🔬 Show Full Pipeline Details"):

        st.markdown("**Cleaned Content (after 5-stage NLP normalisation):**")
        display_cleaned = cleaned[:600] + ("..." if len(cleaned) > 600 else "")
        st.code(display_cleaned if display_cleaned else "(empty after cleaning)", language=None)

        d1, d2, d3 = st.columns(3)
        d1.metric("Word Count (post-clean)", f"{wc:,}")
        d2.metric("VADER Sentiment", f"{sent:+.4f}")
        d3.metric("Sentiment Interpretation",
                  "Positive" if sent > 0.05 else "Negative" if sent < -0.05 else "Neutral")

        st.markdown("**Feature Matrix Breakdown:**")
        st.markdown(f"""
| Component | Details |
|-----------|---------|
| TF-IDF features | 20,000 (unigrams + bigrams, `max_features=20,000`) |
| Numerical features | 2 (`word_count`, `sentiment`) |
| **Total dimensions** | **20,002** |
| Vectoriser type | `TfidfVectorizer` fitted on 743-sample training corpus |
| Matrix format | `scipy.sparse.csr_matrix` |
| Prediction model | `XGBClassifier` (tuned: `learning_rate=0.05`, `max_depth=3`, `n_estimators=200`) |
        """)

        st.markdown("**All Model Results (from training notebook):**")
        st.markdown("""
| Model | Accuracy | ROC-AUC | F1 (Fake) | F1 (Real) |
|-------|----------|---------|-----------|-----------|
| **Tuned XGBoost ✅** | **86.75%** | **0.9393** | **0.87** | **0.87** |
| XGBoost (Baseline) | 84.34% | 0.9051 | 0.84 | 0.84 |
| Logistic Regression | 79.52% | 0.8560 | 0.78 | 0.80 |
| Random Forest | 79.52% | 0.8656 | 0.80 | 0.79 |
| Tuned Logistic Regression | 78.31% | 0.8513 | 0.78 | 0.79 |
| Decision Tree | 72.29% | 0.7227 | 0.73 | 0.72 |
| SVC (RBF kernel) | 68.67% | 0.7311 | 0.74 | 0.61 |
        """)


# ════════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='footer'>
    Built by <strong>Adewale Samson Adeagbo</strong> &nbsp;|&nbsp;
    <a href='https://linkedin.com/in/adewalesamsonadeagbo' target='_blank'>LinkedIn</a> &nbsp;·&nbsp;
    <a href='https://github.com/cssadewale' target='_blank'>GitHub</a> &nbsp;·&nbsp;
    adewalesamsonadeagbo@gmail.com &nbsp;·&nbsp; 08100866322
    <br>
    3MTT Capstone Project 2 &nbsp;·&nbsp; The TruthLens Institute &nbsp;·&nbsp;
    Tuned XGBoost &nbsp;·&nbsp; Accuracy: 86.75% &nbsp;·&nbsp; AUC: 0.9393
</div>
""", unsafe_allow_html=True)
