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
# PAGE CONFIGURATION  — must be the very first Streamlit call
# ════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title            = "Fake News Detector | TruthLens",
    page_icon             = "🔍",
    layout                = "wide",
    initial_sidebar_state = "expanded"
)


# ════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALISATION
# CRITICAL: Keys must be initialised BEFORE any widgets are rendered.
# This prevents StreamlitAPIException when example buttons write to
# the same keys that text widgets read from.
# ════════════════════════════════════════════════════════════════════
if "title_val" not in st.session_state:
    st.session_state["title_val"] = ""

if "body_val" not in st.session_state:
    st.session_state["body_val"] = ""


# ════════════════════════════════════════════════════════════════════
# SAMPLE ARTICLES
# ════════════════════════════════════════════════════════════════════
EXAMPLES = {
    "fake": {
        "title": "SHOCKING: Government Secretly Controls Weather Machines",
        "body": (
            "Sources close to the deep state have confirmed that shadow elites are "
            "using HAARP technology to manipulate storms and droughts across the country. "
            "Citizens are being kept in the dark about this horrifying conspiracy. "
            "Share this before it gets deleted! The mainstream media won't tell you "
            "the truth about what is really happening behind closed doors. "
            "Wake up America — the elites are coming for your freedom and they will "
            "stop at nothing to silence the brave patriots who dare to speak out. "
            "This is the biggest cover-up in modern history and you deserve to know."
        )
    },
    "real": {
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
    "ambiguous": {
        "title": "New Study Claims Daily Coffee Consumption May Extend Lifespan",
        "body": (
            "Researchers at a European university published findings suggesting that "
            "moderate coffee consumption of three to four cups per day is linked to a "
            "reduced mortality risk of up to 17 percent. The observational study tracked "
            "50,000 participants over 15 years across multiple countries. Critics note "
            "the research was funded by a coffee industry advocacy group and that "
            "correlation does not imply causation. The findings have not yet been "
            "independently peer-reviewed. Nutritionists caution against drawing firm "
            "conclusions until additional studies replicate the results in diverse "
            "populations with controlled confounding variables."
        )
    }
}


# ════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .verdict-fake {
        background: linear-gradient(135deg, #ff4757, #c0392b);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: 2px;
        margin: 1rem 0;
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
        margin: 1rem 0;
    }
    .metric-card {
        background: #f8f9fa;
        border-left: 4px solid #3498db;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        margin: 0.4rem 0;
    }
    .footer {
        text-align: center;
        color: #888;
        font-size: 0.82em;
        padding: 1rem 0;
        border-top: 1px solid #eee;
        margin-top: 2rem;
    }
    #MainMenu {visibility: hidden;}
    footer     {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# LOAD MODELS  — cached so they load only ONCE per session
# ════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_models():
    tfidf = joblib.load('tfidf_vectorizer.joblib')
    model = joblib.load('tuned_xgboost_fake_news.joblib')
    le    = joblib.load('label_encoder.joblib')
    return tfidf, model, le


@st.cache_resource
def load_nlp_tools():
    sw        = set(stopwords.words('english'))
    lem       = WordNetLemmatizer()
    sentiment = SentimentIntensityAnalyzer()
    return sw, lem, sentiment


try:
    tfidf, model, le             = load_models()
    stop_words, lemmatizer, sia  = load_nlp_tools()
    MODELS_LOADED = True
except Exception as e:
    MODELS_LOADED = False
    MODEL_ERROR   = str(e)


# ════════════════════════════════════════════════════════════════════
# TEXT PREPROCESSING — identical to training notebook clean_text()
# ════════════════════════════════════════════════════════════════════
def clean_text(text: str) -> str:
    """Five-stage NLP normalisation. Must match training notebook exactly."""
    if not isinstance(text, str):
        return ''
    sentence = re.sub('[^a-zA-Z]', ' ', text)
    sentence = sentence.lower()
    sentence = ''.join([c for c in sentence if c not in string.punctuation])
    tokens   = sentence.split()
    clean_tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(clean_tokens)


def predict_article(title: str, body: str) -> tuple:
    """Full 7-step prediction pipeline mirroring the training notebook."""
    raw = (title.strip() + ': ' + body.strip()).strip(': ')
    if not raw:
        return None, None, None, None, '', 0, 0.0

    cleaned      = clean_text(raw)
    word_count   = len(cleaned.split())
    sentiment    = sia.polarity_scores(cleaned)['compound']
    num_features = np.array([[word_count, sentiment]])

    X_tfidf = tfidf.transform([cleaned])
    X_final = scipy.sparse.hstack([X_tfidf, num_features])

    pred_int   = model.predict(X_final)[0]
    proba      = model.predict_proba(X_final)[0]
    label      = le.inverse_transform([pred_int])[0]
    fake_prob  = float(proba[0])
    real_prob  = float(proba[1])
    confidence = max(fake_prob, real_prob)

    return label, confidence, fake_prob, real_prob, cleaned, word_count, sentiment


# ════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════
with st.sidebar:
    try:
        st.image(
            "https://drive.google.com/uc?export=view&id=1BSdTj6PVZwEnSCqucDa5DVHUGcPV_6jK",
            use_column_width=True
        )
    except Exception:
        st.markdown("### 🔍 TruthLens Institute")

    st.markdown("---")
    st.markdown("### 📊 Model Performance")
    for k, v in {
        "Model":        "Tuned XGBoost",
        "Accuracy":     "86.75%",
        "ROC-AUC":      "0.9393",
        "F1 (Fake)":    "0.87",
        "F1 (Real)":    "0.87",
        "Features":     "20,002",
        "Training set": "743 articles",
        "CV AUC":       "0.8987",
    }.items():
        st.markdown(
            f"<div class='metric-card'><b>{k}</b>: {v}</div>",
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("### 🧰 NLP Pipeline")
    st.markdown("""
1. Title + Body → `content`
2. `clean_text()` — 5 stages
   - Remove special chars & digits
   - Lowercase
   - Remove punctuation
   - Tokenise
   - Remove stopwords + lemmatise
3. Word count
4. VADER sentiment score
5. TF-IDF (20,000 features)
6. `scipy.sparse.hstack` → 20,002 dims
7. XGBoost → prediction + probability
    """)

    st.markdown("---")
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

if not MODELS_LOADED:
    st.error(
        f"⚠️ **Model files not found.** Ensure `tfidf_vectorizer.joblib`, "
        f"`tuned_xgboost_fake_news.joblib`, and `label_encoder.joblib` are "
        f"in the same folder as `app.py`.\n\n**Error:** `{MODEL_ERROR}`"
    )
    st.stop()

st.markdown("---")


# ════════════════════════════════════════════════════════════════════
# EXAMPLE BUTTONS
# WHY BUTTONS ARE ABOVE THE WIDGETS:
# When a button is clicked, it writes to st.session_state["title_val"]
# and st.session_state["body_val"], then calls st.rerun().
# On the rerun, the text widgets below read those values via value=.
# This is the correct Streamlit pattern for pre-filling text inputs
# from a button click — the key= binding pattern causes the error
# you experienced because Streamlit forbids writing to a session_state
# key that is already bound to a rendered widget in the same run.
# ════════════════════════════════════════════════════════════════════
st.markdown("### 📝 Enter the News Article")
st.markdown("##### 💡 Try a sample article:")

btn_col1, btn_col2, btn_col3 = st.columns(3)

if btn_col1.button("🔴 Fake Example", use_container_width=True):
    st.session_state["title_val"] = EXAMPLES["fake"]["title"]
    st.session_state["body_val"]  = EXAMPLES["fake"]["body"]
    st.rerun()

if btn_col2.button("🟢 Real Example", use_container_width=True):
    st.session_state["title_val"] = EXAMPLES["real"]["title"]
    st.session_state["body_val"]  = EXAMPLES["real"]["body"]
    st.rerun()

if btn_col3.button("🟡 Ambiguous Example", use_container_width=True):
    st.session_state["title_val"] = EXAMPLES["ambiguous"]["title"]
    st.session_state["body_val"]  = EXAMPLES["ambiguous"]["body"]
    st.rerun()

st.markdown("<br>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# INPUT WIDGETS
# WHY value= IS USED INSTEAD OF key=:
# Using value= reads from session_state without creating a two-way
# binding. This lets the example buttons write to session_state freely
# without triggering the "widget already rendered" exception.
# Manual user edits are synced back to session_state at the bottom
# of this block so they persist across reruns.
# ════════════════════════════════════════════════════════════════════
col_title, col_body = st.columns([1, 2])

with col_title:
    st.markdown("**Article Title** *(optional but recommended)*")
    title_input = st.text_input(
        label            = "Article Title",
        value            = st.session_state["title_val"],
        placeholder      = "e.g. Breaking: New Government Policy Announced...",
        label_visibility = "collapsed"
    )

with col_body:
    st.markdown("**Article Body** *(paste the full text)*")
    body_input = st.text_area(
        label            = "Article Body",
        value            = st.session_state["body_val"],
        placeholder      = "Paste the full article text here...",
        height           = 180,
        label_visibility = "collapsed"
    )

# Sync user edits back to session state so they survive reruns
st.session_state["title_val"] = title_input
st.session_state["body_val"]  = body_input


# ════════════════════════════════════════════════════════════════════
# ANALYSE BUTTON
# ════════════════════════════════════════════════════════════════════
st.markdown("---")
predict_clicked = st.button(
    "🚀  Analyse Article",
    type                = "primary",
    use_container_width = True
)

if predict_clicked:

    if not title_input.strip() and not body_input.strip():
        st.warning(
            "⚠️ Please enter an article title, body text, or both before clicking Analyse."
        )
        st.stop()

    with st.spinner("Running NLP pipeline — cleaning, extracting features, predicting..."):
        label, confidence, fake_prob, real_prob, cleaned, wc, sent = predict_article(
            title_input, body_input
        )

    if label is None:
        st.warning("⚠️ The text was empty after cleaning. Please provide more content.")
        st.stop()

    # ────────────────────────────────────────────────────────────────
    # RESULTS
    # ────────────────────────────────────────────────────────────────
    st.markdown("### 🎯 Prediction Result")

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

    m1, m2, m3 = st.columns(3)
    m1.metric("Verdict",         label.upper())
    m2.metric("Confidence",      f"{confidence * 100:.1f}%")
    m3.metric("VADER Sentiment", f"{sent:+.4f}")

    st.markdown("---")

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

    st.markdown("#### 🔎 Confidence Interpretation")
    if confidence >= 0.85:
        st.success(
            f"**High confidence ({confidence*100:.1f}%)** — The model is strongly "
            f"confident this article is **{label.upper()}**. The linguistic patterns "
            f"align clearly with {'fabricated' if label == 'fake' else 'credible'} news."
        )
    elif confidence >= 0.65:
        st.info(
            f"**Moderate confidence ({confidence*100:.1f}%)** — The model leans toward "
            f"**{label.upper()}** but detects mixed signals. Human review is advisable."
        )
    else:
        st.warning(
            f"**Low confidence ({confidence*100:.1f}%)** — This article is borderline. "
            f"The model cannot classify it reliably. "
            f"**Human editorial review is strongly recommended.**"
        )

    st.markdown("---")

    st.markdown("#### ⚠️ Important Disclaimer")
    st.markdown("""
> This tool is a **machine learning classifier** trained on US political news (2015–2016).
> Use it as a **screening aid**, not a definitive verdict. Always cross-check important
> articles with Snopes, PolitiFact, or AFP Fact Check.
    """)

    with st.expander("🔬 Show Full Pipeline Details"):
        st.markdown("**Cleaned Content (after 5-stage NLP normalisation):**")
        st.code(
            (cleaned[:600] + "..." if len(cleaned) > 600 else cleaned) or "(empty)",
            language=None
        )

        d1, d2, d3 = st.columns(3)
        d1.metric("Word Count (post-clean)", f"{wc:,}")
        d2.metric("VADER Sentiment Score",   f"{sent:+.4f}")
        d3.metric(
            "Sentiment",
            "Positive" if sent > 0.05 else "Negative" if sent < -0.05 else "Neutral"
        )

        st.markdown("**Feature Matrix:**")
        st.markdown("""
| Component | Value |
|-----------|-------|
| TF-IDF features | 20,000 (unigrams + bigrams) |
| Numerical features | 2 (word_count, sentiment) |
| **Total dimensions** | **20,002** |
| Model | XGBClassifier (learning_rate=0.05, max_depth=3, n_estimators=200) |
        """)

        st.markdown("**All 7 Model Results:**")
        st.markdown("""
| Model | Accuracy | ROC-AUC | F1 (Fake) | F1 (Real) |
|-------|----------|---------|-----------|-----------|
| **Tuned XGBoost ✅** | **86.75%** | **0.9393** | **0.87** | **0.87** |
| XGBoost (Baseline) | 84.34% | 0.9051 | 0.84 | 0.84 |
| Logistic Regression | 79.52% | 0.8560 | 0.78 | 0.80 |
| Random Forest | 79.52% | 0.8656 | 0.80 | 0.79 |
| Tuned LR | 78.31% | 0.8513 | 0.78 | 0.79 |
| Decision Tree | 72.29% | 0.7227 | 0.73 | 0.72 |
| SVC (RBF) | 68.67% | 0.7311 | 0.74 | 0.61 |
        """)


# ════════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='footer'>
    Built by <strong>Adewale Samson Adeagbo</strong> &nbsp;|&nbsp;
    <a href='https://linkedin.com/in/adewalesamsonadeagbo' target='_blank'>LinkedIn</a>
    &nbsp;·&nbsp;
    <a href='https://github.com/cssadewale' target='_blank'>GitHub</a>
    &nbsp;·&nbsp;
    adewalesamsonadeagbo@gmail.com &nbsp;·&nbsp; 08100866322
    <br>
    3MTT Capstone Project 2 &nbsp;·&nbsp; The TruthLens Institute &nbsp;·&nbsp;
    Tuned XGBoost &nbsp;·&nbsp; Accuracy: 86.75% &nbsp;·&nbsp; AUC: 0.9393
</div>
""", unsafe_allow_html=True)
