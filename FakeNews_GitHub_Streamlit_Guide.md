# 🚀 Complete Deployment Guide
## From Google Colab to GitHub to Streamlit — Fake News Detector
### Adewale Samson Adeagbo | adewalesamsonadeagbo@gmail.com | github.com/cssadewale

---

## 📋 Table of Contents

1. [What You Will Build](#1-what-you-will-build)
2. [What You Need Before Starting](#2-what-you-need-before-starting)
3. [Step 1 — Save Your Models and Vectoriser from Colab](#3-step-1--save-your-models-and-vectoriser-from-colab)
4. [Step 2 — Build the Streamlit App File](#4-step-2--build-the-streamlit-app-file)
5. [Step 3 — Create the requirements.txt File](#5-step-3--create-the-requirementstxt-file)
6. [Step 4 — Create the README.md File](#6-step-4--create-the-readmemd-file)
7. [Step 5 — Set Up Your GitHub Repository](#7-step-5--set-up-your-github-repository)
8. [Step 6 — Upload Everything to GitHub](#8-step-6--upload-everything-to-github)
9. [Step 7 — Deploy on Streamlit Cloud](#9-step-7--deploy-on-streamlit-cloud)
10. [Step 8 — Test Your Live App](#10-step-8--test-your-live-app)
11. [Troubleshooting Common Errors](#11-troubleshooting-common-errors)
12. [Your Final Portfolio Entry](#12-your-final-portfolio-entry)

---

## 1. What You Will Build

By the end of this guide you will have:

- ✅ A **live web app** at a public URL (e.g. `https://fakenews-truthlens-adewale.streamlit.app`)
- ✅ A **GitHub repository** at `github.com/cssadewale/fake-news-detector`
- ✅ A fully working **fake news classifier** that anyone can use in their browser
- ✅ A professional portfolio entry you can link on LinkedIn and your CV

The app will take a news article title and body text as input, clean and process it exactly the way your notebook does, and return a **FAKE or REAL prediction** with a confidence percentage — powered by your Tuned XGBoost model (Accuracy: 86.75%, AUC: 0.9393).

---

## 2. What You Need Before Starting

Before you take a single step, make sure you have all of these ready:

| Item | Where to Get It | Status |
|------|----------------|--------|
| Google account (for Colab) | Already have it | ✅ |
| GitHub account | github.com — free | Check |
| Streamlit account | streamlit.io — free | Check |

**Create your GitHub account** (if you do not have one):
1. Go to **github.com**
2. Click **Sign up**
3. Use username: **cssadewale** (to match your portfolio)
4. Use email: **adewalesamsonadeagbo@gmail.com**
5. Verify your email

**Create your Streamlit account** (if you do not have one):
1. Go to **streamlit.io**
2. Click **Sign up**
3. Choose **Continue with GitHub** — this links both accounts automatically. This is important.

---

## 3. Step 1 — Save Your Models and Vectoriser from Colab

Your Streamlit app needs three files that live inside your Google Colab session right now. You must save them permanently to Google Drive so they survive after the session ends.

### Why three files?

When a user types an article into your app, the app must:
1. **Clean the text** — using your `clean_text()` function
2. **Vectorise it** — using the exact same `TfidfVectorizer` that was fitted on your training data
3. **Predict** — using your Tuned XGBoost model

If you only save the model and not the vectoriser, the app will crash because the model expects 20,002 features in a specific order. The vectoriser was fitted on your 826-record training corpus and remembers that exact vocabulary.

### Run this code at the END of your Colab notebook

Open your Colab notebook. Scroll to the very bottom. Add a new code cell and paste this entire block:

```python
# ════════════════════════════════════════════════════════
# SAVE ALL DEPLOYMENT ARTEFACTS TO GOOGLE DRIVE
# Run this cell after all models have been trained
# ════════════════════════════════════════════════════════
import joblib
import os
from google.colab import drive

# ── Mount Google Drive ────────────────────────────────
drive.mount('/content/drive')

# ── Create a dedicated folder for deployment files ───
save_dir = '/content/drive/MyDrive/FakeNews_Deployment'
os.makedirs(save_dir, exist_ok=True)

# ── Save the TF-IDF Vectoriser (CRITICAL — must match training) ──
tfidf_path = os.path.join(save_dir, 'tfidf_vectorizer.joblib')
joblib.dump(tfidf, tfidf_path)
print(f"✓ TF-IDF vectoriser saved: {tfidf_path}")

# ── Save the Tuned XGBoost (Best / Production Model) ────────────
xgb_path = os.path.join(save_dir, 'tuned_xgboost_fake_news.joblib')
joblib.dump(tuned_xgb_model, xgb_path)
print(f"✓ Tuned XGBoost saved: {xgb_path}")

# ── Save the Tuned LR (Interpretability Model) ──────────────────
lr_path = os.path.join(save_dir, 'tuned_logistic_regression_fake_news.joblib')
joblib.dump(tuned_log_reg_model, lr_path)
print(f"✓ Tuned LR saved: {lr_path}")

# ── Save the Label Encoder (maps 0/1 back to fake/real) ─────────
le_path = os.path.join(save_dir, 'label_encoder.joblib')
joblib.dump(le, le_path)
print(f"✓ Label encoder saved: {le_path}")

# ── Confirm all four files exist ─────────────────────────────────
print("\n── All deployment artefacts saved ──────────────────────")
for fname in ['tfidf_vectorizer.joblib', 'tuned_xgboost_fake_news.joblib',
              'tuned_logistic_regression_fake_news.joblib', 'label_encoder.joblib']:
    full_path = os.path.join(save_dir, fname)
    size_kb = os.path.getsize(full_path) / 1024
    print(f"  {fname:<45}  ({size_kb:.1f} KB)")

print(f"\n✓ Deployment folder: {save_dir}")
print("  Go to Google Drive → MyDrive → FakeNews_Deployment to download these files.")
```

Run the cell. You will see a Google Drive authorisation popup — click **Allow**. Wait for all four files to save.

### Download the files to your computer

1. Go to **drive.google.com**
2. Navigate to **My Drive → FakeNews_Deployment**
3. Download all four files:
   - `tfidf_vectorizer.joblib`
   - `tuned_xgboost_fake_news.joblib`
   - `tuned_logistic_regression_fake_news.joblib`
   - `label_encoder.joblib`
4. Save them all to a folder on your computer called `fake-news-detector`

> ⚠️ **Important:** The XGBoost model file may be large (several MB). This is normal. Do NOT rename any of these files — the app code references them by exact name.

---

## 4. Step 2 — Build the Streamlit App File

This is the core file that powers your web app. Every line below is precisely designed to replicate your notebook's preprocessing pipeline so that predictions are consistent.

### Create a file called `app.py`

Inside your `fake-news-detector` folder, create a new file called exactly `app.py`. Copy and paste this complete code:

```python
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
nltk.download('stopwords',    quiet=True)
nltk.download('wordnet',      quiet=True)
nltk.download('vader_lexicon', quiet=True)


# ════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title  = "Fake News Detector | TruthLens",
    page_icon   = "🔍",
    layout      = "wide",
    initial_sidebar_state = "expanded"
)


# ════════════════════════════════════════════════════════════════════
# LOAD MODELS (cached so they only load once per session)
# ════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_models():
    """Load the TF-IDF vectoriser, XGBoost model, and label encoder.
    Uses @st.cache_resource so models load only once — not on every click."""
    tfidf    = joblib.load('tfidf_vectorizer.joblib')
    model    = joblib.load('tuned_xgboost_fake_news.joblib')
    le       = joblib.load('label_encoder.joblib')
    return tfidf, model, le

tfidf, model, le = load_models()


# ════════════════════════════════════════════════════════════════════
# TEXT PREPROCESSING — IDENTICAL TO NOTEBOOK PIPELINE
# This function must exactly match the clean_text() used in training
# ════════════════════════════════════════════════════════════════════
stop_words  = set(stopwords.words('english'))
lemmatizer  = WordNetLemmatizer()
sia         = SentimentIntensityAnalyzer()

def clean_text(text: str) -> str:
    """Five-stage NLP normalisation pipeline.
    Must be identical to the training notebook's clean_text() function."""
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


def predict_article(title: str, body: str):
    """Run the full prediction pipeline on a single article.
    
    Replicates the exact feature engineering from the training notebook:
    - Concatenate title + ': ' + body
    - Apply clean_text()
    - Extract word_count and VADER sentiment
    - Transform with fitted TF-IDF vectoriser
    - Stack with numerical features using scipy.sparse.hstack
    - Predict with Tuned XGBoost
    
    Returns:
        label       (str)  : 'fake' or 'real'
        confidence  (float): prediction probability for the predicted class (0–1)
        fake_prob   (float): raw probability of being fake
        real_prob   (float): raw probability of being real
    """
    # ── Step 1: Combine and clean ─────────────────────────────────────────────
    raw_content     = title + ': ' + body
    cleaned_content = clean_text(raw_content)

    # ── Step 2: Extract numerical features ───────────────────────────────────
    word_count  = len(cleaned_content.split())
    sentiment   = sia.polarity_scores(cleaned_content)['compound']
    num_features = np.array([[word_count, sentiment]])

    # ── Step 3: TF-IDF vectorisation ─────────────────────────────────────────
    X_tfidf = tfidf.transform([cleaned_content])   # shape: (1, 20000)

    # ── Step 4: Stack into final feature matrix ───────────────────────────────
    X_final = scipy.sparse.hstack([X_tfidf, num_features])   # shape: (1, 20002)

    # ── Step 5: Predict ───────────────────────────────────────────────────────
    pred_int    = model.predict(X_final)[0]          # 0 = fake, 1 = real
    proba       = model.predict_proba(X_final)[0]    # [P(fake), P(real)]

    label       = le.inverse_transform([pred_int])[0]
    fake_prob   = float(proba[0])
    real_prob   = float(proba[1])
    confidence  = max(fake_prob, real_prob)

    return label, confidence, fake_prob, real_prob


# ════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image(
        "https://drive.google.com/uc?export=view&id=1BSdTj6PVZwEnSCqucDa5DVHUGcPV_6jK",
        use_column_width=True
    )
    st.markdown("---")
    st.markdown("### 📊 Model Information")
    st.markdown("""
| Attribute | Value |
|-----------|-------|
| **Model** | Tuned XGBoost |
| **Accuracy** | 86.75% |
| **ROC-AUC** | 0.9393 |
| **F1 (Fake)** | 0.87 |
| **F1 (Real)** | 0.87 |
| **Features** | 20,002 |
| **Training set** | 743 articles |
    """)

    st.markdown("---")
    st.markdown("### 🧰 Pipeline")
    st.markdown("""
1. Title + Body → `content`
2. `clean_text()` (5-stage NLP)
3. VADER sentiment score
4. Word count extraction
5. TF-IDF (20,000 features)
6. Feature stacking
7. XGBoost prediction
    """)

    st.markdown("---")
    st.markdown("### 👤 About the Author")
    st.markdown("""
**Adewale Samson Adeagbo**  
Lead Data Scientist / ML Engineer  
Mathematics Teacher | 10+ Years Experience

📧 adewalesamsonadeagbo@gmail.com  
📱 08100866322  
🔗 [LinkedIn](https://linkedin.com/in/adewalesamsonadeagbo)  
💻 [GitHub](https://github.com/cssadewale)  

*3MTT Capstone Project 2*  
*The TruthLens Institute*
    """)


# ════════════════════════════════════════════════════════════════════
# MAIN PAGE HEADER
# ════════════════════════════════════════════════════════════════════
st.title("🔍 Fake News Detector")
st.markdown("#### Powered by Tuned XGBoost · Accuracy: 86.75% · AUC: 0.9393")
st.markdown("""
This tool classifies news articles as **FAKE** or **REAL** using Natural Language Processing
and Machine Learning. Paste an article's title and body text below to get an instant prediction.

> Built as part of the **3MTT Data Science Capstone Programme** for
> **The TruthLens Institute** — a global research organisation combating misinformation.
""")
st.markdown("---")


# ════════════════════════════════════════════════════════════════════
# INPUT SECTION
# ════════════════════════════════════════════════════════════════════
st.markdown("### 📝 Enter the News Article")

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("**Article Title**")
    title_input = st.text_input(
        label       = "title_hidden",
        placeholder = "e.g. Breaking: New Government Policy Announced...",
        label_visibility = "collapsed"
    )

with col2:
    st.markdown("**Article Body**")
    body_input = st.text_area(
        label       = "body_hidden",
        placeholder = "Paste the full article text here...",
        height      = 200,
        label_visibility = "collapsed"
    )

# ── Example articles button ───────────────────────────────────────────────────
st.markdown("##### 💡 Try a sample article:")
col_ex1, col_ex2, col_ex3 = st.columns(3)

EXAMPLES = {
    "🔴 Fake Example": {
        "title": "SHOCKING: Government Secretly Controls Weather Machines",
        "body":  "Sources close to the deep state have confirmed that shadow elites "
                 "are using HAARP technology to manipulate storms and drought across "
                 "the country. Citizens are being kept in the dark. Share this before "
                 "it gets deleted! The mainstream media won't tell you the truth about "
                 "what's really happening behind closed doors. Wake up America!"
    },
    "🟢 Real Example": {
        "title": "Senate Passes Infrastructure Bill After Bipartisan Vote",
        "body":  "The United States Senate passed a $1.2 trillion infrastructure bill "
                 "on Tuesday following weeks of bipartisan negotiations. The legislation "
                 "allocates funding for road repairs, broadband expansion, and public "
                 "transit upgrades. President Biden praised the vote as a historic "
                 "achievement, while Republican Senator Rob Portman called it a victory "
                 "for the American people. The bill now moves to the House for consideration."
    },
    "🟡 Ambiguous Example": {
        "title": "New Study Claims Coffee May Extend Lifespan",
        "body":  "Researchers at a European university published findings suggesting "
                 "that moderate coffee consumption is linked to reduced mortality risk. "
                 "The study tracked 50,000 participants over 15 years. Critics note the "
                 "research was funded by a coffee industry group and that correlation "
                 "does not imply causation. The findings have not yet been peer-reviewed "
                 "by independent scientists."
    }
}

if col_ex1.button("🔴 Fake Example"):
    title_input = EXAMPLES["🔴 Fake Example"]["title"]
    body_input  = EXAMPLES["🔴 Fake Example"]["body"]
    st.rerun()

if col_ex2.button("🟢 Real Example"):
    title_input = EXAMPLES["🟢 Real Example"]["title"]
    body_input  = EXAMPLES["🟢 Real Example"]["body"]
    st.rerun()

if col_ex3.button("🟡 Ambiguous Example"):
    title_input = EXAMPLES["🟡 Ambiguous Example"]["title"]
    body_input  = EXAMPLES["🟡 Ambiguous Example"]["body"]
    st.rerun()


# ════════════════════════════════════════════════════════════════════
# PREDICTION SECTION
# ════════════════════════════════════════════════════════════════════
st.markdown("---")

predict_btn = st.button("🚀 Analyse Article", type="primary", use_container_width=True)

if predict_btn:
    if not title_input.strip() and not body_input.strip():
        st.warning("⚠️ Please enter at least a title or some article text before clicking Analyse.")
    else:
        with st.spinner("Running NLP pipeline and predicting..."):
            label, confidence, fake_prob, real_prob = predict_article(
                title_input.strip(), body_input.strip()
            )

        # ── Result display ────────────────────────────────────────────────────
        st.markdown("### 🎯 Prediction Result")

        res_col1, res_col2, res_col3 = st.columns(3)

        if label == 'fake':
            res_col1.error(f"🚨 **FAKE NEWS**")
            verdict_color = "#e74c3c"
        else:
            res_col1.success(f"✅ **REAL NEWS**")
            verdict_color = "#2ecc71"

        res_col2.metric("Confidence", f"{confidence * 100:.1f}%")
        res_col3.metric("Verdict", label.upper())

        # ── Probability breakdown ─────────────────────────────────────────────
        st.markdown("#### 📊 Probability Breakdown")
        prob_col1, prob_col2 = st.columns(2)

        prob_col1.markdown("**Probability: FAKE**")
        prob_col1.progress(fake_prob)
        prob_col1.markdown(f"`{fake_prob * 100:.1f}%`")

        prob_col2.markdown("**Probability: REAL**")
        prob_col2.progress(real_prob)
        prob_col2.markdown(f"`{real_prob * 100:.1f}%`")

        # ── Confidence interpretation ─────────────────────────────────────────
        st.markdown("#### 🔎 Confidence Interpretation")
        if confidence >= 0.85:
            st.success(f"**High confidence** ({confidence*100:.1f}%) — The model is strongly confident in this prediction.")
        elif confidence >= 0.65:
            st.info(f"**Moderate confidence** ({confidence*100:.1f}%) — The model leans toward {label.upper()} but the article has mixed signals.")
        else:
            st.warning(f"**Low confidence** ({confidence*100:.1f}%) — The article is borderline. Human review is recommended before acting on this prediction.")

        # ── Pipeline trace ────────────────────────────────────────────────────
        with st.expander("🔬 Show Pipeline Details"):
            cleaned = clean_text(title_input + ': ' + body_input)
            wc   = len(cleaned.split())
            sent = sia.polarity_scores(cleaned)['compound']

            st.markdown("**Preprocessing Output:**")
            st.code(cleaned[:500] + ("..." if len(cleaned) > 500 else ""), language=None)

            detail_col1, detail_col2 = st.columns(2)
            detail_col1.metric("Word Count (after cleaning)", wc)
            detail_col2.metric("VADER Sentiment Score", f"{sent:.4f}")

            st.markdown("""
**Feature Matrix:**
- TF-IDF features: 20,000 (unigrams + bigrams, max_features=20,000)
- Numerical features: 2 (word_count, sentiment)
- Total dimensions: **20,002**
- Vectoriser fitted on: 743-sample balanced training corpus
            """)


# ════════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 0.85em;'>
Built by <strong>Adewale Samson Adeagbo</strong> |
<a href='https://linkedin.com/in/adewalesamsonadeagbo'>LinkedIn</a> ·
<a href='https://github.com/cssadewale'>GitHub</a> ·
adewalesamsonadeagbo@gmail.com<br>
3MTT Capstone Project 2 — The TruthLens Institute |
Tuned XGBoost Classifier | Accuracy: 86.75% | AUC: 0.9393
</div>
""", unsafe_allow_html=True)
```

Save the file as `app.py` inside your `fake-news-detector` folder.

---

## 5. Step 3 — Create the requirements.txt File

Streamlit Cloud needs to know exactly which Python libraries to install when it builds your app. This file tells it.

Create a file called exactly `requirements.txt` (no capital letters, no `.py` extension). Put it in the same `fake-news-detector` folder as `app.py`. Copy this content exactly:

```
streamlit>=1.28.0
joblib>=1.3.0
scikit-learn>=1.3.0
xgboost>=1.7.0
nltk>=3.8.0
scipy>=1.11.0
numpy>=1.24.0
```

> **Why these specific versions?** These are the minimum versions that work together reliably. Streamlit Cloud will install these automatically when it deploys your app.

---

## 6. Step 4 — Create the README.md File

Your README is the front page of your GitHub repository. It is the first thing recruiters, hiring managers, and colleagues see. Make it professional.

Create a file called `README.md` in your `fake-news-detector` folder. Copy this content:

```markdown
# 🔍 Fake News Detector — TruthLens Institute
### NLP + Machine Learning | 3MTT Capstone Project 2

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

---

## 🌐 Live Demo
**→ [Launch the App](https://your-app-url.streamlit.app)**

---

## 📌 Project Overview

This project builds a **binary fake news classifier** for The TruthLens Institute —
a global research organisation dedicated to combating misinformation.

Given the title and body text of a news article, the app predicts whether it is
**FAKE** or **REAL** using a complete NLP preprocessing pipeline and a
Tuned XGBoost Classifier.

| Attribute | Value |
|-----------|-------|
| **Best Model** | Tuned XGBoost Classifier |
| **Accuracy** | 86.75% |
| **ROC-AUC** | 0.9393 |
| **F1-Score (Fake)** | 0.87 |
| **F1-Score (Real)** | 0.87 |
| **Dataset** | 12,999 news articles (8 misinformation categories) |
| **Final Training Set** | 826 balanced records (413 fake / 413 real) |
| **Feature Matrix** | 20,002 dimensions (TF-IDF + word_count + sentiment) |

---

## 🧠 Technical Pipeline

```
Raw Article (title + body)
        ↓
  clean_text() — 5-stage NLP normalisation
  (remove special chars → lowercase → remove punctuation
   → tokenise → remove stopwords + lemmatise)
        ↓
  word_count  ←→  VADER sentiment score
        ↓
  TF-IDF Vectoriser (20,000 features, unigrams + bigrams)
        ↓
  scipy.sparse.hstack → (1, 20,002) feature matrix
        ↓
  Tuned XGBoost → FAKE / REAL + confidence score
```

---

## 📊 Model Comparison

| Model | Accuracy | ROC-AUC | F1 (Fake) | F1 (Real) |
|-------|----------|---------|-----------|-----------|
| **Tuned XGBoost ✅** | **86.75%** | **0.9393** | **0.87** | **0.87** |
| XGBoost (Baseline) | 84.34% | 0.9051 | 0.84 | 0.84 |
| Logistic Regression | 79.52% | 0.8560 | 0.78 | 0.80 |
| Random Forest | 79.52% | 0.8656 | 0.80 | 0.79 |
| Tuned Logistic Regression | 78.31% | 0.8513 | 0.78 | 0.79 |
| Decision Tree | 72.29% | 0.7227 | 0.73 | 0.72 |
| SVC (RBF kernel) | 68.67% | 0.7311 | 0.74 | 0.61 |

---

## 🗂️ Repository Structure

```
fake-news-detector/
├── app.py                                  ← Streamlit web application
├── requirements.txt                        ← Python dependencies
├── tfidf_vectorizer.joblib                 ← Fitted TF-IDF vectoriser
├── tuned_xgboost_fake_news.joblib          ← Production model (XGBoost)
├── tuned_logistic_regression_fake_news.joblib  ← Interpretability model
├── label_encoder.joblib                    ← Label encoder (fake=0, real=1)
└── README.md                               ← This file
```

---

## 🚀 Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/cssadewale/fake-news-detector.git
cd fake-news-detector

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

---

## 📁 Dataset

The original dataset contains **12,999 news articles** with 20 columns including
article content, publication metadata, and engagement signals. The `type` column
contains 8 misinformation categories (`bs`, `bias`, `conspiracy`, `hate`, `satire`,
`state`, `junksci`, `fake`), which were mapped to a binary `fake`/`real` label.

After mapping and downsampling to address a 29.8:1 class imbalance, the final
dataset contains **826 balanced records** (413 per class).

---

## 👤 Author

**Adewale Samson Adeagbo**
Lead Data Scientist / ML Engineer | Mathematics Teacher (10+ Years)

- 📧 Email: adewalesamsonadeagbo@gmail.com
- 📱 Phone: 08100866322
- 🔗 LinkedIn: [linkedin.com/in/adewalesamsonadeagbo](https://linkedin.com/in/adewalesamsonadeagbo)
- 💻 GitHub: [github.com/cssadewale](https://github.com/cssadewale)
- 🌐 Portfolio: [hmgconcepts.business.site](https://hmgconcepts.business.site)

---

## 📜 Programme

3MTT (3 Million Technical Talent) Data Science Capstone Project 2
Nigeria, 2025
```

Save this file. You will update the two placeholder URLs (`your-app-url.streamlit.app`) in Step 9 after deployment.

---

## 7. Step 5 — Set Up Your GitHub Repository

### Check your folder

Before going to GitHub, confirm your `fake-news-detector` folder on your computer looks exactly like this:

```
fake-news-detector/
├── app.py
├── requirements.txt
├── README.md
├── tfidf_vectorizer.joblib
├── tuned_xgboost_fake_news.joblib
├── tuned_logistic_regression_fake_news.joblib
└── label_encoder.joblib
```

If any of these 7 files is missing, stop and go back to the earlier step.

### Create the repository on GitHub

1. Open your browser and go to **github.com**
2. Make sure you are logged in as **cssadewale**
3. Click the **+** icon in the top right corner
4. Click **New repository**
5. Fill in the form exactly as shown:

| Field | What to Enter |
|-------|--------------|
| **Repository name** | `fake-news-detector` |
| **Description** | `NLP fake news classifier · Tuned XGBoost · AUC 0.9393 · 3MTT Capstone 2` |
| **Visibility** | ✅ Public (required for Streamlit Cloud free tier) |
| **Add a README file** | ❌ Do NOT tick this (you have your own README) |
| **Add .gitignore** | None |
| **Choose a licence** | None |

6. Click the green **Create repository** button
7. You will land on an empty repository page. **Leave this tab open** — you need the URL in the next step.

---

## 8. Step 6 — Upload Everything to GitHub

You have two options. **Option A** (drag and drop) is the easiest for most people. **Option B** (Git command line) is more professional and faster once you learn it.

---

### Option A — Upload via GitHub Website (Easiest, Recommended)

**This works entirely in your browser. No software to install.**

#### Upload the code files first

1. On your empty repository page, click **uploading an existing file** (in the blue text)
2. Open your `fake-news-detector` folder on your computer
3. Select these **three files only** first:
   - `app.py`
   - `requirements.txt`
   - `README.md`
4. Drag them into the GitHub upload window
5. At the bottom of the page, in the **Commit changes** section:
   - First box: `Add Streamlit app, requirements, and README`
   - Second box: leave empty
6. Make sure **Commit directly to the `main` branch** is selected
7. Click **Commit changes**

GitHub will process the upload. Wait for it to finish.

#### Upload the model files

The `.joblib` files are binary files. Upload them the same way:

1. On your repository page, click **Add file → Upload files**
2. Drag in all four `.joblib` files:
   - `tfidf_vectorizer.joblib`
   - `tuned_xgboost_fake_news.joblib`
   - `tuned_logistic_regression_fake_news.joblib`
   - `label_encoder.joblib`
3. In the **Commit changes** section:
   - First box: `Add trained models and vectoriser`
4. Click **Commit changes**

Wait for the upload. Large files (your XGBoost model) may take a minute.

#### Verify all 7 files are present

After both uploads, your repository's file list should show:

```
📄 README.md
📄 app.py
📄 label_encoder.joblib
📄 requirements.txt
📄 tuned_logistic_regression_fake_news.joblib
📄 tuned_xgboost_fake_news.joblib
📄 tfidf_vectorizer.joblib
```

If you see all 7, you are ready for deployment.

---

### Option B — Upload via Git (Command Line, More Professional)

Use this if you have Git installed on your computer. If you are not sure, use Option A.

**Check if Git is installed:** Open a terminal (Windows: Command Prompt or PowerShell; Mac/Linux: Terminal) and type:
```bash
git --version
```
If you see a version number, Git is installed. If you see an error, download Git from **git-scm.com** and install it, then come back.

**Run these commands in order:**

```bash
# 1. Navigate to your project folder
#    (Replace the path with where you actually saved your folder)
cd /path/to/fake-news-detector

# 2. Initialise a local Git repository
git init

# 3. Connect it to your GitHub repository
#    (Replace cssadewale if you used a different username)
git remote add origin https://github.com/cssadewale/fake-news-detector.git

# 4. Stage all 7 files for commit
git add .

# 5. Create your first commit
git commit -m "Initial commit: Streamlit app, models, and README"

# 6. Set the branch name to main
git branch -M main

# 7. Push to GitHub
git push -u origin main
```

GitHub will ask for your username and password. For the password, use a **Personal Access Token**, not your GitHub login password. To create one: GitHub → Settings → Developer Settings → Personal Access Tokens → Tokens (classic) → Generate new token → tick `repo` → copy the token and use it as your password.

---

## 9. Step 7 — Deploy on Streamlit Cloud

Your code is on GitHub. Now you connect it to Streamlit Cloud and get a public URL.

### Connect GitHub to Streamlit Cloud

1. Go to **share.streamlit.io** in your browser
2. Click **Sign in with GitHub**
3. Authorise Streamlit to access your GitHub account
4. You will land on the Streamlit Cloud dashboard

### Create a new deployment

1. Click **New app** (top right, blue button)
2. Fill in the deployment form:

| Field | What to Enter |
|-------|--------------|
| **Repository** | `cssadewale/fake-news-detector` |
| **Branch** | `main` |
| **Main file path** | `app.py` |
| **App URL** (optional) | `fakenews-truthlens-adewale` |

3. Leave all other settings at their defaults
4. Click **Deploy!**

### Watch the build logs

Streamlit Cloud will now:
1. Clone your GitHub repository
2. Install all packages listed in `requirements.txt`
3. Start your app

This takes **3 to 8 minutes** on the first deployment. You will see a build log with green output. If you see red error text, go to the Troubleshooting section below.

### Get your public URL

When the build finishes, your app will be live at:
```
https://fakenews-truthlens-adewale.streamlit.app
```
(or a similar URL if that name was taken)

Copy this URL. You will need it in the next step.

### Update your README with the live URL

1. Go back to your GitHub repository
2. Click `README.md` in the file list
3. Click the pencil icon (✏️) to edit
4. Find both instances of `your-app-url.streamlit.app` and replace them with your actual URL
5. Scroll down → **Commit changes** → **Commit directly to main**

---

## 10. Step 8 — Test Your Live App

Open your app URL in your browser. Test it thoroughly before sharing it.

### Test checklist

**Test 1 — The fake example:**
1. Click the **🔴 Fake Example** button
2. Click **Analyse Article**
3. ✅ Expected: **FAKE NEWS** prediction with high confidence (>70%)

**Test 2 — The real example:**
1. Click the **🟢 Real Example** button
2. Click **Analyse Article**
3. ✅ Expected: **REAL NEWS** prediction with high confidence (>70%)

**Test 3 — Empty input:**
1. Clear both fields
2. Click **Analyse Article**
3. ✅ Expected: Yellow warning message, no crash

**Test 4 — Your own article:**
1. Find a news article online
2. Copy its headline into the title field
3. Copy its first two paragraphs into the body field
4. Click **Analyse Article**
5. ✅ Expected: A prediction with a probability score

**Test 5 — Pipeline details:**
1. Submit any article
2. Click **🔬 Show Pipeline Details**
3. ✅ Expected: Cleaned text, word count, sentiment score, and feature matrix info visible

If all 5 tests pass, your app is working correctly.

---

## 11. Troubleshooting Common Errors

### Error: `ModuleNotFoundError: No module named 'xgboost'`

**Cause:** The `requirements.txt` was not found or has a typo.

**Fix:**
1. On GitHub, click `requirements.txt`
2. Confirm it contains `xgboost>=1.7.0`
3. On Streamlit Cloud → your app → **Reboot app**

---

### Error: `FileNotFoundError: tfidf_vectorizer.joblib`

**Cause:** The `.joblib` files were not uploaded to GitHub, or were uploaded to a subfolder.

**Fix:**
1. Go to your GitHub repository
2. Confirm `tfidf_vectorizer.joblib` appears at the **root level** (not inside any folder)
3. If it is in a subfolder, click it → Download → re-upload at root level

---

### Error: `NLTK resource not found`

**Cause:** NLTK downloads failed silently.

**Fix:** Add this to the top of `app.py`, right after the imports:
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
nltk.download('punkt')
```
Then commit the change to GitHub. Streamlit will pick it up automatically.

---

### Error: `ValueError: X has N features, expected 20002`

**Cause:** The `tfidf_vectorizer.joblib` file does not match the model — either a different vectoriser was saved or the file got corrupted.

**Fix:** Go back to your Colab notebook. Re-run the save cell from Step 1 of this guide. Download the fresh files and re-upload them to GitHub.

---

### App is very slow on first load

**Cause:** Streamlit Cloud spins down inactive apps. The first visit after inactivity triggers a cold start.

**Fix:** This is normal behaviour on the free tier. The `@st.cache_resource` decorator on `load_models()` ensures models only load once per session, not on every click.

---

### Build fails with `ERROR: Could not find a version that satisfies the requirement`

**Cause:** A package version in `requirements.txt` is not available.

**Fix:** Change the version constraints to be less strict. Replace `>=` with nothing, for example:
```
xgboost
scikit-learn
streamlit
```
Streamlit will install the latest available version.

---

## 12. Your Final Portfolio Entry

Once the app is live, here is how to present it across your professional profiles.

---

### LinkedIn Post

Copy and adapt this for your LinkedIn:

---

🚀 **Excited to share my 3MTT Capstone Project 2 — now live as a web app!**

🔍 **Fake News Detector** built for The TruthLens Institute — a global organisation dedicated to combating misinformation.

**What it does:**
Given a news article, it classifies it as FAKE or REAL using NLP + Machine Learning.

**Technical highlights:**
- Dataset: 12,999 real-world news articles across 8 misinformation categories
- Pipeline: 5-stage text normalisation → TF-IDF (20,000 features) → XGBoost
- Best model: Tuned XGBoost — **86.75% accuracy, AUC 0.9393**
- 5 models evaluated: Logistic Regression, SVC, Decision Tree, Random Forest, XGBoost

**Try it live:** [your-app-url.streamlit.app]

This project is part of my ongoing transition from Mathematics teaching (10+ years) into Data Science — building real tools, not just notebooks.

Built entirely on mobile (Android tablet + Acode editor) 📱

#DataScience #MachineLearning #NLP #FakeNews #3MTT #Python #XGBoost #Streamlit

---

### GitHub README badge

Your README already includes this badge — confirm it appears on your repository:
```
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)
```

### CV / Resume entry

```
Fake News Detection System | NLP + Machine Learning               2025
The TruthLens Institute | 3MTT Capstone Project 2

• Built end-to-end NLP classification pipeline on 12,999 real-world news articles
• Evaluated 5 classifiers; Tuned XGBoost selected as production model
  (Accuracy: 86.75%, ROC-AUC: 0.9393, F1: 0.87 both classes)
• Features: TF-IDF (20,000 unigrams + bigrams) + VADER sentiment + word count
• Deployed as live Streamlit web application on Streamlit Cloud
• GitHub: github.com/cssadewale/fake-news-detector
• Live app: [your-app-url.streamlit.app]
Tech stack: Python · XGBoost · scikit-learn · NLTK · Streamlit · joblib
```

---

### Your complete portfolio links

Once deployed, your portfolio now includes:

| Project | Tech | Live App | GitHub |
|---------|------|----------|--------|
| Fake News Detector | XGBoost, NLP, TF-IDF | ✅ Streamlit | github.com/cssadewale/fake-news-detector |
| Income Prediction | ML, Streamlit | ✅ Deployed | github.com/cssadewale |
| Burn Rate Prediction | Gradient Boosting | ✅ Deployed | github.com/cssadewale |
| Churn Prediction | ML | ✅ Deployed | github.com/cssadewale |
| CBT.ng | HTML/CSS/JS, Supabase | ✅ Live | github.com/cssadewale |

---

## ✅ Final Checklist

Before you share anything, confirm every item below:

- [ ] All 4 `.joblib` files downloaded from Colab to your computer
- [ ] `app.py` created with complete code
- [ ] `requirements.txt` created with 7 packages
- [ ] `README.md` created with your details
- [ ] GitHub repository `fake-news-detector` created as **Public**
- [ ] All 7 files visible at the root of the GitHub repository
- [ ] Streamlit deployment completed without build errors
- [ ] App live at a public URL
- [ ] All 5 test cases pass in the live app
- [ ] README updated with the real Streamlit URL
- [ ] LinkedIn post drafted with the live URL
- [ ] CV entry updated

---

*Guide prepared for Adewale Samson Adeagbo | adewalesamsonadeagbo@gmail.com | github.com/cssadewale*
*3MTT Data Science Programme | 2025*
