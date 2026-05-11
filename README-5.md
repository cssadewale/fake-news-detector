# 🔍 Through the Lens of Truth: Fake News Detector
### The TruthLens Institute · NLP + Machine Learning · Binary Classification

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://adewale-fake-news-detector.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-86.75%25-brightgreen)
![AUC](https://img.shields.io/badge/ROC--AUC-0.9393-brightgreen)
![3MTT](https://img.shields.io/badge/3MTT-Cohort%203-blue)
![Darey.io](https://img.shields.io/badge/Platform-Darey.io-purple)
![Project](https://img.shields.io/badge/Project-2%20of%204-informational)

---

## 🌐 Live Demo

**→ [Launch the Live App](https://adewale-fake-news-detector.streamlit.app)**
**→ [View Training Notebook on GitHub](https://github.com/cssadewale/fake-news-detector)**
**→ [Author Portfolio](https://cssadewale.pages.dev)**

---

## 📌 Programme Context

This project is **Project 2 of 4** in the **3MTT (Three Million Technical Talent) Data Science Track**, delivered on the **Darey.io** training platform.

### What is 3MTT?

The **3 Million Technical Talent (3MTT) Programme** is the flagship initiative of Nigeria's **Federal Ministry of Communications, Innovation & Digital Economy**, launched in 2023 under President Bola Ahmed Tinubu's *Renewed Hope* agenda. The programme aims to:

- Train **3 million Nigerians** in high-demand technical skills by 2027
- Create **2 million digital jobs** for the Nigerian economy
- Position Nigeria as a **net exporter of technical talent** globally
- Bridge the nation's tech talent gap across software development, data science, AI/ML, cybersecurity, and more

The programme is executed in partnership with NITDA (National Information Technology Development Agency) and multiple accredited training providers. Fellows receive fully government-funded training, practical project experience, and pathways to industry placement.

**Adewale Samson Adeagbo is a Cohort 3 Fellow** of the 3MTT Data Science Track.

### What is Darey.io?

**Darey.io** is an accredited 3MTT training partner — a Nigerian EdTech company founded by Dare Olufunmilayo. Originally focused on DevOps and Cloud Computing, Darey.io has expanded its curriculum to include data science, cybersecurity, product management, and AI. The platform provides:

- **Structured learning tracks** with hands-on projects
- **Xterns** — a companion platform for building real-world experience and a verifiable digital portfolio
- **AI-driven, project-based learning** aligned with global employer demands
- Access to live classes, recordings, an AI assistant, and dedicated programme support

Darey.io's motto is: *"Darey.io is the university where you gain skills, and Xterns is where you gain experience."*

The 3MTT Data Science track on Darey.io is structured around **four capstone projects** that require fellows to build and deploy end-to-end machine learning systems — from raw data exploration through to live web applications.

---

## 🎯 Project Overview

This project builds a **binary fake news classifier** for **The TruthLens Institute** — a fictionalised global research organisation dedicated to combating misinformation and promoting digital media literacy.

The business problem: given the **title and body text** of a news article, predict with high confidence whether it is **FAKE** or **REAL**.

The solution is a complete end-to-end pipeline — from raw dataset ingestion, exploratory data analysis, NLP preprocessing, feature engineering, and multi-model evaluation, through to hyperparameter tuning and deployment as a live interactive web application.

| Attribute | Value |
|-----------|-------|
| **Best Model** | Tuned XGBoost Classifier |
| **Accuracy** | 86.75% |
| **ROC-AUC** | 0.9393 |
| **F1-Score (Fake)** | 0.87 |
| **F1-Score (Real)** | 0.87 |
| **Cross-Validation AUC** | 0.8987 (5-fold stratified) |
| **Dataset** | 12,999 real-world news articles (8 misinformation categories) |
| **Final Training Set** | 826 balanced records (413 fake / 413 real) |
| **Feature Matrix** | 20,002 dimensions (TF-IDF + word_count + sentiment) |
| **Models Evaluated** | 7 (including 2 tuned variants) |
| **Notebook Cells** | 119 |

---

## 🧠 End-to-End Technical Pipeline

```
Raw Dataset (12,999 articles, 20 columns)
          │
          ▼
  ┌─────────────────────────────────────────┐
  │  STEP 2: DATA LOADING & CLEANING        │
  │  • Column standardisation               │
  │  • Target variable inspection           │
  │  • Label standardisation                │
  │  • Title + Text → unified content field │
  │  • clean_text() — 5-stage NLP           │
  │  • Duplicate detection and removal      │
  └─────────────────────────────────────────┘
          │
          ▼
  ┌─────────────────────────────────────────┐
  │  STEP 3: EDA & FEATURE ENGINEERING      │
  │  • Numerical feature extraction         │
  │    (word_count, char_count, etc.)        │
  │  • WordCloud analysis (8 categories)    │
  │  • Binary label mapping                 │
  │    (bias → real; all others → fake)     │
  │  • Class balancing: downsampling        │
  │    (29.8:1 → 1:1 balanced corpus)       │
  │  • Top bigrams visualisation            │
  │  • Metadata extraction attempt          │
  │  • TF-IDF construction                  │
  │    (max_features=20,000, ngram (1,2))   │
  │  • scipy.sparse.hstack → 20,002 dims    │
  │  • Label encoding + train/test split    │
  └─────────────────────────────────────────┘
          │
          ▼
  ┌─────────────────────────────────────────┐
  │  STEP 4: BASELINE MODELLING             │
  │  • Logistic Regression (baseline)       │
  │  • Support Vector Classifier (RBF)      │
  │  • Decision Tree Classifier             │
  │  • Random Forest Classifier             │
  │  • XGBoost Classifier (baseline)        │
  │  • ROC-AUC comparative analysis         │
  └─────────────────────────────────────────┘
          │
          ▼
  ┌─────────────────────────────────────────┐
  │  STEP 5: HYPERPARAMETER TUNING          │
  │  • Tuned Logistic Regression            │
  │  • Tuned XGBoost                        │
  │    (learning_rate=0.05, max_depth=3,    │
  │     n_estimators=200)                   │
  │  • Full model comparison table          │
  │  • Model serialisation (.joblib)        │
  └─────────────────────────────────────────┘
          │
          ▼
  ┌─────────────────────────────────────────┐
  │  STEP 6: INSIGHTS & RECOMMENDATIONS     │
  │  • Model recommendation summary         │
  │  • Key data findings                    │
  │  • Actionable business recommendations  │
  │  • Limitations and next steps           │
  │  • Project summary                      │
  └─────────────────────────────────────────┘
          │
          ▼
  ┌─────────────────────────────────────────┐
  │  DEPLOYMENT — Streamlit Web App         │
  │  • Real-time article classification     │
  │  • 7-step prediction pipeline           │
  │  • Confidence scoring + interpretation  │
  │  • Sample article examples (3 types)    │
  │  • Pipeline trace expander              │
  │  • Hosted: Streamlit Community Cloud    │
  └─────────────────────────────────────────┘
```

---

## 🔬 The clean_text() NLP Pipeline

All text passes through a 5-stage normalisation function before any feature extraction. The same function runs identically in both the training notebook and the deployed Streamlit app — this is critical for prediction consistency.

| Stage | Operation | Example Input → Output |
|-------|-----------|----------------------|
| 1 | Remove special characters and digits | `"COVID-19 outbreak!"` → `"COVID  outbreak "` |
| 2 | Lowercase conversion | `"BREAKING News"` → `"breaking news"` |
| 3 | Punctuation removal | `"won't stop"` → `"wont stop"` |
| 4 | Tokenisation | `"fake news today"` → `["fake", "news", "today"]` |
| 5 | Stopword removal + WordNet Lemmatisation | `["the", "lies", "spreading"]` → `["lie", "spread"]` |

---

## 📊 Full Model Comparison

Seven classifiers were trained and evaluated on the same 20,002-feature matrix. Evaluation metrics include Accuracy, ROC-AUC (threshold-independent), and F1-scores for both classes. The Tuned XGBoost was selected as the production model.

| Rank | Model | Accuracy | ROC-AUC | F1 (Fake) | F1 (Real) | Notes |
|------|-------|----------|---------|-----------|-----------|-------|
| 🥇 1 | **Tuned XGBoost** | **86.75%** | **0.9393** | **0.87** | **0.87** | ✅ **Production** |
| 🥈 2 | XGBoost (Baseline) | 84.34% | 0.9051 | 0.84 | 0.84 | Strong baseline |
| 🥉 3 | Random Forest | 79.52% | 0.8656 | 0.80 | 0.79 | Good ensemble |
| 4 | Logistic Regression | 79.52% | 0.8560 | 0.78 | 0.80 | Fast, interpretable |
| 5 | Tuned Logistic Regression | 78.31% | 0.8513 | 0.78 | 0.79 | Interpretability model |
| 6 | Decision Tree | 72.29% | 0.7227 | 0.73 | 0.72 | High variance |
| 7 | SVC (RBF kernel) | 68.67% | 0.7311 | 0.74 | 0.61 | Class imbalance sensitive |

**Why ROC-AUC was chosen as the primary tuning objective:**
ROC-AUC is threshold-independent — it measures the model's ability to rank fake articles above real ones regardless of the decision boundary. For a content screening tool where both false positives (blocking legitimate news) and false negatives (passing fake news) carry real-world costs, AUC gives a more robust picture than raw accuracy.

---

## 🗃️ Dataset Deep Dive

The dataset is a real-world news crawl of **12,999 articles** published between 2015 and 2016 — the height of online political misinformation in the US — containing 20 columns including article content, publication metadata, source information, and engagement signals.

### Target Variable: `type` (8 misinformation categories)

| Category | Count | Description | Binary Label |
|----------|-------|-------------|-------------|
| `bs` (bullshit) | 11,492 | Entirely fabricated content | **fake** |
| `bias` | 443 | Grounded but ideologically slanted | **real** (proxy) |
| `conspiracy` | 430 | Conspiracy theory narratives | **fake** |
| `hate` | 246 | Hate speech content | **fake** |
| `satire` | 146 | Satirical/parody articles | **fake** |
| `state` | 121 | State-sponsored propaganda | **fake** |
| `junksci` | 102 | Junk science / pseudoscience | **fake** |
| `fake` | 19 | Explicitly labelled fake | **fake** |

### Critical Design Decisions

**Why `bias` → real?**
The `bias` category represents ideologically slanted but factually grounded reporting. It is the only category in the dataset with a verifiable factual basis. All other categories involve fabricated content, conspiracy narratives, or deliberate disinformation. Using `bias` as the real-class proxy is an imperfect but principled choice — it gives the model a real-world example of grounded journalism, even if opinionated.

**Why downsampling over SMOTE?**
The raw class distribution was 29.8:1 (fake:real). SMOTE was rejected because it generates *synthetic text patterns* by interpolating in feature space — synthetic TF-IDF vectors do not represent real article writing styles and would introduce noise the model cannot generalise from. Downsampling to 413 records per class preserves the authenticity of every training example.

**Why TF-IDF over word embeddings?**
The balanced training corpus is small (826 records). Deep embeddings (Word2Vec, BERT) require large corpora to learn meaningful representations. TF-IDF with bigrams (e.g., "white house" vs "white" + "house") captures phrase-level discriminative signals effectively for this dataset size, without the risk of overfitting to an embedding space trained on insufficient data.

---

## 🔧 Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Binary mapping | `bias` → real, all others → fake | Only `bias` represents grounded reporting |
| Class balancing | Random downsampling to 413 per class | Avoids synthetic text patterns from SMOTE |
| Text features | TF-IDF, max_features=20,000, ngram_range=(1,2) | Bigrams capture phrase-level patterns |
| Numerical features | word_count + VADER compound sentiment | Supplements TF-IDF with stylometric signals |
| Feature stacking | scipy.sparse.hstack → (1, 20,002) | Merges sparse TF-IDF with dense numerical features |
| Scaling | None | TF-IDF values are in [0,1]; tree models are scale-invariant |
| Tuning objective | ROC-AUC | Threshold-independent; robust for screening tasks |
| Production model | Tuned XGBoost | Highest accuracy, AUC, and balanced F1 across both classes |
| Interpretability model | Tuned Logistic Regression | Coefficient-based explanations for editorial teams |

---

## ⚙️ Streamlit Application — Feature Breakdown

The live web application is built with `Streamlit` and mirrors the training notebook's preprocessing pipeline exactly.

### Application Features

| Feature | Description |
|---------|-------------|
| **Real-time classification** | Classifies any news article as FAKE or REAL in under 1 second |
| **Confidence scoring** | Displays prediction probability for both classes (0–100%) |
| **Confidence interpretation** | Contextual guidance: High (≥85%) / Moderate (65–85%) / Low (<65%) |
| **Sample articles** | Three pre-loaded examples: Fake / Real / Ambiguous |
| **Pipeline trace** | Expandable panel showing cleaned text, word count, sentiment score, and feature matrix details |
| **Model performance sidebar** | Live display of all model metrics and the 7-step NLP pipeline |
| **Full model comparison table** | All 7 models with accuracy, AUC, and F1 scores (inside expander) |
| **Disclaimer panel** | Clear guidance on responsible use and fact-checking resources |

### Session State Architecture

The app uses Streamlit's `st.session_state` for managing example button state. This is a deliberate design decision: example buttons write to `session_state` keys *before* the text input widgets are rendered, then `st.rerun()` propagates the values correctly. This avoids the `StreamlitAPIException` that occurs when buttons and widgets share the same key binding in the same run.

### Model Loading

Both the TF-IDF vectoriser and XGBoost model are loaded via `@st.cache_resource` — they are instantiated once per session and reused for all subsequent predictions. This eliminates the overhead of deserialising large `.joblib` files on every click.

---

## 🗂️ Repository Structure

```
fake-news-detector/
├── app.py                                       ← Streamlit web application
├── requirements.txt                             ← Python dependencies
├── tfidf_vectorizer.joblib                      ← Fitted TF-IDF vectoriser (20,000 features)
├── tuned_xgboost_fake_news.joblib               ← Production model (Tuned XGBoost)
├── tuned_logistic_regression_fake_news.joblib   ← Interpretability model (Tuned LR)
├── label_encoder.joblib                         ← Label encoder (fake=0, real=1)
└── README.md                                    ← This file
```

> ⚠️ **Note on model files:** The `.joblib` files are generated by the training notebook (Google Colab). They are not stored in the repository by default due to file size. To regenerate: run all cells in the training notebook through the model-saving step.

---

## 🚀 Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/cssadewale/fake-news-detector.git
cd fake-news-detector

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Mac / Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Ensure your .joblib model files are in the root directory
#    (generate from the training notebook, or request from the author)

# 5. Launch the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## 📦 Dependencies

```
streamlit>=1.28.0
joblib>=1.3.0
scikit-learn>=1.3.0
xgboost>=1.7.0
nltk>=3.8.0
scipy>=1.11.0
numpy>=1.24.0
```

NLTK corpora (`stopwords`, `wordnet`, `vader_lexicon`, `punkt`, `omw-1.4`) are downloaded automatically on first launch via `nltk.download(..., quiet=True)`.

---

## ⚠️ Known Limitations

| Limitation | Detail |
|------------|--------|
| **Domain specificity** | Trained on US political news (2015–2016). Performance on Nigerian news, scientific misinformation, or post-2020 content is untested. |
| **Proxy ground truth** | The "real" class is approximated by `bias` articles — ideologically slanted but factually grounded content. This is an imperfect gold standard. |
| **Training set size** | 826 records after balancing. The model may exhibit higher variance on article styles absent from training. |
| **Metadata exclusion** | `site_url`, `published`, `domain_rank`, and `spam_score` were excluded from the pipeline. Incorporating them could improve AUC by an estimated 2–5 points. |
| **No multilingual support** | English-only. NLTK stopwords and lemmatisation are English-specific. |
| **Tool, not verdict** | This is a screening aid. It should always be used alongside editorial judgement and established fact-checking platforms. |

---

## 🔮 Future Improvements

- [ ] Add metadata features (`domain_rank`, `spam_score`, `published`) to the feature matrix
- [ ] Train on a more recent and geographically diverse dataset
- [ ] Explore BERT-based embeddings once a larger training corpus is available
- [ ] Add SHAP (SHapley Additive exPlanations) for per-prediction feature attribution
- [ ] Integrate with fact-checking APIs (Snopes, PolitiFact) for real-time cross-referencing
- [ ] Add multilingual support via multilingual TF-IDF or mBERT

---

## 👤 Author

**Adewale Samson Adeagbo**
*Data Scientist · Machine Learning Engineer · STEM Educator (15+ Years)*
*Director & Data Lead, HMG Concepts | HMG Academy*

| Contact | Details |
|---------|---------|
| 📧 Email | adeagboadewalesamson@gmail.com |
| 📱 Phone | +234 810 086 6322 |
| 🌐 Portfolio | [cssadewale.pages.dev](https://cssadewale.pages.dev) |
| 🔗 LinkedIn | [linkedin.com/in/adewalesamsonadeagbo](https://linkedin.com/in/adewalesamsonadeagbo) |
| 💻 GitHub | [github.com/cssadewale](https://github.com/cssadewale) |

---

## 📚 Full Portfolio

| # | Project | Tech Stack | Live App | Repository |
|---|---------|-----------|----------|-----------|
| 1 | Yakub Trading Group — Staff Promotion Prediction | Random Forest, Scikit-learn, Streamlit | [Live](https://yakub-promotion-prediction.streamlit.app) | [GitHub](https://github.com/cssadewale/yakub-promotion-prediction) |
| **2** | **The TruthLens Institute — Fake News Detector** | **XGBoost, NLP, TF-IDF, VADER, NLTK** | **[Live](https://adewale-fake-news-detector.streamlit.app)** | **[GitHub](https://github.com/cssadewale/fake-news-detector)** |
| 3 | Insurance Claim Prediction | ML Classification, Streamlit | [Live](https://adewale-insurance-claim-prediction.streamlit.app) | [GitHub](https://github.com/cssadewale/insurance-claim-prediction) |
| 4 | Bank Customer Churn Prediction | ML, Streamlit | [Live](https://adewale-bank-customer-churn-prediction.streamlit.app) | [GitHub](https://github.com/cssadewale/bank-customer-churn-prediction) |

*Additional projects: [Income Level Prediction](https://adewale-income-level-prediction.streamlit.app) · [SwiftChain Delivery Delay Prediction](https://adewale-swiftchain-delivery-prediction.streamlit.app) · [NeuroWell Employee Burnout Predictor](https://adewale-burnout-prediction.streamlit.app) · [Student At-Risk Predictor](https://student-at-risk-predictor.streamlit.app) · [CBT.ng (HTML/JS/Supabase)](https://cssadewale.github.io/cbt-system/student.html)*

---

## 📜 Programme

**3MTT (Three Million Technical Talent) Programme**
*An initiative of the Federal Ministry of Communications, Innovation & Digital Economy, Federal Republic of Nigeria*
*Training Platform: Darey.io · Cohort 3 · Data Science Track · Project 2 of 4*

---

## 📄 Licence

This project is open-source under the [MIT Licence](LICENSE).

---

*Built end-to-end on an Android tablet (itel Vistatab 30S) using Google Colab, GitHub mobile interface, and Streamlit Community Cloud — no desktop, no terminal.*
