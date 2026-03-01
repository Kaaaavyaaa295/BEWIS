# 🧠 BEWIS — Behavioral Early Warning & Intervention System

> An AI-driven behavioral decision intelligence system that predicts student burnout and dropout risk **before** academic performance declines.

---

## 🎯 Problem Statement

Universities lose students every semester to burnout and dropout. The warning signs appear **weeks before grades fall** — but nobody is watching for them. By the time poor grades are noticed, it's already too late to intervene effectively.

**BEWIS detects invisible behavioral drift before academic decline happens.**

---

## 📊 Dataset Information

- **Dataset Type:** Synthetic
- **Why Synthetic:** No real-world student behavioral dataset was publicly available with the required granularity of weekly behavioral signals.
- **How Generated:** Simulated using Python (NumPy, Pandas) with realistic distributions and behavioral decay patterns based on risk profiles.
- **Number of Records:** 1,600 rows (200 students × 8 weeks)
- **Number of Features:** 18 (9 raw + 7 engineered + 2 target labels)

### Raw Features
| Feature | Description |
|---------|-------------|
| lms_logins | Weekly LMS login count |
| attendance_pct | Weekly attendance percentage |
| sentiment_score | Feedback sentiment (-1 to +1) |
| submission_delay | Average days late on assignments |
| forum_posts | Weekly forum participation count |
| video_completion | Video lecture completion rate |
| true_risk | Ground truth risk label (Low/Medium/High) |

---

## 🔧 Feature Engineering

7 advanced behavioral features engineered from raw data:

| Engineered Feature | Description |
|-------------------|-------------|
| login_slope | Week-over-week login trend direction |
| attendance_drop | Drop in attendance vs Week 1 baseline |
| delay_variance | Variance in submission delays over time |
| sentiment_decline | Sentiment drop vs Week 1 baseline |
| engagement_volatility | Std deviation of logins / mean logins |
| sudden_drop | Binary flag if any metric drops 50%+ in one week |
| behavioral_drift_score | Composite score combining login slope, attendance drop, sentiment decline |

---

## 🤖 Models Built

### Model 1 — Burnout Risk Classifier
- **Algorithm:** Random Forest
- **Output:** Low / Medium / High risk classification
- **Accuracy:** 98.44%

### Model 2 — Dropout Probability Model
- **Algorithm:** Logistic Regression
- **Output:** 0–100% dropout probability
- **Accuracy:** 99.50%

### Model 3 — Behavioral Clustering
- **Algorithm:** K-Means (k=4)
- **Output:** 4 behavioral pattern groups
  - Gradual Burnout
  - Sudden Drop-off
  - Emotional Distress
  - Chronic Disengagement

---

## 📊 Risk Scoring Engine

Final Risk Score (0–100) calculated as:
```
Risk Score = (0.30 × engagement decline) + 
             (0.25 × attendance drop) + 
             (0.20 × submission delay) + 
             (0.15 × sentiment fall) + 
             (0.10 × behavioral drift score)
```

| Score Range | Risk Level |
|-------------|------------|
| 0–30 | 🟢 Low |
| 31–60 | 🟡 Medium |
| 61–100 | 🔴 High |

---

## 🖥️ Dashboard Features

- 📈 **Risk Evolution Timeline** — 8-week risk progression per student
- 🎯 **Risk Gauge** — Speedometer-style risk score display
- 🎛️ **What-If Intervention Simulator** — Shows impact of early counselor intervention
- 🔍 **Key Risk Triggers** — Top 3 behavioral factors driving risk
- 💡 **Intervention Recommendations** — Cluster and risk-specific action plans
- 🏫 **Cohort Overview** — Full class risk distribution

---

## 🚀 How To Run

### Prerequisites
```bash
pip install pandas numpy scikit-learn xgboost streamlit plotly shap faker
```

### Steps
```bash
# 1. Generate dataset
python generate_data.py

# 2. Engineer features
python feature_engineering.py

# 3. Train models
python train_models.py

# 4. Calculate risk scores
python risk_engine.py

# 5. Launch dashboard
streamlit run app.py
```

---

## 📁 Project Structure
```
bewis/
├── data/
│   ├── raw_data.csv
│   ├── engineered_data.csv
│   ├── final_data.csv
│   └── scored_data.csv
├── models/
│   ├── burnout_classifier.pkl
│   ├── dropout_model.pkl
│   ├── scaler.pkl
│   ├── kmeans.pkl
│   └── cluster_names.pkl
├── generate_data.py
├── feature_engineering.py
├── train_models.py
├── risk_engine.py
└── app.py
```

---

## 🏆 Key Differentiators

- ✅ Detects risk **before** grades fall
- ✅ Three models working together
- ✅ Behavioral drift score (custom engineered feature)
- ✅ Risk evolution over 8 weeks
- ✅ Explainable AI — shows WHY a student is at risk
- ✅ What-If Simulator — shows impact of early intervention
- ✅ Cluster-specific intervention recommendations

---

## 👩‍💻 Built By

Kavya — Gen AI Exchange Hackathon 2026
