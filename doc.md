# Customer Churn Prediction — Project Documentation

## Table of Contents

- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Dataset](#dataset)
- [ML Pipeline (Notebook)](#ml-pipeline-notebook)
- [Model Details](#model-details)
- [Streamlit Web App](#streamlit-web-app)
- [Deployment](#deployment)
- [Key Observations](#key-observations)

---

## Project Overview

An **end-to-end Machine Learning project** that predicts customer churn using historical behavioral data. It includes:

- A **Jupyter Notebook** for data exploration, model training, and evaluation
- A **Streamlit web app** for real-time churn predictions
- Deployment on **Hugging Face Spaces**

This is **Milestone 1** of a larger initiative to build an agentic AI retention strategist.

---

## System Architecture

Data flows from user input to prediction output through four stages:

```
┌─────────────────────────────────────────────────────────┐
│  STAGE 1: USER INPUT                                    │
│  Customer details entered via Streamlit sidebar widgets  │
│  (Age, Gender, Tenure, Usage Frequency, etc.)           │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 2: PREPROCESSING (inside Pipeline)               │
│  StandardScaler → 7 numeric features (zero mean, σ=1)   │
│  OneHotEncoder  → 3 categorical features (binary cols)  │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 3: MODEL INFERENCE                               │
│  Decision Tree Classifier (max_depth=5)                 │
│  Outputs: binary prediction (0/1) + probability score   │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 4: OUTPUT DISPLAY                                │
│  Churn probability (%) with progress bar                │
│  Risk level indicator (HIGH / LOW) with color coding    │
│  Full customer profile summary table                    │
└─────────────────────────────────────────────────────────┘
```

**Component diagram:**

```
┌──────────────┐     ┌────────────────────────────────┐     ┌──────────────┐
│  Google Colab │────▶│  churn_model.pkl                │◀────│  app.py      │
│  (Training)   │     │  (Pipeline: Preprocessor +      │     │  (Streamlit) │
│               │     │   DecisionTreeClassifier)        │     │              │
└──────────────┘     └────────────────────────────────┘     └──────┬───────┘
                                                                    │
                                                                    ▼
                                                           ┌──────────────┐
                                                           │ Hugging Face │
                                                           │   Spaces     │
                                                           │  (Hosting)   │
                                                           └──────────────┘
```

---

## Project Structure

```
customer-churn-prediction/
├── Data/
│   └── customer_churn_dataset-testing-master.csv   # 64,374 rows of customer data
├── Model/
│   └── churn_model.pkl                             # Serialized Decision Tree pipeline (~3.6 KB)
├── Notebook/
│   └── GenAi-Capstone.ipynb                        # Training & evaluation notebook (Colab)
├── app.py                                          # Streamlit web app (89 lines)
├── requirements.txt                                # Python dependencies
├── runtime.txt                                     # Python 3.12
└── doc.md                                          # This documentation
```

---

## Setup Instructions

### Prerequisites

- **Python 3.12** (or compatible)
- **pip** package manager

### Local Setup

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd customer-churn-prediction
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   This installs:
   | Package | Version | Purpose |
   |---|---|---|
   | streamlit | 1.33.0 | Web app framework |
   | pandas | 2.2.2 | Data manipulation |
   | numpy | 1.26.4 | Numerical operations |
   | scikit-learn | 1.4.2 | ML pipeline & model |
   | joblib | 1.3.2 | Model serialization |

3. **Run the Streamlit app:**

   ```bash
   streamlit run app.py
   ```

   The app will open at `http://localhost:8501`.

> **Note:** The app expects `churn_model.pkl` in the working directory. If running from the project root, ensure the model file is accessible (it may be in the `Model/` directory).

### Retraining the Model

1. Open `Notebook/GenAi-Capstone.ipynb` in Google Colab or Jupyter.
2. Update the data path if needed: `filePath="../Data/customer_churn_dataset-testing-master.csv"`.
3. Run all cells. The last cell exports `churn_model.pkl`.
4. Copy the exported `.pkl` file to the project root for the Streamlit app.

---

## Dataset

**File:** `Data/customer_churn_dataset-testing-master.csv`

- **64,374 rows** × **12 columns**
- **Zero missing values**

| Column | Type | Range / Values | Description |
|---|---|---|---|
| CustomerID | int | 1–64,374 | Unique identifier (**dropped before training**) |
| Age | int | 18–65 | Customer's age in years |
| Gender | category | Male, Female | Customer's gender |
| Tenure | int | 1–60 | Months as a customer |
| Usage Frequency | int | 1–30 | Service usage frequency |
| Support Calls | int | 0–10 | Number of support calls |
| Payment Delay | int | 0–30 | Avg payment delay in days |
| Subscription Type | category | Basic, Standard, Premium | Subscription tier |
| Contract Length | category | Monthly, Quarterly, Annual | Contract duration |
| Total Spend | int | 100–1,000 | Total monetary spend ($) |
| Last Interaction | int | 1–30 | Days since last interaction |
| **Churn** (target) | binary | 0 (Stayed), 1 (Churned) | Target variable |

---

## ML Pipeline (Notebook)

The `GenAi-Capstone.ipynb` notebook follows this workflow:

1. **Import libraries** — pandas, numpy, scikit-learn
2. **Load data** — `pd.read_csv()`, inspect with `.head()` and `.shape`
3. **Drop `CustomerID`** — sequential identifier with no predictive value
4. **Check nulls** — `df.isnull().sum()` → all zeros
5. **`dropna()`** — defensive measure for production robustness
6. **Split features/target** — `X = df.drop("Churn")`, `y = df["Churn"]`
7. **Train-test split** — 80/20, `random_state=42`
8. **Identify feature types** — 7 numeric, 3 categorical
9. **Build `ColumnTransformer`** — StandardScaler (numeric) + OneHotEncoder (categorical)
10. **Train Logistic Regression** — baseline model → 83.17% accuracy
11. **Train Decision Tree** — `max_depth=5` → **95.97% accuracy, 98.24% recall**
12. **Overfitting check** — train (95.63%) ≈ test (95.97%) → no overfitting
13. **Feature importance** — Payment Delay dominates at 47.9%
14. **Export model** — `joblib.dump(tree_pipeline, "churn_model.pkl", compress=3)`

---

## Model Details

### Model Comparison

| Metric | Logistic Regression | Decision Tree |
|---|---|---|
| Accuracy | 83.17% | **95.97%** |
| Precision | 81.63% | — |
| Recall | 83.06% | **98.24%** |
| F1-Score | 82.34% | — |

### Feature Importance (Decision Tree)

| Rank | Feature | Importance |
|---|---|---|
| 1 | Payment Delay | 0.4787 |
| 2 | Support Calls | 0.1440 |
| 3 | Tenure | 0.0991 |
| 4 | Usage Frequency | 0.0910 |
| 5 | Gender (Female) | 0.0828 |
| 6 | Age | 0.0431 |
| 7 | Gender (Male) | 0.0327 |
| 8 | Total Spend | 0.0212 |
| 9 | Contract Length (Annual) | 0.0074 |
| 10 | Last Interaction | 0.0000 |

### Confusion Matrix (Logistic Regression)

|  | Predicted: Stayed (0) | Predicted: Churned (1) |
|---|---|---|
| **Actual: Stayed (0)** | 5,656 | 1,137 |
| **Actual: Churned (1)** | 1,030 | 5,052 |

---

## Streamlit Web App

**File:** `app.py` (89 lines)

### Features

- **Sidebar inputs:** Sliders and dropdowns for all 10 features
- **Prediction output:** Churn probability (%) with progress bar
- **Risk indicator:** Color-coded HIGH (red) / LOW (green) alert
- **Profile summary:** Table showing all input features

### How It Works

1. Loads `churn_model.pkl` at startup via `joblib.load()`
2. Collects user inputs through Streamlit sidebar widgets
3. Constructs a single-row `pandas.DataFrame` matching the training schema
4. Calls `model.predict()` for binary result + `model.predict_proba()` for probability
5. Displays results with visual indicators

---

## Deployment

### Hugging Face Spaces

The app is deployed on **Hugging Face Spaces**, which:

- Natively supports Streamlit apps
- Automatically builds from the Git repository
- Provides a public URL for anyone to test predictions
- Requires zero infrastructure management

### Deployment Steps

1. Push `app.py`, `churn_model.pkl`, and `requirements.txt` to a Hugging Face Space
2. Hugging Face detects the Streamlit SDK and installs dependencies
3. The app is live at the Space's public URL

---

## Key Observations

1. **Payment Delay is the #1 churn predictor** (47.9% feature importance) — a retention strategy targeting payment delays could significantly reduce churn
2. **Decision Tree chosen over Logistic Regression** due to +12.8pp accuracy gain driven by non-linear feature interactions
3. **No overfitting** — train (95.63%) ≈ test (95.97%) with `max_depth=5`
4. **Pipeline design** ensures the deployed model accepts raw categorical strings with no external preprocessing needed
5. **Last Interaction has zero importance** — recency of contact alone does not predict churn
6. **App slider ranges exceed training data** (e.g., Total Spend up to 100K vs training max of 1,000) — predictions on out-of-distribution inputs may be unreliable
