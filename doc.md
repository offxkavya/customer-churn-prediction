# Customer Churn Prediction — Full Codebase Walkthrough

## Project Overview

This is an **end-to-end Machine Learning project** that predicts whether a customer will churn (leave) based on their demographic and behavioral data. It includes a Jupyter notebook for model training/evaluation and a **Streamlit web app** (with Plotly visualizations) for real-time predictions, deployed on **Hugging Face Spaces**.

---

## System Architecture

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
│  Logistic Regression Classifier                         │
│  Outputs: binary prediction (0/1) + probability score   │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 4: OUTPUT & VISUALIZATION                        │
│  Gauge chart (churn probability %)                      │
│  Risk level indicator (HIGH / LOW) with color coding    │
│  Risk interpretation (3-tier contextual message)        │
│  Customer metrics bar chart + profile summary table     │
└─────────────────────────────────────────────────────────┘
```

```
┌──────────────┐     ┌────────────────────────────────┐     ┌──────────────┐
│  Google Colab │────▶│  churn_model.pkl                │◀────│  app.py      │
│  (Training)   │     │  (Pipeline: Preprocessor +      │     │  (Streamlit  │
│               │     │   LogisticRegression)            │     │  + Plotly)   │
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
   | streamlit | latest | Web app framework |
   | pandas | latest | Data manipulation |
   | scikit-learn | 1.6.1 | ML pipeline & model (pinned to match training) |
   | joblib | latest | Model serialization |
   | plotly | latest | Gauge chart & bar chart visualizations |

3. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```
   The app will open at `http://localhost:8501`.
   The app is also deployed on Hugging Face Spaces: [https://offxkavya-customer-churn-prediction.hf.space/](https://offxkavya-customer-churn-prediction.hf.space/)

> **Note:** The app expects `churn_model.pkl` in the working directory. If running from the project root, ensure the model file is accessible (it may be in the `Model/` directory).

### Retraining the Model

1. Open `Notebook/GenAi-Capstone.ipynb` in Google Colab or Jupyter.
2. Update the data path if needed.
3. Run all cells. The last cell exports `churn_model.pkl` (Logistic Regression pipeline).
4. Copy the exported `.pkl` file to the project root for the Streamlit app.

---

## Project Structure

```
customer-churn-prediction/
├── Assets/                                          (reserved for future assets)
├── Data/
│   └── customer_churn_dataset-testing-master.csv   (64,374 rows of customer data)
├── Model/
│   └── churn_model.pkl                             (serialized Logistic Regression pipeline)
├── Notebook/
│   └── GenAi-Capstone.ipynb                        (training & evaluation notebook)
├── app.py                                          (Streamlit web app — 180 lines)
├── requirements.txt                                (Python dependencies)
├── doc.md                                          (this documentation)
└── README.md                                       (project README)
```

---

## File-by-File Breakdown

### 1. `Data/customer_churn_dataset-testing-master.csv`

A CSV with **64,374 rows** and **12 columns**:

| Column | Type | Range / Values |
|---|---|---|
| CustomerID | int | 1–64,374 (unique identifier, dropped before training) |
| Age | int | 18–65 |
| Gender | category | Male, Female |
| Tenure | int | 1–60 months |
| Usage Frequency | int | 1–30 |
| Support Calls | int | 0–10 |
| Payment Delay | int | 0–30 days |
| Subscription Type | category | Basic, Standard, Premium |
| Contract Length | category | Monthly, Quarterly, Annual |
| Total Spend | int | 100–1,000 |
| Last Interaction | int | 1–30 days |
| **Churn** | int (target) | 0 (stayed) or 1 (churned) |

- **No missing values** in the dataset.

---

### 2. `Notebook/GenAi-Capstone.ipynb`

A Google Colab notebook that performs the full ML workflow:

#### Step-by-step pipeline:

1. **Install pinned scikit-learn** — `!pip install scikit-learn==1.6.1` (ensures version consistency between training and deployment)

2. **Imports** — pandas, numpy, scikit-learn (train_test_split, StandardScaler, OneHotEncoder, ColumnTransformer, Pipeline, LogisticRegression, DecisionTreeClassifier, metrics)

3. **Data Loading** — reads the CSV into a DataFrame, inspects with `.head()` and `.shape()`

4. **Preprocessing**
   - Drops `CustomerID` column
   - Confirms **zero null values**, then calls `dropna()` as a safety measure
   - Splits features (`X`) and target (`y = Churn`)
   - 80/20 train-test split (`random_state=42`)

5. **Feature Engineering**
   - **Numeric features** (7): Age, Tenure, Usage Frequency, Support Calls, Payment Delay, Total Spend, Last Interaction → **StandardScaler**
   - **Categorical features** (3): Gender, Subscription Type, Contract Length → **OneHotEncoder** (`handle_unknown="ignore"`)
   - Combined via `ColumnTransformer`

6. **Model 1: Logistic Regression Pipeline**
   - Results:
     - Accuracy: **83.17%**
     - Precision: **81.63%**
     - Recall: **83.06%**
     - F1 Score: **82.34%**

7. **Model 2: Decision Tree Pipeline** (`max_depth=5, random_state=42`)
   - Results:
     - Accuracy: **95.97%**
     - Recall: **98.24%**
     - No sign of overfitting (training accuracy ≈ 95.63%, testing ≈ 95.97%)

8. **Feature Importance** (from Decision Tree):
   - **Payment Delay** dominates at 47.9%
   - Support Calls: 14.4%
   - Tenure: 9.9%
   - Usage Frequency: 9.1%
   - Gender (Female): 8.3%
   - Last Interaction: 0% importance

9. **Model Export** — saves the **Logistic Regression pipeline** as `churn_model.pkl` using `joblib.dump(compress=3)`. Logistic Regression was chosen for deployment because it produces smoother, well-calibrated probability estimates suitable for the gauge chart and risk interpretation in the web app.

---

### 3. `Model/churn_model.pkl`

The serialized **Logistic Regression pipeline**. It includes both the `ColumnTransformer` (preprocessing) and the `LogisticRegression` classifier — so raw input with categorical strings can be passed directly.

---

### 4. `app.py` — Streamlit Web Application (180 lines)

The front-end inference app with **Plotly visualizations** and a **tabbed interface**:

- **Page config**: title "Customer Churn Predictor", 📊 icon, wide layout
- **Custom CSS**: centered title/subtitle styling, adjusted block padding
- **Model loading**: `joblib.load("churn_model.pkl")`
- **Sidebar inputs** (10 features):
  - Sliders: Age (18–80), Tenure (0–120), Usage Frequency (0–100), Support Calls (0–20), Payment Delay (0–60), Days Since Last Interaction (0–365)
  - Number input: Total Spend (0–100,000)
  - Dropdowns: Gender, Subscription Type, Contract Length
- **Predict button**: 🚀 Predict Churn

#### Tab 1 — 📈 Prediction:
- **Gauge Chart** (Plotly): Speedometer-style churn probability visualization
  - Green zone (0–40%), Yellow zone (40–70%), Red zone (70–100%)
  - Bar color changes: green if <50%, red if ≥50%
- **Risk Level Display**: Error/success alert with HIGH/LOW label
- **Risk Interpretation**: Three-tier contextual message:
  - <30%: "Customer is stable."
  - 30–70%: "Monitor closely."
  - >70%: "Immediate retention action recommended."

#### Tab 2 — 📊 Analytics:
- **Customer Profile Summary**: DataFrame displaying all input features
- **Customer Metrics Overview**: Plotly bar chart showing the 7 numeric input features

**Why these charts?**
- **Gauge chart** → visualizes the model's main output (probability) in a dashboard-friendly, professional format
- **Customer metrics bar chart** → helps business users visually compare the customer's profile and understand risk drivers (e.g., high Payment Delay, low Tenure)
- No pie charts or random line charts — this is an ML prediction app, so visualizations are chosen to serve interpretability

---

### 5. `requirements.txt`

```
streamlit
pandas
scikit-learn==1.6.1
joblib
plotly
```

- scikit-learn **pinned to 1.6.1** (must match the training version to avoid model deserialization errors)
- **plotly** added for gauge chart and bar chart visualizations
- Other packages unpinned (latest compatible versions)

---

## Key Observations

1. **Two models trained, one deployed**: Decision Tree achieved 96% accuracy and provided feature importance insights; Logistic Regression produces smoother probability estimates and was chosen for deployment in the Streamlit app
2. **Payment Delay is the single most important feature** for churn prediction (47.9% importance from Decision Tree analysis)
3. The model is a **full pipeline** (preprocessor + classifier), meaning raw data with categorical strings can be passed directly without manual encoding
4. The app uses **Plotly** for professional visualizations (gauge chart + bar chart) with a **tabbed interface** (Prediction + Analytics)
5. scikit-learn is **pinned to 1.6.1** in both the notebook and requirements.txt for version consistency
6. App slider ranges (e.g., Total Spend up to 100K) still exceed training data ranges (max 1,000) — predictions on extreme values may be less reliable
