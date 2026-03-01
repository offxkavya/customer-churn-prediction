# Customer Churn Prediction — Full Codebase Walkthrough

## Project Overview

This is an **end-to-end Machine Learning project** that predicts whether a customer will churn (leave) based on their demographic and behavioral data. It includes a Jupyter notebook for model training/evaluation and a **Streamlit web app** for real-time predictions.

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

## Project Structure

```
customer-churn-prediction/
├── Data/
│   └── customer_churn_dataset-testing-master.csv   (64,374 rows of customer data)
├── Model/
│   └── churn_model.pkl                             (serialized Decision Tree pipeline)
├── Notebook/
│   └── GenAi-Capstone.ipynb                        (training & evaluation notebook)
├── app.py                                          (Streamlit web app)
├── requirements.txt                                (Python dependencies)
└── runtime.txt                                     (Python 3.12)
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

1. **Imports** — pandas, numpy, scikit-learn (train_test_split, StandardScaler, OneHotEncoder, ColumnTransformer, Pipeline, LogisticRegression, DecisionTreeClassifier, metrics)

2. **Data Loading** — reads the CSV into a DataFrame, inspects with `.head()` and `.shape()`

3. **Preprocessing**
   - Drops `CustomerID` column
   - Confirms **zero null values**, then calls `dropna()` as a safety measure
   - Splits features (`X`) and target (`y = Churn`)
   - 80/20 train-test split (`random_state=42`)

4. **Feature Engineering**
   - **Numeric features** (7): Age, Tenure, Usage Frequency, Support Calls, Payment Delay, Total Spend, Last Interaction → **StandardScaler**
   - **Categorical features** (3): Gender, Subscription Type, Contract Length → **OneHotEncoder** (`handle_unknown="ignore"`)
   - Combined via `ColumnTransformer`

5. **Model 1: Logistic Regression Pipeline**
   - Results:
     - Accuracy: **83.17%**
     - Precision: **81.63%**
     - Recall: **83.06%**
     - F1 Score: **82.34%**

6. **Model 2: Decision Tree Pipeline** (`max_depth=5, random_state=42`)
   - Results:
     - Accuracy: **95.97%**
     - Recall: **98.24%**
     - No sign of overfitting (training accuracy ≈ 95.63%, testing ≈ 95.97%)

7. **Feature Importance** (from Decision Tree):
   - **Payment Delay** dominates at 47.9%
   - Support Calls: 14.4%
   - Tenure: 9.9%
   - Usage Frequency: 9.1%
   - Gender (Female): 8.3%
   - Last Interaction: 0% importance

8. **Model Export** — saves the **Decision Tree pipeline** as `churn_model.pkl` using `joblib.dump(compress=3)`, then downloads it via Colab's `files.download()`

---

### 3. `Model/churn_model.pkl`

The serialized **Decision Tree pipeline** (~3.6 KB compressed). This is the model loaded by the Streamlit app. It includes both the `ColumnTransformer` (preprocessing) and the `DecisionTreeClassifier` — so raw input with categorical strings can be passed directly.

---

### 4. `app.py` — Streamlit Web Application (89 lines)

The front-end inference app:

- **Page config**: title "Customer Churn Predictor", wide layout
- **Model loading**: `joblib.load("churn_model.pkl")` — ⚠️ loads from the current working directory, not from `Model/`
- **Sidebar inputs** (10 features):
  - Sliders: Age, Tenure, Usage Frequency, Support Calls, Payment Delay, Days Since Last Interaction
  - Number input: Total Spend
  - Dropdowns: Gender, Subscription Type, Contract Length
- **Prediction logic**:
  - Constructs a single-row DataFrame matching the training feature names
  - Calls `model.predict()` and `model.predict_proba()`
  - Displays churn probability with a progress bar
  - Shows HIGH/LOW risk based on the prediction
  - Displays the full customer profile summary as a table

> **Note:** The slider ranges in `app.py` (e.g., Age up to 80, Tenure up to 120, Total Spend up to 100,000) are **wider** than the training data ranges (Age 18–65, Tenure 1–60, Total Spend 100–1,000). Predictions on out-of-distribution inputs may be unreliable.

---

### 5. `requirements.txt`

```
streamlit==1.33.0
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.4.2
joblib==1.3.2
```

---

### 6. `runtime.txt`

```
python-3.12
```

Used by hosting platforms (e.g., Hugging Face Spaces) to specify the Python version.

---

## Key Observations

1. **The Decision Tree model was chosen** over Logistic Regression due to significantly better performance (~96% vs ~83% accuracy)
2. **Payment Delay is the single most important feature** for churn prediction (47.9% importance)
3. The model is a **full pipeline** (preprocessor + classifier), meaning raw data with categorical strings can be passed directly without manual encoding
4. The `app.py` loads the model from `"churn_model.pkl"` in the current directory — but the actual file lives in `Model/churn_model.pkl`. This means the app must be run from the `Model/` directory or the pkl file must be copied to the project root
5. The dataset is relatively clean — zero null values, balanced enough for a Decision Tree to achieve high recall (98.24%)
