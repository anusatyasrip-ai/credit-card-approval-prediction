# Credit Card Approval Prediction System using Machine Learning and IBM Watson Machine Learning

A complete, production-grade end-to-end Machine Learning web application designed for financial institutions to automate preliminary credit card applicant screening (`Approved` vs `Rejected`).

Built with **Python, Scikit-Learn, XGBoost, Flask, SQLite, and Bootstrap 5**, complete with **IBM Watson Machine Learning** deployment automation.

---

## Key Features

- **Multi-Model Benchmark Pipeline**: Automatically trains and evaluates 4 classifiers (**Logistic Regression, Decision Tree, Random Forest, and XGBoost**) and selects the optimal model based on maximum **F1 Score**.
- **Reusable Data Preprocessor**: Scikit-Learn `ColumnTransformer` with `StandardScaler` for numerical feature scaling and `OneHotEncoder` for categorical feature encoding to eliminate data leakage.
- **Derived Feature Engineering**: Calculates `debt_to_income_ratio`, `income_per_employment_year`, `high_credit_inquiry_flag`, and maps `payment_status` to binary risk signals.
- **Exploratory Data Analysis (EDA)**: Script outputs statistical summaries and generates high-resolution charts in `visualizations/`.
- **Modern Banking UI**: Responsive Bootstrap 5.3 financial interface with KPI metric cards, visual decision badges, probability gauges, and pre-set sample loaders.
- **Prediction History & Audit**: SQLite local database stores applicant details, predictions, approval probabilities, and model metadata. Supports live search and filter.
- **REST API Endpoint**: `/api/predict` accepts JSON payloads with input validation and returns structured predictions and probability scores.
- **IBM Watson ML & COS Ready**: Includes `ibm_cloud/deploy.py` and step-by-step documentation for deploying models to IBM Cloud Watson Machine Learning.

---

## 🛠️ Technology Stack

- **Machine Learning**: Python 3.x, NumPy, Pandas, Matplotlib, Seaborn, Scikit-Learn, XGBoost, Joblib
- **Backend Framework**: Flask (Python REST API & Jinja2 Templates)
- **Frontend**: HTML5, CSS3, JavaScript (ES6), Bootstrap 5.3, FontAwesome Icons
- **Database**: SQLite3 (`predictions.db`)
- **Cloud & Deployment**: IBM Cloud, IBM Watson Machine Learning, IBM Cloud Object Storage

---

## 📁 Project Structure

```text
credit-card-approval-prediction/
│
├── app.py                      # Flask web application server & REST API
├── train.py                    # ML model training, benchmarking, F1 selection & artifact saving
├── eda.py                      # Exploratory Data Analysis & chart generator
├── config.py                   # Central paths, feature lists, and settings
├── requirements.txt            # Python dependencies
├── README.md                   # Complete system documentation & guide
├── .env.example                # Sample environment variables template
├── .gitignore                  # Git ignore rules
│
├── data/
│   ├── generate_dataset.py     # Realistic synthetic dataset generator
│   └── credit_card_applications.csv # Primary credit card dataset (1,500 records)
│
├── models/
│   ├── best_model.pkl          # Serialized top-performing ML model
│   ├── preprocessor.pkl        # Serialized Scikit-Learn preprocessor pipeline
│   └── metrics.json            # Model evaluation metrics & selection metadata
│
├── services/
│   ├── __init__.py
│   ├── database.py             # SQLite predictions database manager
│   └── predictor.py            # ML inference engine & input validator
│
├── templates/
│   ├── base.html               # Base layout, navbar, and footer
│   ├── index.html              # Dashboard KPI summary & system capabilities
│   ├── predict.html            # Applicant evaluation form with sample loaders
│   ├── result.html             # Visual decision outcome page & probability gauge
│   ├── history.html            # Searchable SQLite prediction audit history
│   └── about.html              # Architecture diagram & model evaluation charts
│
├── static/
│   ├── css/
│   │   └── style.css           # Banking UI custom stylesheet
│   ├── js/
│   │   └── script.js           # Client-side validation & sample filler
│   └── images/
│
├── visualizations/             # Generated EDA & model comparison charts (.png)
│
├── tests/
│   ├── __init__.py
│   └── test_app.py             # Automated Pytest suite (Flask, API, DB, Predictor)
│
└── ibm_cloud/
    ├── deploy.py               # Automated IBM Watson ML deployment script
    └── README.md               # IBM Cloud & COS deployment guide
```

---

## 🚀 Quick Start Guide

### 1. Environment Setup

Clone or open the repository directory:

```bash
cd "c:\credit card"
```

Create a virtual environment (optional but recommended):

```bash
python -m venv venv
```

Activate the virtual environment:

- **Windows (PowerShell)**: `venv\Scripts\Activate.ps1`
- **Windows (CMD)**: `venv\Scripts\activate.bat`
- **Linux/macOS**: `source venv/bin/activate`

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

---

### 2. Exploratory Data Analysis (EDA)

Run the EDA script to inspect dataset statistics and generate visualizations:

```bash
python eda.py
```

Generated charts will be saved inside `visualizations/`:
- `approval_distribution.png`
- `income_distribution.png`
- `education_vs_approval.png`
- `payment_status_vs_approval.png`
- `employment_status_vs_approval.png`

---

### 3. Model Training & Automatic Selection

Train all four machine learning classifiers (**Logistic Regression, Decision Tree, Random Forest, XGBoost**):

```bash
python train.py
```

`train.py` outputs a performance comparison table:

```text
Model                  | Accuracy  | Precision | Recall    | F1 Score 
----------------------------------------------------------------------
Logistic Regression    | 0.9867    | 0.9904    | 0.9904    | 0.9904   
Decision Tree          | 0.9233    | 0.9387    | 0.9522    | 0.9454   
Random Forest          | 0.9267    | 0.9431    | 0.9522    | 0.9476   
XGBoost                | 0.9400    | 0.9484    | 0.9665    | 0.9573   
```

The top model is automatically saved to `models/best_model.pkl` along with `preprocessor.pkl` and `metrics.json`.

---

### 4. Start Flask Web Application

Run the Flask server:

```bash
python app.py
```

Open your browser and navigate to:

```text
http://127.0.0.1:5000
```

---

## 🧪 Automated Testing

Execute the test suite to verify endpoints, validation rules, and database persistence:

```bash
python -m pytest tests/test_app.py -v
```

---

## 📡 REST API Documentation

### Endpoint

`POST /api/predict`

### Request Headers

`Content-Type: application/json`

### Example Request Body

```json
{
  "gender": "Female",
  "income_type": "Salary",
  "annual_income": 75000,
  "employment_years": 6.0,
  "education_level": "Higher education",
  "employment_status": "Employed",
  "existing_loan_balance": 5000,
  "credit_inquiries": 1,
  "credit_history_years": 8.5,
  "payment_status": "No Past Due"
}
```

### Example Success Response (HTTP 200)

```json
{
  "status": "success",
  "prediction": 1,
  "label": "Approved",
  "approval_probability": 0.9985,
  "approval_percentage": "99.85%",
  "model": "Logistic Regression"
}
```

---

## ☁️ IBM Watson Machine Learning Deployment

1. Copy `.env.example` to `.env` and fill in your IBM Cloud credentials:

```ini
IBM_WML_API_KEY=your_ibm_cloud_api_key
IBM_WML_PROJECT_ID=your_project_id
IBM_WML_SPACE_ID=your_deployment_space_id
IBM_WML_URL=https://us-south.ml.cloud.ibm.com
```

2. Execute the automated deployment script:

```bash
python ibm_cloud/deploy.py
```

Refer to [`ibm_cloud/README.md`](file:///c:/credit%20card/ibm_cloud/README.md) for step-by-step instructions.
