# IBM Watson Machine Learning & Cloud Object Storage Deployment Guide

This guide explains how to deploy the **Credit Card Approval Prediction Model** to **IBM Watson Machine Learning (WML)** and store artifacts in **IBM Cloud Object Storage (COS)**.

---

## 1. Architecture & Cloud Artifacts

Local Development -> IBM Cloud Object Storage -> IBM Watson ML -> Production REST Endpoint

Artifacts moved to IBM Cloud:
- `models/best_model.pkl` - Serialized Scikit-Learn / XGBoost best model.
- `models/preprocessor.pkl` - Reusable ColumnTransformer scaling & encoding pipeline.
- `models/metrics.json` - Benchmark evaluation metrics & selection metadata.

---

## 2. IBM Cloud Prerequisites

1. Create an **IBM Cloud Account** at [cloud.ibm.com](https://cloud.ibm.com).
2. Provision an instance of **Watson Machine Learning (WML)**.
3. Provision an instance of **IBM Cloud Object Storage (COS)**.
4. Create a **Deployment Space** in IBM Watson Studio / Cloud Pak for Data.
5. Create an **IBM Cloud API Key**:
   - Go to **Manage > Access (IAM) > API keys**.
   - Click **Create an IBM Cloud API key**.
   - Copy the API Key value.

---

## 3. Environment Variables Configuration

Copy `.env.example` to `.env` in the root directory:

```bash
cp .env.example .env
```

Update `.env` with your IBM Cloud credentials:

```ini
IBM_WML_API_KEY=your_actual_ibm_cloud_api_key
IBM_WML_PROJECT_ID=your_ibm_watson_project_id
IBM_WML_SPACE_ID=your_ibm_watson_deployment_space_id
IBM_WML_URL=https://us-south.ml.cloud.ibm.com
```

---

## 4. Running the Automated Deployment Script

Execute the deployment script:

```bash
python ibm_cloud/deploy.py
```

The deployment script will:
1. Read IBM Cloud credentials securely from `.env`.
2. Connect to the specified IBM Watson Machine Learning Deployment Space.
3. Upload `best_model.pkl` to the IBM WML Repository.
4. Deploy the model as an active **Online Real-Time REST Scoring Endpoint**.
5. Output the **Deployment ID** and **Scoring URL**.

---

## 5. Scoring API Endpoint Usage

Once deployed, send a `POST` request to the scoring URL with IBM IAM bearer token authentication:

```json
{
  "input_data": [
    {
      "fields": [
        "gender", "income_type", "annual_income", "employment_years",
        "education_level", "employment_status", "existing_loan_balance",
        "credit_inquiries", "credit_history_years", "payment_status"
      ],
      "values": [
        ["Female", "Salary", 75000, 6.0, "Higher education", "Employed", 5000, 1, 8.5, "No Past Due"]
      ]
    }
  ]
}
```

The IBM WML scoring endpoint returns the prediction (`1` for Approved, `0` for Rejected) and class probability array.
