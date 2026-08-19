import json
import joblib
import pandas as pd
import numpy as np
import config
from services.feature_engineering import apply_feature_engineering

class CreditPredictor:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.model_name = "Trained ML Model"
        self._load_artifacts()

    def _load_artifacts(self):
       

        self.model = joblib.load(config.BEST_MODEL_PATH)
        self.preprocessor = joblib.load(config.PREPROCESSOR_PATH)

        if config.METRICS_PATH.exists():
            with open(config.METRICS_PATH, "r") as f:
                metrics = json.load(f)
                self.model_name = metrics.get("best_model_name", "Trained ML Model")

    def validate_input(self, raw_input: dict):
        """
        Validates raw user input values. Raises ValueError on invalid data.
        """
        required_fields = [
            "gender", "income_type", "annual_income", "employment_years",
            "education_level", "employment_status", "existing_loan_balance",
            "credit_inquiries", "credit_history_years", "payment_status"
        ]

        for field in required_fields:
            if field not in raw_input or raw_input[field] is None or str(raw_input[field]).strip() == "":
                raise ValueError(f"Required field '{field}' is missing or empty.")

        try:
            annual_income = float(raw_input["annual_income"])
            if annual_income < 0:
                raise ValueError("Annual income cannot be negative.")
        except ValueError as e:
            if "negative" in str(e):
                raise
            raise ValueError("Annual income must be a valid number.")

        try:
            employment_years = float(raw_input["employment_years"])
            if employment_years < 0:
                raise ValueError("Employment duration cannot be negative.")
        except ValueError as e:
            if "negative" in str(e):
                raise
            raise ValueError("Employment duration must be a valid number.")

        try:
            loan_balance = float(raw_input["existing_loan_balance"])
            if loan_balance < 0:
                raise ValueError("Existing loan balance cannot be negative.")
        except ValueError as e:
            if "negative" in str(e):
                raise
            raise ValueError("Existing loan balance must be a valid number.")

        try:
            inquiries = int(raw_input["credit_inquiries"])
            if inquiries < 0:
                raise ValueError("Credit inquiries cannot be negative.")
        except ValueError as e:
            if "negative" in str(e):
                raise
            raise ValueError("Credit inquiries must be a non-negative integer.")

        try:
            credit_history = float(raw_input["credit_history_years"])
            if credit_history < 0:
                raise ValueError("Credit history duration cannot be negative.")
        except ValueError as e:
            if "negative" in str(e):
                raise
            raise ValueError("Credit history duration must be a valid number.")

    def predict(self, raw_input: dict) -> dict:
        self.validate_input(raw_input)

        # Convert input dictionary into DataFrame
        df_input = pd.DataFrame([{
            "gender": str(raw_input["gender"]),
            "income_type": str(raw_input["income_type"]),
            "annual_income": float(raw_input["annual_income"]),
            "employment_years": float(raw_input["employment_years"]),
            "education_level": str(raw_input["education_level"]),
            "employment_status": str(raw_input["employment_status"]),
            "existing_loan_balance": float(raw_input["existing_loan_balance"]),
            "credit_inquiries": int(raw_input["credit_inquiries"]),
            "credit_history_years": float(raw_input["credit_history_years"]),
            "payment_status": str(raw_input["payment_status"])
        }])

        # Apply exact consistent feature engineering
        df_fe = apply_feature_engineering(df_input)

        X_input = df_fe[config.NUMERICAL_FEATURES + config.CATEGORICAL_FEATURES]

        # Transform using fitted preprocessor
        X_proc = self.preprocessor.transform(X_input)

        # Predict class & probability
        pred_class = int(self.model.predict(X_proc)[0])

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_proc)[0]
            approval_prob = float(probs[1])
        else:
            approval_prob = 1.0 if pred_class == 1 else 0.0

        label = "Approved" if pred_class == 1 else "Rejected"

        return {
            "prediction": pred_class,
            "label": label,
            "approval_probability": round(approval_prob, 4),
            "approval_percentage": f"{approval_prob * 100:.2f}%",
            "model": self.model_name
        }

# Global singleton predictor instance
predictor = CreditPredictor()
