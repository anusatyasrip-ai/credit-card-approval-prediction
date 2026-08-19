import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
DATA_FILE = DATA_DIR / "credit_card_applications.csv"

MODELS_DIR = BASE_DIR / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
METRICS_PATH = MODELS_DIR / "metrics.json"

VISUALIZATIONS_DIR = BASE_DIR / "visualizations"
DATABASE_PATH = BASE_DIR / os.getenv("DATABASE_PATH", "predictions.db")

# Model Columns Configuration
NUMERICAL_FEATURES = [
    "annual_income",
    "employment_years",
    "existing_loan_balance",
    "credit_inquiries",
    "credit_history_years",
    "debt_to_income_ratio",
    "income_per_employment_year"
]

CATEGORICAL_FEATURES = [
    "gender",
    "income_type",
    "education_level",
    "employment_status",
    "payment_status",
    "high_credit_inquiry_flag"
]

RAW_INPUT_COLUMNS = [
    "gender",
    "income_type",
    "annual_income",
    "employment_years",
    "education_level",
    "employment_status",
    "existing_loan_balance",
    "credit_inquiries",
    "credit_history_years",
    "payment_status"
]

TARGET_COLUMN = "approved"

# Ensure directories exist
