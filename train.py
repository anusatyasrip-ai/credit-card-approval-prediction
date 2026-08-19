import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

import config

def apply_feature_engineering(df_raw):
    """
    Applies consistent feature engineering to raw applicant DataFrame.
    """
    df = df_raw.copy()

    # Section 9 Requirement: Convert payment status into binary label if it's text
    if "payment_status" in df.columns:
        if df["payment_status"].dtype == object:
            df["payment_status"] = df["payment_status"].map({
                "No Past Due": 0,
                "Past Due": 1
            }).fillna(0).astype(int)

    # Derived Features
    annual_inc = df["annual_income"].fillna(df["annual_income"].median() if len(df) > 1 else 50000.0)
    emp_yrs = df["employment_years"].fillna(df["employment_years"].median() if len(df) > 1 else 2.0)
    loan_bal = df["existing_loan_balance"].fillna(0.0)
    inquiries = df["credit_inquiries"].fillna(0)

    df["debt_to_income_ratio"] = np.round(loan_bal / (annual_inc + 1.0), 4)
    df["income_per_employment_year"] = np.round(annual_inc / (emp_yrs + 1.0), 2)
    df["high_credit_inquiry_flag"] = (inquiries >= 3).astype(int)

    return df

def build_preprocessor():
    """
    Builds a Scikit-Learn ColumnTransformer pipeline.
    """
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', num_pipeline, config.NUMERICAL_FEATURES),
        ('cat', cat_pipeline, config.CATEGORICAL_FEATURES)
    ])

    return preprocessor

def train_and_evaluate_models():
    print("=" * 60)
    print(" CREDIT CARD APPROVAL PREDICTION SYSTEM - MODEL TRAINING ")
    print("=" * 60)

    # 1. Load Data
    if not config.DATA_FILE.exists():
        print(f"Dataset missing at {config.DATA_FILE}. Generating new dataset...")
        from data.generate_dataset import generate_credit_data
        df = generate_credit_data()
        df.to_csv(config.DATA_FILE, index=False)
    else:
        df = pd.read_csv(config.DATA_FILE)

    # Remove duplicates
    df = df.drop_duplicates()

    # 2. Feature Engineering
    print("\n--- 1. FEATURE ENGINEERING ---")
    df_fe = apply_feature_engineering(df)
    print("Features engineered: debt_to_income_ratio, income_per_employment_year, high_credit_inquiry_flag, payment_status binary.")

    X = df_fe[config.NUMERICAL_FEATURES + config.CATEGORICAL_FEATURES]
    y = df_fe[config.TARGET_COLUMN]

    # 3. Train / Test Split (80% train, 20% test, Stratified)
    print("\n--- 2. TRAIN / TEST SPLIT (80% / 20% Stratified) ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape:  {X_test.shape}")

    # 4. Preprocessing Pipeline
    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # 5. Define ML Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, eval_metric='logloss')
    }

    metrics_results = {}
    fitted_models = {}
    confusion_matrices = {}

    print("\n--- 3. MODEL TRAINING & EVALUATION ---")
    print(f"{'Model':<22} | {'Accuracy':<9} | {'Precision':<9} | {'Recall':<9} | {'F1 Score':<9}")
    print("-" * 70)

    for name, model in models.items():
        # Train
        model.fit(X_train_proc, y_train)
        y_pred = model.predict(X_test_proc)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        metrics_results[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "classification_report": classification_report(y_test, y_pred, output_dict=True)
        }

        fitted_models[name] = model
        confusion_matrices[name] = cm.tolist()

        print(f"{name:<22} | {acc:<9.4f} | {prec:<9.4f} | {rec:<9.4f} | {f1:<9.4f}")

    # 6. Select Best Model (based on F1 Score)
    best_model_name = max(metrics_results, key=lambda k: metrics_results[k]["f1_score"])
    best_model = fitted_models[best_model_name]
    best_metrics = metrics_results[best_model_name]

    print("\n" + "=" * 60)
    print(f" BEST MODEL AUTOMATICALLY SELECTED: {best_model_name}")
    print(f" Top F1 Score: {best_metrics['f1_score']:.4f} (Accuracy: {best_metrics['accuracy']:.4f})")
    print("=" * 60)

    # 7. Save Models and Metadata
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, config.BEST_MODEL_PATH)
    joblib.dump(preprocessor, config.PREPROCESSOR_PATH)

    metrics_payload = {
        "best_model_name": best_model_name,
        "all_models": metrics_results,
        "confusion_matrices": confusion_matrices,
        "features": {
            "numerical": config.NUMERICAL_FEATURES,
            "categorical": config.CATEGORICAL_FEATURES
        }
    }

    with open(config.METRICS_PATH, "w") as f:
        json.dump(metrics_payload, f, indent=4)

    print(f"\nSaved artifacts:")
    print(f" - Best Model:   {config.BEST_MODEL_PATH}")
    print(f" - Preprocessor: {config.PREPROCESSOR_PATH}")
    print(f" - Metrics:      {config.METRICS_PATH}")

    # 8. Generate Visualizations for Models
    generate_model_visualizations(metrics_results, confusion_matrices)

def generate_model_visualizations(metrics_results, confusion_matrices):
    config.VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)

    # Chart 1: Model Comparison Bar Chart
    models = list(metrics_results.keys())
    f1_scores = [metrics_results[m]["f1_score"] for m in models]
    accuracies = [metrics_results[m]["accuracy"] for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    rects1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', color='#1e88e5')
    rects2 = ax.bar(x + width/2, f1_scores, width, label='F1 Score', color='#43a047')

    ax.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax.set_title('Machine Learning Model Performance Comparison', fontsize=13, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontweight='bold')
    ax.legend()
    ax.set_ylim(0, 1.1)

    for rect in rects1 + rects2:
        h = rect.get_height()
        ax.annotate(f'{h:.3f}', (rect.get_x() + rect.get_width() / 2., h),
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    comp_path = config.VISUALIZATIONS_DIR / "model_comparison.png"
    plt.savefig(comp_path, dpi=300)
    plt.close()
    print(f" Saved: {comp_path.name}")

    # Chart 2: Confusion Matrices Grid
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for idx, (name, cm) in enumerate(confusion_matrices.items()):
        cm_arr = np.array(cm)
        sns.heatmap(cm_arr, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False,
                    xticklabels=['Rejected', 'Approved'], yticklabels=['Rejected', 'Approved'])
        axes[idx].set_title(f"{name}", fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('Predicted Label')
        axes[idx].set_ylabel('True Label')

    plt.tight_layout()
    cm_path = config.VISUALIZATIONS_DIR / "confusion_matrices.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f" Saved: {cm_path.name}")

if __name__ == "__main__":
    train_and_evaluate_models()
