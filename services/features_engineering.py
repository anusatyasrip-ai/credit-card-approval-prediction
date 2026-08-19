import numpy as np


def apply_feature_engineering(df_raw):
    """
    Applies consistent feature engineering to raw applicant DataFrame.
    """
    df = df_raw.copy()

    # Convert payment status to binary
    if "payment_status" in df.columns:
        if df["payment_status"].dtype == object:
            df["payment_status"] = df["payment_status"].map({
                "No Past Due": 0,
                "Past Due": 1
            }).fillna(0).astype(int)

    # Derived features
    annual_inc = df["annual_income"].fillna(
        df["annual_income"].median() if len(df) > 1 else 50000.0
    )

    emp_yrs = df["employment_years"].fillna(
        df["employment_years"].median() if len(df) > 1 else 2.0
    )

    loan_bal = df["existing_loan_balance"].fillna(0.0)
    inquiries = df["credit_inquiries"].fillna(0)

    df["debt_to_income_ratio"] = np.round(
        loan_bal / (annual_inc + 1.0), 4
    )

    df["income_per_employment_year"] = np.round(
        annual_inc / (emp_yrs + 1.0), 2
    )

    df["high_credit_inquiry_flag"] = (
        inquiries >= 3
    ).astype(int)

    return df
