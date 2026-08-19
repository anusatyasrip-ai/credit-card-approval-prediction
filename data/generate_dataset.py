import numpy as np
import pandas as pd
from pathlib import Path

def generate_credit_data(n_samples=1500, random_seed=42):
    np.random.seed(random_seed)

    genders = np.random.choice(["Male", "Female"], size=n_samples, p=[0.48, 0.52])
    income_types = np.random.choice(
        ["Salary", "Self-Employed", "Business", "Pensioner", "Commercial"],
        size=n_samples,
        p=[0.45, 0.20, 0.15, 0.12, 0.08]
    )
    education_levels = np.random.choice(
        ["Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary", "Academic degree"],
        size=n_samples,
        p=[0.50, 0.35, 0.08, 0.05, 0.02]
    )
    employment_statuses = np.random.choice(
        ["Employed", "Self-Employed", "Unemployed", "Pensioner", "Student"],
        size=n_samples,
        p=[0.60, 0.20, 0.08, 0.08, 0.04]
    )
    payment_statuses = np.random.choice(
        ["No Past Due", "Past Due"],
        size=n_samples,
        p=[0.75, 0.25]
    )

    annual_incomes = np.round(np.random.lognormal(mean=10.8, sigma=0.55, size=n_samples), -2)
    annual_incomes = np.clip(annual_incomes, 18000, 350000)

    employment_years = np.round(np.random.exponential(scale=6.0, size=n_samples), 1)
    employment_years = np.clip(employment_years, 0.0, 40.0)

    for i in range(n_samples):
        if employment_statuses[i] == "Unemployed":
            employment_years[i] = 0.0
        elif employment_statuses[i] == "Student" and employment_years[i] > 4.0:
            employment_years[i] = np.round(np.random.uniform(0.0, 3.0), 1)

    existing_loan_balances = np.round(np.random.exponential(scale=15000, size=n_samples), -2)
    existing_loan_balances = np.clip(existing_loan_balances, 0, 120000)

    credit_inquiries = np.random.poisson(lam=1.8, size=n_samples)
    credit_inquiries = np.clip(credit_inquiries, 0, 12)

    credit_history_years = np.round(employment_years + np.random.uniform(1.0, 10.0, size=n_samples), 1)
    credit_history_years = np.clip(credit_history_years, 0.5, 45.0)

    # Realistic approval decision formula
    score = np.zeros(n_samples)
    for i in range(n_samples):
        score[i] = (
            (annual_incomes[i] / 50000.0) * 1.2 +
            (employment_years[i] / 5.0) * 0.8 +
            (credit_history_years[i] / 8.0) * 0.6 +
            (1.5 if education_levels[i] in ["Higher education", "Academic degree"] else 0.0) -
            (existing_loan_balances[i] / (annual_incomes[i] + 1.0)) * 2.5 -
            (credit_inquiries[i] * 0.6) -
            (3.0 if payment_statuses[i] == "Past Due" else -1.0) -
            (2.5 if employment_statuses[i] == "Unemployed" else 0.0)
        )

    probs = 1.0 / (1.0 + np.exp(-score))
    approved = (probs >= 0.52).astype(int)

    df = pd.DataFrame({
        "gender": genders,
        "income_type": income_types,
        "annual_income": annual_incomes,
        "employment_years": employment_years,
        "education_level": education_levels,
        "employment_status": employment_statuses,
        "existing_loan_balance": existing_loan_balances,
        "credit_inquiries": credit_inquiries,
        "credit_history_years": credit_history_years,
        "payment_status": payment_statuses,
        "approved": approved
    })

    # Introduce minor missing values (~1%) to demonstrate pipeline imputation
    missing_idx_inc = np.random.choice(n_samples, size=int(n_samples * 0.015), replace=False)
    df.loc[missing_idx_inc, "annual_income"] = np.nan

    missing_idx_edu = np.random.choice(n_samples, size=int(n_samples * 0.01), replace=False)
    df.loc[missing_idx_edu, "education_level"] = np.nan

    return df

if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "credit_card_applications.csv"
    df = generate_credit_data()
    df.to_csv(csv_path, index=False)
    print(f"Dataset created successfully at: {csv_path}")
    print(f"Shape: {df.shape}")
    print(f"Target distribution:\n{df['approved'].value_counts(normalize=True)}")
