import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import config

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

COLOR_APPROVED = '#2E7D32'  # Emerald Green
COLOR_REJECTED = '#C62828'  # Crimson Red
PALETTE = [COLOR_REJECTED, COLOR_APPROVED]

def run_eda():
    print("=" * 60)
    print(" CREDIT CARD APPROVAL PREDICTION SYSTEM - EDA ")
    print("=" * 60)

    # 1. Load Data
    if not config.DATA_FILE.exists():
        print(f"Data file not found at {config.DATA_FILE}. Generating dataset...")
        from data.generate_dataset import generate_credit_data
        df = generate_credit_data()
        df.to_csv(config.DATA_FILE, index=False)
    else:
        df = pd.read_csv(config.DATA_FILE)

    # 2. Dataset Shape & Overview
    print("\n--- 1. DATASET SHAPE & TYPES ---")
    print(f"Total Rows: {df.shape[0]}")
    print(f"Total Columns: {df.shape[1]}")
    print("\nColumn Data Types:")
    print(df.dtypes)

    # 3. Missing Values & Duplicates
    print("\n--- 2. MISSING VALUES & DUPLICATE RECORDS ---")
    missing = df.isnull().sum()
    print("Missing values per column:")
    print(missing[missing > 0] if missing.sum() > 0 else "No missing values found.")
    duplicates = df.duplicated().sum()
    print(f"Duplicate records found: {duplicates}")

    # 4. Statistical Summary
    print("\n--- 3. NUMERICAL FEATURES SUMMARY ---")
    print(df.describe().T.to_string())

    print("\n--- 4. CATEGORICAL FEATURES UNIQUE VALUES ---")
    cat_cols = df.select_dtypes(include=['object', 'str']).columns
    for col in cat_cols:
        print(f"{col}: {df[col].nunique()} unique values -> {df[col].unique()}")

    # 5. Target Distribution
    print("\n--- 5. TARGET DISTRIBUTION (APPROVED VS REJECTED) ---")
    target_counts = df[config.TARGET_COLUMN].value_counts()
    target_props = df[config.TARGET_COLUMN].value_counts(normalize=True) * 100
    print(f"Rejected (0): {target_counts.get(0, 0)} ({target_props.get(0, 0):.2f}%)")
    print(f"Approved (1): {target_counts.get(1, 0)} ({target_props.get(1, 0):.2f}%)")

    # 6. Generate Visualizations
    config.VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
    print("\n--- 6. GENERATING EDA VISUALIZATIONS ---")

    # Chart 1: Target Distribution
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.countplot(
        data=df, 
        x=config.TARGET_COLUMN, 
        palette=PALETTE, 
        ax=ax,
        hue=config.TARGET_COLUMN,
        legend=False
    )
    ax.set_title("Credit Card Application Outcomes (Target Distribution)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Application Outcome", fontsize=11, fontweight='bold')
    ax.set_ylabel("Count of Applicants", fontsize=11, fontweight='bold')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Rejected (0)", "Approved (1)"])
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{int(height)}\n({height/len(df)*100:.1f}%)',
                    (p.get_x() + p.get_width() / 2., height / 2),
                    ha='center', va='center', fontsize=11, color='white', fontweight='bold')
    plt.tight_layout()
    chart1_path = config.VISUALIZATIONS_DIR / "approval_distribution.png"
    plt.savefig(chart1_path, dpi=300)
    plt.close()
    print(f" Saved: {chart1_path.name}")

    # Chart 2: Annual Income Distribution
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.kdeplot(
        data=df, 
        x="annual_income", 
        hue=config.TARGET_COLUMN, 
        palette=PALETTE, 
        fill=True, 
        common_norm=False, 
        alpha=0.4, 
        linewidth=2,
        ax=ax
    )
    ax.set_title("Annual Income Distribution by Approval Outcome", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Annual Income ($)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Density", fontsize=11, fontweight='bold')
    ax.legend(title="Outcome", labels=["Approved (1)", "Rejected (0)"])
    plt.tight_layout()
    chart2_path = config.VISUALIZATIONS_DIR / "income_distribution.png"
    plt.savefig(chart2_path, dpi=300)
    plt.close()
    print(f" Saved: {chart2_path.name}")

    # Chart 3: Education Level vs Approval
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(
        data=df, 
        y="education_level", 
        hue=config.TARGET_COLUMN, 
        palette=PALETTE, 
        ax=ax
    )
    ax.set_title("Approval Rate across Education Levels", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Count", fontsize=11, fontweight='bold')
    ax.set_ylabel("Education Level", fontsize=11, fontweight='bold')
    ax.legend(title="Outcome", labels=["Rejected", "Approved"])
    plt.tight_layout()
    chart3_path = config.VISUALIZATIONS_DIR / "education_vs_approval.png"
    plt.savefig(chart3_path, dpi=300)
    plt.close()
    print(f" Saved: {chart3_path.name}")

    # Chart 4: Payment Status vs Approval
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(
        data=df, 
        x="payment_status", 
        hue=config.TARGET_COLUMN, 
        palette=PALETTE, 
        ax=ax
    )
    ax.set_title("Credit Card Approval by Payment Status", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Payment History Status", fontsize=11, fontweight='bold')
    ax.set_ylabel("Count of Applicants", fontsize=11, fontweight='bold')
    ax.legend(title="Outcome", labels=["Rejected", "Approved"])
    plt.tight_layout()
    chart4_path = config.VISUALIZATIONS_DIR / "payment_status_vs_approval.png"
    plt.savefig(chart4_path, dpi=300)
    plt.close()
    print(f" Saved: {chart4_path.name}")

    # Chart 5: Employment Status vs Approval
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.countplot(
        data=df, 
        x="employment_status", 
        hue=config.TARGET_COLUMN, 
        palette=PALETTE, 
        ax=ax
    )
    ax.set_title("Approval Outcome by Employment Status", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Employment Status", fontsize=11, fontweight='bold')
    ax.set_ylabel("Count", fontsize=11, fontweight='bold')
    ax.legend(title="Outcome", labels=["Rejected", "Approved"])
    plt.tight_layout()
    chart5_path = config.VISUALIZATIONS_DIR / "employment_status_vs_approval.png"
    plt.savefig(chart5_path, dpi=300)
    plt.close()
    print(f" Saved: {chart5_path.name}")

    print("\nEDA completed successfully!")

if __name__ == "__main__":
    run_eda()
