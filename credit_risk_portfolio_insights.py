"""
credit_risk_portfolio_insights.py

End‑to‑end credit risk portfolio project in ONE file:

- Load accepted + rejected Lending‑Club style datasets
- Prepare default target for accepted loans
- Basic EDA and business summaries
- PD model (Probability of Default) with SMOTE + Logistic Regression
- Compare with Random Forest
- Portfolio Expected Loss (EL = PD * LGD * EAD) on accepted book
- Score rejected applications with PD to get insights about missed opportunities and risk
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# ============================================================
# 1. CONFIG
# ============================================================

# Paths to your two datasets (put them in project root or adjust paths)
ACCEPTED_PATH = "accepted_2007_to_2018Q4.csv"
REJECTED_PATH = "rejected_2007_to_2018Q4.csv"

# Target column created from loan_status for accepted loans
TARGET_COL = "default"           # 1 = default/bad, 0 = non-default/good
SOURCE_COL = "application_status"  # accepted / rejected

# Example Lending Club–style features (rename if needed).[web:55][web:59]
NUMERIC_FEATURES = [
    "loan_amnt",        # loan amount
    "int_rate",         # interest rate
    "annual_inc",       # annual income
    "dti",              # debt-to-income
    "emp_length_num",   # numeric employment length you create from emp_length
    "fico_range_low"    # proxy credit score
]

CATEGORICAL_FEATURES = [
    "purpose",
    "grade",
    "home_ownership",
    "addr_state"
]

# Optional LGD/EAD proxies
LOSS_COL = "loss_rate"      # if you engineer it; else EL part will be skipped
EAD_COL = "loan_amnt"       # exposure at default


# ============================================================
# 2. DATA LOADING & PREPARATION
# ============================================================

def load_and_prepare_data():
    """
    Load accepted and rejected datasets and:
    - Create SOURCE_COL flag (accepted / rejected)
    - Derive a binary default label for accepted loans from loan_status
      (good vs bad definition simplified for teaching).[web:55][web:59]
    """

    accepted = pd.read_csv(ACCEPTED_PATH, low_memory=False)
    rejected = pd.read_csv(REJECTED_PATH, low_memory=False)

    accepted[SOURCE_COL] = "accepted"
    rejected[SOURCE_COL] = "rejected"

    # Create binary default label from loan_status (simplified).
    # Good statuses: Fully Paid.
    # Bad statuses: Charged Off, Default, Late (31-120 days), etc.[web:55][web:65]
    if "loan_status" not in accepted.columns:
        raise ValueError("Column 'loan_status' not found in accepted data. Adjust mapping in load_and_prepare_data().")

    good_statuses = [
        "Fully Paid"
    ]
    bad_statuses = [
        "Charged Off",
        "Default",
        "Does not meet the credit policy. Status:Charged Off",
        "Late (31-120 days)"
    ]

    def map_default(status):
        if status in good_statuses:
            return 0
        if status in bad_statuses:
            return 1
        # Other statuses (Current, In Grace Period, etc.) have no final outcome; drop later.[web:60][web:62]
        return np.nan

    accepted[TARGET_COL] = accepted["loan_status"].apply(map_default)
    # Keep only rows where default label is known
    accepted = accepted.dropna(subset=[TARGET_COL])
    accepted[TARGET_COL] = accepted[TARGET_COL].astype(int)

    # Example: create numeric emp_length from text like "10+ years", "< 1 year", etc.
    if "emp_length" in accepted.columns:
        accepted["emp_length_num"] = (
            accepted["emp_length"]
            .fillna("0")
            .replace({
                "10+ years": "10",
                "< 1 year": "0",
                "n/a": "0"
            })
            .str.extract(r"(\d+)")
            .astype(float)
        )
    if "emp_length" in rejected.columns:
        rejected["emp_length_num"] = (
            rejected["emp_length"]
            .fillna("0")
            .replace({
                "10+ years": "10",
                "< 1 year": "0",
                "n/a": "0"
            })
            .str.extract(r"(\d+)")
            .astype(float)
        )

    return accepted, rejected


# ============================================================
# 3. EDA & UTILITIES
# ============================================================

def basic_eda(df: pd.DataFrame, target: str, name: str):
    print(f"\n===== BASIC INFO: {name} =====")
    print(df.info())
    print("\n===== HEAD =====")
    print(df.head())
    if target in df.columns:
        print("\n===== TARGET VALUE COUNTS =====")
        print(df[target].value_counts(dropna=False))
        default_rate = df[target].mean()
        print(f"\nOverall default rate ({name}): {default_rate:.2%}")


def plot_default_rate_by_category(df: pd.DataFrame, target: str, category: str):
    if category not in df.columns:
        return
    rates = df.groupby(category)[target].mean().sort_values(ascending=False)
    plt.figure(figsize=(8, 4))
    sns.barplot(x=rates.index, y=rates.values)
    plt.title(f"Default rate by {category}")
    plt.ylabel("Default rate")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def build_preprocessor(numeric_features, categorical_features):
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder="drop"
    )
    return preprocessor


def train_pd_model(X_train, y_train, preprocessor, model_type="log_reg"):
    if model_type == "log_reg":
        model = LogisticRegression(max_iter=500)
    elif model_type == "rf":
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=42,
            class_weight="balanced"
        )
    else:
        raise ValueError("model_type must be 'log_reg' or 'rf'")

    clf = ImbPipeline(steps=[
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("model", model)
    ])
    clf.fit(X_train, y_train)
    return clf


def evaluate_pd_model(clf, X_test, y_test, threshold=0.5, name="Model"):
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    print(f"\n===== {name} EVALUATION =====")
    print("ROC AUC:", roc_auc_score(y_test, y_pred_proba))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    return y_pred_proba, y_pred


def analyze_pd_cutoffs(test_df, target, pd_col="pd_pred", thresholds=None):
    if thresholds is None:
        thresholds = [0.05, 0.1, 0.2, 0.3]

    print("\n===== PD CUT-OFF ANALYSIS (ACCEPTED) =====")
    for thr in thresholds:
        approved = test_df[test_df[pd_col] <= thr]
        declined = test_df[test_df[pd_col] > thr]
        approval_rate = len(approved) / len(test_df) if len(test_df) > 0 else 0
        default_rate_approved = approved[target].mean() if len(approved) > 0 else 0

        print(f"\nPD threshold: {thr:.2f}")
        print(f"Approval rate: {approval_rate:.2%}")
        print(f"Default rate among approved: {default_rate_approved:.2%}")
        print(f"Number approved: {len(approved)}, declined: {len(declined)}")


def compute_portfolio_expected_loss(df, features, clf, loss_col, ead_col, target):
    if loss_col not in df.columns or ead_col not in df.columns:
        print("\nLoss or EAD column missing; skipping expected loss computation.")
        return None

    df = df.copy()
    df["pd_pred"] = clf.predict_proba(df[features])[:, 1]
    df["LGD"] = df[loss_col].clip(lower=0, upper=1)
    df["EAD"] = df[ead_col].clip(lower=0)

    df["expected_loss"] = df["pd_pred"] * df["LGD"] * df["EAD"]
    total_el = df["expected_loss"].sum()
    avg_el = df["expected_loss"].mean()

    print("\n===== PORTFOLIO EXPECTED LOSS (ACCEPTED) =====")
    print(f"Total portfolio expected loss: {total_el:.2f}")
    print(f"Average expected loss per loan: {avg_el:.2f}")

    if "grade" in df.columns:
        print("\nExpected loss by grade:")
        print(df.groupby("grade")["expected_loss"].sum().sort_values(ascending=False))

    return df


def score_rejected_applications(rejected_df, features, clf):
    common_cols = [c for c in features if c in rejected_df.columns]
    if not common_cols:
        print("\nNo overlapping features between accepted and rejected for scoring.")
        return None

    rejected = rejected_df.copy()
    rejected["pd_pred"] = clf.predict_proba(rejected[common_cols])[:, 1]

    print("\n===== REJECTED APPLICATIONS – PD SUMMARY =====")
    print(rejected["pd_pred"].describe())
    print("\nHigh-risk rejected (pd_pred > 0.3):", (rejected["pd_pred"] > 0.3).mean())
    print("Very low-risk rejected (pd_pred < 0.05):", (rejected["pd_pred"] < 0.05).mean())
    return rejected


# ============================================================
# 4. MAIN PIPELINE
# ============================================================

def main():
    # 1) Load data
    accepted_df, rejected_df = load_and_prepare_data()

    # EDA on accepted (with default label)
    basic_eda(accepted_df, TARGET_COL, "ACCEPTED")
    for cat in ["grade", "purpose", "addr_state"]:
        plot_default_rate_by_category(accepted_df, TARGET_COL, cat)

    # 2) Select features and split
    features = [f for f in NUMERIC_FEATURES + CATEGORICAL_FEATURES if f in accepted_df.columns]
    missing = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES) - set(features)
    if missing:
        print(f"\nWARNING: these configured features are not in accepted data and will be ignored: {missing}")

    X = accepted_df[features]
    y = accepted_df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    numeric_used = [f for f in NUMERIC_FEATURES if f in features]
    categorical_used = [f for f in CATEGORICAL_FEATURES if f in features]
    preprocessor = build_preprocessor(numeric_used, categorical_used)

    # 3) Train PD models
    log_reg_clf = train_pd_model(X_train, y_train, preprocessor, model_type="log_reg")
    y_pred_proba_lr, _ = evaluate_pd_model(
        log_reg_clf, X_test, y_test, threshold=0.5, name="Logistic Regression (PD)"
    )

    rf_clf = train_pd_model(X_train, y_train, preprocessor, model_type="rf")
    _rf_proba, _ = evaluate_pd_model(
        rf_clf, X_test, y_test, threshold=0.5, name="Random Forest (PD)"
    )

    # 4) Cut-off analysis on accepted test set
    test_results = X_test.copy()
    test_results[TARGET_COL] = y_test.values
    test_results["pd_pred"] = y_pred_proba_lr
    test_results["risk_score"] = (1 - test_results["pd_pred"]) * 100
    analyze_pd_cutoffs(test_results, TARGET_COL, pd_col="pd_pred")

    # 5) Expected loss on accepted portfolio (if LOSS_COL exists)
    compute_portfolio_expected_loss(
        df=accepted_df,
        features=features,
        clf=log_reg_clf,
        loss_col=LOSS_COL,
        ead_col=EAD_COL,
        target=TARGET_COL
    )

    # 6) Score rejected applications with PD model
    score_rejected_applications(rejected_df, features, log_reg_clf)


if __name__ == "__main__":
    main()
