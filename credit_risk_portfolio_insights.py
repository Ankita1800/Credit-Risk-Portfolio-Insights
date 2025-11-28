"""
credit_risk_portfolio_insights.py  — Updated, memory-friendly, ready-to-run single-file script.

Key changes vs original:
- Proper categorical encoding (OneHot for low-cardinality, Ordinal for high-cardinality)
- Automatic cardinality split using the actual accepted data
- Default: no SMOTE for large datasets (use class_weight instead). SMOTE can be enabled on a smaller sample.
- Preprocessor sanity checks before heavy training
- Save trained model to disk (joblib)
- Clear warnings about memory/cardinality
"""

import os
import warnings
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import joblib

# ============================================================
# 1. CONFIG
# ============================================================
ACCEPTED_PATH = "accepted_2007_to_2018Q4.csv"
REJECTED_PATH = "rejected_2007_to_2018Q4.csv"

TARGET_COL = "default"           # 1 = default/bad, 0 = non-default/good
SOURCE_COL = "application_status"  # accepted / rejected

NUMERIC_FEATURES = [
    "loan_amnt",
    "int_rate",
    "annual_inc",
    "dti",
    "emp_length_num",
    "fico_range_low"
]

CATEGORICAL_FEATURES = [
    "purpose",
    "grade",
    "home_ownership",
    "addr_state"
]

LOSS_COL = "loss_rate"
EAD_COL = "loan_amnt"

# Memory / behavior flags
USE_SMOTE = False              # Default: False (recommended for very large data)
SMOTE_SAMPLE_SIZE = 200_000    # If you want to try SMOTE, use it on a sampled subset of this many rows
HIGH_CARD_THRESHOLD = 30       # cardinality threshold for switching to ordinal encoding
MODEL_OUTPUT_PATH = "credit_pd_model.joblib"

# ============================================================
# 2. DATA LOADING & PREPARATION
# ============================================================
def load_and_prepare_data():
    accepted = pd.read_csv(ACCEPTED_PATH, low_memory=False)
    rejected = pd.read_csv(REJECTED_PATH, low_memory=False)

    accepted[SOURCE_COL] = "accepted"
    rejected[SOURCE_COL] = "rejected"

    if "loan_status" not in accepted.columns:
        raise ValueError("Column 'loan_status' not found in accepted data. Adjust mapping in load_and_prepare_data().")

    good_statuses = ["Fully Paid"]
    bad_statuses = ["Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off", "Late (31-120 days)"]

    def map_default(status):
        if status in good_statuses:
            return 0
        if status in bad_statuses:
            return 1
        return np.nan

    accepted[TARGET_COL] = accepted["loan_status"].apply(map_default)
    accepted = accepted.dropna(subset=[TARGET_COL])
    accepted[TARGET_COL] = accepted[TARGET_COL].astype(int)

    # numeric emp_length
    if "emp_length" in accepted.columns:
        accepted["emp_length_num"] = (
            accepted["emp_length"]
            .fillna("0")
            .replace({"10+ years": "10", "< 1 year": "0", "n/a": "0"})
            .str.extract(r"(\d+)")
            .astype(float)
        )
    if "emp_length" in rejected.columns:
        rejected["emp_length_num"] = (
            rejected["emp_length"]
            .fillna("0")
            .replace({"10+ years": "10", "< 1 year": "0", "n/a": "0"})
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

# ============================================================
# 4. PREPROCESSOR BUILDING (AUTOMATIC CARDINALITY SPLIT)
# ============================================================
def build_preprocessor_from_data(df, numeric_features, categorical_features, high_card_threshold=HIGH_CARD_THRESHOLD):
    """
    Build a ColumnTransformer using the actual df to decide which categorical features
    are low-cardinality (OneHot) vs high-cardinality (Ordinal).
    Returns: preprocessor, low_card_cats, high_card_cats
    """
    # Filter features to those present in df
    numeric_used = [c for c in numeric_features if c in df.columns]
    categorical_used = [c for c in categorical_features if c in df.columns]

    low_card = []
    high_card = []
    for c in categorical_used:
        nunique = df[c].nunique(dropna=True)
        if nunique <= high_card_threshold:
            low_card.append(c)
        else:
            high_card.append(c)

    # numeric pipeline
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # low-card: one-hot
    onehot_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
    ])

    # high-card: ordinal encoding (memory friendly)
    ordinal_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])

    transformers = [("num", numeric_transformer, numeric_used)]
    if low_card:
        transformers.append(("cat_onehot", onehot_transformer, low_card))
    if high_card:
        transformers.append(("cat_ord", ordinal_transformer, high_card))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor, numeric_used, low_card, high_card

# ============================================================
# 5. MODEL TRAINING & EVALUATION
# ============================================================
def train_pd_model(X_train, y_train, preprocessor, model_type="log_reg", use_smote=USE_SMOTE):
    # Choose model with class_weight if not using SMOTE
    if model_type == "log_reg":
        if use_smote:
            model = LogisticRegression(max_iter=500)
        else:
            model = LogisticRegression(max_iter=500, class_weight="balanced", solver="saga")
    elif model_type == "rf":
        if use_smote:
            model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight=None)
        else:
            model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    else:
        raise ValueError("model_type must be 'log_reg' or 'rf'")

    if use_smote:
        clf = ImbPipeline(steps=[
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("model", model)
        ])
    else:
        clf = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])

    clf.fit(X_train, y_train)
    return clf

def evaluate_pd_model(clf, X_test, y_test, threshold=0.5, name="Model"):
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    print(f"\n===== {name} EVALUATION =====")
    try:
        print("ROC AUC:", roc_auc_score(y_test, y_pred_proba))
    except Exception as ex:
        print("ROC AUC could not be computed:", ex)
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
    # we need to transform using the pipeline's preprocessor + model; easiest is predict_proba on the df[features]
    try:
        df["pd_pred"] = clf.predict_proba(df[features])[:, 1]
    except Exception as ex:
        print("Could not compute PDs for EL:", ex)
        return None

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
    try:
        rejected["pd_pred"] = clf.predict_proba(rejected[common_cols])[:, 1]
    except Exception as ex:
        print("Could not score rejected applications (need same preprocessor/features):", ex)
        return None

    print("\n===== REJECTED APPLICATIONS – PD SUMMARY =====")
    print(rejected["pd_pred"].describe())
    print("\nHigh-risk rejected (pd_pred > 0.3):", (rejected["pd_pred"] > 0.3).mean())
    print("Very low-risk rejected (pd_pred < 0.05):", (rejected["pd_pred"] < 0.05).mean())
    return rejected

# ============================================================
# 6. MAIN PIPELINE
# ============================================================
def main():
    warnings.filterwarnings("ignore")
    print("Loading data...")
    accepted_df, rejected_df = load_and_prepare_data()

    basic_eda(accepted_df, TARGET_COL, "ACCEPTED")
    for cat in ["grade", "purpose", "addr_state"]:
        plot_default_rate_by_category(accepted_df, TARGET_COL, cat)

    # Select features
    features = [f for f in NUMERIC_FEATURES + CATEGORICAL_FEATURES if f in accepted_df.columns]
    missing = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES) - set(features)
    if missing:
        print(f"\nWARNING: these configured features are not in accepted data and will be ignored: {missing}")

    X = accepted_df[features]
    y = accepted_df[TARGET_COL]

    # Time-safe split note: if you have issue_d or similar, prefer a time split. Here we keep simple stratified split.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Build preprocessor based on training data cardinality
    print("\nBuilding preprocessor (automatic cardinality split)...")
    preprocessor, numeric_used, low_card_cats, high_card_cats = build_preprocessor_from_data(
        pd.concat([X_train, X_test], ignore_index=True),
        NUMERIC_FEATURES,
        CATEGORICAL_FEATURES,
        high_card_threshold=HIGH_CARD_THRESHOLD
    )

    print("Numeric used:", numeric_used)
    print("OneHot (low-card) categorical:", low_card_cats)
    print("Ordinal (high-card) categorical:", high_card_cats)

    # Sanity test: transform a small slice
    try:
        X_small = X_train.head(100)
        transformed = preprocessor.fit_transform(X_small)
        print("Preprocessor transform successful. Transformed shape (100 rows):", transformed.shape)
    except Exception as ex:
        print("Preprocessor failed on small sample — check encoders and data:", ex)
        return

    # Decide whether to use SMOTE. For large full datasets we default to class_weight approach.
    if USE_SMOTE:
        # If dataset is very large, create a sample for SMOTE training
        n_train = len(X_train)
        if n_train > SMOTE_SAMPLE_SIZE:
            print(f"\nDataset large ({n_train} rows). Creating stratified sample of {SMOTE_SAMPLE_SIZE} rows for SMOTE training.")
            X_train_sample, _, y_train_sample, _ = train_test_split(
                X_train, y_train, train_size=SMOTE_SAMPLE_SIZE, stratify=y_train, random_state=42
            )
            train_for_smote_X = X_train_sample
            train_for_smote_y = y_train_sample
        else:
            train_for_smote_X = X_train
            train_for_smote_y = y_train

        print("Training LogisticRegression with SMOTE on sample...")
        log_reg_clf = train_pd_model(train_for_smote_X, train_for_smote_y, preprocessor, model_type="log_reg", use_smote=True)
    else:
        print("\nTraining LogisticRegression WITHOUT SMOTE (using class_weight='balanced') — memory-friendly default.")
        log_reg_clf = train_pd_model(X_train, y_train, preprocessor, model_type="log_reg", use_smote=False)

    # Evaluate logistic regression
    y_pred_proba_lr, _ = evaluate_pd_model(log_reg_clf, X_test, y_test, threshold=0.5, name="Logistic Regression (PD)")

    # Train RandomForest (memory friendly default, no SMOTE)
    print("\nTraining RandomForest (memory-friendly default)...")
    rf_clf = train_pd_model(X_train, y_train, preprocessor, model_type="rf", use_smote=False)
    _rf_proba, _ = evaluate_pd_model(rf_clf, X_test, y_test, threshold=0.5, name="Random Forest (PD)")

    # Cut-off analysis on accepted test set (using LR PDs)
    test_results = X_test.copy()
    test_results[TARGET_COL] = y_test.values
    test_results["pd_pred"] = y_pred_proba_lr
    test_results["risk_score"] = (1 - test_results["pd_pred"]) * 100
    analyze_pd_cutoffs(test_results, TARGET_COL, pd_col="pd_pred")

    # Expected loss (if LGD/EAD exists)
    compute_portfolio_expected_loss(
        df=accepted_df,
        features=features,
        clf=log_reg_clf,
        loss_col=LOSS_COL,
        ead_col=EAD_COL,
        target=TARGET_COL
    )

    # Score rejected applications
    score_rejected_applications(rejected_df, features, log_reg_clf)

    # Save the logistic model
    try:
        joblib.dump(log_reg_clf, MODEL_OUTPUT_PATH)
        print(f"\nSaved trained logistic PD model to {MODEL_OUTPUT_PATH}")
    except Exception as ex:
        print("Could not save model:", ex)

if __name__ == "__main__":
    main()
