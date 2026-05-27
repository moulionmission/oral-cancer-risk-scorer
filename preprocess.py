"""
preprocess.py
Full reproducible preprocessing pipeline for SEER oral cancer dataset.
Outputs: X_train, X_test, y_train, y_test, feature names, preprocessor object.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer

SEED = 42
DATA_PATH = "/home/claude/oral_cancer/data/seer_oral_cancer_simulated.csv"
OUT_DIR = Path("/home/claude/oral_cancer/artifacts")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"Loaded: {df.shape}")

# ── Drop non-feature columns ──────────────────────────────────────────────────
DROP = ["patient_id", "survival_months", "vital_status", "mortality_5yr"]
X = df.drop(columns=DROP)
y = df["mortality_5yr"]

# ── Column groups ─────────────────────────────────────────────────────────────
NUMERIC = ["age", "poverty_pct", "median_income", "cci_score",
           "diabetes", "hypertension", "immunosuppressed", "prior_cancer"]

CATEGORICAL = ["sex", "race", "insurance", "primary_site", "stage",
               "grade", "hpv_status", "tobacco_use", "alcohol_use", "treatment"]

# ── Pipelines ─────────────────────────────────────────────────────────────────
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
])

preprocessor = ColumnTransformer([
    ("num", num_pipe, NUMERIC),
    ("cat", cat_pipe, CATEGORICAL),
], remainder="drop")

# ── Split ─────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y
)

# ── Fit & transform ───────────────────────────────────────────────────────────
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc  = preprocessor.transform(X_test)

# ── Feature names ─────────────────────────────────────────────────────────────
feature_names = NUMERIC + CATEGORICAL

# ── Persist ───────────────────────────────────────────────────────────────────
joblib.dump(preprocessor, OUT_DIR / "preprocessor.pkl")
np.save(OUT_DIR / "X_train.npy", X_train_proc)
np.save(OUT_DIR / "X_test.npy",  X_test_proc)
np.save(OUT_DIR / "y_train.npy", y_train.values)
np.save(OUT_DIR / "y_test.npy",  y_test.values)

# Save raw splits for Cox model
X_train.to_csv(OUT_DIR / "X_train_raw.csv", index=False)
X_test.to_csv(OUT_DIR  / "X_test_raw.csv",  index=False)
pd.Series(y_train.values, name="mortality_5yr").to_csv(OUT_DIR / "y_train.csv", index=False)
pd.Series(y_test.values,  name="mortality_5yr").to_csv(OUT_DIR / "y_test.csv",  index=False)

# Save survival columns for Cox
df[["survival_months", "vital_status"]].iloc[X_train.index].to_csv(
    OUT_DIR / "survival_train.csv", index=False)
df[["survival_months", "vital_status"]].iloc[X_test.index].to_csv(
    OUT_DIR / "survival_test.csv",  index=False)

import json
with open(OUT_DIR / "feature_names.json", "w") as f:
    json.dump(feature_names, f)

print("Preprocessing complete.")
print(f"  Train: {X_train_proc.shape}  |  Test: {X_test_proc.shape}")
print(f"  Class balance (train): {y_train.mean():.3f} positive")
print(f"  Features: {feature_names}")
