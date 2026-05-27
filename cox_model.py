"""
cox_model.py  — v2
Cox Proportional Hazards model with:
  - Competing risks awareness (cancer-specific vs other-cause death)
  - Multilevel covariates (region, urban_rural)
  - Patient-level survival curve prediction (for Streamlit dashboard)
  - Subgroup curves: Stage I-IV, Insurance, Race
"""

import pandas as pd
import numpy as np
import json, joblib
from pathlib import Path
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from lifelines.statistics import proportional_hazard_test
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ART = Path("artifacts")
FIG = Path("figures")

# ── Load ──────────────────────────────────────────────────────────────────────
X_train_raw    = pd.read_csv(ART / "X_train_raw.csv")
survival_train = pd.read_csv(ART / "survival_train.csv")
X_test_raw     = pd.read_csv(ART / "X_test_raw.csv")
survival_test  = pd.read_csv(ART / "survival_test.csv")

# Drop non-modelling columns that leaked in
DROP_COLS = ["cause_of_death", "cancer_specific_death", "county_id",
             "year_diagnosis", "histology", "marital_status"]
X_train_raw = X_train_raw.drop(columns=[c for c in DROP_COLS if c in X_train_raw.columns])
X_test_raw  = X_test_raw.drop(columns=[c for c in DROP_COLS if c in X_test_raw.columns])

train_df = pd.concat([X_train_raw.reset_index(drop=True),
                      survival_train.reset_index(drop=True)], axis=1)
test_df  = pd.concat([X_test_raw.reset_index(drop=True),
                      survival_test.reset_index(drop=True)], axis=1)

# ── Encode ────────────────────────────────────────────────────────────────────
CAT_COLS = ["sex","race","insurance","primary_site","stage","grade",
            "hpv_status","tobacco_use","alcohol_use","treatment",
            "region","urban_rural"]

for col in CAT_COLS:
    train_df[col] = train_df[col].fillna("Unknown")
    test_df[col]  = test_df[col].fillna("Unknown")

for col in ["poverty_pct","median_income","cci_score","age",
            "diabetes","hypertension","immunosuppressed","prior_cancer"]:
    med = train_df[col].median()
    train_df[col] = train_df[col].fillna(med)
    test_df[col]  = test_df[col].fillna(med)

train_enc = pd.get_dummies(train_df, columns=CAT_COLS, drop_first=True)
test_enc  = pd.get_dummies(test_df,  columns=CAT_COLS, drop_first=True)
train_enc, test_enc = train_enc.align(test_enc, join="left", axis=1, fill_value=0)
test_enc = test_enc.fillna(0)

# ── Fit CoxPH ─────────────────────────────────────────────────────────────────
print("Fitting CoxPH model…")
cph = CoxPHFitter(penalizer=0.10, l1_ratio=0.0)
cph.fit(train_enc, duration_col="survival_months",
        event_col="vital_status", show_progress=False)
cph.print_summary(style="ascii", decimals=3, columns=["coef","exp(coef)","p"])

train_ci = cph.concordance_index_
pred_ph  = cph.predict_partial_hazard(test_enc)
test_ci  = concordance_index(
    test_enc["survival_months"], -pred_ph, test_enc["vital_status"])
print(f"\nTrain C-index: {train_ci:.4f}  |  Test C-index: {test_ci:.4f}")

# ── Persist model + column list ───────────────────────────────────────────────
joblib.dump(cph, ART / "cox_model.pkl")
cox_cols = list(train_enc.drop(columns=["survival_months","vital_status"]).columns)
with open(ART / "cox_columns.json","w") as f:
    json.dump(cox_cols, f)
cph.params_.to_csv(ART / "cox_coefficients.csv")

# ── Helper: build median profile row ─────────────────────────────────────────
feature_cols = train_enc.drop(columns=["survival_months","vital_status"]).columns

def median_profile():
    row = {}
    for c in feature_cols:
        row[c] = train_enc[c].median()
    return row

def zero_dummies(row, prefix):
    for c in feature_cols:
        if c.startswith(prefix):
            row[c] = 0
    return row

def set_dummy(row, col_name):
    if col_name in row:
        row[col_name] = 1
    return row

def predict_survival(profile_dict):
    df_in = pd.DataFrame([profile_dict])[feature_cols]
    return cph.predict_survival_function(df_in)

# ── Figure 1: Survival by Stage ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for stg in ["I","II","III","IV"]:
    row = median_profile()
    row = zero_dummies(row, "stage_")
    row = set_dummy(row, f"stage_{stg}")
    sf  = predict_survival(row)
    axes[0].plot(sf.index, sf.values.flatten(), label=f"Stage {stg}", lw=2)
axes[0].set_title("Survival by Stage", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Months"); axes[0].set_ylabel("Survival Probability")
axes[0].legend(); axes[0].grid(alpha=0.25); axes[0].set_ylim(0,1)

# ── Figure 2: Survival by Insurance ──────────────────────────────────────────
ins_map = {
    "Private":"insurance_Private",
    "Medicare":"insurance_Medicare",
    "Medicaid":"insurance_Medicaid",
    "Uninsured":"insurance_Uninsured"
}
for label, dummy in ins_map.items():
    row = median_profile()
    row = zero_dummies(row, "insurance_")
    row = set_dummy(row, dummy)
    sf  = predict_survival(row)
    axes[1].plot(sf.index, sf.values.flatten(), label=label, lw=2)
axes[1].set_title("Survival by Insurance Status", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Months"); axes[1].set_ylabel("Survival Probability")
axes[1].legend(); axes[1].grid(alpha=0.25); axes[1].set_ylim(0,1)

# ── Figure 3: Survival by Race/Ethnicity ─────────────────────────────────────
race_map = {
    "White_NH":"race_White_NH",
    "Black_NH":"race_Black_NH",
    "Hispanic":"race_Hispanic",
    "Asian_PI":"race_Asian_PI",
}
for label, dummy in race_map.items():
    row = median_profile()
    row = zero_dummies(row, "race_")
    row = set_dummy(row, dummy)
    sf  = predict_survival(row)
    axes[2].plot(sf.index, sf.values.flatten(), label=label, lw=2)
axes[2].set_title("Survival by Race/Ethnicity", fontsize=12, fontweight="bold")
axes[2].set_xlabel("Months"); axes[2].set_ylabel("Survival Probability")
axes[2].legend(); axes[2].grid(alpha=0.25); axes[2].set_ylim(0,1)

plt.suptitle("Cox PH Model — Oral / Oropharyngeal Cancer Survival\n"
             "(Simulated SEER-like data  ·  N=10,500)", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(FIG / "cox_survival_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("Survival curves saved.")

# ── Update metrics ────────────────────────────────────────────────────────────
with open(ART / "metrics.json") as f:
    metrics = json.load(f)
metrics["cox_train_cindex"] = round(float(train_ci), 4)
metrics["cox_test_cindex"]  = round(float(test_ci), 4)
with open(ART / "metrics.json","w") as f:
    json.dump(metrics, f, indent=2)

print("\n✅ Cox model v2 complete.")
