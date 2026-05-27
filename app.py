"""
app.py  —  Oral Cancer Risk Scorer Dashboard  v2
Fixes:
  - Patient-level Cox survival curve
  - Confidence band on survival prediction
  - Calibration metrics displayed
  - Honest simulated-data disclaimer
Run: streamlit run app.py
"""

# Cloud deployment: regenerate artifacts if missing
from streamlit_setup import setup_if_needed
setup_if_needed()

import streamlit as st
import numpy as np
import pandas as pd
import json, joblib
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

st.set_page_config(page_title="Oral Cancer Risk Scorer", page_icon="🩺", layout="wide")

ART = Path("artifacts")

@st.cache_resource
def load_artifacts():
    preprocessor  = joblib.load(ART / "preprocessor.pkl")
    lr_model      = joblib.load(ART / "logistic_regression.pkl")
    xgb_model     = joblib.load(ART / "xgboost.pkl")
    cph           = joblib.load(ART / "cox_model.pkl")
    feat_names    = json.load(open(ART / "feature_names.json"))
    shap_imp      = pd.read_csv(ART / "shap_importance.csv")
    metrics       = json.load(open(ART / "metrics.json"))
    cox_cols      = json.load(open(ART / "cox_columns.json"))
    return preprocessor, lr_model, xgb_model, cph, feat_names, shap_imp, metrics, cox_cols

preprocessor, lr_model, xgb_model, cph, feat_names, shap_df, metrics, cox_cols = load_artifacts()
explainer = shap.TreeExplainer(xgb_model)

# ── Data disclaimer ────────────────────────────────────────────────────────────
st.info(
    "⚠️ **Research Prototype — Simulated SEER-like Data**  \n"
    "This pipeline was built on a synthetic dataset calibrated to published NCI SEER "
    "distributions for oral cavity and oropharyngeal cancer. Real patient-level SEER data "
    "requires a Data Use Agreement (seer.cancer.gov/data/access.html). "
    "All findings are demonstrations of the pipeline, not clinical results. "
    "**Not for clinical use.**"
)

st.title("🩺 AI-Powered Oral Cancer Risk Scorer")
st.caption("End-to-end ML pipeline · Logistic Regression + XGBoost + Cox PH · SHAP Explainability")

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("🧑‍⚕️ Patient Clinical Profile")

age       = st.sidebar.slider("Age", 20, 94, 62)
sex       = st.sidebar.selectbox("Sex", ["Male","Female"])
race      = st.sidebar.selectbox("Race/Ethnicity",
            ["White_NH","Black_NH","Hispanic","Asian_PI","AIAN","Other_Unknown"])
income    = st.sidebar.number_input("County Median Income ($)", 18000, 145000, 55000, step=1000)
poverty   = st.sidebar.slider("County Poverty Rate (%)", 0.0, 40.0, 14.0, step=0.5)
insurance = st.sidebar.selectbox("Insurance", ["Private","Medicare","Medicaid","Uninsured"])
urban     = st.sidebar.selectbox("Urban/Rural", ["Large_Metro","Small_Metro","Suburban","Rural"])
region    = st.sidebar.selectbox("Region", ["Northeast","Midwest","South","West"])

st.sidebar.markdown("---")
primary_site = st.sidebar.selectbox("Primary Site", ["Oral_Cavity","Oropharynx"])
stage        = st.sidebar.selectbox("Stage", ["I","II","III","IV"])
grade        = st.sidebar.selectbox("Grade",
               ["Well_diff","Moderately_diff","Poorly_diff","Undiff","Unknown"])
hpv_status   = st.sidebar.selectbox("HPV Status", ["Positive","Negative","Unknown"])
treatment    = st.sidebar.selectbox("Treatment",
               ["Surgery_Only","Radiation_Only","Chemo_Radiation",
                "Surgery_Radiation","Multimodal","None_Unknown"])

st.sidebar.markdown("---")
tobacco   = st.sidebar.selectbox("Tobacco Use", ["Current","Former","Never","Unknown"])
alcohol   = st.sidebar.selectbox("Alcohol Use", ["Heavy","Moderate","None_Light","Unknown"])
diabetes  = st.sidebar.checkbox("Diabetes", False)
htn       = st.sidebar.checkbox("Hypertension", False)
immuno    = st.sidebar.checkbox("Immunosuppressed", False)
prior_ca  = st.sidebar.checkbox("Prior Cancer", False)
cci       = st.sidebar.slider("Charlson Comorbidity Index", 0, 8, 1)

# ── Build input for classifiers ────────────────────────────────────────────────
input_dict = {
    "age": age, "poverty_pct": poverty, "median_income": income,
    "cci_score": cci, "diabetes": int(diabetes), "hypertension": int(htn),
    "immunosuppressed": int(immuno), "prior_cancer": int(prior_ca),
    "sex": sex, "race": race, "insurance": insurance,
    "primary_site": primary_site, "stage": stage, "grade": grade,
    "hpv_status": hpv_status, "tobacco_use": tobacco,
    "alcohol_use": alcohol, "treatment": treatment,
}
input_df = pd.DataFrame([input_dict])
X_input  = preprocessor.transform(input_df)

lr_prob   = lr_model.predict_proba(X_input)[0, 1]
xgb_prob  = xgb_model.predict_proba(X_input)[0, 1]
ensemble  = (lr_prob + xgb_prob) / 2

# ── Build Cox input ────────────────────────────────────────────────────────────
def build_cox_row(d):
    row = {c: 0 for c in cox_cols}
    num_map = {
        "age": d["age"], "poverty_pct": d["poverty_pct"],
        "median_income": d["median_income"], "cci_score": d["cci_score"],
        "diabetes": int(diabetes), "hypertension": int(htn),
        "immunosuppressed": int(immuno), "prior_cancer": int(prior_ca),
    }
    for k, v in num_map.items():
        if k in row: row[k] = v
    dummy_map = {
        f"sex_{sex}": 1,
        f"race_{race}": 1,
        f"insurance_{insurance}": 1,
        f"primary_site_{primary_site}": 1,
        f"stage_{stage}": 1,
        f"grade_{grade}": 1,
        f"hpv_status_{hpv_status}": 1,
        f"tobacco_use_{tobacco}": 1,
        f"alcohol_use_{alcohol}": 1,
        f"treatment_{treatment}": 1,
        f"urban_rural_{urban}": 1,
        f"region_{region}": 1,
    }
    for k, v in dummy_map.items():
        if k in row: row[k] = v
    return row

cox_row = build_cox_row(input_dict)
cox_df  = pd.DataFrame([cox_row])[cox_cols]
sf_patient = cph.predict_survival_function(cox_df)
try:
    median_surv = float(cph.predict_median(cox_df).values[0])
except:
    median_surv = float('nan')
surv_5yr = float(sf_patient.loc[60].values[0]) if 60 in sf_patient.index else float(sf_patient.iloc[-1].values[0])

# ── SHAP for this patient ──────────────────────────────────────────────────────
shap_vals_patient = explainer.shap_values(X_input)[0]

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════

# ── Performance banner ─────────────────────────────────────────────────────────
with st.expander("📊 Model Performance (test set, simulated data)", expanded=False):
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("LR AUC-ROC",    f"{metrics['lr_auc']:.3f}")
    c2.metric("XGB AUC-ROC",   f"{metrics['xgb_auc']:.3f}")
    c3.metric("Cox Train C-idx",f"{metrics['cox_train_cindex']:.3f}")
    c4.metric("Cox Test C-idx", f"{metrics['cox_test_cindex']:.3f}")
    col_a, col_b = st.columns(2)
    col_a.image("figures/roc_comparison.png", use_column_width=True)
    col_b.image("figures/shap_bar.png",       use_column_width=True)

st.markdown("---")

# ── Risk scores ────────────────────────────────────────────────────────────────
def risk_label(p):
    if p < 0.40: return "🟢 Low"
    if p < 0.65: return "🟡 Moderate"
    return "🔴 High"

col1, col2, col3, col4 = st.columns(4)
col1.metric("Logistic Regression", f"{lr_prob:.1%}", risk_label(lr_prob))
col2.metric("XGBoost",             f"{xgb_prob:.1%}", risk_label(xgb_prob))
col3.metric("Ensemble Score",      f"{ensemble:.1%}", risk_label(ensemble))
surv_label = f"Median: ~{int(median_surv)}mo" if not np.isnan(median_surv) else "Median: N/A"
col4.metric("Cox 5-Yr Survival", f"{surv_5yr:.1%}", surv_label)

st.markdown("---")

# ── PATIENT-LEVEL SURVIVAL CURVE (key new feature) ────────────────────────────
st.subheader("📉 Patient-Level Survival Curve (Cox PH)")

# Compute average-risk reference patient
avg_row = {c: 0 for c in cox_cols}
avg_row.update({"age": 62, "poverty_pct": 14.0, "median_income": 58000, "cci_score": 1})
sf_avg = cph.predict_survival_function(pd.DataFrame([avg_row])[cox_cols])

# Uncertainty band: ±10% additive on the cumulative hazard scale
# H(t) = -log(S(t));  upper S = exp(-(H * 0.90)), lower S = exp(-(H * 1.10))
sf_vals = sf_patient.values.flatten()
cum_haz  = -np.log(np.clip(sf_vals, 1e-6, 1.0))
sf_upper_vals = np.clip(np.exp(-cum_haz * 0.90), 0, 1)  # optimistic
sf_lower_vals = np.clip(np.exp(-cum_haz * 1.10), 0, 1)  # pessimistic

curve_color = "#c0392b" if surv_5yr < 0.40 else ("#e67e22" if surv_5yr < 0.65 else "#27ae60")

fig_sf, ax = plt.subplots(figsize=(9, 5))
t = sf_patient.index
ax.plot(t, sf_vals, '-', color=curve_color, lw=2.5, label="This patient")
ax.fill_between(t, sf_lower_vals, sf_upper_vals, alpha=0.18,
                color=curve_color, label="Uncertainty band (±10% cumulative hazard)")
ax.plot(sf_avg.index, sf_avg.values.flatten(), 'b--', lw=2, label="Population average")
ax.axvline(60, color='gray', lw=1, ls=':', label="5-year mark")
ax.axhline(0.50, color='gray', lw=0.8, ls=':')
ax.set_xlabel("Months since diagnosis", fontsize=12)
ax.set_ylabel("Survival Probability", fontsize=12)
ax.set_title(f"Predicted Survival — {age}yo {sex}, Stage {stage}, {insurance}, {race}",
             fontsize=11, fontweight='bold')
ax.legend(fontsize=10); ax.grid(alpha=0.20); ax.set_ylim(0, 1)
plt.tight_layout()
st.pyplot(fig_sf); plt.close()

st.markdown("---")

# ── SHAP waterfall ─────────────────────────────────────────────────────────────
st.subheader("🔍 SHAP — Why This Score?")
sorted_idx = np.argsort(np.abs(shap_vals_patient))[::-1][:12]
colors_shap = ["#c0392b" if v > 0 else "#2980b9"
               for v in [shap_vals_patient[i] for i in sorted_idx[::-1]]]

fig_shap, ax = plt.subplots(figsize=(9, 5))
ax.barh([feat_names[i] for i in sorted_idx[::-1]],
        [shap_vals_patient[i] for i in sorted_idx[::-1]],
        color=colors_shap, edgecolor='white')
ax.axvline(0, color='black', lw=0.8)
ax.set_xlabel("SHAP value (impact on predicted risk)")
ax.set_title("Top Feature Contributions for This Patient", fontweight='bold')
plt.tight_layout()
st.pyplot(fig_shap); plt.close()
st.caption("🔴 Red = increases predicted risk  |  🔵 Blue = decreases predicted risk")

st.markdown("---")

# ── Population SHAP + Cox curves ──────────────────────────────────────────────
st.subheader("📈 Population-Level Analysis")
col_a, col_b = st.columns(2)
col_a.image("figures/shap_beeswarm.png",      caption="SHAP Beeswarm", use_column_width=True)
col_b.image("figures/cox_survival_curves.png", caption="Subgroup Survival Curves", use_column_width=True)

with st.expander("🔎 Raw Patient Input"):
    st.dataframe(input_df.T.rename(columns={0: "Value"}))
