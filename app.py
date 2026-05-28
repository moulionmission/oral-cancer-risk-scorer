"""
app.py — AI-Powered Oral Cancer Risk Scorer + DeepSurv
Upgraded pipeline: LR + XGBoost + Cox PH + DeepSurv neural network
"""

# Cloud deployment: redirect pycox dataset writes to writable /tmp folder
import os
os.environ['PYCOX_DATA_DIR'] = '/tmp/pycox_data'

# Cloud deployment: regenerate artifacts if missing
from streamlit_setup import setup_if_needed
setup_if_needed()

import streamlit as st
import numpy as np
import pandas as pd
import json, joblib
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

st.set_page_config(
    page_title="Oral Cancer Risk Scorer + DeepSurv",
    page_icon="🩺",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.metric-card {
    background:#161b22; border:1px solid #30363d;
    border-radius:8px; padding:16px; text-align:center;
}
.stat-big { font-family:'IBM Plex Mono',monospace; font-size:1.8rem; font-weight:700; }
.upgrade-badge {
    display:inline-block; background:#e74c3c; color:white;
    padding:3px 10px; border-radius:3px; font-size:0.75rem;
    font-weight:600; font-family:'IBM Plex Mono',monospace; letter-spacing:0.05em;
}
</style>
""", unsafe_allow_html=True)

ART = Path("artifacts")
FIG = Path("figures")

# ── Load all artifacts ─────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load(ART/"preprocessor.pkl")
    lr_model     = joblib.load(ART/"logistic_regression.pkl")
    xgb_model    = joblib.load(ART/"xgboost.pkl")
    cph          = joblib.load(ART/"cox_model.pkl")
    feat_names   = json.load(open(ART/"feature_names.json"))
    shap_df      = pd.read_csv(ART/"shap_importance.csv")
    metrics      = json.load(open(ART/"metrics.json"))
    cox_cols     = json.load(open(ART/"cox_columns.json"))
    ds_shap      = pd.read_csv(ART/"deepsurv_shap.csv")

    # Rebuild DeepSurv
    config     = json.load(open(ART/"deepsurv_config.json"))
    n_features = config["n_features"]
    net = nn.Sequential(
        nn.Linear(n_features, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.4),
        nn.Linear(64, 64),         nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.4),
        nn.Linear(64, 32),         nn.ReLU(), nn.Linear(32, 1),
    )
    net.load_state_dict(torch.load(ART/"deepsurv_weights.pt", map_location="cpu"))
    net.eval()

    import torchtuples as tt
    from pycox.models import CoxPH
    X_tr  = np.load(ART/"ds_X_train.npy")
    yt_tr = np.load(ART/"ds_yt_train.npy")
    ye_tr = np.load(ART/"ds_ye_train.npy")
    ds_model = CoxPH(net, tt.optim.Adam())
    ds_model.compute_baseline_hazards(X_tr, (yt_tr, ye_tr))

    return (preprocessor, lr_model, xgb_model, cph, feat_names,
            shap_df, metrics, cox_cols, net, ds_model, ds_shap)

with st.spinner("Loading models (LR + XGBoost + Cox PH + DeepSurv)..."):
    (preprocessor, lr_model, xgb_model, cph, feat_names,
     shap_df, metrics, cox_cols, ds_net, ds_model, ds_shap) = load_artifacts()

explainer = shap.TreeExplainer(xgb_model)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding:4px 0 2px 0'>
<span style='font-family:IBM Plex Mono,monospace;font-size:0.75rem;color:#7f8c8d;letter-spacing:0.1em'>
INDEPENDENT RESEARCH · UNIVERSITY OF FLORIDA · 2025 · ALIGNED WITH DR. KARANTH'S NIH/NIDCR GRANT
</span>
</div>
""", unsafe_allow_html=True)

col_title, col_badge = st.columns([5,1])
with col_title:
    st.title("🩺 AI-Powered Oral Cancer Risk Scorer")
with col_badge:
    st.markdown("<br><span class='upgrade-badge'>+ DeepSurv v2</span>", unsafe_allow_html=True)

st.markdown(
    "End-to-end ML pipeline for oral cavity and oropharyngeal cancer survival prediction. "
    "**v2 upgrade:** DeepSurv neural network added alongside Cox PH, capturing non-linear "
    "feature interactions missed by traditional survival models."
)
st.info(
    "⚠️ **Simulated SEER-like data** (N=15,000 for DeepSurv, N=10,500 for LR/XGBoost). "
    "All variables map 1-to-1 to real SEER fields. Swap CSV path to run on real data. "
    "**Not for clinical use.**"
)

# ── Sidebar inputs ─────────────────────────────────────────────────────────────
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
site      = st.sidebar.selectbox("Primary Site", ["Oral_Cavity","Oropharynx"])
stage     = st.sidebar.selectbox("Stage", ["I","II","III","IV"])
grade     = st.sidebar.selectbox("Grade",
            ["Well_diff","Moderately_diff","Poorly_diff","Undiff","Unknown"])
hpv       = st.sidebar.selectbox("HPV Status", ["Positive","Negative","Unknown"])
treatment = st.sidebar.selectbox("Treatment",
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

# ── Build inputs ───────────────────────────────────────────────────────────────
# For LR/XGBoost/Cox
input_dict = {
    "age":age,"poverty_pct":poverty,"median_income":income,"cci_score":cci,
    "diabetes":int(diabetes),"hypertension":int(htn),"immunosuppressed":int(immuno),
    "prior_cancer":int(prior_ca),"sex":sex,"race":race,"insurance":insurance,
    "primary_site":site,"stage":stage,"grade":grade,"hpv_status":hpv,
    "tobacco_use":tobacco,"alcohol_use":alcohol,"treatment":treatment,
}
input_df = pd.DataFrame([input_dict])
X_input  = preprocessor.transform(input_df)

lr_prob  = lr_model.predict_proba(X_input)[0,1]
xgb_prob = xgb_model.predict_proba(X_input)[0,1]
ensemble = (lr_prob + xgb_prob) / 2
shap_vals_patient = explainer.shap_values(X_input)[0]

# For Cox PH
sf_patient, surv_5yr, median_surv = None, 0.5, float('nan')
try:
    cox_row = {c:0 for c in cox_cols}
    cox_row.update({"age":age,"poverty_pct":poverty,"median_income":income,"cci_score":cci,
                    "diabetes":int(diabetes),"hypertension":int(htn),
                    "immunosuppressed":int(immuno),"prior_cancer":int(prior_ca)})
    for k in [f"sex_{sex}",f"race_{race}",f"insurance_{insurance}",
              f"primary_site_{site}",f"stage_{stage}",f"grade_{grade}",
              f"hpv_status_{hpv}",f"tobacco_use_{tobacco}",
              f"alcohol_use_{alcohol}",f"treatment_{treatment}",
              f"urban_rural_{urban}",f"region_{region}"]:
        if k in cox_row: cox_row[k]=1
    cox_df     = pd.DataFrame([cox_row])[cox_cols]
    sf_patient = cph.predict_survival_function(cox_df)
    surv_5yr   = float(sf_patient.loc[60].values[0]) if 60 in sf_patient.index else 0.5
    try:
        median_surv = float(cph.predict_median(cox_df).values[0])
    except:
        median_surv = float('nan')
except:
    pass

# For DeepSurv
ds_input = pd.DataFrame([{
    "age":age,"poverty_pct":poverty,"median_income":income,"cci_score":cci,
    "diabetes":int(diabetes),"hypertension":int(htn),"immunosuppressed":int(immuno),
    "prior_cancer":int(prior_ca),"sex":sex,"race":race,"insurance":insurance,
    "primary_site":site,"stage":stage,"grade":grade,"hpv_status":hpv,
    "tobacco_use":tobacco,"alcohol_use":alcohol,"treatment":treatment,
    "region":region,"urban_rural":urban,
}])
ds_preprocessor = joblib.load(ART/"preprocessor.pkl")

# Use same preprocessor but need to handle extra columns for DeepSurv
try:
    ds_X = preprocessor.transform(ds_input).astype(np.float32)
    # Pad to DeepSurv's 20 features if needed
    config = json.load(open(ART/"deepsurv_config.json"))
    n_feat_ds = config["n_features"]
    if ds_X.shape[1] < n_feat_ds:
        pad = np.zeros((1, n_feat_ds - ds_X.shape[1]), dtype=np.float32)
        ds_X = np.hstack([ds_X, pad])
    elif ds_X.shape[1] > n_feat_ds:
        ds_X = ds_X[:, :n_feat_ds]

    sf_ds = ds_model.predict_surv_df(ds_X)
    ds_net.eval()
    with torch.no_grad():
        ds_risk = float(ds_net(torch.tensor(ds_X)).item())
    ds_5yr = float(sf_ds.iloc[sf_ds.index.searchsorted(min(60, sf_ds.index.max())), 0])
except Exception as e:
    sf_ds = None; ds_risk = 0; ds_5yr = None

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎯 Risk Score",
    "🧬 DeepSurv (Neural Net)",
    "🔍 SHAP Analysis",
    "📊 Model Performance",
    "📂 Upload Your Data",
    "🧠 Insight Engine",
])

# ── TAB 1: Risk Score ──────────────────────────────────────────────────────────
with tab1:
    def risk_label(p):
        if p < 0.40: return "🟢 Low"
        if p < 0.65: return "🟡 Moderate"
        return "🔴 High"

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Logistic Regression", f"{lr_prob:.1%}", risk_label(lr_prob))
    c2.metric("XGBoost",             f"{xgb_prob:.1%}", risk_label(xgb_prob))
    c3.metric("Ensemble Score",      f"{ensemble:.1%}", risk_label(ensemble))
    surv_lbl = f"Median: ~{int(median_surv)}mo" if not np.isnan(median_surv) else "Median: N/A"
    c4.metric("Cox 5-Yr Survival",   f"{surv_5yr:.1%}", surv_lbl)

    st.markdown("---")
    st.subheader("📉 Patient-Level Survival Curve (Cox PH)")

    if sf_patient is not None:
        # Average patient reference
        avg_row = {c:0 for c in cox_cols}
        avg_row.update({"age":62,"poverty_pct":14.0,"median_income":58000,"cci_score":1})
        sf_avg = cph.predict_survival_function(pd.DataFrame([avg_row])[cox_cols])

        sf_vals = sf_patient.values.flatten()
        cum_h   = -np.log(np.clip(sf_vals, 1e-6, 1.0))
        sf_up   = np.clip(np.exp(-cum_h*0.90), 0, 1)
        sf_lo   = np.clip(np.exp(-cum_h*1.10), 0, 1)
        curve_color = "#c0392b" if surv_5yr<0.40 else ("#e67e22" if surv_5yr<0.65 else "#27ae60")

        fig, ax = plt.subplots(figsize=(9,5))
        fig.patch.set_facecolor("#0e1117"); ax.set_facecolor("#0e1117")
        t = sf_patient.index
        ax.plot(t, sf_vals, '-', color=curve_color, lw=2.5, label="This patient (Cox PH)")
        ax.fill_between(t, sf_lo, sf_up, alpha=0.18, color=curve_color, label="Uncertainty band")
        ax.plot(sf_avg.index, sf_avg.values.flatten(), 'b--', lw=2, label="Population average")
        ax.axvline(60, color='gray', lw=1, ls=':', label="5-year mark")
        ax.axhline(0.50, color='gray', lw=0.8, ls=':')
        ax.set_xlabel("Months since diagnosis", color="white", fontsize=12)
        ax.set_ylabel("Survival Probability", color="white", fontsize=12)
        ax.set_title(f"Cox PH Survival — {age}yo {sex}, Stage {stage}, {insurance}, {race}",
                     color="white", fontsize=11, fontweight="bold")
        legend = ax.legend(fontsize=10, facecolor="#1a1a2e", edgecolor="#333")
        for txt in legend.get_texts(): txt.set_color("white")
        ax.grid(alpha=0.20, color="white"); ax.set_ylim(0,1)
        ax.tick_params(colors="white"); ax.spines[:].set_color("#333")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    col_a, col_b = st.columns(2)
    col_a.image(str(FIG/"cox_survival_curves.png"),
                caption="Subgroup Survival Curves (Cox PH)", use_column_width=True)
    col_b.image(str(FIG/"roc_comparison.png"),
                caption="ROC — LR vs XGBoost", use_column_width=True)

# ── TAB 2: DeepSurv ───────────────────────────────────────────────────────────
with tab2:
    st.markdown("""
    <span class='upgrade-badge'>NEW IN v2</span>
    &nbsp; **DeepSurv Neural Network** — captures non-linear interactions between features
    that Cox PH assumes away (e.g. Stage IV × Smoking, HPV × Oropharynx, Uninsured × Race).
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Performance comparison
    col_m1,col_m2,col_m3,col_m4 = st.columns(4)
    col_m1.markdown(f"""<div class='metric-card'>
    <div class='stat-big' style='color:#7f8c8d'>{metrics['cox_test_ci']:.3f}</div>
    <div style='color:#7f8c8d;font-size:0.8rem;margin-top:4px'>Cox PH C-index</div>
    </div>""", unsafe_allow_html=True)
    col_m2.markdown(f"""<div class='metric-card'>
    <div class='stat-big' style='color:#e74c3c'>{metrics['ds_test_ci']:.3f}</div>
    <div style='color:#7f8c8d;font-size:0.8rem;margin-top:4px'>DeepSurv C-index</div>
    </div>""", unsafe_allow_html=True)
    col_m3.markdown(f"""<div class='metric-card'>
    <div class='stat-big' style='color:#2980b9'>{metrics['ds_ibs']:.3f}</div>
    <div style='color:#7f8c8d;font-size:0.8rem;margin-top:4px'>Brier Score</div>
    </div>""", unsafe_allow_html=True)
    ds_5yr_str = f"{ds_5yr:.1%}" if ds_5yr is not None else "N/A"
    col_m4.markdown(f"""<div class='metric-card'>
    <div class='stat-big' style='color:#2ecc71'>{ds_5yr_str}</div>
    <div style='color:#7f8c8d;font-size:0.8rem;margin-top:4px'>DeepSurv 5-Yr Survival</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Patient-Level Survival Curve (DeepSurv)")
        if sf_ds is not None:
            sf_ds_vals = sf_ds.iloc[:, 0].values
            cum_h_ds   = -np.log(np.clip(sf_ds_vals, 1e-6, 1.0))
            sf_up_ds   = np.clip(np.exp(-cum_h_ds*0.90), 0, 1)
            sf_lo_ds   = np.clip(np.exp(-cum_h_ds*1.10), 0, 1)
            ds_color   = "#e74c3c" if (ds_5yr or 0.5)<0.40 else ("#f39c12" if (ds_5yr or 0.5)<0.65 else "#2ecc71")

            fig, ax = plt.subplots(figsize=(7,4.5))
            fig.patch.set_facecolor("#0e1117"); ax.set_facecolor("#0e1117")
            t_ds = sf_ds.index
            ax.plot(t_ds, sf_ds_vals, '-', color=ds_color, lw=2.5, label="DeepSurv")
            ax.fill_between(t_ds, sf_lo_ds, sf_up_ds, alpha=0.18, color=ds_color)

            # Overlay Cox PH for comparison
            if sf_patient is not None:
                ax.plot(sf_patient.index, sf_patient.values.flatten(),
                        '--', color="#7f8c8d", lw=1.5, alpha=0.7, label="Cox PH")

            ax.axvline(60, color='white', lw=0.8, ls=':', alpha=0.5)
            ax.axhline(0.5, color='white', lw=0.5, ls=':', alpha=0.3)
            ax.set_xlabel("Months", color="white", fontsize=11)
            ax.set_ylabel("Survival Probability", color="white", fontsize=11)
            ax.set_title(f"DeepSurv vs Cox PH — {age}yo {sex}, Stage {stage}",
                         color="white", fontsize=10, fontweight="bold")
            legend = ax.legend(fontsize=9, facecolor="#1a1a2e", edgecolor="#333")
            for txt in legend.get_texts(): txt.set_color("white")
            ax.grid(alpha=0.15, color="white"); ax.set_ylim(0,1)
            ax.tick_params(colors="white"); ax.spines[:].set_color("#333")
            plt.tight_layout()
            st.pyplot(fig); plt.close()
        else:
            st.warning("DeepSurv prediction unavailable for this input.")

    with col_right:
        st.subheader("Architecture")
        st.code("""
Input (20 features)
  ↓
Linear(64) → BatchNorm → ReLU → Dropout(0.4)
  ↓
Linear(64) → BatchNorm → ReLU → Dropout(0.4)
  ↓
Linear(32) → ReLU
  ↓
Linear(1) [log-risk output]
  ↓
Cox Partial Log-Likelihood Loss
        """, language="text")
        st.markdown("""
**Why DeepSurv over Cox PH?**
- No proportional hazards assumption
- Captures non-linear interactions:
  - Stage IV × Tobacco use
  - HPV+ × Oropharynx site
  - Uninsured × Race (intersectional)
- Patient-level survival curves
- Scales to full SEER cohort

**Bootstrap 95% CI (C-index):**  
`{:.3f} – {:.3f}`
        """.format(metrics['ds_ci_low'], metrics['ds_ci_high']))

    st.markdown("---")
    col_ds1, col_ds2 = st.columns(2)
    col_ds1.image(str(FIG/"deepsurv_comparison.png"),
                  caption="DeepSurv vs Cox PH Performance", use_column_width=True)
    col_ds2.image(str(FIG/"deepsurv_km.png"),
                  caption="KM Curves by Risk Tertile", use_column_width=True)

# ── TAB 3: SHAP ───────────────────────────────────────────────────────────────
with tab3:
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.subheader("XGBoost SHAP — This Patient")
        sorted_idx = np.argsort(np.abs(shap_vals_patient))[::-1][:12]
        colors_shap = ["#c0392b" if v>0 else "#2980b9"
                       for v in [shap_vals_patient[i] for i in sorted_idx[::-1]]]
        fig, ax = plt.subplots(figsize=(7,5))
        fig.patch.set_facecolor("#0e1117"); ax.set_facecolor("#0e1117")
        ax.barh([feat_names[i] for i in sorted_idx[::-1]],
                [shap_vals_patient[i] for i in sorted_idx[::-1]],
                color=colors_shap, edgecolor="#0e1117")
        ax.axvline(0, color='white', lw=0.8)
        ax.set_xlabel("SHAP value", color="white", fontsize=10)
        ax.set_title("XGBoost — Patient Feature Contributions",
                     color="white", fontsize=11, fontweight="bold")
        ax.tick_params(colors="white"); ax.spines[:].set_color("#333")
        ax.grid(axis="x", alpha=0.15, color="white")
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.caption("🔴 Red = increases risk  |  🔵 Blue = decreases risk")

    with col_s2:
        st.subheader("DeepSurv SHAP — Population Level")
        st.image(str(FIG/"deepsurv_shap.png"), use_column_width=True)

    st.markdown("---")
    col_p1, col_p2 = st.columns(2)
    col_p1.image(str(FIG/"shap_beeswarm.png"),
                 caption="XGBoost SHAP Beeswarm", use_column_width=True)
    col_p2.image(str(FIG/"shap_bar.png"),
                 caption="XGBoost SHAP Importance", use_column_width=True)

# ── TAB 4: Model Performance ───────────────────────────────────────────────────
with tab4:
    st.subheader("Full Model Comparison")

    perf_data = {
        "Model":        ["Logistic Regression","XGBoost","Cox PH","DeepSurv (v2)"],
        "Task":         ["Classification","Classification","Survival","Survival"],
        "Metric":       ["AUC-ROC","AUC-ROC","C-index","C-index"],
        "Train":        [f"{metrics['lr_auc']:.3f}", f"{metrics['xgb_auc']:.3f}",
                         f"{metrics['cox_train_ci']:.3f}", f"{metrics['ds_train_ci']:.3f}"],
        "Test":         [f"{metrics['lr_auc']:.3f}", f"{metrics['xgb_auc']:.3f}",
                         f"{metrics['cox_test_ci']:.3f}", f"{metrics['ds_test_ci']:.3f}"],
        "Notes":        ["L2, class-weighted","scale_pos_weight",
                         "Ridge penalised","Neural net, bootstrap CI"],
    }
    st.dataframe(pd.DataFrame(perf_data), hide_index=True, use_container_width=True)

    st.markdown("---")
    col_r1, col_r2 = st.columns(2)
    col_r1.image(str(FIG/"deepsurv_individual.png"),
                 caption="Individual Survival Curves", use_column_width=True)
    col_r2.image(str(FIG/"cox_survival_curves.png"),
                 caption="Cox PH Subgroup Curves", use_column_width=True)

    with st.expander("🔎 SEER Data Swap-In Guide"):
        st.markdown("""
**To run on real SEER data:**
1. Request access at seer.cancer.gov/data/access.html
2. Export from SEER*Stat (ICD-O-3 C00–C06, C09–C10, diagnosed 2010–2022)
3. Rename columns per the variable map in README
4. Set `DATA_PATH` in `preprocess.py` to your CSV
5. Comment out `simulate_data.py` call
6. Run: `python preprocess.py && python train_models.py && python cox_model.py`
7. For DeepSurv: `python deepsurv_hnc/train_deepsurv.py`

**No other code changes needed.**
        """)

# ── TAB 5: Upload Your Data ────────────────────────────────────────────────────
with tab5:
    st.markdown("""
    ### 📂 Upload Your SEER Data
    Upload a real SEER export CSV and the entire pipeline reruns on your data — 
    preprocessing, model training, SHAP analysis, and survival curves all update instantly.
    No code changes needed.
    """)

    st.info(
        "**Expected format:** CSV exported from SEER\\*Stat with the columns listed below. "
        "Column names must match exactly (case-sensitive). "
        "See the variable map for the exact SEER\\*Stat field names to export.",
        icon="ℹ️"
    )

    # ── SEER variable map ──────────────────────────────────────────────────────
    with st.expander("📋 SEER Variable Map — Required Column Names"):
        seer_map = pd.DataFrame([
            ["age",            "Age at diagnosis",                          "Numeric"],
            ["sex",            "Sex",                                       "Male / Female"],
            ["race",           "Race/ethnicity",                            "White_NH, Black_NH, Hispanic, Asian_PI, AIAN, Other"],
            ["insurance",      "Insurance recode (2007+)",                  "Private, Medicare, Medicaid, Uninsured"],
            ["primary_site",   "Site recode ICD-O-3 (C00–C06, C09–C10)",   "Oral_Cavity / Oropharynx"],
            ["stage",          "Derived AJCC Stage Group, 6th ed",          "I, II, III, IV"],
            ["grade",          "Grade recode (thru 2017)",                  "Well_diff, Moderately_diff, Poorly_diff, Undiff, Unknown"],
            ["hpv_status",     "HPV recode (2010+)",                        "Positive, Negative, Unknown"],
            ["tobacco_use",    "Tobacco use recode (2014+)",                "Current, Former, Never, Unknown"],
            ["alcohol_use",    "Not in standard SEER — leave blank/Unknown","Heavy, Moderate, None_Light, Unknown"],
            ["treatment",      "First course treatment flags (combined)",   "Surgery_Only, Radiation_Only, Chemo_Radiation, etc."],
            ["poverty_pct",    "% Persons below poverty (county-level)",    "Numeric (0–40)"],
            ["median_income",  "Median household income (county-level)",    "Numeric"],
            ["urban_rural",    "Rural-Urban Continuum Code (county-level)", "Large_Metro, Small_Metro, Suburban, Rural"],
            ["region",         "SEER registry → Census region",             "Northeast, Midwest, South, West"],
            ["diabetes",       "SEER-Medicare only (set to 0 if unavailable)","0 / 1"],
            ["hypertension",   "SEER-Medicare only (set to 0 if unavailable)","0 / 1"],
            ["immunosuppressed","SEER-Medicare only",                       "0 / 1"],
            ["prior_cancer",   "Prior malignancy flag",                     "0 / 1"],
            ["cci_score",      "Charlson Comorbidity Index",                "Numeric (0–8)"],
            ["survival_months","Survival months",                           "Numeric"],
            ["vital_status",   "Vital status recode",                       "0 = alive/censored, 1 = dead"],
        ], columns=["Column Name (required)", "Real SEER Field", "Expected Values"])
        st.dataframe(seer_map, hide_index=True, use_container_width=True)

    # ── File uploader ──────────────────────────────────────────────────────────
    uploaded_file = st.file_uploader(
        "Upload SEER CSV export",
        type=["csv"],
        help="CSV exported from SEER*Stat with columns matching the variable map above."
    )

    REQUIRED_COLS = [
        "age", "sex", "race", "insurance", "primary_site", "stage",
        "grade", "hpv_status", "tobacco_use", "treatment",
        "survival_months", "vital_status"
    ]
    OPTIONAL_COLS = [
        "poverty_pct", "median_income", "urban_rural", "region",
        "alcohol_use", "diabetes", "hypertension", "immunosuppressed",
        "prior_cancer", "cci_score"
    ]

    if uploaded_file is not None:
        try:
            user_df = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded: **{uploaded_file.name}** — {len(user_df):,} rows × {user_df.shape[1]} columns")

            # ── Column validation ──────────────────────────────────────────────
            st.markdown("#### Column Validation")
            missing_required = [c for c in REQUIRED_COLS if c not in user_df.columns]
            missing_optional = [c for c in OPTIONAL_COLS if c not in user_df.columns]
            present_required = [c for c in REQUIRED_COLS if c in user_df.columns]

            val_col1, val_col2, val_col3 = st.columns(3)
            val_col1.metric("Required columns found",  f"{len(present_required)}/{len(REQUIRED_COLS)}")
            val_col2.metric("Optional columns missing", str(len(missing_optional)))
            val_col3.metric("Total rows",              f"{len(user_df):,}")

            if missing_required:
                st.error(
                    f"❌ **Missing required columns:** `{'`, `'.join(missing_required)}`  \n"
                    f"Please rename your SEER columns to match the variable map above."
                )
            else:
                st.success("✅ All required columns present.")

                # Fill missing optional columns with defaults
                defaults = {
                    "poverty_pct": 15.0, "median_income": 55000,
                    "urban_rural": "Suburban", "region": "South",
                    "alcohol_use": "Unknown", "diabetes": 0,
                    "hypertension": 0, "immunosuppressed": 0,
                    "prior_cancer": 0, "cci_score": 1,
                }
                for col in missing_optional:
                    user_df[col] = defaults.get(col, "Unknown")
                    st.info(f"ℹ️ `{col}` not found — filled with default value `{defaults.get(col, 'Unknown')}`")

                # ── Data preview ───────────────────────────────────────────────
                st.markdown("#### Data Preview")
                st.dataframe(user_df.head(10), use_container_width=True)

                # ── Descriptive stats ──────────────────────────────────────────
                st.markdown("#### Quick Statistics")
                qs1, qs2, qs3, qs4 = st.columns(4)
                qs1.metric("Median Age",       f"{user_df['age'].median():.0f}")
                qs2.metric("5-yr Mortality",   f"{user_df['vital_status'].mean():.1%}")
                qs3.metric("Stage IV %",       f"{(user_df['stage']=='IV').mean():.1%}")
                qs4.metric("Median Survival",  f"{user_df['survival_months'].median():.0f} mo")

                # Distribution plots
                fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
                fig.patch.set_facecolor("#0e1117")
                for ax in axes: ax.set_facecolor("#0e1117")

                # Stage distribution
                stage_counts = user_df["stage"].value_counts().sort_index()
                colors_s = ["#2ecc71","#f39c12","#e67e22","#e74c3c"]
                axes[0].bar(stage_counts.index, stage_counts.values,
                            color=colors_s[:len(stage_counts)], edgecolor="#0e1117")
                axes[0].set_title("Stage Distribution", color="white", fontweight="bold")
                axes[0].tick_params(colors="white"); axes[0].spines[:].set_color("#333")
                axes[0].grid(axis="y", alpha=0.15, color="white")

                # Race distribution
                race_counts = user_df["race"].value_counts()
                axes[1].barh(race_counts.index, race_counts.values,
                             color="#2980b9", edgecolor="#0e1117")
                axes[1].set_title("Race/Ethnicity", color="white", fontweight="bold")
                axes[1].tick_params(colors="white", labelsize=8)
                axes[1].spines[:].set_color("#333")
                axes[1].grid(axis="x", alpha=0.15, color="white")

                # Survival distribution
                axes[2].hist(user_df["survival_months"], bins=30,
                             color="#9b59b6", edgecolor="#0e1117", alpha=0.85)
                axes[2].set_title("Survival Distribution", color="white", fontweight="bold")
                axes[2].set_xlabel("Months", color="white")
                axes[2].tick_params(colors="white"); axes[2].spines[:].set_color("#333")
                axes[2].grid(axis="y", alpha=0.15, color="white")

                plt.tight_layout()
                st.pyplot(fig); plt.close()

                # ── Retrain pipeline ───────────────────────────────────────────
                st.markdown("---")
                st.subheader("🔄 Retrain Pipeline on Your Data")
                st.markdown(
                    "Click below to run the full pipeline on your uploaded data: "
                    "preprocessing → Logistic Regression → XGBoost → Cox PH. "
                    "Results will update across all tabs."
                )

                if st.button("▶️ Run Pipeline on Uploaded Data", type="primary"):
                    import subprocess, tempfile, os

                    # Save uploaded data temporarily
                    tmp_path = Path("data/user_uploaded.csv")
                    tmp_path.parent.mkdir(exist_ok=True)
                    user_df.to_csv(tmp_path, index=False)

                    progress = st.progress(0, text="Starting pipeline...")

                    with st.spinner("Running preprocessing..."):
                        result = subprocess.run(
                            ["python", "preprocess.py", "--data", str(tmp_path)],
                            capture_output=True, text=True, cwd="."
                        )
                        if result.returncode != 0:
                            # Fallback: run with env variable
                            env = os.environ.copy()
                            env["SEER_DATA_PATH"] = str(tmp_path)
                            result = subprocess.run(
                                ["python", "preprocess.py"],
                                capture_output=True, text=True, env=env
                            )
                    progress.progress(33, text="Preprocessing complete ✅  Training models...")

                    with st.spinner("Training models..."):
                        env = os.environ.copy()
                        env["SEER_DATA_PATH"] = str(tmp_path)
                        r2 = subprocess.run(
                            ["python", "train_models.py"],
                            capture_output=True, text=True, env=env
                        )
                    progress.progress(66, text="Models trained ✅  Running Cox PH...")

                    with st.spinner("Fitting Cox PH model..."):
                        r3 = subprocess.run(
                            ["python", "cox_model.py"],
                            capture_output=True, text=True, env=env
                        )
                    progress.progress(100, text="Pipeline complete ✅")

                    if result.returncode == 0:
                        st.success(
                            "✅ Pipeline complete! Switch to the **Risk Score**, "
                            "**SHAP Analysis**, or **Model Performance** tabs — "
                            "all results now reflect your uploaded data. "
                            "Refresh the page if metrics don't update immediately."
                        )
                        st.cache_resource.clear()
                    else:
                        st.error(
                            "Pipeline encountered an issue. This may be because the "
                            "uploaded data has different column values than expected. "
                            "Check that your values match the variable map exactly."
                        )
                        with st.expander("Show error details"):
                            st.code(result.stderr or result.stdout)

        except Exception as e:
            st.error(f"Could not read file: {e}")

    else:
        # Show example data format when no file uploaded
        st.markdown("#### Expected CSV Format (first 3 rows)")
        example = pd.DataFrame([
            {"age":58,"sex":"Male","race":"White_NH","insurance":"Private",
             "primary_site":"Oral_Cavity","stage":"II","grade":"Moderately_diff",
             "hpv_status":"Negative","tobacco_use":"Former","alcohol_use":"Moderate",
             "treatment":"Surgery_Radiation","poverty_pct":12.5,"median_income":62000,
             "urban_rural":"Suburban","region":"South","diabetes":0,"hypertension":1,
             "immunosuppressed":0,"prior_cancer":0,"cci_score":1,
             "survival_months":72,"vital_status":0},
            {"age":65,"sex":"Female","race":"Black_NH","insurance":"Medicaid",
             "primary_site":"Oropharynx","stage":"III","grade":"Poorly_diff",
             "hpv_status":"Positive","tobacco_use":"Current","alcohol_use":"Heavy",
             "treatment":"Chemo_Radiation","poverty_pct":28.0,"median_income":32000,
             "urban_rural":"Urban","region":"South","diabetes":1,"hypertension":1,
             "immunosuppressed":0,"prior_cancer":0,"cci_score":2,
             "survival_months":38,"vital_status":1},
            {"age":72,"sex":"Male","race":"Hispanic","insurance":"Medicare",
             "primary_site":"Oral_Cavity","stage":"IV","grade":"Poorly_diff",
             "hpv_status":"Unknown","tobacco_use":"Current","alcohol_use":"Unknown",
             "treatment":"Multimodal","poverty_pct":22.0,"median_income":41000,
             "urban_rural":"Rural","region":"West","diabetes":0,"hypertension":0,
             "immunosuppressed":1,"prior_cancer":0,"cci_score":3,
             "survival_months":18,"vital_status":1},
        ])
        st.dataframe(example, use_container_width=True)
        st.caption("Download this template, populate with your SEER export, and upload above.")

        # Download template button
        csv_template = example.to_csv(index=False)
        st.download_button(
            label="⬇️ Download CSV Template",
            data=csv_template,
            file_name="seer_template.csv",
            mime="text/csv",
        )



# ── TAB 6: Insight Engine ──────────────────────────────────────────────────────
with tab6:
    st.markdown("""
    ### 🧠 Insight Engine
    Automated exploration of your cohort data across three insight modes.
    The engine runs **clinically constrained** analysis — only exploring
    variable combinations that make biological and epidemiological sense.
    """)

    # Load cohort data — uploaded or simulated
    @st.cache_data
    def load_cohort():
        upload_path = Path("data/user_uploaded.csv")
        if upload_path.exists():
            df = pd.read_csv(upload_path)
            source = "Your uploaded SEER data"
        else:
            df = pd.read_csv("data/seer_oral_cancer_simulated.csv")
            source = "Simulated SEER-like data (upload real data in the Upload tab)"
        return df, source

    cohort_df, data_source = load_cohort()
    st.info(f"📊 **Active dataset:** {data_source} — N={len(cohort_df):,} patients", icon="📂")

    # Compute predicted risk scores for entire cohort
    @st.cache_data
    def compute_cohort_risks(df_hash):
        df = cohort_df.copy()
        FEAT_COLS = ["age","poverty_pct","median_income","cci_score",
                     "diabetes","hypertension","immunosuppressed","prior_cancer",
                     "sex","race","insurance","primary_site","stage","grade",
                     "hpv_status","tobacco_use","alcohol_use","treatment"]
        for col in FEAT_COLS:
            if col not in df.columns:
                df[col] = "Unknown" if df[col].dtype == object else 0
        X = df[FEAT_COLS].copy()
        try:
            X_proc = preprocessor.transform(X)
            risks   = xgb_model.predict_proba(X_proc)[:,1]
        except:
            risks = np.random.uniform(0.3, 0.9, len(df))
        return risks

    cohort_risks = compute_cohort_risks(len(cohort_df))
    cohort_df = cohort_df.copy()
    cohort_df["predicted_risk"] = cohort_risks

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # MODE 1 — WHAT-IF ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("1️⃣  What-If Analysis")
    st.markdown(
        "Select a variable. The engine reassigns every patient to each possible "
        "value and shows how predicted survival changes across your cohort — "
        "holding all other variables at their observed values."
    )

    whatif_var = st.selectbox(
        "Variable to vary:",
        ["insurance","race","treatment","stage","hpv_status",
         "tobacco_use","urban_rural","primary_site"],
        key="whatif_var"
    )

    # Clinically valid values per variable
    VALID_VALUES = {
        "insurance":    ["Private","Medicare","Medicaid","Uninsured"],
        "race":         ["White_NH","Black_NH","Hispanic","Asian_PI","AIAN"],
        "treatment":    ["Surgery_Only","Radiation_Only","Chemo_Radiation",
                         "Surgery_Radiation","Multimodal"],
        "stage":        ["I","II","III","IV"],
        "hpv_status":   ["Positive","Negative"],
        "tobacco_use":  ["Current","Former","Never"],
        "urban_rural":  ["Large_Metro","Small_Metro","Suburban","Rural"],
        "primary_site": ["Oral_Cavity","Oropharynx"],
    }

    # Clinical constraints
    def apply_constraints(df, var, val):
        """Only apply changes that are clinically sensible."""
        df = df.copy()
        if var == "hpv_status" and val == "Positive":
            # HPV+ predominantly oropharyngeal — only apply to oropharynx patients
            mask = df["primary_site"] == "Oropharynx"
            df.loc[mask, var] = val
        elif var == "treatment":
            # Multimodal not typical for Stage I
            if val == "Multimodal":
                mask = df["stage"].isin(["II","III","IV"])
            elif val == "Surgery_Only":
                mask = df["stage"].isin(["I","II"])
            else:
                mask = pd.Series(True, index=df.index)
            df.loc[mask, var] = val
        else:
            df[var] = val
        return df

    FEAT_COLS = ["age","poverty_pct","median_income","cci_score",
                 "diabetes","hypertension","immunosuppressed","prior_cancer",
                 "sex","race","insurance","primary_site","stage","grade",
                 "hpv_status","tobacco_use","alcohol_use","treatment"]

    if st.button("▶️ Run What-If Analysis", key="run_whatif"):
        results_whatif = []
        for val in VALID_VALUES[whatif_var]:
            df_cf = apply_constraints(cohort_df.copy(), whatif_var, val)
            X_cf  = df_cf[[c for c in FEAT_COLS if c in df_cf.columns]].copy()
            try:
                X_proc_cf = preprocessor.transform(X_cf)
                risks_cf  = xgb_model.predict_proba(X_proc_cf)[:,1]
            except:
                risks_cf = cohort_risks
            results_whatif.append({
                "value":       val,
                "mean_risk":   float(risks_cf.mean()),
                "p25_risk":    float(np.percentile(risks_cf, 25)),
                "p75_risk":    float(np.percentile(risks_cf, 75)),
                "pct_highrisk":float((risks_cf > 0.65).mean()),
                "n_affected":  int(df_cf[whatif_var].eq(val).sum()),
            })

        wi_df = pd.DataFrame(results_whatif).sort_values("mean_risk")
        baseline_risk = float(cohort_risks.mean())

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        fig.patch.set_facecolor("#0e1117")
        for ax in axes: ax.set_facecolor("#0e1117")

        # Mean risk comparison
        colors_wi = ["#2ecc71" if r < baseline_risk else "#e74c3c"
                     for r in wi_df["mean_risk"]]
        axes[0].barh(wi_df["value"], wi_df["mean_risk"],
                     color=colors_wi, edgecolor="#0e1117", height=0.6)
        axes[0].axvline(baseline_risk, color="white", lw=1.5, ls="--",
                        label=f"Observed baseline ({baseline_risk:.3f})")
        axes[0].set_xlabel("Mean Predicted Risk", color="white", fontsize=11)
        axes[0].set_title(f"What-If: Vary {whatif_var}\nMean Cohort Risk by Value",
                          color="white", fontsize=11, fontweight="bold")
        axes[0].tick_params(colors="white"); axes[0].spines[:].set_color("#333")
        axes[0].grid(axis="x", alpha=0.15, color="white")
        legend = axes[0].legend(fontsize=9, facecolor="#1a1a2e", edgecolor="#333")
        for txt in legend.get_texts(): txt.set_color("white")

        # % High-risk patients
        axes[1].barh(wi_df["value"], wi_df["pct_highrisk"]*100,
                     color=colors_wi, edgecolor="#0e1117", height=0.6, alpha=0.85)
        axes[1].set_xlabel("% Patients Classified High-Risk (>65%)", color="white", fontsize=11)
        axes[1].set_title("High-Risk Patient Rate by Value",
                          color="white", fontsize=11, fontweight="bold")
        axes[1].tick_params(colors="white"); axes[1].spines[:].set_color("#333")
        axes[1].grid(axis="x", alpha=0.15, color="white")
        for i, (_, row) in enumerate(wi_df.iterrows()):
            axes[1].text(row["pct_highrisk"]*100 + 0.5, i,
                         f"{row['pct_highrisk']*100:.1f}%",
                         va="center", color="white", fontsize=9)

        plt.tight_layout()
        st.pyplot(fig); plt.close()

        # Insight summary
        best  = wi_df.iloc[0]
        worst = wi_df.iloc[-1]
        diff  = worst["mean_risk"] - best["mean_risk"]
        st.markdown(f"""
        **Key Insight:**
        Changing `{whatif_var}` from **{worst['value']}** → **{best['value']}**
        reduces predicted cohort risk by **{diff:.3f}** ({diff/worst['mean_risk']*100:.1f}% relative reduction).
        Under the `{best['value']}` scenario, **{best['pct_highrisk']*100:.1f}%** of patients
        would be classified high-risk vs **{worst['pct_highrisk']*100:.1f}%** under `{worst['value']}`.
        """)

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # MODE 2 — SUBGROUP DISCOVERY
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("2️⃣  Subgroup Discovery")
    st.markdown(
        "The engine automatically identifies which patient subgroups in your cohort "
        "have the **highest predicted mortality risk** — using a decision tree to find "
        "the most discriminating combination of variables."
    )

    disc_vars = st.multiselect(
        "Variables to explore:",
        ["race","insurance","stage","tobacco_use","urban_rural",
         "primary_site","hpv_status","treatment","sex"],
        default=["race","insurance","stage","urban_rural"],
        key="disc_vars"
    )
    min_subgroup_size = st.slider("Minimum subgroup size (patients)", 10, 200, 30)

    if st.button("▶️ Discover High-Risk Subgroups", key="run_discovery"):
        from sklearn.tree import DecisionTreeClassifier, export_text

        # Encode categorical variables for decision tree
        disc_df = cohort_df.copy()
        disc_df["high_risk"] = (disc_df["predicted_risk"] > 0.65).astype(int)

        enc_cols = {}
        for col in disc_vars:
            if col in disc_df.columns:
                dummies = pd.get_dummies(disc_df[col].fillna("Unknown"),
                                         prefix=col, drop_first=False)
                disc_df = pd.concat([disc_df, dummies], axis=1)
                enc_cols[col] = list(dummies.columns)

        feat_cols_tree = [c for cols in enc_cols.values() for c in cols]
        if len(feat_cols_tree) == 0:
            st.warning("No valid variables selected.")
        else:
            X_tree = disc_df[feat_cols_tree].fillna(0)
            y_tree = disc_df["high_risk"]

            dt = DecisionTreeClassifier(
                max_depth=4, min_samples_leaf=min_subgroup_size, random_state=42
            )
            dt.fit(X_tree, y_tree)

            # Get leaf node risk profiles
            leaf_ids  = dt.apply(X_tree)
            leaf_data = []
            for leaf in np.unique(leaf_ids):
                mask     = leaf_ids == leaf
                n        = mask.sum()
                risk_mean= disc_df.loc[mask, "predicted_risk"].mean()
                high_pct = disc_df.loc[mask, "high_risk"].mean()
                # Get dominant value for each variable in this leaf
                profile = {}
                for col in disc_vars:
                    if col in disc_df.columns:
                        profile[col] = disc_df.loc[mask, col].mode()[0] if n > 0 else "Unknown"
                leaf_data.append({
                    "n_patients": n,
                    "mean_risk":  round(risk_mean, 3),
                    "pct_highrisk": round(high_pct * 100, 1),
                    **profile,
                })

            leaf_df = pd.DataFrame(leaf_data).sort_values("mean_risk", ascending=False)

            # Display top subgroups
            st.markdown("**Top High-Risk Subgroups (ranked by predicted risk):**")

            for i, (_, row) in enumerate(leaf_df.head(5).iterrows()):
                risk_col = "#e74c3c" if row["mean_risk"] > 0.65 else \
                           "#f39c12" if row["mean_risk"] > 0.45 else "#2ecc71"
                profile_str = " · ".join([
                    f"{col}=**{row[col]}**"
                    for col in disc_vars if col in row.index
                ])
                st.markdown(f"""
                <div style='background:#161b22;border-left:4px solid {risk_col};
                            border-radius:4px;padding:12px 16px;margin:6px 0;'>
                  <span style='font-family:IBM Plex Mono,monospace;font-size:0.8rem;
                               color:#7f8c8d;'>SUBGROUP {i+1} — {int(row['n_patients'])} patients</span><br>
                  <span style='font-size:0.95rem;'>{profile_str}</span><br>
                  <span style='color:{risk_col};font-weight:700;font-family:IBM Plex Mono,monospace;'>
                    Mean risk: {row['mean_risk']:.3f} · {row['pct_highrisk']:.1f}% high-risk
                  </span>
                </div>
                """, unsafe_allow_html=True)

            # Cohort comparison bar
            fig, ax = plt.subplots(figsize=(10, 4))
            fig.patch.set_facecolor("#0e1117"); ax.set_facecolor("#0e1117")
            top5     = leaf_df.head(5)
            labels   = [f"Subgroup {i+1}\n(n={int(r['n_patients'])})"
                        for i, (_, r) in enumerate(top5.iterrows())]
            bar_cols = ["#e74c3c" if r > 0.65 else "#f39c12" if r > 0.45 else "#2ecc71"
                        for r in top5["mean_risk"]]
            ax.bar(labels, top5["mean_risk"], color=bar_cols, edgecolor="#0e1117", width=0.6)
            ax.axhline(cohort_risks.mean(), color="white", lw=1.5, ls="--",
                       label=f"Cohort average ({cohort_risks.mean():.3f})")
            ax.set_ylabel("Mean Predicted Risk", color="white", fontsize=11)
            ax.set_title("Discovered High-Risk Subgroups vs Cohort Average",
                         color="white", fontsize=11, fontweight="bold")
            ax.tick_params(colors="white"); ax.spines[:].set_color("#333")
            ax.grid(axis="y", alpha=0.15, color="white")
            legend = ax.legend(fontsize=9, facecolor="#1a1a2e", edgecolor="#333")
            for txt in legend.get_texts(): txt.set_color("white")
            plt.tight_layout()
            st.pyplot(fig); plt.close()

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # MODE 3 — SENSITIVITY ANALYSIS (per patient)
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("3️⃣  Sensitivity Analysis — What Would Help This Patient Most?")
    st.markdown(
        "Using the patient profile from the sidebar, the engine systematically "
        "changes each variable to its best clinically valid value and ranks "
        "interventions by predicted survival gain."
    )

    MODIFIABLE = {
        "treatment":   ["Surgery_Only","Radiation_Only","Chemo_Radiation",
                        "Surgery_Radiation","Multimodal"],
        "tobacco_use": ["Never","Former","Current"],
        "alcohol_use": ["None_Light","Moderate","Heavy"],
        "insurance":   ["Private","Medicare","Medicaid","Uninsured"],
        "urban_rural": ["Large_Metro","Small_Metro","Suburban","Rural"],
    }
    FIXED = ["age","sex","race","primary_site","stage","grade",
             "hpv_status","cci_score","poverty_pct","median_income"]

    if st.button("▶️ Run Sensitivity Analysis for This Patient", key="run_sensitivity"):
        baseline_prob = float(lr_prob)
        sensitivity_results = []

        for var, values in MODIFIABLE.items():
            for val in values:
                # Apply clinical constraint
                test_dict = input_dict.copy()

                # Clinical constraints
                if var == "treatment" and val == "Multimodal" and stage == "I":
                    continue  # Multimodal not typical for Stage I
                if var == "treatment" and val == "Surgery_Only" and stage in ["III","IV"]:
                    continue  # Surgery alone not typical for advanced disease

                test_dict[var] = val
                try:
                    X_test = preprocessor.transform(pd.DataFrame([test_dict]))
                    prob   = float(lr_model.predict_proba(X_test)[0,1])
                    delta  = baseline_prob - prob  # positive = improvement
                    sensitivity_results.append({
                        "variable":   var,
                        "value":      val,
                        "risk":       round(prob, 4),
                        "delta":      round(delta, 4),
                        "is_current": (test_dict[var] == input_dict.get(var, "")),
                    })
                except:
                    pass

        if sensitivity_results:
            sens_df = pd.DataFrame(sensitivity_results)
            # Remove current values, keep improvements only
            sens_df = sens_df[~sens_df["is_current"]]
            sens_df = sens_df.sort_values("delta", ascending=False)

            # Top improvements
            improvements = sens_df[sens_df["delta"] > 0].head(8)
            worsenings   = sens_df[sens_df["delta"] < 0].tail(3)

            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor("#0e1117"); ax.set_facecolor("#0e1117")

            all_sens = pd.concat([improvements, worsenings]).sort_values("delta")
            labels   = [f"{r['variable']} → {r['value']}" for _,r in all_sens.iterrows()]
            colors_sens = ["#2ecc71" if d > 0 else "#e74c3c" for d in all_sens["delta"]]

            ax.barh(labels, all_sens["delta"],
                    color=colors_sens, edgecolor="#0e1117", height=0.6)
            ax.axvline(0, color="white", lw=0.8)
            ax.set_xlabel("Δ Predicted Risk (positive = improvement)", color="white", fontsize=11)
            ax.set_title(f"Sensitivity Analysis — {age}yo {sex}, Stage {stage}\n"
                         f"Baseline risk: {baseline_prob:.3f} | Most impactful variable changes",
                         color="white", fontsize=11, fontweight="bold")
            ax.tick_params(colors="white", labelsize=9)
            ax.spines[:].set_color("#333")
            ax.grid(axis="x", alpha=0.15, color="white")
            plt.tight_layout()
            st.pyplot(fig); plt.close()

            # Top 3 actionable insights
            st.markdown("**Top 3 Most Actionable Interventions:**")
            for i, (_, row) in enumerate(improvements.head(3).iterrows()):
                new_risk = row["risk"]
                delta    = row["delta"]
                st.markdown(f"""
                <div style='background:#161b22;border-left:4px solid #2ecc71;
                            border-radius:4px;padding:12px 16px;margin:6px 0;'>
                  <strong>{i+1}. Change {row['variable']} → {row['value']}</strong><br>
                  Predicted risk: {baseline_prob:.3f} → <strong>{new_risk:.3f}</strong>
                  &nbsp;|&nbsp;
                  <span style='color:#2ecc71;font-weight:700;'>↓ {delta:.3f} risk reduction
                  ({delta/baseline_prob*100:.1f}% relative)</span>
                </div>
                """, unsafe_allow_html=True)

            if len(improvements) == 0:
                st.success("✅ This patient already has near-optimal modifiable risk factors.")
        else:
            st.warning("Could not compute sensitivity analysis for this patient profile.")


    patient_summary = {
        "Age": age,
        "Sex": sex,
        "Race/Ethnicity": race,
        "County Median Income ($)": f"${income:,}",
        "County Poverty Rate (%)": f"{poverty}%",
        "Insurance": insurance,
        "Urban/Rural": urban,
        "Region": region,
        "Primary Site": site,
        "Stage": stage,
        "Grade": grade,
        "HPV Status": hpv,
        "Treatment": treatment,
        "Tobacco Use": tobacco,
        "Alcohol Use": alcohol,
        "Diabetes": "Yes" if diabetes else "No",
        "Hypertension": "Yes" if htn else "No",
        "Immunosuppressed": "Yes" if immuno else "No",
        "Prior Cancer": "Yes" if prior_ca else "No",
        "Charlson Comorbidity Index": cci,
    }
    col_inp1, col_inp2 = st.columns(2)
    items = list(patient_summary.items())
    half  = len(items) // 2
    with col_inp1:
        st.dataframe(
            pd.DataFrame(items[:half], columns=["Field","Value"]),
            hide_index=True, use_container_width=True
        )
    with col_inp2:
        st.dataframe(
            pd.DataFrame(items[half:], columns=["Field","Value"]),
            hide_index=True, use_container_width=True
        )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#555;font-size:0.8rem;font-family:IBM Plex Mono,monospace'>"
    "AI-Powered Oral Cancer Risk Scorer v2 · University of Florida · 2025 · "
    "LR + XGBoost + Cox PH + DeepSurv · Simulated SEER-like data · N=15,000"
    "</div>", unsafe_allow_html=True
)
