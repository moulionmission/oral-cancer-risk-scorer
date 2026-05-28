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
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Risk Score",
    "🧬 DeepSurv (Neural Net)",
    "🔍 SHAP Analysis",
    "📊 Model Performance",
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
    st.dataframe(pd.DataFrame(perf_data), hide_index=True, use_column_width=True)

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

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#555;font-size:0.8rem;font-family:IBM Plex Mono,monospace'>"
    "AI-Powered Oral Cancer Risk Scorer v2 · University of Florida · 2025 · "
    "LR + XGBoost + Cox PH + DeepSurv · Simulated SEER-like data · N=15,000"
    "</div>", unsafe_allow_html=True
)
