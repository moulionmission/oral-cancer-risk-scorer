# AI-Powered Oral Cancer Risk Scorer
**Independent Research Project — Gainesville, FL | 2025**

> Aligned with Dr. Karanth's NIH/NIDCR-funded grant on AI-derived multilevel risk scores for oral cavity and oropharyngeal cancers.

---

## Project Overview

This pipeline builds an end-to-end ML system for predicting 5-year cancer-specific mortality and analyzing sociodemographic drivers of risk using SEER-like data.

### Research Questions
1. Which clinical and sociodemographic features are the strongest predictors of 5-year mortality in oral cavity/oropharyngeal cancer?
2. Do insurance status, income, and race show independent prognostic value after adjusting for stage and treatment?
3. Can a Cox PH model accurately stratify survival outcomes across patient subgroups?

---

## Repository Structure

```
oral_cancer/
├── simulate_data.py        # SEER-like dataset generation (10,500 records)
├── preprocess.py           # Preprocessing pipeline (imputation, encoding, scaling)
├── train_models.py         # LR + XGBoost training + SHAP analysis
├── cox_model.py            # Cox Proportional Hazards survival model
├── app.py                  # Streamlit dashboard
├── requirements.txt
├── data/
│   └── seer_oral_cancer_simulated.csv
├── artifacts/
│   ├── preprocessor.pkl
│   ├── logistic_regression.pkl
│   ├── xgboost.pkl
│   ├── cox_model.pkl
│   ├── shap_importance.csv
│   ├── cox_coefficients.csv
│   └── metrics.json
└── figures/
    ├── roc_comparison.png
    ├── shap_bar.png
    ├── shap_beeswarm.png
    └── cox_survival_curves.png
```

---

## Dataset

Simulated to mirror SEER Research Data File structure for:
- **Cancer sites**: Oral Cavity (C00–C06, C14) and Oropharynx (C09–C10)
- **N = 10,500** records with ~5% structured missingness
- **Features**: Age, sex, race/ethnicity, insurance, income, poverty rate, stage, grade, HPV status, tobacco/alcohol use, comorbidities (CCI), treatment modality

### Key Variables
| Variable | Type | Notes |
|---|---|---|
| `age` | Numeric | 18–95 |
| `race` | Categorical | White, Black, Hispanic, Asian/PI, AI/AN |
| `stage` | Ordinal | I, II, III, IV |
| `hpv_status` | Categorical | Strong protective factor for oropharynx |
| `insurance` | Categorical | SES proxy; uninsured = higher risk |
| `poverty_pct` | Numeric | County-level; SEER-linked |
| `cci_score` | Numeric | Charlson Comorbidity Index (0–8) |
| `mortality_5yr` | Binary | Outcome for classification models |
| `survival_months` + `vital_status` | Survival pair | Outcome for Cox model |

---

## Models

### 1. Logistic Regression
- Regularization: L2, C=0.5
- Class-weighted to handle imbalance
- Interpretable coefficient-based baseline

### 2. XGBoost
- 400 estimators, depth=5, LR=0.05
- `scale_pos_weight` for class imbalance
- Best performing classification model

### 3. Cox Proportional Hazards (lifelines)
- Ridge penalization (penalizer=0.10)
- Breslow baseline hazard estimator
- Subgroup survival curves: Stage I–IV, Insurance status

---

## SHAP Analysis

SHAP (SHapley Additive exPlanations) applied to XGBoost to identify sociodemographic drivers:

**Top features by mean |SHAP|:**
1. `stage` — dominant clinical predictor
2. `tobacco_use` — strongest behavioral predictor
3. `age` — continuous risk gradient
4. `median_income` / `poverty_pct` — SES gradient confirmed
5. `alcohol_use`
6. `cci_score` — comorbidity burden
7. `hpv_status` — protective for oropharynx
8. `treatment`

---

## Streamlit Dashboard

Interactive dashboard enabling patient-level risk score inference:

```bash
cd oral_cancer
streamlit run app.py
```

**Features:**
- Sidebar clinical input form (18 variables)
- Ensemble risk score (LR + XGBoost)
- Patient-level SHAP waterfall explanation
- Population-level SHAP bar + beeswarm plots
- Cox survival curves by stage and insurance
- Model performance metrics

---

## How to Run (Full Pipeline)

```bash
pip install -r requirements.txt

python simulate_data.py    # Generate data
python preprocess.py       # Build preprocessing pipeline
python train_models.py     # Train LR + XGBoost + SHAP
python cox_model.py        # Cox PH survival model
streamlit run app.py       # Launch dashboard
```

---

## Replacing with Real SEER Data

To use actual SEER data:
1. Request access at https://seer.cancer.gov/data/access.html
2. Filter for ICD-O-3 site codes C00–C06, C09–C10, C14
3. Export with variables matching `data/seer_oral_cancer_simulated.csv` column names
4. Drop `simulate_data.py`; all downstream scripts are data-agnostic

---

## Limitations

- Simulated data: coefficients and AUC values reflect the DGP, not real SEER patterns
- Cox C-index ~0.56: moderate discrimination; real SEER data expected to perform better
- No external validation cohort
- HPV testing not universally available in SEER historical records

---

## References

- Karanth S et al. (2023). AI-driven multilevel risk score framework for HNC. *NIDCR R01*.
- SEER Program. National Cancer Institute. https://seer.cancer.gov
- Lundberg SM & Lee S-I (2017). A unified approach to interpreting model predictions. *NeurIPS*.
- Davidson-Pilon C (2019). lifelines: survival analysis in Python. *JOSS*.
