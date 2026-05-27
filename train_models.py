"""
train_models.py
Trains Logistic Regression and XGBoost classifiers, evaluates them,
and runs SHAP-based feature importance analysis.
"""

import numpy as np
import pandas as pd
import json, joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, classification_report,
                             RocCurveDisplay, ConfusionMatrixDisplay,
                             average_precision_score)
import xgboost as xgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
ART = Path("artifacts")
FIG = Path("figures"); FIG.mkdir(exist_ok=True)

# ── Load preprocessed data ────────────────────────────────────────────────────
X_train = np.load(ART / "X_train.npy")
X_test  = np.load(ART / "X_test.npy")
y_train = np.load(ART / "y_train.npy")
y_test  = np.load(ART / "y_test.npy")

with open(ART / "feature_names.json") as f:
    feat_names = json.load(f)

# ── 1. Logistic Regression ─────────────────────────────────────────────────────
print("=== Logistic Regression ===")
lr = LogisticRegression(max_iter=1000, C=0.5, class_weight="balanced",
                        random_state=SEED, solver="lbfgs")
lr.fit(X_train, y_train)
lr_proba = lr.predict_proba(X_test)[:, 1]
lr_auc   = roc_auc_score(y_test, lr_proba)
lr_ap    = average_precision_score(y_test, lr_proba)
print(f"  AUC-ROC: {lr_auc:.4f}  |  Avg Precision: {lr_ap:.4f}")
print(classification_report(y_test, (lr_proba > 0.5).astype(int), digits=3))
joblib.dump(lr, ART / "logistic_regression.pkl")

# ── 2. XGBoost ─────────────────────────────────────────────────────────────────
print("=== XGBoost ===")
scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
xgb_model = xgb.XGBClassifier(
    n_estimators=400, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=scale_pos, eval_metric="auc",
    use_label_encoder=False, random_state=SEED, verbosity=0,
)
xgb_model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)
xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
xgb_auc   = roc_auc_score(y_test, xgb_proba)
xgb_ap    = average_precision_score(y_test, xgb_proba)
print(f"  AUC-ROC: {xgb_auc:.4f}  |  Avg Precision: {xgb_ap:.4f}")
print(classification_report(y_test, (xgb_proba > 0.5).astype(int), digits=3))
joblib.dump(xgb_model, ART / "xgboost.pkl")

# ── 3. ROC comparison plot ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
RocCurveDisplay.from_predictions(y_test, lr_proba,  name=f"Logistic Reg (AUC={lr_auc:.3f})",  ax=ax)
RocCurveDisplay.from_predictions(y_test, xgb_proba, name=f"XGBoost (AUC={xgb_auc:.3f})", ax=ax)
ax.plot([0,1],[0,1],"k--", lw=0.8)
ax.set_title("ROC Curves — Oral Cancer 5-Year Mortality", fontsize=13)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
plt.tight_layout()
plt.savefig(FIG / "roc_comparison.png", dpi=150)
plt.close()
print("ROC plot saved.")

# ── 4. SHAP Analysis (XGBoost) ────────────────────────────────────────────────
print("Running SHAP…")
explainer   = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

# Summary bar plot
fig, ax = plt.subplots(figsize=(9, 6))
shap.summary_plot(shap_values, X_test, feature_names=feat_names,
                  plot_type="bar", show=False, max_display=18)
plt.title("SHAP Feature Importance — XGBoost (5-Yr Mortality)", fontsize=12)
plt.tight_layout()
plt.savefig(FIG / "shap_bar.png", dpi=150, bbox_inches="tight")
plt.close()

# Beeswarm plot
fig, ax = plt.subplots(figsize=(9, 7))
shap.summary_plot(shap_values, X_test, feature_names=feat_names,
                  show=False, max_display=18)
plt.title("SHAP Beeswarm — Sociodemographic Drivers of Mortality", fontsize=11)
plt.tight_layout()
plt.savefig(FIG / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()

# Save mean |SHAP| per feature as CSV
mean_shap = pd.DataFrame({
    "feature": feat_names,
    "mean_abs_shap": np.abs(shap_values).mean(axis=0)
}).sort_values("mean_abs_shap", ascending=False)
mean_shap.to_csv(ART / "shap_importance.csv", index=False)
print("SHAP plots saved.")
print(mean_shap.head(10).to_string(index=False))

# Persist SHAP values for dashboard
np.save(ART / "shap_values_test.npy", shap_values)

print("\n✅ All models trained and artifacts saved.")
metrics = {
    "lr_auc": round(lr_auc, 4), "lr_ap": round(lr_ap, 4),
    "xgb_auc": round(xgb_auc, 4), "xgb_ap": round(xgb_ap, 4),
}
with open(ART / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
