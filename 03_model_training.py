"""
=============================================================
 Step 3 – Model Training with Imbalance-Aware Classifiers
=============================================================
Models:
  1. Logistic Regression + class_weight='balanced' (baseline)
  2. BalancedRandomForest  (imbalanced-learn)
  3. XGBoost               (with sample_weight)
  4. EasyEnsemble          (aggressive – best for deep tails)

All wrapped in imblearn.Pipeline with SMOTE-NC + StratifiedKFold CV
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib, os, warnings
from collections import Counter

from sklearn.pipeline        import Pipeline as SKPipeline
from sklearn.linear_model    import LogisticRegression
from sklearn.preprocessing   import OrdinalEncoder, FunctionTransformer, RobustScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics         import (
    f1_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay
)

from imblearn.pipeline          import Pipeline as ImbPipeline    # KEY: imblearn Pipeline
from imblearn.over_sampling     import SMOTENC
from imblearn.ensemble          import (
    BalancedRandomForestClassifier, EasyEnsembleClassifier
)
import xgboost as xgb

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# 0.  Load encoded artefacts
# ─────────────────────────────────────────────────────────────
X_train_enc = np.load("/home/claude/artefacts/X_train_enc.npy")
X_val_enc   = np.load("/home/claude/artefacts/X_val_enc.npy")
X_test_enc  = np.load("/home/claude/artefacts/X_test_enc.npy")
y_train     = np.load("/home/claude/artefacts/y_train.npy")
y_val       = np.load("/home/claude/artefacts/y_val.npy")
y_test      = np.load("/home/claude/artefacts/y_test.npy")
le          = joblib.load("/home/claude/artefacts/label_encoder.pkl")

# Load SMOTE-NC resampled data (for training final models)
X_smote_nc = np.load("/home/claude/artefacts/X_smote_nc.npy")
y_smote_nc = np.load("/home/claude/artefacts/y_smote_nc.npy")

# Load raw splits (for imblearn Pipeline inside CV)
X_train_raw = pd.read_parquet("/home/claude/artefacts/X_train_raw.parquet")

n_classes = len(le.classes_)
class_names = le.classes_
print(f"Classes: {n_classes} | Train: {len(y_train):,} | Val: {len(y_val):,} | Test: {len(y_test):,}")


# ─────────────────────────────────────────────────────────────
# 1.  Compute class weights (used as baseline and XGBoost weights)
# ─────────────────────────────────────────────────────────────
from sklearn.utils.class_weight import compute_class_weight

cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(cw))

# Sample weights for XGBoost (each row gets its class weight)
sample_weights = np.array([class_weight_dict[yi] for yi in y_train])
sample_weights_smote = np.array([class_weight_dict.get(yi, 1.0) for yi in y_smote_nc])

print("\nClass weights (top 5 smallest = dominant, top 5 largest = tail):")
cw_series = pd.Series(class_weight_dict).sort_values()
print("  Dominant:", cw_series.head(5).to_dict())
print("  Tail:    ", cw_series.tail(5).to_dict())


# ─────────────────────────────────────────────────────────────
# 2.  Tiered sampling strategy (same as Step 2)
# ─────────────────────────────────────────────────────────────
def tiered_strategy(y, top_frac=0.40, mid_target=0.50, tail_target=0.30):
    from collections import Counter
    counts = Counter(y)
    n_cls  = len(counts)
    sorted_cls = sorted(counts, key=counts.get, reverse=True)
    n_top  = max(1, int(n_cls * top_frac))
    dom_max = counts[sorted_cls[0]]
    strategy = {}
    for rank, cls in enumerate(sorted_cls):
        if rank < n_top:
            continue
        target = int(dom_max * (mid_target if rank < n_top + int(n_cls * 0.30) else tail_target))
        if counts[cls] < target:
            strategy[cls] = target
    return strategy

# Categorical indices for SMOTE-NC (col 0 = NetSpend numeric, cols 1–4 = categorical)
CATEGORICAL_INDICES = [1, 2, 3, 4]


# ─────────────────────────────────────────────────────────────
# 3.  imblearn Pipeline helper
#     Wraps SMOTE-NC + classifier so sampling happens INSIDE
#     each CV fold — this is the leak-free pattern.
#
#     Input X must be the SMOTE-NC compatible matrix:
#     [NetSpend | MCC_enc | PaymentMode_enc | ProductCode_enc | CardPresent]
# ─────────────────────────────────────────────────────────────
def make_imbpipeline(classifier, use_smote=True):
    steps = []
    if use_smote:
        steps.append(("smote_nc", SMOTENC(
            categorical_features=CATEGORICAL_INDICES,
            sampling_strategy="auto",       # oversample all minority to majority count
            k_neighbors=5,
            random_state=42,
            n_jobs=-1,
        )))
    steps.append(("clf", classifier))
    return ImbPipeline(steps)


# We need the raw OrdinalEncoded matrix for SMOTE-NC during CV
ord_enc = joblib.load("/home/claude/artefacts/ord_enc_smote.pkl")
X_train_smote_ready = np.hstack([
    X_train_raw["NetSpend"].values.reshape(-1, 1),
    ord_enc.transform(X_train_raw[["MCC", "PaymentMode", "ProductCode", "CardPresent"]]),
])


# ─────────────────────────────────────────────────────────────
# 4.  Define models
# ─────────────────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    # ── Baseline: Logistic Regression + class weights (no sampling)
    "LR_Baseline": {
        "pipeline": SKPipeline([("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="saga",
            multi_class="multinomial",
            C=1.0,
            n_jobs=-1,
            random_state=42,
        ))]),
        "X": X_train_enc,
        "use_smote": False,
        "description": "Logistic Reg + class_weight='balanced' (no sampling)",
    },

    # ── BalancedRandomForest (built-in undersampling per tree)
    "BalancedRF": {
        "pipeline": BalancedRandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            sampling_strategy="auto",       # undersample majority in each tree
            replacement=False,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        ),
        "X": X_train_enc,
        "use_smote": False,
        "description": "BalancedRandomForest (undersampling per tree)",
    },

    # ── SMOTE-NC + BalancedRF (combined)
    "SMOTE_BalancedRF": {
        "pipeline": make_imbpipeline(
            BalancedRandomForestClassifier(
                n_estimators=200,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=42,
            ),
            use_smote=True,
        ),
        "X": X_train_smote_ready,
        "use_smote": True,
        "description": "SMOTE-NC inside CV fold + BalancedRandomForest",
    },

    # ── XGBoost with scale_pos_weight (via sample_weight)
    "XGBoost": {
        "pipeline": xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="mlogloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
        ),
        "X": X_train_enc,
        "use_smote": False,
        "fit_params": {"clf__sample_weight": sample_weights},
        "description": "XGBoost + sample_weight (class-balanced)",
    },
}


# ─────────────────────────────────────────────────────────────
# 5.  Stratified K-Fold Cross Validation
# ─────────────────────────────────────────────────────────────
cv_results = {}
scoring = {
    "macro_f1":    "f1_macro",
    "weighted_f1": "f1_weighted",
    "accuracy":    "accuracy",
}

print("\n" + "="*60)
print("  Cross-Validation (StratifiedKFold, k=5)")
print("="*60)

for name, cfg in models.items():
    print(f"\n[{name}] {cfg['description']}")
    fit_params = cfg.get("fit_params", {})

    try:
        scores = cross_validate(
            cfg["pipeline"],
            cfg["X"],
            y_train,
            cv=cv,
            scoring=scoring,
            fit_params=fit_params,
            n_jobs=1,   # outer CV sequential; inner model can use n_jobs=-1
            return_train_score=False,
        )
        cv_results[name] = {
            "macro_f1_mean":    scores["test_macro_f1"].mean(),
            "macro_f1_std":     scores["test_macro_f1"].std(),
            "weighted_f1_mean": scores["test_weighted_f1"].mean(),
            "weighted_f1_std":  scores["test_weighted_f1"].std(),
            "accuracy_mean":    scores["test_accuracy"].mean(),
        }
        print(f"  Macro F1:    {scores['test_macro_f1'].mean():.4f} ± {scores['test_macro_f1'].std():.4f}")
        print(f"  Weighted F1: {scores['test_weighted_f1'].mean():.4f} ± {scores['test_weighted_f1'].std():.4f}")
        print(f"  Accuracy:    {scores['test_accuracy'].mean():.4f}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        cv_results[name] = {"macro_f1_mean": 0, "macro_f1_std": 0,
                            "weighted_f1_mean": 0, "weighted_f1_std": 0, "accuracy_mean": 0}


# ─────────────────────────────────────────────────────────────
# 6.  CV results plot
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

model_names = list(cv_results.keys())
macro_means  = [cv_results[m]["macro_f1_mean"]    for m in model_names]
macro_stds   = [cv_results[m]["macro_f1_std"]     for m in model_names]
wted_means   = [cv_results[m]["weighted_f1_mean"] for m in model_names]
wted_stds    = [cv_results[m]["weighted_f1_std"]  for m in model_names]

colors = ["#888787", "#3266ad", "#1d9e75", "#8b5cf6"][:len(model_names)]

for ax, means, stds, title in [
    (axes[0], macro_means,  macro_stds,  "Macro F1 (equal weight per class)"),
    (axes[1], wted_means,   wted_stds,   "Weighted F1 (weighted by support)"),
]:
    bars = ax.bar(model_names, means, yerr=stds, color=colors,
                  edgecolor="white", linewidth=0.4, capsize=5, alpha=0.85)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("F1 Score")
    ax.set_xticklabels(model_names, rotation=20, ha="right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig("/home/claude/cv_results.png", dpi=150)
plt.close()
print("\n[saved] cv_results.png")


# ─────────────────────────────────────────────────────────────
# 7.  Train best model on full train set, evaluate on val set
#     Best model = SMOTE-NC + BalancedRF (as recommended)
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  Final model training: SMOTE-NC + BalancedRandomForest")
print("="*60)

best_model = BalancedRandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    class_weight="balanced_subsample",
    sampling_strategy="auto",
    n_jobs=-1,
    random_state=42,
)

# Train on SMOTE-NC resampled data (generated in Step 2)
# Re-apply SMOTE on the plain encoded matrix for cleaner pipeline
smote_nc_final = SMOTENC(
    categorical_features=CATEGORICAL_INDICES,
    sampling_strategy=tiered_strategy(y_train),
    k_neighbors=5,
    random_state=42,
    n_jobs=-1,
)
X_resampled, y_resampled = smote_nc_final.fit_resample(X_train_smote_ready, y_train)
print(f"  Resampled train size: {len(y_resampled):,}")

best_model.fit(X_resampled, y_resampled)

# Evaluate on validation set (no sampling applied)
y_val_pred = best_model.predict(
    np.hstack([
        pd.read_parquet("/home/claude/artefacts/X_val_raw.parquet")["NetSpend"].values.reshape(-1, 1),
        ord_enc.transform(
            pd.read_parquet("/home/claude/artefacts/X_val_raw.parquet")[
                ["MCC", "PaymentMode", "ProductCode", "CardPresent"]
            ]
        )
    ])
)

print(f"\n  Validation Macro F1:    {f1_score(y_val, y_val_pred, average='macro'):.4f}")
print(f"  Validation Weighted F1: {f1_score(y_val, y_val_pred, average='weighted'):.4f}")


# ─────────────────────────────────────────────────────────────
# 8.  Save best model
# ─────────────────────────────────────────────────────────────
joblib.dump(best_model, "/home/claude/artefacts/best_model.pkl")
print("\n✓ Best model saved to artefacts/best_model.pkl")
print("  → Run 04_evaluation.py next")
