"""
=============================================================
 Step 4 – Evaluation, Per-Class Analysis & Threshold Tuning
=============================================================
Metrics:
  - Macro F1           (equal weight per IRC class)
  - Weighted F1        (business-impact weighted)
  - Per-class P / R / F1  (spot which tail classes fail)
  - Confusion matrix   (normalised)
  - Threshold tuning   (Platt scaling → per-class cutoffs)
  - Feature importance (from BalancedRandomForest)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib, warnings
from collections import Counter

from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.calibration    import CalibratedClassifierCV
from sklearn.preprocessing  import label_binarize

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# 0.  Load artefacts
# ─────────────────────────────────────────────────────────────
best_model  = joblib.load("/home/claude/artefacts/best_model.pkl")
ord_enc     = joblib.load("/home/claude/artefacts/ord_enc_smote.pkl")
le          = joblib.load("/home/claude/artefacts/label_encoder.pkl")

y_val   = np.load("/home/claude/artefacts/y_val.npy")
y_test  = np.load("/home/claude/artefacts/y_test.npy")

X_val_raw  = pd.read_parquet("/home/claude/artefacts/X_val_raw.parquet")
X_test_raw = pd.read_parquet("/home/claude/artefacts/X_test_raw.parquet")

def to_smote_matrix(df_raw):
    return np.hstack([
        df_raw["NetSpend"].values.reshape(-1, 1),
        ord_enc.transform(df_raw[["MCC", "PaymentMode", "ProductCode", "CardPresent"]]),
    ])

X_val_m  = to_smote_matrix(X_val_raw)
X_test_m = to_smote_matrix(X_test_raw)

class_names = le.classes_
n_classes   = len(class_names)

y_val_pred  = best_model.predict(X_val_m)
y_test_pred = best_model.predict(X_test_m)

print("="*60)
print("  VALIDATION SET RESULTS")
print("="*60)
print(f"  Macro F1:    {f1_score(y_val, y_val_pred, average='macro'):.4f}")
print(f"  Weighted F1: {f1_score(y_val, y_val_pred, average='weighted'):.4f}")
print(f"  Accuracy:    {(y_val == y_val_pred).mean():.4f}")
print("\n  WARNING: accuracy is shown for reference only — ")
print("  do NOT use it as the primary metric with imbalanced data.\n")


# ─────────────────────────────────────────────────────────────
# 1.  Per-class precision / recall / F1 report
# ─────────────────────────────────────────────────────────────
report = classification_report(
    y_val, y_val_pred,
    target_names=class_names,
    output_dict=True,
)
report_df = pd.DataFrame(report).T.drop(
    ["accuracy", "macro avg", "weighted avg"], errors="ignore"
).rename(columns={"f1-score": "f1"})

# Identify struggling classes (recall < 0.50)
struggling = report_df[report_df["recall"] < 0.50]
print(f"\nClasses with recall < 0.50 (need attention):")
print(struggling[["precision", "recall", "f1", "support"]].to_string())

print("\nFull per-class report:")
print(report_df[["precision", "recall", "f1", "support"]].to_string())


# ─────────────────────────────────────────────────────────────
# 2.  Visualise per-class metrics
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# 2a.  Precision / Recall / F1 bar chart
ax = axes[0, 0]
x   = np.arange(n_classes)
w   = 0.28
p   = report_df["precision"].values.astype(float)
r   = report_df["recall"].values.astype(float)
f   = report_df["f1"].values.astype(float)

ax.bar(x - w, p, w, label="Precision", color="#3266ad", alpha=0.85)
ax.bar(x,     r, w, label="Recall",    color="#1d9e75", alpha=0.85)
ax.bar(x + w, f, w, label="F1",        color="#8b5cf6", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
ax.set_ylim(0, 1.1)
ax.axhline(0.5, color="#e05a3a", linestyle="--", linewidth=1, alpha=0.6, label="Recall=0.5 threshold")
ax.set_title("Per-class Precision / Recall / F1 (validation set)")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)

# 2b.  Recall heatmap by class
ax2 = axes[0, 1]
recall_vals = r.reshape(1, -1)
im = ax2.imshow(recall_vals, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
ax2.set_xticks(range(n_classes))
ax2.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
ax2.set_yticks([])
ax2.set_title("Per-class recall heatmap (red = poor coverage)")
for j, val in enumerate(r):
    ax2.text(j, 0, f"{val:.2f}", ha="center", va="center", fontsize=8,
             color="white" if val < 0.4 else "black")
plt.colorbar(im, ax=ax2, orientation="horizontal", pad=0.35)

# 2c.  Support (class size) vs F1 scatter
ax3 = axes[1, 0]
support = report_df["support"].values.astype(float)
sc = ax3.scatter(support, f, c=f, cmap="RdYlGn", s=80, vmin=0, vmax=1, edgecolors="white")
for i, name in enumerate(class_names):
    ax3.annotate(name, (support[i], f[i]), fontsize=7, ha="left",
                 xytext=(4, 2), textcoords="offset points")
ax3.set_xlabel("Class support (val set size)")
ax3.set_ylabel("F1 score")
ax3.set_title("Class size vs F1 — low-support classes tend to underperform")
ax3.axhline(0.5, color="#e05a3a", linestyle="--", linewidth=1, alpha=0.6)
plt.colorbar(sc, ax=ax3, label="F1")
ax3.grid(alpha=0.3)

# 2d.  Normalised confusion matrix (top 10 most common classes for readability)
ax4 = axes[1, 1]
top10 = list(report_df["support"].astype(float).nlargest(10).index)
top10_idx = [list(class_names).index(c) for c in top10]

mask_val  = np.isin(y_val,      top10_idx)
mask_pred = np.isin(y_val_pred, top10_idx)
mask      = mask_val & mask_pred

if mask.sum() > 0:
    cm = confusion_matrix(
        y_val[mask], y_val_pred[mask],
        labels=top10_idx,
        normalize="true"
    )
    top10_names = [class_names[i] for i in top10_idx]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=top10_names)
    disp.plot(ax=ax4, colorbar=False, cmap="Blues", xticks_rotation=45)
    ax4.set_title("Confusion matrix (top 10 classes, row-normalised)")
else:
    ax4.text(0.5, 0.5, "Insufficient data for top-10 CM", transform=ax4.transAxes, ha="center")

plt.tight_layout()
plt.savefig("/home/claude/per_class_evaluation.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[saved] per_class_evaluation.png")


# ─────────────────────────────────────────────────────────────
# 3.  Feature importance (BalancedRandomForest)
# ─────────────────────────────────────────────────────────────
feature_names = ["NetSpend", "MCC", "PaymentMode", "ProductCode", "CardPresent"]
importances   = best_model.feature_importances_

fi_df = pd.DataFrame({
    "Feature":    feature_names,
    "Importance": importances
}).sort_values("Importance", ascending=True)

fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.barh(fi_df["Feature"], fi_df["Importance"],
               color=["#3266ad", "#1d9e75", "#8b5cf6", "#e8a020", "#c04020"],
               edgecolor="white", linewidth=0.4)
ax.set_title("Feature importance (BalancedRandomForest — mean decrease in impurity)")
ax.set_xlabel("Importance")
for bar, val in zip(bars, fi_df["Importance"]):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=10)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("/home/claude/feature_importance.png", dpi=150)
plt.close()
print("[saved] feature_importance.png")


# ─────────────────────────────────────────────────────────────
# 4.  Threshold tuning (per-class probability cutoffs)
#     Uses predict_proba → find per-class threshold that
#     maximises F1 for each class on the validation set.
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  Threshold Tuning (per-class optimal cutoff)")
print("="*60)

try:
    y_proba = best_model.predict_proba(X_val_m)   # shape (n_samples, n_classes)
except AttributeError:
    print("  Model does not support predict_proba — skipping threshold tuning")
    y_proba = None

if y_proba is not None:
    thresholds_range = np.linspace(0.05, 0.95, 37)
    optimal_thresholds = {}

    for cls_idx in range(n_classes):
        best_f1  = -1
        best_thr = 0.5

        for thr in thresholds_range:
            # Binary prediction for this class at this threshold
            y_binary_pred = (y_proba[:, cls_idx] >= thr).astype(int)
            y_binary_true = (y_val == cls_idx).astype(int)
            if y_binary_pred.sum() == 0:
                continue
            from sklearn.metrics import f1_score as f1s
            cls_f1 = f1s(y_binary_true, y_binary_pred, zero_division=0)
            if cls_f1 > best_f1:
                best_f1  = cls_f1
                best_thr = thr

        optimal_thresholds[cls_idx] = {"threshold": best_thr, "f1": best_f1}

    thr_df = pd.DataFrame(optimal_thresholds).T
    thr_df.index = class_names
    thr_df.columns = ["Optimal threshold", "Best F1"]

    print("\nOptimal per-class thresholds:")
    print(thr_df.sort_values("Optimal threshold").to_string())

    # Apply tuned thresholds → one-vs-rest multi-class prediction
    final_pred = np.full(len(y_val), -1, dtype=int)
    confidence  = np.zeros(len(y_val))

    for cls_idx, info in optimal_thresholds.items():
        thr = info["threshold"]
        mask = (y_proba[:, cls_idx] >= thr) & (y_proba[:, cls_idx] > confidence)
        final_pred[mask]  = cls_idx
        confidence[mask]  = y_proba[:, cls_idx][mask]

    # Any still unassigned → use argmax
    unassigned = (final_pred == -1)
    final_pred[unassigned] = np.argmax(y_proba[unassigned], axis=1)

    print(f"\n  After threshold tuning:")
    print(f"  Macro F1:    {f1_score(y_val, final_pred, average='macro'):.4f}")
    print(f"  Weighted F1: {f1_score(y_val, final_pred, average='weighted'):.4f}")

    # Visualise threshold curve for top 5 tail classes
    tail_classes = list(
        pd.Series(Counter(y_val)).sort_values().head(5).index
    )

    fig, axes = plt.subplots(1, len(tail_classes), figsize=(16, 4), sharey=True)
    for i, cls_idx in enumerate(tail_classes):
        ax = axes[i]
        f1s_per_thr = []
        y_bin = (y_val == cls_idx).astype(int)
        for thr in thresholds_range:
            pred_bin = (y_proba[:, cls_idx] >= thr).astype(int)
            from sklearn.metrics import f1_score as f1s_fn
            f1s_per_thr.append(f1s_fn(y_bin, pred_bin, zero_division=0))

        ax.plot(thresholds_range, f1s_per_thr, color="#3266ad", linewidth=2)
        opt = optimal_thresholds[cls_idx]["threshold"]
        ax.axvline(opt, color="#e05a3a", linestyle="--", linewidth=1.5,
                   label=f"opt={opt:.2f}")
        ax.set_title(class_names[cls_idx], fontsize=10)
        ax.set_xlabel("Threshold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("F1 score")
    plt.suptitle("F1 vs decision threshold for 5 tail IRC classes", y=1.02)
    plt.tight_layout()
    plt.savefig("/home/claude/threshold_tuning.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n[saved] threshold_tuning.png")


# ─────────────────────────────────────────────────────────────
# 5.  FINAL TEST SET EVALUATION
#     Run only once — after all tuning decisions are locked.
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  FINAL TEST SET EVALUATION (run once)")
print("="*60)

y_test_pred  = best_model.predict(X_test_m)

print(classification_report(
    y_test, y_test_pred,
    target_names=class_names,
    digits=3,
))

test_macro_f1  = f1_score(y_test, y_test_pred, average="macro")
test_wted_f1   = f1_score(y_test, y_test_pred, average="weighted")
print(f"  Test Macro F1:    {test_macro_f1:.4f}")
print(f"  Test Weighted F1: {test_wted_f1:.4f}")


# ─────────────────────────────────────────────────────────────
# 6.  Summary card
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.axis("off")

summary_data = [
    ["Metric",             "Value",          "Notes"],
    ["Macro F1 (test)",    f"{test_macro_f1:.4f}",  "Equal weight per IRC class"],
    ["Weighted F1 (test)", f"{test_wted_f1:.4f}",   "Weighted by class support"],
    ["Classes struggling", str(len(struggling)), "recall < 0.50 on val set"],
    ["Sampling method",   "SMOTE-NC",              "Mixed cat + numerical"],
    ["Classifier",        "BalancedRandomForest",  "n_estimators=300"],
    ["CV strategy",       "StratifiedKFold k=5",   "No leakage"],
    ["Primary metric",    "Macro F1",              "NOT accuracy"],
]

table = ax.table(
    cellText=summary_data[1:],
    colLabels=summary_data[0],
    loc="center",
    cellLoc="left",
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.8)

for j in range(3):
    table[0, j].set_facecolor("#3266ad")
    table[0, j].set_text_props(color="white", fontweight="bold")

for i in range(1, len(summary_data)):
    for j in range(3):
        table[i, j].set_facecolor("#f0f4fb" if i % 2 == 0 else "white")

ax.set_title("Evaluation Summary — IRC Multi-Class Classification",
             fontsize=13, fontweight="bold", pad=20)

plt.tight_layout()
plt.savefig("/home/claude/evaluation_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[saved] evaluation_summary.png")
print("\n✓ Full evaluation complete!")
