"""
=============================================================
 Step 2 – Imbalance Mitigation & Sampling Strategies
=============================================================
Techniques covered:
  A) SMOTE-NC      (primary – mixed categorical + numerical)
  B) Random OS / US
  C) SMOTE-Tomek   (hybrid)
  D) Class-weight baseline (no resampling)

SMOTE-NC column order requirement:
  Numerical columns MUST come BEFORE categorical columns.
  We reorder: [NetSpend, MCC, PaymentMode, ProductCode, CardPresent]
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib
from collections import Counter

from imblearn.over_sampling  import SMOTENC, RandomOverSampler, ADASYN
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, ClusterCentroids
from imblearn.combine        import SMOTETomek, SMOTEENN

import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# 0.  Load raw training data (original dtypes preserved)
# ─────────────────────────────────────────────────────────────
X_train = pd.read_parquet("/home/claude/artefacts/X_train_raw.parquet")
y_train = np.load("/home/claude/artefacts/y_train.npy")
le      = joblib.load("/home/claude/artefacts/label_encoder.pkl")

print(f"Training set: {X_train.shape[0]:,} rows | {Counter(y_train)}")
print(f"\nClass counts (before sampling):")
cc = pd.Series(Counter(y_train)).sort_index()
print(cc.to_string())


# ─────────────────────────────────────────────────────────────
# 1.  Prepare SMOTE-NC compatible matrix
#     Numerical FIRST, then categoricals (SMOTE-NC requirement)
#     categorical_features = indices of categorical columns
# ─────────────────────────────────────────────────────────────

# Encode categoricals as integers so SMOTE-NC can operate on them
from sklearn.preprocessing import OrdinalEncoder

cat_cols = ["MCC", "PaymentMode", "ProductCode", "CardPresent"]
num_cols = ["NetSpend"]

ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_cat_enc = ord_enc.fit_transform(X_train[cat_cols])

# Concatenate: numerical | categorical
X_smote_ready = np.hstack([
    X_train[num_cols].values,          # column 0 → NetSpend
    X_cat_enc                          # columns 1,2,3,4 → categorical
])

# Indices of categorical columns in the combined matrix
categorical_indices = list(range(len(num_cols), len(num_cols) + len(cat_cols)))
print(f"\nSMOTE-NC categorical indices: {categorical_indices}")


# ─────────────────────────────────────────────────────────────
# 2.  Sampling strategy function
#     sampling_strategy = dict {class_label: target_n_samples}
#
#     We use a TIERED approach:
#       - Dominant classes (top 40%)  → keep original counts
#       - Mid-tier classes            → raise to 50% of dominant max
#       - Tail classes                → raise to 30% of dominant max
#     This avoids over-inflating dominant class signal.
# ─────────────────────────────────────────────────────────────

def tiered_strategy(y, top_frac=0.40, mid_target=0.50, tail_target=0.30):
    """
    Build a sampling_strategy dict for minority classes.
    
    Parameters
    ----------
    y           : array of integer labels
    top_frac    : fraction of classes considered "dominant"
    mid_target  : fraction of dominant_max to assign to mid-tier
    tail_target : fraction of dominant_max to assign to tail classes
    
    Returns
    -------
    dict {class_label: n_samples}  — only classes that need upsampling
    """
    counts  = Counter(y)
    n_cls   = len(counts)
    sorted_cls = sorted(counts, key=counts.get, reverse=True)

    n_top   = max(1, int(n_cls * top_frac))
    dom_max = counts[sorted_cls[0]]

    mid_n  = int(dom_max * mid_target)
    tail_n = int(dom_max * tail_target)

    strategy = {}
    for rank, cls in enumerate(sorted_cls):
        if rank < n_top:
            # dominant — do not resample upward (keep as-is)
            continue
        elif rank < n_top + int(n_cls * 0.30):
            target = mid_n
        else:
            target = tail_n

        if counts[cls] < target:
            strategy[cls] = target

    return strategy


sampling_strat = tiered_strategy(y_train)
print("\nTiered sampling targets (minority classes only):")
for cls, tgt in list(sampling_strat.items())[:8]:
    print(f"  Class {le.classes_[cls]:10s}  "
          f"original: {Counter(y_train)[cls]:,}  →  target: {tgt:,}")


# ─────────────────────────────────────────────────────────────
# 3A.  SMOTE-NC  (PRIMARY RECOMMENDATION)
# ─────────────────────────────────────────────────────────────
smote_nc = SMOTENC(
    categorical_features=categorical_indices,
    sampling_strategy=sampling_strat,
    k_neighbors=5,
    random_state=42,
    n_jobs=-1,
)

print("\n[SMOTE-NC] Resampling …")
X_smote_nc, y_smote_nc = smote_nc.fit_resample(X_smote_ready, y_train)
print(f"  Before: {len(y_train):,}  →  After: {len(y_smote_nc):,}")
print(f"  New synthetic samples: {len(y_smote_nc) - len(y_train):,}")


# ─────────────────────────────────────────────────────────────
# 3B.  Random Oversampling (simple baseline to compare)
# ─────────────────────────────────────────────────────────────
ros = RandomOverSampler(sampling_strategy=sampling_strat, random_state=42)
X_ros, y_ros = ros.fit_resample(X_smote_ready, y_train)
print(f"\n[Random OS] Before: {len(y_train):,}  →  After: {len(y_ros):,}")


# ─────────────────────────────────────────────────────────────
# 3C.  SMOTE-Tomek  (hybrid: oversample minorities + clean border)
# ─────────────────────────────────────────────────────────────
smote_tomek = SMOTETomek(
    smotenc=SMOTENC(
        categorical_features=categorical_indices,
        sampling_strategy=sampling_strat,
        random_state=42
    ),
    random_state=42,
)
# NOTE: SMOTETomek with SMOTENC needs imblearn ≥ 0.11
# Fallback: run SMOTE-NC first, then TomekLinks separately
X_st_temp, y_st_temp = smote_nc.fit_resample(X_smote_ready, y_train)
tl = TomekLinks(n_jobs=-1)
X_smote_tomek, y_smote_tomek = tl.fit_resample(X_st_temp, y_st_temp)
print(f"\n[SMOTE-Tomek] Before: {len(y_train):,}  →  After: {len(y_smote_tomek):,}")
print(f"  Tomek links removed: {len(y_st_temp) - len(y_smote_tomek):,} borderline samples")


# ─────────────────────────────────────────────────────────────
# 4.  Visualise before / after distributions
# ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 10))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

datasets = {
    "Original (no sampling)":   (y_train,       "#888787"),
    "Random Oversampling":       (y_ros,         "#3266ad"),
    "SMOTE-NC (recommended)":    (y_smote_nc,    "#1d9e75"),
    "SMOTE-Tomek (hybrid)":      (y_smote_tomek, "#8b5cf6"),
}

axes_positions = [(0,0), (0,1), (0,2), (1,0)]

for (row, col), (title, (y_arr, color)) in zip(axes_positions, datasets.items()):
    ax = fig.add_subplot(gs[row, col])
    cnt = pd.Series(Counter(y_arr)).sort_index()
    ax.bar(range(len(cnt)), cnt.values, color=color, alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("IRC class index", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.set_xticks(range(len(cnt)))
    ax.set_xticklabels([f"{i}" for i in range(len(cnt))], fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    ir = cnt.max() / cnt.min()
    ax.set_title(f"{title}\n(IR = {ir:.1f}:1)", fontsize=9)

# IR comparison bar chart
ax_ir = fig.add_subplot(gs[1, 1:])
labels  = ["Original", "Random OS", "SMOTE-NC", "SMOTE-Tomek"]
ir_vals = []
for _, (y_arr, _) in datasets.items():
    c = Counter(y_arr)
    ir_vals.append(max(c.values()) / max(1, min(c.values())))

bar_colors = ["#888787", "#3266ad", "#1d9e75", "#8b5cf6"]
bars = ax_ir.bar(labels, ir_vals, color=bar_colors, edgecolor="white", linewidth=0.4)
ax_ir.set_title("Imbalance Ratio (IR) comparison\n(lower = more balanced)", fontsize=10)
ax_ir.set_ylabel("IR  (max / min class count)")
for bar, val in zip(bars, ir_vals):
    ax_ir.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
               f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax_ir.grid(axis="y", alpha=0.3)

plt.suptitle("Class distribution: before and after sampling strategies", fontsize=13, y=1.01)
plt.savefig("/home/claude/sampling_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[saved] sampling_comparison.png")


# ─────────────────────────────────────────────────────────────
# 5.  Validate synthetic samples – spot-check
#     Confirm SMOTE-NC interpolated NetSpend numerically
#     and kept MCC values as existing categories (not invented)
# ─────────────────────────────────────────────────────────────
n_real     = len(y_train)
syn_spend  = X_smote_nc[n_real:, 0]    # NetSpend for synthetic rows
real_spend = X_smote_nc[:n_real, 0]

print("\n[Validation] Net Spend – real vs synthetic:")
print(f"  Real  → min: {real_spend.min():.2f}  max: {real_spend.max():.2f}  "
      f"mean: {real_spend.mean():.2f}")
print(f"  Synth → min: {syn_spend.min():.2f}  max: {syn_spend.max():.2f}  "
      f"mean: {syn_spend.mean():.2f}")

# Check that synthetic MCC values (col 1) are in the original MCC vocabulary
real_mcc_vals  = set(X_smote_nc[:n_real, 1])
synth_mcc_vals = set(X_smote_nc[n_real:, 1])
unseen = synth_mcc_vals - real_mcc_vals
print(f"\n  MCC vocab – real: {len(real_mcc_vals)}  "
      f"synth new values: {len(unseen)} (should be 0)")


# ─────────────────────────────────────────────────────────────
# 6.  Save resampled data
# ─────────────────────────────────────────────────────────────
import os, joblib
os.makedirs("/home/claude/artefacts", exist_ok=True)

np.save("/home/claude/artefacts/X_smote_nc.npy",    X_smote_nc)
np.save("/home/claude/artefacts/y_smote_nc.npy",    y_smote_nc)
np.save("/home/claude/artefacts/X_smote_tomek.npy", X_smote_tomek)
np.save("/home/claude/artefacts/y_smote_tomek.npy", y_smote_tomek)
joblib.dump(ord_enc, "/home/claude/artefacts/ord_enc_smote.pkl")

print("\n✓ Sampling complete. Resampled artefacts saved.")
print("  → Run 03_model_training.py next")
