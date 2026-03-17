"""
=============================================================
 Step 1 – Feature Preprocessing Pipeline
 Multi-label Interchange Rate Code (IRC) Classification
=============================================================
Features
  - MCC                (high-cardinality categorical)
  - Mode of Payment    (low-cardinality categorical)
  - Product Code       (categorical)
  - CP / CNP Indicator (boolean)
  - Net Spend          (skewed numerical, range 1 → 1M)
Target : Interchange Rate Code (multi-class, imbalanced)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OrdinalEncoder, OneHotEncoder, RobustScaler, LabelEncoder
)
from sklearn.model_selection import train_test_split
from category_encoders import TargetEncoder          # pip install category-encoders
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# 0.  Synthetic dataset (replace with your real data load)
# ─────────────────────────────────────────────────────────────
np.random.seed(42)
N = 50_000

MCC_CODES = [f"MCC_{i:04d}" for i in range(600)]           # 600 unique MCCs
PAYMENT_MODES = ["POS", "Wallet", "ApplePay", "GPay", "Contactless", "CNP_Online"]
PRODUCT_CODES = ["Debit_Classic", "Credit_Gold", "Credit_Platinum", "Prepaid", "Business_Credit"]
IRC_CLASSES = [f"IRC_{i:02d}" for i in range(1, 21)]       # 20 interchange rate codes

# Imbalanced IRC distribution (40% of codes → 90%+ of volume)
irc_probs = np.array(
    [0.22, 0.18, 0.14, 0.10, 0.08, 0.06, 0.04, 0.03, 0.02, 0.02,
     0.015, 0.012, 0.009, 0.008, 0.006, 0.005, 0.004, 0.003, 0.002, 0.002]
)
irc_probs /= irc_probs.sum()

df = pd.DataFrame({
    "MCC":           np.random.choice(MCC_CODES,    size=N),
    "PaymentMode":   np.random.choice(PAYMENT_MODES, size=N),
    "ProductCode":   np.random.choice(PRODUCT_CODES, size=N),
    "CardPresent":   np.random.randint(0, 2, size=N),
    # Heavy right-skewed spend: mix of log-normal peaks
    "NetSpend":      np.clip(
        np.random.lognormal(mean=4.5, sigma=2.0, size=N), 1, 1_000_000
    ),
    "IRC":           np.random.choice(IRC_CLASSES, size=N, p=irc_probs),
})

print("Dataset shape:", df.shape)
print("\nClass distribution (top 10):")
print(df["IRC"].value_counts().head(10))


# ─────────────────────────────────────────────────────────────
# 1.  CRITICAL – Train / Val / Test split BEFORE any sampling
# ─────────────────────────────────────────────────────────────
X = df.drop(columns="IRC")
y = df["IRC"]

# Encode target labels to integers
le = LabelEncoder()
y_enc = le.fit_transform(y)

X_temp, X_test,  y_temp, y_test  = train_test_split(
    X, y_enc, test_size=0.15, stratify=y_enc, random_state=42
)
X_train, X_val,  y_train, y_val  = train_test_split(
    X_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=42  # 0.176 ≈ 15% of total
)

print(f"\nSplit sizes → train: {len(X_train):,}  val: {len(X_val):,}  test: {len(X_test):,}")


# ─────────────────────────────────────────────────────────────
# 2.  Net Spend – visualise skew before / after log1p
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

axes[0].hist(X_train["NetSpend"], bins=80, color="#3266ad", edgecolor="white", linewidth=0.3)
axes[0].set_title("Net Spend – raw (extreme right skew)")
axes[0].set_xlabel("Spend (£)")

log_spend = np.log1p(X_train["NetSpend"])
axes[1].hist(log_spend, bins=80, color="#1d9e75", edgecolor="white", linewidth=0.3)
axes[1].set_title("Net Spend – after log1p")
axes[1].set_xlabel("log1p(Spend)")

from sklearn.preprocessing import RobustScaler as RS
rs_tmp = RS()
scaled = rs_tmp.fit_transform(log_spend.values.reshape(-1, 1)).ravel()
axes[2].hist(scaled, bins=80, color="#8b5cf6", edgecolor="white", linewidth=0.3)
axes[2].set_title("Net Spend – log1p → RobustScaler")
axes[2].set_xlabel("Scaled value")

plt.tight_layout()
plt.savefig("/home/claude/spend_distribution.png", dpi=150)
plt.close()
print("\n[saved] spend_distribution.png")


# ─────────────────────────────────────────────────────────────
# 3.  Class imbalance visualisation
# ─────────────────────────────────────────────────────────────
class_counts = pd.Series(y_train).map(dict(enumerate(le.classes_))).value_counts()
colors = ["#3266ad" if i < 8 else "#91b3e0" if i < 14 else "#c8dcf2"
          for i in range(len(class_counts))]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

class_counts.plot(kind="bar", ax=ax1, color=colors, edgecolor="white", linewidth=0.4)
ax1.set_title("IRC class distribution (training set)")
ax1.set_xlabel("Interchange Rate Code")
ax1.set_ylabel("Sample count")
ax1.tick_params(axis="x", rotation=45)

cum = class_counts.cumsum() / class_counts.sum() * 100
ax2.plot(range(len(cum)), cum.values, marker="o", markersize=5,
         color="#3266ad", linewidth=2)
ax2.axhline(90, color="#e05a3a", linestyle="--", linewidth=1.5, label="90% threshold")
ax2.fill_between(range(len(cum)), cum.values, alpha=0.15, color="#3266ad")
ax2.set_title("Cumulative transaction share by IRC")
ax2.set_xlabel("IRC rank (most → least frequent)")
ax2.set_ylabel("Cumulative % of transactions")
ax2.set_xticks(range(len(cum)))
ax2.set_xticklabels(cum.index, rotation=45, ha="right", fontsize=8)
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("/home/claude/class_imbalance.png", dpi=150)
plt.close()
print("[saved] class_imbalance.png")


# ─────────────────────────────────────────────────────────────
# 4.  Preprocessing pipeline
#     MCC        → TargetEncoder    (high-cardinality, 600 codes)
#     PaymentMode→ OrdinalEncoder   (6 categories, ordinal)
#     ProductCode→ OrdinalEncoder   (5 categories)
#     CardPresent→ pass-through (already 0/1 integer)
#     NetSpend   → log1p + RobustScaler
# ─────────────────────────────────────────────────────────────
from sklearn.preprocessing import FunctionTransformer

log1p_transformer = FunctionTransformer(np.log1p, validate=True)


def build_preprocessor():
    """Return a ColumnTransformer for all 5 features."""

    # MCC: target encoding (smoothed mean of target per MCC)
    mcc_enc = TargetEncoder(cols=["MCC"], smoothing=10, handle_missing="value")

    # Low-cardinality categoricals
    cat_enc = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1
    )

    # Numerical: log1p → RobustScaler
    num_pipe = Pipeline([
        ("log1p",  log1p_transformer),
        ("scaler", RobustScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("mcc",     mcc_enc,  ["MCC"]),
            ("cat",     cat_enc,  ["PaymentMode", "ProductCode"]),
            ("bool",    "passthrough", ["CardPresent"]),
            ("num",     num_pipe, ["NetSpend"]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


preprocessor = build_preprocessor()

# Fit on train, transform train + val + test
# NOTE: TargetEncoder needs y during fit → pass y_train here
X_train_enc = preprocessor.fit_transform(X_train, y_train)
X_val_enc   = preprocessor.transform(X_val)
X_test_enc  = preprocessor.transform(X_test)

print(f"\nEncoded feature matrix shape: {X_train_enc.shape}")
print("Feature columns:", preprocessor.get_feature_names_out().tolist())

# ─────────────────────────────────────────────────────────────
# 5.  Save artefacts for next step
# ─────────────────────────────────────────────────────────────
import joblib, os
os.makedirs("/home/claude/artefacts", exist_ok=True)

joblib.dump(preprocessor, "/home/claude/artefacts/preprocessor.pkl")
joblib.dump(le,           "/home/claude/artefacts/label_encoder.pkl")

np.save("/home/claude/artefacts/X_train_enc.npy", X_train_enc)
np.save("/home/claude/artefacts/X_val_enc.npy",   X_val_enc)
np.save("/home/claude/artefacts/X_test_enc.npy",  X_test_enc)
np.save("/home/claude/artefacts/y_train.npy",     y_train)
np.save("/home/claude/artefacts/y_val.npy",       y_val)
np.save("/home/claude/artefacts/y_test.npy",      y_test)

# Also save raw splits (needed by SMOTE-NC which operates on original dtypes)
X_train.to_parquet("/home/claude/artefacts/X_train_raw.parquet")
X_val.to_parquet("/home/claude/artefacts/X_val_raw.parquet")
X_test.to_parquet("/home/claude/artefacts/X_test_raw.parquet")

print("\n✓ Preprocessing complete. Artefacts saved to /home/claude/artefacts/")
print("  → Run 02_sampling.py next")
