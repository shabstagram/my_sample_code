"""
============================================================
  Multi-Label Imbalance Handling Pipeline
  Target: Interchange Rate Code Classification
============================================================

PIPELINE LAYERS
  1. Data generation & EDA          - simulate / inspect imbalance
  2. Data splitting                  - iterative stratification
  3. Preprocessing                   - encoding, scaling, imputation
  4. Sampling (inside CV folds)      - undersample + MLSMOTE + ENN
  5. Model training                  - LightGBM + class weights
  6. Threshold tuning                - per-label F1-maximising search
  7. Evaluation                      - macro F1, hamming, per-label report
  8. Full cross-validated pipeline   - everything wired together

INSTALL (pip):
  pip install lightgbm scikit-learn imbalanced-learn
      category_encoders skmultilearn numpy pandas

NOTE: mlsmote is a lightweight helper included inline below
      (avoids the unmaintained mlsmote PyPI package).
============================================================
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import warnings
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    f1_score, hamming_loss, classification_report,
    roc_curve, precision_recall_curve,
    label_ranking_average_precision_score,
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.neighbors import NearestNeighbors

from lightgbm import LGBMClassifier

from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours
from imblearn.over_sampling import RandomOverSampler

try:
    from skmultilearn.model_selection import iterative_train_test_split
    ITERATIVE_SPLIT = True
except ImportError:
    from sklearn.model_selection import train_test_split
    ITERATIVE_SPLIT = False
    warnings.warn("skmultilearn not found — falling back to random split.")

import category_encoders as ce

warnings.filterwarnings("ignore")
np.random.seed(42)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 ─ Synthetic data generation
#   Mirrors your real feature set and target imbalance:
#   40% of labels cover >90% of transactions
# ══════════════════════════════════════════════════════════════════════════════

def generate_synthetic_data(n_samples: int = 50_000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate a synthetic transaction dataset that mirrors the described
    imbalance: 40% of rate codes dominate 90%+ of transactions.

    Returns
    -------
    X : pd.DataFrame  — feature matrix (11 features)
    Y : pd.DataFrame  — binary label matrix (20 interchange rate codes)
    """
    rng = np.random.default_rng(42)

    # ── Features ─────────────────────────────────────────────────────────────
    mcc_codes        = [str(c) for c in range(1000, 1200)]   # 200 MCC codes
    merch_desc       = [f"MERCHANT_{i:04d}" for i in range(500)]
    modes            = ["POS", "Wallet", "ApplePay", "GPay", "NFC", "Online"]
    product_codes    = ["VISA_SIG", "VISA_INF", "MC_WORLD", "MC_PLAT",
                        "AMEX_GOLD", "AMEX_PLAT", "DISC_PREM"]
    merch_categories = ["Retail", "Grocery", "Travel", "Dining",
                        "Fuel", "Healthcare", "Entertainment", "Utilities"]
    locations        = [f"LOC_{i:03d}" for i in range(50)]
    origins          = ["US", "UK", "EU", "APAC", "LATAM", "MEA"]
    zipcodes         = [f"{z:05d}" for z in rng.integers(10000, 99999, 300)]

    X = pd.DataFrame({
        "mcc":               rng.choice(mcc_codes, n_samples),
        "merchant_desc":     rng.choice(merch_desc, n_samples),
        "mode_of_payment":   rng.choice(modes, n_samples,
                                        p=[0.40, 0.18, 0.15, 0.12, 0.10, 0.05]),
        "is_domestic":       rng.choice([0, 1], n_samples, p=[0.25, 0.75]),
        "product_code":      rng.choice(product_codes, n_samples),
        "merchant_category": rng.choice(merch_categories, n_samples,
                                        p=[0.25, 0.20, 0.15, 0.12,
                                           0.10, 0.08, 0.06, 0.04]),
        "merchant_location": rng.choice(locations, n_samples),
        "merchant_origin":   rng.choice(origins, n_samples,
                                        p=[0.50, 0.15, 0.15, 0.10, 0.06, 0.04]),
        "txn_origin_zip":    rng.choice(zipcodes, n_samples),
        # Heavy right-skew: log-normal spend & interchange
        "net_spend":         np.clip(rng.lognormal(mean=4.0, sigma=2.5, size=n_samples),
                                     1, 1_000_000),
        "interchange":       np.clip(rng.lognormal(mean=2.5, sigma=2.0, size=n_samples),
                                     1, 1_000_000),
    })

    # ── Labels: 20 interchange rate codes ────────────────────────────────────
    # Majority codes (0–7): ~90% of transactions  → high base probability
    # Minority codes (8–19): ~10% of transactions → low base probability
    n_labels    = 20
    maj_codes   = list(range(8))       # 40% of label space
    min_codes   = list(range(8, 20))   # 60% of label space

    label_probs = np.array(
        [0.45, 0.40, 0.38, 0.35, 0.30, 0.28, 0.25, 0.22]   # majority
        + [0.04, 0.03, 0.03, 0.02, 0.02, 0.02, 0.01,        # minority
           0.01, 0.01, 0.01, 0.005, 0.005]
    )

    Y_raw = rng.random((n_samples, n_labels)) < label_probs
    # Ensure every sample has at least one label
    no_label_mask = ~Y_raw.any(axis=1)
    Y_raw[no_label_mask, rng.integers(0, 8, no_label_mask.sum())] = True

    Y = pd.DataFrame(
        Y_raw.astype(int),
        columns=[f"RC_{i:02d}" for i in range(n_labels)]
    )

    return X, Y


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 ─ EDA helpers — inspect the imbalance before touching anything
# ══════════════════════════════════════════════════════════════════════════════

def print_imbalance_report(Y: pd.DataFrame) -> None:
    """Print a concise per-label imbalance report."""
    print("\n" + "═" * 60)
    print("  LABEL IMBALANCE REPORT")
    print("═" * 60)
    print(f"{'Label':<10} {'Positives':>10} {'%':>8} {'IR (neg/pos)':>14} {'Group':<10}")
    print("─" * 60)

    maj_codes = [c for c in Y.columns if int(c.split("_")[1]) < 8]
    min_codes = [c for c in Y.columns if int(c.split("_")[1]) >= 8]

    total = len(Y)
    for col in Y.columns:
        pos    = int(Y[col].sum())
        pct    = pos / total * 100
        neg    = total - pos
        ir     = neg / max(pos, 1)
        group  = "MAJORITY" if col in maj_codes else "minority"
        flag   = " <--" if ir > 20 else ""
        print(f"{col:<10} {pos:>10,} {pct:>7.2f}% {ir:>14.1f}  {group}{flag}")

    maj_txn = Y[maj_codes].any(axis=1).sum()
    min_txn = Y[min_codes].any(axis=1).sum()
    print("─" * 60)
    print(f"\nSamples with ≥1 MAJORITY label : {maj_txn:,}  ({maj_txn/total*100:.1f}%)")
    print(f"Samples with ≥1 MINORITY label : {min_txn:,}  ({min_txn/total*100:.1f}%)")
    print(f"Compound imbalance ratio        : {maj_txn/max(min_txn,1):.1f}:1")
    print("═" * 60 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 ─ Preprocessing pipeline
#   - High-cardinality cats   → Target encoding  (MCC, description, zip)
#   - Low-cardinality cats    → Ordinal encoding  (mode, product, category, etc.)
#   - Binary flag             → pass-through
#   - Numericals              → log1p → RobustScaler
# ══════════════════════════════════════════════════════════════════════════════

HIGH_CARD_CATS = ["mcc", "merchant_desc", "txn_origin_zip"]
LOW_CARD_CATS  = ["mode_of_payment", "product_code", "merchant_category",
                  "merchant_location", "merchant_origin"]
BINARY_COLS    = ["is_domestic"]
NUMERIC_COLS   = ["net_spend", "interchange"]


def build_preprocessor() -> ColumnTransformer:
    """
    Build the sklearn ColumnTransformer.
    Target encoding for high-cardinality features is applied manually
    inside the CV loop (to prevent leakage — see train_fold_pipeline).
    This transformer handles everything except target encoding.
    """
    low_card_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )),
    ])

    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("log1p",  FunctionTransformer(np.log1p)),
        ("scale",  RobustScaler()),
    ])

    return ColumnTransformer([
        ("low_card", low_card_pipe, LOW_CARD_CATS + BINARY_COLS),
        ("numeric",  numeric_pipe,  NUMERIC_COLS),
        # high-card columns handled via target encoding below
    ], remainder="drop")


# sklearn doesn't export FunctionTransformer at top-level — import it here
from sklearn.preprocessing import FunctionTransformer


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 ─ MLSMOTE — Multi-Label SMOTE (implemented inline)
#   Generates synthetic minority samples by interpolating in feature space
#   and taking the union of label sets from nearest neighbours.
# ══════════════════════════════════════════════════════════════════════════════

def mlsmote(
    X_min: np.ndarray,
    Y_min: np.ndarray,
    n_synthetic: int = 500,
    k_neighbors: int = 5,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    MLSMOTE: Multi-Label Synthetic Minority Oversampling Technique.

    For each minority sample:
      1. Find k nearest neighbours in feature space (Euclidean).
      2. Randomly pick one neighbour.
      3. Interpolate: x_new = x_i + λ * (x_nn - x_i), λ ~ Uniform(0,1).
      4. y_new = union of y_i and y_nn label sets.

    Parameters
    ----------
    X_min       : np.ndarray  — feature matrix of minority samples only
    Y_min       : np.ndarray  — label matrix of minority samples only
    n_synthetic : int         — number of synthetic samples to generate
    k_neighbors : int         — neighbours used for interpolation
    random_state: int

    Returns
    -------
    X_new : np.ndarray  — synthetic feature rows
    Y_new : np.ndarray  — synthetic label rows
    """
    rng = np.random.default_rng(random_state)
    n   = X_min.shape[0]

    if n < k_neighbors + 1:
        # Not enough samples — fall back to random oversampling
        idx = rng.integers(0, n, n_synthetic)
        return X_min[idx], Y_min[idx]

    nn = NearestNeighbors(n_neighbors=k_neighbors + 1, algorithm="auto")
    nn.fit(X_min)
    _, indices = nn.kneighbors(X_min)          # shape (n, k+1) — includes self

    X_new = np.empty((n_synthetic, X_min.shape[1]), dtype=float)
    Y_new = np.empty((n_synthetic, Y_min.shape[1]), dtype=int)

    for s in range(n_synthetic):
        i      = rng.integers(0, n)
        nn_idx = rng.choice(indices[i][1:])    # exclude self (index 0)
        lam    = rng.random()

        X_new[s] = X_min[i] + lam * (X_min[nn_idx] - X_min[i])
        Y_new[s] = np.logical_or(Y_min[i], Y_min[nn_idx]).astype(int)

    return X_new, Y_new


def get_minority_mask(Y: np.ndarray, minority_label_indices: list[int]) -> np.ndarray:
    """Return a boolean mask of rows that contain at least one minority label."""
    return Y[:, minority_label_indices].any(axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 ─ Sampling pipeline (applied per training fold)
#   Order matters:
#     a) Undersample pure-majority rows           (reduce majority volume)
#     b) MLSMOTE on minority-containing rows     (boost minority samples)
#     c) ENN cleaning on binary-relevance basis  (remove noisy boundaries)
# ══════════════════════════════════════════════════════════════════════════════

def apply_sampling(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    minority_label_indices: list[int],
    undersample_majority_ratio: float = 0.30,
    mlsmote_n_synthetic: int = 1000,
    mlsmote_k: int = 5,
    apply_enn: bool = True,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Full sampling pipeline for one training fold.

    Parameters
    ----------
    X_train                   : preprocessed feature array
    Y_train                   : binary label matrix
    minority_label_indices    : indices of the minority (rare) labels
    undersample_majority_ratio: fraction of pure-majority rows to KEEP (0–1)
    mlsmote_n_synthetic       : number of synthetic minority samples to generate
    mlsmote_k                 : MLSMOTE nearest-neighbour count
    apply_enn                 : whether to run ENN noise cleaning
    verbose                   : print sample counts before/after

    Returns
    -------
    X_resampled, Y_resampled : np.ndarray
    """
    if verbose:
        print(f"\n  [Sampling] Input  : {X_train.shape[0]:,} samples, "
              f"{Y_train.sum():,} positive label instances")

    # ── a) Undersample pure-majority rows ────────────────────────────────────
    minority_mask = get_minority_mask(Y_train, minority_label_indices)
    pure_maj_idx  = np.where(~minority_mask)[0]
    min_cont_idx  = np.where(minority_mask)[0]

    n_keep = max(1, int(len(pure_maj_idx) * undersample_majority_ratio))
    keep   = np.random.choice(pure_maj_idx, n_keep, replace=False)
    idx_after_us = np.concatenate([min_cont_idx, keep])

    X_us = X_train[idx_after_us]
    Y_us = Y_train[idx_after_us]

    if verbose:
        print(f"  [Sampling] After undersample : {X_us.shape[0]:,} samples")

    # ── b) MLSMOTE on minority-containing rows ────────────────────────────────
    X_min_rows = X_us[get_minority_mask(Y_us, minority_label_indices)]
    Y_min_rows = Y_us[get_minority_mask(Y_us, minority_label_indices)]

    if len(X_min_rows) > 0:
        X_syn, Y_syn = mlsmote(
            X_min_rows.astype(float),
            Y_min_rows.astype(int),
            n_synthetic=mlsmote_n_synthetic,
            k_neighbors=mlsmote_k,
        )
        X_combined = np.vstack([X_us, X_syn])
        Y_combined = np.vstack([Y_us, Y_syn])
    else:
        X_combined, Y_combined = X_us, Y_us

    if verbose:
        print(f"  [Sampling] After MLSMOTE    : {X_combined.shape[0]:,} samples")

    # ── c) ENN cleaning (per majority label, then re-combine) ─────────────────
    if apply_enn and len(X_combined) > 10:
        try:
            # Apply ENN on the most dominant majority label as a proxy clean
            dominant_label = int(Y_combined[:, :len(minority_label_indices)].sum(0).argmax())
            enn = EditedNearestNeighbours(n_neighbors=3, kind_sel="all")
            _, y_enn_col = enn.fit_resample(X_combined, Y_combined[:, dominant_label])
            enn_mask      = np.isin(np.arange(len(X_combined)),
                                    enn.sample_indices_)
            X_combined = X_combined[enn_mask]
            Y_combined = Y_combined[enn_mask]
        except Exception:
            pass    # ENN may fail on tiny folds — safe to skip

    if verbose:
        print(f"  [Sampling] After ENN clean  : {X_combined.shape[0]:,} samples")
        new_minority = get_minority_mask(Y_combined, minority_label_indices).sum()
        ir = (len(X_combined) - new_minority) / max(new_minority, 1)
        print(f"  [Sampling] Final IR         : {ir:.1f}:1")

    return X_combined, Y_combined


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 ─ Model: LightGBM with per-label class weights
#   Wrapped in a simple multi-label loop (binary relevance).
#   Each label gets its own scale_pos_weight derived from training fold.
# ══════════════════════════════════════════════════════════════════════════════

class MultiLabelLGBM:
    """
    Binary-relevance multi-label classifier using LightGBM.
    One LGBMClassifier per label, each with its own scale_pos_weight.
    """

    def __init__(self, n_estimators: int = 300, learning_rate: float = 0.05,
                 num_leaves: int = 63, n_jobs: int = -1):
        self.n_estimators   = n_estimators
        self.learning_rate  = learning_rate
        self.num_leaves     = num_leaves
        self.n_jobs         = n_jobs
        self.classifiers_   = []
        self.n_labels_      = 0

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "MultiLabelLGBM":
        self.n_labels_   = Y.shape[1]
        self.classifiers_ = []

        for i in range(self.n_labels_):
            y_col = Y[:, i]
            n_neg = (y_col == 0).sum()
            n_pos = max((y_col == 1).sum(), 1)
            spw   = n_neg / n_pos          # scale_pos_weight = neg/pos ratio

            clf = LGBMClassifier(
                n_estimators   = self.n_estimators,
                learning_rate  = self.learning_rate,
                num_leaves     = self.num_leaves,
                n_jobs         = self.n_jobs,
                scale_pos_weight = spw,
                verbose        = -1,
                random_state   = 42,
            )
            clf.fit(X, y_col)
            self.classifiers_.append(clf)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return (n_samples, n_labels) probability matrix."""
        return np.column_stack([
            clf.predict_proba(X)[:, 1] for clf in self.classifiers_
        ])

    def predict(self, X: np.ndarray,
                thresholds: np.ndarray | None = None) -> np.ndarray:
        proba = self.predict_proba(X)
        if thresholds is None:
            thresholds = np.full(self.n_labels_, 0.5)
        return (proba >= thresholds).astype(int)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 ─ Per-label threshold optimisation
#   Three strategies — use F1-maximising as default.
# ══════════════════════════════════════════════════════════════════════════════

def find_f1_thresholds(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    search_range: np.ndarray | None = None,
) -> np.ndarray:
    """
    For each label, sweep candidate thresholds and return the one
    that maximises binary F1.  Safe for zero-positive labels.

    Returns
    -------
    thresholds : np.ndarray of shape (n_labels,)
    """
    if search_range is None:
        search_range = np.arange(0.05, 0.90, 0.01)

    n_labels   = y_true.shape[1]
    thresholds = np.full(n_labels, 0.5)

    for i in range(n_labels):
        best_t, best_f1 = 0.5, -1.0
        for t in search_range:
            preds = (y_proba[:, i] >= t).astype(int)
            f1    = f1_score(y_true[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[i] = best_t

    return thresholds


def find_youden_thresholds(y_true: np.ndarray, y_proba: np.ndarray) -> np.ndarray:
    """Youden's J: threshold that maximises TPR − FPR per label."""
    n_labels   = y_true.shape[1]
    thresholds = np.full(n_labels, 0.5)

    for i in range(n_labels):
        if y_true[:, i].sum() == 0:
            continue
        fpr, tpr, thr = roc_curve(y_true[:, i], y_proba[:, i])
        j_idx          = np.argmax(tpr - fpr)
        thresholds[i]  = thr[j_idx]

    return thresholds


def find_precision_floor_thresholds(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    min_precision: float = 0.80,
) -> np.ndarray:
    """
    Per label: highest recall subject to precision >= min_precision.
    Falls back to 0.5 if the floor can't be met.
    """
    n_labels   = y_true.shape[1]
    thresholds = np.full(n_labels, 0.5)

    for i in range(n_labels):
        if y_true[:, i].sum() == 0:
            continue
        prec, rec, thr = precision_recall_curve(y_true[:, i], y_proba[:, i])
        valid           = prec[:-1] >= min_precision
        if valid.any():
            best_idx       = np.argmax(rec[:-1][valid])
            thresholds[i]  = thr[valid][best_idx]

    return thresholds


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 ─ Evaluation helpers
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    label_names: list[str],
    minority_label_indices: list[int],
    title: str = "Evaluation Results",
) -> dict:
    """
    Compute and print a comprehensive multi-label evaluation report.

    Returns a dict with all metrics for downstream comparison.
    """
    macro_f1  = f1_score(y_true, y_pred, average="macro",  zero_division=0)
    micro_f1  = f1_score(y_true, y_pred, average="micro",  zero_division=0)
    weighted  = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    hl        = hamming_loss(y_true, y_pred)
    exact     = np.all(y_true == y_pred, axis=1).mean()
    lrap      = label_ranking_average_precision_score(y_true, y_proba)

    print("\n" + "═" * 60)
    print(f"  {title}")
    print("═" * 60)
    print(f"  Macro F1            : {macro_f1:.4f}  ← primary metric")
    print(f"  Micro F1            : {micro_f1:.4f}")
    print(f"  Weighted F1         : {weighted:.4f}")
    print(f"  Hamming loss        : {hl:.4f}  ({hl*100:.2f}% labels wrong)")
    print(f"  Exact match ratio   : {exact:.4f}")
    print(f"  Label ranking AP    : {lrap:.4f}")

    # ── Per-label breakdown ──────────────────────────────────────────────────
    per_label_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_label_rc = []
    for i in range(y_true.shape[1]):
        from sklearn.metrics import recall_score
        rc = recall_score(y_true[:, i], y_pred[:, i], zero_division=0)
        per_label_rc.append(rc)

    print(f"\n  {'Label':<10} {'F1':>8} {'Recall':>8} {'Support':>9} {'Group':<10}")
    print("  " + "─" * 50)
    for i, name in enumerate(label_names):
        group   = "MINORITY" if i in minority_label_indices else "majority"
        support = int(y_true[:, i].sum())
        flag    = " **" if i in minority_label_indices and per_label_f1[i] < 0.40 else ""
        print(f"  {name:<10} {per_label_f1[i]:>8.3f} {per_label_rc[i]:>8.3f} "
              f"{support:>9,}  {group}{flag}")

    maj_idx   = [i for i in range(len(label_names)) if i not in minority_label_indices]
    min_f1    = per_label_f1[minority_label_indices].mean()
    maj_f1    = per_label_f1[maj_idx].mean()

    print("  " + "─" * 50)
    print(f"  Avg F1 — majority labels  : {maj_f1:.4f}")
    print(f"  Avg F1 — minority labels  : {min_f1:.4f}  ← watch this")
    print(f"  F1 gap (maj − min)        : {maj_f1 - min_f1:.4f}")
    print("═" * 60 + "\n")

    return {
        "macro_f1":       macro_f1,
        "micro_f1":       micro_f1,
        "hamming_loss":   hl,
        "exact_match":    exact,
        "lrap":           lrap,
        "minority_f1":    min_f1,
        "majority_f1":    maj_f1,
        "per_label_f1":   per_label_f1,
    }


# ══════════════════════════════════════════════════════════════════════════════
# STEP 9 ─ Target encoding (leakage-safe, applied inside CV)
# ══════════════════════════════════════════════════════════════════════════════

def target_encode_fold(
    X_train_df: pd.DataFrame,
    X_val_df: pd.DataFrame,
    y_train: np.ndarray,
    high_card_cols: list[str],
    smoothing: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit target encoder on training fold, transform both train and val.
    Uses the mean of all labels as the target proxy (multi-label average).
    Smoothing prevents high-variance encoding on rare categories.
    """
    y_mean = y_train.mean(axis=1)          # aggregate target across all labels
    enc    = ce.TargetEncoder(cols=high_card_cols, smoothing=smoothing)
    enc.fit(X_train_df[high_card_cols], y_mean)

    X_tr = X_train_df.copy()
    X_vl = X_val_df.copy()
    X_tr[high_card_cols] = enc.transform(X_train_df[high_card_cols])
    X_vl[high_card_cols] = enc.transform(X_val_df[high_card_cols])

    return X_tr, X_vl


# ══════════════════════════════════════════════════════════════════════════════
# STEP 10 ─ Full end-to-end pipeline (single train/test run)
#   For production use, wrap this inside k-fold cross-validation.
# ══════════════════════════════════════════════════════════════════════════════

def run_full_pipeline(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    minority_label_indices: list[int],
    test_size: float = 0.20,
    val_size: float  = 0.15,
    threshold_strategy: str = "f1",       # "f1" | "youden" | "precision_floor"
    min_precision_floor: float = 0.80,    # used only if strategy == "precision_floor"
    verbose: bool = True,
) -> dict:
    """
    End-to-end pipeline:
      split → target encode → preprocess → sample → train → tune thresholds → evaluate.

    Parameters
    ----------
    X                       : raw feature DataFrame
    Y                       : binary label DataFrame
    minority_label_indices  : indices of minority labels in Y
    test_size               : hold-out test fraction
    val_size                : validation fraction (for threshold tuning)
    threshold_strategy      : "f1" | "youden" | "precision_floor"
    min_precision_floor     : minimum precision when using "precision_floor"
    verbose                 : print progress
    """
    label_names = list(Y.columns)
    X_arr       = X.values
    Y_arr       = Y.values.astype(int)

    # ── Split ────────────────────────────────────────────────────────────────
    if ITERATIVE_SPLIT:
        X_temp, Y_temp, X_test, Y_test = iterative_train_test_split(
            X_arr, Y_arr, test_size=test_size
        )
        X_train, Y_train, X_val, Y_val = iterative_train_test_split(
            X_temp, Y_temp, test_size=val_size / (1 - test_size)
        )
        # Rebuild DataFrames for target encoding
        X_train_df = pd.DataFrame(X_train, columns=X.columns)
        X_val_df   = pd.DataFrame(X_val,   columns=X.columns)
        X_test_df  = pd.DataFrame(X_test,  columns=X.columns)
    else:
        from sklearn.model_selection import train_test_split
        X_temp_df, X_test_df, Y_temp, Y_test = train_test_split(
            X, Y_arr, test_size=test_size, random_state=42)
        X_train_df, X_val_df, Y_train, Y_val = train_test_split(
            X_temp_df, Y_temp, test_size=val_size / (1 - test_size), random_state=42)

    if verbose:
        print(f"\n  Split → train: {len(Y_train):,} | val: {len(Y_val):,} | "
              f"test: {len(Y_test):,}")

    # ── Target encoding (fit on train, apply to val + test) ──────────────────
    X_train_df, X_val_df   = target_encode_fold(
        X_train_df, X_val_df, Y_train, HIGH_CARD_CATS)
    X_train_df2, X_test_df = target_encode_fold(
        X_train_df, X_test_df, Y_train, HIGH_CARD_CATS)

    # ── Structural preprocessing ─────────────────────────────────────────────
    preprocessor = build_preprocessor()
    X_train_pp   = preprocessor.fit_transform(X_train_df)
    X_val_pp     = preprocessor.transform(X_val_df)
    X_test_pp    = preprocessor.transform(X_test_df)

    # ── Sampling (on training fold only) ─────────────────────────────────────
    X_train_s, Y_train_s = apply_sampling(
        X_train_pp, Y_train,
        minority_label_indices=minority_label_indices,
        undersample_majority_ratio=0.35,
        mlsmote_n_synthetic=2000,
        mlsmote_k=5,
        apply_enn=True,
        verbose=verbose,
    )

    # ── Model training ────────────────────────────────────────────────────────
    if verbose:
        print("\n  [Model] Training LightGBM (binary relevance, per-label weights)...")
    model = MultiLabelLGBM(n_estimators=200, learning_rate=0.05, num_leaves=63)
    model.fit(X_train_s, Y_train_s)

    # ── Threshold tuning on validation set ───────────────────────────────────
    if verbose:
        print(f"\n  [Threshold] Strategy: {threshold_strategy}")
    val_proba = model.predict_proba(X_val_pp)

    if threshold_strategy == "f1":
        thresholds = find_f1_thresholds(Y_val, val_proba)
    elif threshold_strategy == "youden":
        thresholds = find_youden_thresholds(Y_val, val_proba)
    elif threshold_strategy == "precision_floor":
        thresholds = find_precision_floor_thresholds(
            Y_val, val_proba, min_precision=min_precision_floor)
    else:
        thresholds = np.full(Y_arr.shape[1], 0.5)

    if verbose:
        print("\n  Per-label thresholds (default was 0.5 for all):")
        for i, (name, t) in enumerate(zip(label_names, thresholds)):
            marker = " <<< lowered (minority)" if t < 0.40 else ""
            print(f"    {name}: {t:.2f}{marker}")

    # ── Evaluation on held-out test set ───────────────────────────────────────
    test_proba = model.predict_proba(X_test_pp)

    # Baseline: default 0.5 thresholds
    baseline_pred = (test_proba >= 0.5).astype(int)
    baseline_metrics = evaluate(
        Y_test, baseline_pred, test_proba,
        label_names, minority_label_indices,
        title="BASELINE — default 0.5 thresholds"
    )

    # Tuned: per-label thresholds
    tuned_pred = model.predict(X_test_pp, thresholds=thresholds)
    tuned_metrics = evaluate(
        Y_test, tuned_pred, test_proba,
        label_names, minority_label_indices,
        title=f"TUNED — per-label {threshold_strategy} thresholds"
    )

    # ── Improvement summary ───────────────────────────────────────────────────
    print("═" * 60)
    print("  IMPROVEMENT SUMMARY (tuned vs baseline)")
    print("═" * 60)
    for metric in ["macro_f1", "minority_f1", "majority_f1", "hamming_loss"]:
        b   = baseline_metrics[metric]
        t   = tuned_metrics[metric]
        # For hamming_loss, lower is better
        imp = (b - t) if metric == "hamming_loss" else (t - b)
        dir_lbl = "improved" if imp > 0 else "unchanged/worse"
        print(f"  {metric:<20}: {b:.4f}  →  {t:.4f}  ({dir_lbl:+.4f})")
    print("═" * 60 + "\n")

    return {
        "model":            model,
        "thresholds":       thresholds,
        "preprocessor":     preprocessor,
        "baseline_metrics": baseline_metrics,
        "tuned_metrics":    tuned_metrics,
    }


# ══════════════════════════════════════════════════════════════════════════════
# STEP 11 ─ Ablation study
#   Compare 4 configurations to quantify the impact of each layer:
#     A) No treatment           — baseline
#     B) Class weights only     — algorithm layer
#     C) Class weights + sample — data + algorithm layers
#     D) Full pipeline          — all four layers
# ══════════════════════════════════════════════════════════════════════════════

def ablation_study(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    minority_label_indices: list[int],
) -> pd.DataFrame:
    """
    Run four ablation configurations and return a comparison DataFrame.
    Useful for justifying each mitigation layer to stakeholders.
    """
    label_names = list(Y.columns)
    results     = []

    configs = [
        dict(name="A — no treatment",
             use_weights=False, use_sampling=False, use_threshold_tuning=False),
        dict(name="B — class weights only",
             use_weights=True,  use_sampling=False, use_threshold_tuning=False),
        dict(name="C — weights + sampling",
             use_weights=True,  use_sampling=True,  use_threshold_tuning=False),
        dict(name="D — full pipeline",
             use_weights=True,  use_sampling=True,  use_threshold_tuning=True),
    ]

    # Shared split (same data each time for fair comparison)
    if ITERATIVE_SPLIT:
        X_arr = X.values
        Y_arr = Y.values.astype(int)
        X_tv, Y_tv, X_test, Y_test = iterative_train_test_split(X_arr, Y_arr, test_size=0.20)
        X_train_arr, Y_train, X_val_arr, Y_val = iterative_train_test_split(X_tv, Y_tv, test_size=0.15/0.80)
        X_train_df = pd.DataFrame(X_train_arr, columns=X.columns)
        X_val_df   = pd.DataFrame(X_val_arr,   columns=X.columns)
        X_test_df  = pd.DataFrame(X_test,      columns=X.columns)
    else:
        from sklearn.model_selection import train_test_split
        X_tv, X_test_df, Y_tv, Y_test = train_test_split(X, Y.values.astype(int), test_size=0.20, random_state=42)
        X_train_df, X_val_df, Y_train, Y_val = train_test_split(X_tv, Y_tv, test_size=0.1875, random_state=42)

    # Target encode once (same for all configs)
    X_tr_enc, X_val_enc = target_encode_fold(X_train_df, X_val_df, Y_train, HIGH_CARD_CATS)
    X_tr_enc2, X_te_enc = target_encode_fold(X_train_df, X_test_df, Y_train, HIGH_CARD_CATS)

    preproc     = build_preprocessor()
    X_tr_pp     = preproc.fit_transform(X_tr_enc)
    X_val_pp    = preproc.transform(X_val_enc)
    X_te_pp     = preproc.transform(X_te_enc)

    for cfg in configs:
        print(f"\n  [Ablation] Running config: {cfg['name']}")

        # Sampling (only if enabled)
        if cfg["use_sampling"]:
            X_fit, Y_fit = apply_sampling(
                X_tr_pp, Y_train, minority_label_indices,
                verbose=False
            )
        else:
            X_fit, Y_fit = X_tr_pp, Y_train

        # Model (class weights baked into LGBMClassifier via scale_pos_weight)
        model = MultiLabelLGBM(n_estimators=150, learning_rate=0.05)
        if not cfg["use_weights"]:
            # Override scale_pos_weight to 1 (no weighting)
            for attr in ["n_estimators", "learning_rate", "num_leaves", "n_jobs"]:
                pass
            model.fit(X_fit, Y_fit)
            for clf in model.classifiers_:
                clf.set_params(scale_pos_weight=1)  # re-fit without weights
                clf.fit(X_fit, Y_fit[:, model.classifiers_.index(clf)])
        else:
            model.fit(X_fit, Y_fit)

        # Threshold tuning (only if enabled)
        val_proba  = model.predict_proba(X_val_pp)
        thresholds = find_f1_thresholds(Y_val, val_proba) if cfg["use_threshold_tuning"] \
                     else np.full(Y_train.shape[1], 0.5)

        test_proba = model.predict_proba(X_te_pp)
        test_pred  = model.predict(X_te_pp, thresholds=thresholds)

        per_f1     = f1_score(Y_test, test_pred, average=None, zero_division=0)
        maj_idx    = [i for i in range(len(label_names)) if i not in minority_label_indices]
        min_f1     = per_f1[minority_label_indices].mean()
        maj_f1     = per_f1[maj_idx].mean()

        results.append({
            "Configuration":       cfg["name"],
            "Macro F1":            round(f1_score(Y_test, test_pred, average="macro", zero_division=0), 4),
            "Minority F1 (avg)":   round(min_f1, 4),
            "Majority F1 (avg)":   round(maj_f1, 4),
            "Hamming Loss":        round(hamming_loss(Y_test, test_pred), 4),
            "Exact Match":         round(np.all(Y_test == test_pred, axis=1).mean(), 4),
        })

    results_df = pd.DataFrame(results)
    print("\n" + "═" * 75)
    print("  ABLATION STUDY — impact of each mitigation layer")
    print("═" * 75)
    print(results_df.to_string(index=False))
    print("═" * 75 + "\n")

    return results_df


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — run the complete pipeline
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("\n" + "═" * 60)
    print("  INTERCHANGE RATE CODE — MULTI-LABEL IMBALANCE PIPELINE")
    print("═" * 60)

    # ── 1. Generate data ──────────────────────────────────────────────────────
    print("\n[1/5] Generating synthetic transaction data...")
    X, Y = generate_synthetic_data(n_samples=50_000)
    print(f"      Dataset: {X.shape[0]:,} rows × {X.shape[1]} features, "
          f"{Y.shape[1]} labels")

    # ── 2. Inspect imbalance ──────────────────────────────────────────────────
    print("\n[2/5] Imbalance report...")
    MINORITY_INDICES = list(range(8, 20))   # RC_08 through RC_19
    print_imbalance_report(Y)

    # ── 3. Full pipeline run ──────────────────────────────────────────────────
    print("\n[3/5] Running full end-to-end pipeline...")
    pipeline_result = run_full_pipeline(
        X, Y,
        minority_label_indices=MINORITY_INDICES,
        test_size=0.20,
        val_size=0.15,
        threshold_strategy="f1",           # "f1" | "youden" | "precision_floor"
        verbose=True,
    )

    # ── 4. Ablation study ─────────────────────────────────────────────────────
    print("\n[4/5] Running ablation study (4 configs)...")
    ablation_df = ablation_study(X, Y, MINORITY_INDICES)

    # ── 5. Inference example ──────────────────────────────────────────────────
    print("\n[5/5] Inference example (first 5 test rows)...")
    model      = pipeline_result["model"]
    thresholds = pipeline_result["thresholds"]
    preproc    = pipeline_result["preprocessor"]

    sample_X = X.head(5).copy()
    sample_X, _ = target_encode_fold(sample_X, sample_X, Y.values[:5], HIGH_CARD_CATS)
    sample_pp   = preproc.transform(sample_X)
    sample_prob = model.predict_proba(sample_pp)
    sample_pred = model.predict(sample_pp, thresholds=thresholds)

    print("\n  Predicted rate codes (1 = active):")
    pred_df = pd.DataFrame(
        sample_pred,
        columns=list(Y.columns)
    )
    pred_df["n_labels_predicted"] = pred_df.sum(axis=1)
    print(pred_df.to_string())
    print("\n  Pipeline complete.")
