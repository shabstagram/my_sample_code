"""
Decision Tree Classifier Pipeline
Target Variable: Interchange Rate Code (INTR_RT_CD)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.metrics import (classification_report, precision_score, recall_score,
                              f1_score, confusion_matrix, ConfusionMatrixDisplay,
                              roc_auc_score)
import graphviz
import os

np.random.seed(42)

# ─────────────────────────────────────────────
# 1. SYNTHETIC DATA GENERATION
# ─────────────────────────────────────────────
print("=" * 65)
print("  DECISION TREE CLASSIFIER — INTERCHANGE RATE CODE PREDICTION")
print("=" * 65)

N = 5000

mrch_cat_options   = ['5411', '5812', '4111', '7011', '5999', '5912', '4900', '6300']
rcur_pymn_options  = ['Y', 'N']
cnp_ind_options    = ['Y', 'N']
payment_method     = ['CHIP', 'CONTACTLESS', 'SWIPE', 'MANUAL_ENTRY', 'ONLINE']
spend_band_options = ['LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH']
intr_rt_options    = ['IR_01', 'IR_02', 'IR_03', 'IR_04', 'IR_05']

df = pd.DataFrame({
    'MRCH_CAT_CD'    : np.random.choice(mrch_cat_options,  N),
    'RCUR_PYMN_IN'   : np.random.choice(rcur_pymn_options, N, p=[0.35, 0.65]),
    'CNP_IND'        : np.random.choice(cnp_ind_options,   N, p=[0.45, 0.55]),
    'Payment_Method' : np.random.choice(payment_method,    N, p=[0.30, 0.25, 0.20, 0.10, 0.15]),
    'SPEND_BAND'     : np.random.choice(spend_band_options, N, p=[0.30, 0.35, 0.25, 0.10]),
    'is_tokenised'   : np.random.choice([True, False],     N, p=[0.60, 0.40]),
})

# Deterministic target mapping for realistic patterns
def assign_target(row):
    score = 0
    if row['SPEND_BAND']     == 'VERY_HIGH'   : score += 3
    elif row['SPEND_BAND']   == 'HIGH'         : score += 2
    elif row['SPEND_BAND']   == 'MEDIUM'       : score += 1
    if row['CNP_IND']        == 'Y'            : score += 2
    if row['RCUR_PYMN_IN']   == 'Y'            : score -= 1
    if row['is_tokenised']                     : score -= 1
    if row['Payment_Method'] == 'ONLINE'       : score += 2
    elif row['Payment_Method'] == 'MANUAL_ENTRY': score += 1
    if row['MRCH_CAT_CD'] in ['5812', '7011'] : score += 1
    score += np.random.randint(-1, 2)
    score = max(0, min(score, 4))
    return intr_rt_options[score]

df['INTR_RT_CD'] = df.apply(assign_target, axis=1)

print(f"\n📦 Dataset shape: {df.shape}")
print(f"\n📊 Target distribution:\n{df['INTR_RT_CD'].value_counts().to_string()}")

# ─────────────────────────────────────────────
# 2. DATA PREPARATION
# ─────────────────────────────────────────────
print("\n" + "─" * 65)
print("  STEP 2 · DATA PREPARATION")
print("─" * 65)

# Boolean → int
df['is_tokenised'] = df['is_tokenised'].astype(int)

# Encode categorical features (OrdinalEncoder)
cat_cols = ['MRCH_CAT_CD', 'RCUR_PYMN_IN', 'CNP_IND', 'Payment_Method', 'SPEND_BAND']
enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
df[cat_cols] = enc.fit_transform(df[cat_cols])

# Encode target
le = LabelEncoder()
df['INTR_RT_CD_ENC'] = le.fit_transform(df['INTR_RT_CD'])

feature_cols = ['MRCH_CAT_CD', 'RCUR_PYMN_IN', 'CNP_IND', 'Payment_Method',
                'SPEND_BAND', 'is_tokenised']
X = df[feature_cols]
y = df['INTR_RT_CD_ENC']

print(f"\n✅ Features encoded:  {feature_cols}")
print(f"✅ Target classes:    {list(le.classes_)}")

# ─────────────────────────────────────────────
# 3. TRAIN / TEST SPLIT (70:30)
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
print(f"\n📂 Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# ─────────────────────────────────────────────
# 4. MODEL TRAINING
# ─────────────────────────────────────────────
print("\n" + "─" * 65)
print("  STEP 3 · MODEL TRAINING")
print("─" * 65)

clf = DecisionTreeClassifier(
    criterion='gini',
    max_depth=6,
    min_samples_split=20,
    min_samples_leaf=10,
    class_weight='balanced',
    random_state=42
)
clf.fit(X_train, y_train)
print(f"\n✅ Model trained  |  Tree depth: {clf.get_depth()}  |  Leaves: {clf.get_n_leaves()}")

# ─────────────────────────────────────────────
# 5. PREDICTIONS & PROBABILITIES
# ─────────────────────────────────────────────
y_train_pred = clf.predict(X_train)
y_test_pred  = clf.predict(X_test)
y_test_proba = clf.predict_proba(X_test)

print(f"\n{'─'*65}")
print("  STEP 4 · PREDICTED PROBABILITIES (first 10 test samples)")
print(f"{'─'*65}")
prob_df = pd.DataFrame(y_test_proba, columns=[f"P({c})" for c in le.classes_])
prob_df['Predicted'] = le.inverse_transform(y_test_pred)
prob_df['Actual']    = le.inverse_transform(y_test.values)
print(prob_df.head(10).to_string(index=False))

# ─────────────────────────────────────────────
# 6. METRICS — TRAIN & TEST
# ─────────────────────────────────────────────
def compute_metrics(y_true, y_pred, label):
    p  = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    r  = recall_score   (y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score       (y_true, y_pred, average='weighted', zero_division=0)
    acc = (y_true == y_pred).mean()
    print(f"\n  [{label}]")
    print(f"    Accuracy          : {acc:.4f}")
    print(f"    Weighted Precision: {p:.4f}")
    print(f"    Weighted Recall   : {r:.4f}")
    print(f"    Weighted F1 Score : {f1:.4f}")
    return acc, p, r, f1

print(f"\n{'─'*65}")
print("  STEP 5 · PERFORMANCE METRICS")
print(f"{'─'*65}")
tr_acc, tr_p, tr_r, tr_f1 = compute_metrics(y_train, y_train_pred, "TRAIN SET")
te_acc, te_p, te_r, te_f1 = compute_metrics(y_test,  y_test_pred,  "TEST  SET")

print(f"\n  Classification Report (Test):\n")
print(classification_report(y_test, y_test_pred,
                             target_names=le.classes_, zero_division=0))

# ─────────────────────────────────────────────
# 7. CROSS-VALIDATION (5-Fold Stratified)
# ─────────────────────────────────────────────
print(f"{'─'*65}")
print("  STEP 6 · 5-FOLD STRATIFIED CROSS-VALIDATION")
print(f"{'─'*65}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = cross_validate(
    clf, X, y, cv=cv,
    scoring=['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted'],
    return_train_score=True
)

for metric in ['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted']:
    tr_scores = cv_results[f'train_{metric}']
    te_scores = cv_results[f'test_{metric}']
    label = metric.replace('_weighted','').capitalize()
    print(f"\n  {label}:")
    print(f"    Train → Mean: {tr_scores.mean():.4f}  Std: {tr_scores.std():.4f}  "
          f"Folds: {[round(s,4) for s in tr_scores]}")
    print(f"    Val   → Mean: {te_scores.mean():.4f}  Std: {te_scores.std():.4f}  "
          f"Folds: {[round(s,4) for s in te_scores]}")

# ─────────────────────────────────────────────
# 8. EXPORT TREE — DOT + SVG
# ─────────────────────────────────────────────
print(f"\n{'─'*65}")
print("  STEP 7 · EXPORTING DECISION TREE")
print(f"{'─'*65}")

dot_data = export_graphviz(
    clf,
    out_file=None,
    feature_names=feature_cols,
    class_names=le.classes_,
    filled=True,
    rounded=True,
    special_characters=True,
    impurity=True,
    proportion=False,
    precision=3
)

dot_path = '/mnt/user-data/outputs/decision_tree.dot'
svg_path = '/mnt/user-data/outputs/decision_tree.svg'

with open(dot_path, 'w') as f:
    f.write(dot_data)
print(f"✅ DOT file saved  → {dot_path}")

try:
    src = graphviz.Source(dot_data)
    src.render('/mnt/user-data/outputs/decision_tree', format='svg', cleanup=True)
    print(f"✅ SVG file saved  → {svg_path}")
except Exception as e:
    print(f"⚠️  SVG render via graphviz failed ({e}); using sklearn export")
    os.system(f"dot -Tsvg {dot_path} -o {svg_path}")

# ─────────────────────────────────────────────
# 9. VISUALISATION DASHBOARD
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(20, 22))
fig.patch.set_facecolor('#0F1117')
DARK = '#0F1117'; PANEL = '#1A1D27'; TEXT = '#E8EAF0'
BLUE = '#4C9BE8'; GREEN = '#4CAF82'; RED = '#E85C4C'; AMBER = '#F5A623'
PURPLE = '#9B59D0'; CYAN = '#4ECDC4'
palette = [BLUE, GREEN, AMBER, RED, PURPLE]

fig.suptitle('Decision Tree Classifier — INTR_RT_CD Prediction Dashboard',
             color=TEXT, fontsize=18, fontweight='bold', y=0.98)

# ── Row 1: Metrics summary bar ──────────────────────────────────
ax0 = fig.add_axes([0.03, 0.90, 0.94, 0.055])
ax0.set_facecolor(PANEL); ax0.axis('off')
metrics_txt = (
    f"  TRAIN  |  Acc: {tr_acc:.3f}   Precision: {tr_p:.3f}   "
    f"Recall: {tr_r:.3f}   F1: {tr_f1:.3f}          "
    f"TEST   |  Acc: {te_acc:.3f}   Precision: {te_p:.3f}   "
    f"Recall: {te_r:.3f}   F1: {te_f1:.3f}"
)
ax0.text(0.5, 0.5, metrics_txt, transform=ax0.transAxes, ha='center', va='center',
         color=CYAN, fontsize=11.5, fontfamily='monospace', fontweight='bold')
for spine in ax0.spines.values():
    spine.set_edgecolor(CYAN); spine.set_linewidth(1.5); spine.set_visible(True)

# ── Row 2a: Feature Importance ──────────────────────────────────
ax1 = fig.add_subplot(3, 3, 1)
ax1.set_facecolor(PANEL)
imp = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=True)
bars = ax1.barh(imp.index, imp.values, color=palette[:len(imp)], edgecolor='none', height=0.6)
for bar, val in zip(bars, imp.values):
    ax1.text(val + 0.002, bar.get_y() + bar.get_height()/2,
             f'{val:.4f}', va='center', color=TEXT, fontsize=8.5)
ax1.set_title('Feature Importance (Gini)', color=TEXT, fontsize=11, pad=8)
ax1.set_facecolor(PANEL); ax1.tick_params(colors=TEXT, labelsize=8)
for sp in ax1.spines.values(): sp.set_color('#2D3148')
ax1.set_xlim(0, imp.max() * 1.20)
ax1.set_xlabel('Importance', color=TEXT, fontsize=9)

# ── Row 2b: Target Distribution ─────────────────────────────────
ax2 = fig.add_subplot(3, 3, 2)
ax2.set_facecolor(PANEL)
class_counts = pd.Series(y_test).value_counts().sort_index()
bars2 = ax2.bar(le.classes_, class_counts.values, color=palette, edgecolor='none', width=0.55)
for b, v in zip(bars2, class_counts.values):
    ax2.text(b.get_x() + b.get_width()/2, v + 5, str(v),
             ha='center', color=TEXT, fontsize=9, fontweight='bold')
ax2.set_title('Target Class Distribution (Test)', color=TEXT, fontsize=11, pad=8)
ax2.tick_params(colors=TEXT, labelsize=9)
for sp in ax2.spines.values(): sp.set_color('#2D3148')
ax2.set_xlabel('Interchange Rate Code', color=TEXT, fontsize=9)
ax2.set_ylabel('Count', color=TEXT, fontsize=9)

# ── Row 2c: Predicted Probability Boxplot ───────────────────────
ax3 = fig.add_subplot(3, 3, 3)
ax3.set_facecolor(PANEL)
bp_data = [y_test_proba[:, i] for i in range(len(le.classes_))]
bp = ax3.boxplot(bp_data, patch_artist=True, notch=False,
                 medianprops=dict(color='white', linewidth=2),
                 whiskerprops=dict(color='#555'), capprops=dict(color='#555'),
                 flierprops=dict(marker='o', color='#666', markersize=2))
for patch, color in zip(bp['boxes'], palette):
    patch.set_facecolor(color); patch.set_alpha(0.8)
ax3.set_xticklabels(le.classes_, color=TEXT, fontsize=8.5)
ax3.set_title('Predicted Probability Distribution', color=TEXT, fontsize=11, pad=8)
ax3.tick_params(colors=TEXT, labelsize=8.5)
for sp in ax3.spines.values(): sp.set_color('#2D3148')
ax3.set_ylabel('Probability', color=TEXT, fontsize=9)

# ── Row 3: Confusion Matrix ──────────────────────────────────────
ax4 = fig.add_subplot(3, 3, 4)
ax4.set_facecolor(PANEL)
cm = confusion_matrix(y_test, y_test_pred)
im = ax4.imshow(cm, cmap='Blues', aspect='auto')
ax4.set_xticks(range(len(le.classes_))); ax4.set_yticks(range(len(le.classes_)))
ax4.set_xticklabels(le.classes_, color=TEXT, fontsize=8)
ax4.set_yticklabels(le.classes_, color=TEXT, fontsize=8)
ax4.set_xlabel('Predicted', color=TEXT, fontsize=9)
ax4.set_ylabel('Actual', color=TEXT, fontsize=9)
ax4.set_title('Confusion Matrix (Test)', color=TEXT, fontsize=11, pad=8)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax4.text(j, i, str(cm[i, j]), ha='center', va='center',
                 color='white' if cm[i, j] > cm.max()/2 else '#333', fontsize=9, fontweight='bold')
ax4.tick_params(colors=TEXT)
for sp in ax4.spines.values(): sp.set_color('#2D3148')

# ── Row 3b: Cross-val F1 per fold ───────────────────────────────
ax5 = fig.add_subplot(3, 3, 5)
ax5.set_facecolor(PANEL)
folds = range(1, 6)
tr_f1_cv = cv_results['train_f1_weighted']
te_f1_cv = cv_results['test_f1_weighted']
ax5.plot(folds, tr_f1_cv, 'o-', color=BLUE,  linewidth=2, markersize=7, label='Train F1')
ax5.plot(folds, te_f1_cv, 's-', color=GREEN, linewidth=2, markersize=7, label='Val F1')
ax5.fill_between(folds, tr_f1_cv, te_f1_cv, alpha=0.12, color=AMBER)
ax5.axhline(tr_f1_cv.mean(), color=BLUE,  linestyle='--', linewidth=1.2, alpha=0.7)
ax5.axhline(te_f1_cv.mean(), color=GREEN, linestyle='--', linewidth=1.2, alpha=0.7)
ax5.set_xticks(list(folds)); ax5.set_xticklabels([f'Fold {f}' for f in folds], color=TEXT, fontsize=8.5)
ax5.set_title('Cross-Validation F1 (5-Fold)', color=TEXT, fontsize=11, pad=8)
ax5.tick_params(colors=TEXT, labelsize=8.5)
ax5.legend(facecolor=PANEL, edgecolor='#2D3148', labelcolor=TEXT, fontsize=8.5)
for sp in ax5.spines.values(): sp.set_color('#2D3148')
ax5.set_ylabel('Weighted F1', color=TEXT, fontsize=9)
ax5.set_ylim(0, 1.05)

# ── Row 3c: Precision / Recall / F1 per class ───────────────────
ax6 = fig.add_subplot(3, 3, 6)
ax6.set_facecolor(PANEL)
report = classification_report(y_test, y_test_pred,
                                target_names=le.classes_,
                                output_dict=True, zero_division=0)
classes = le.classes_
prec_vals = [report[c]['precision'] for c in classes]
rec_vals  = [report[c]['recall']    for c in classes]
f1_vals   = [report[c]['f1-score']  for c in classes]
x = np.arange(len(classes)); w = 0.25
ax6.bar(x - w, prec_vals, w, label='Precision', color=BLUE,   edgecolor='none')
ax6.bar(x,     rec_vals,  w, label='Recall',    color=GREEN,  edgecolor='none')
ax6.bar(x + w, f1_vals,   w, label='F1-Score',  color=AMBER,  edgecolor='none')
ax6.set_xticks(x); ax6.set_xticklabels(classes, color=TEXT, fontsize=8.5)
ax6.set_title('Per-Class Metrics (Test)', color=TEXT, fontsize=11, pad=8)
ax6.tick_params(colors=TEXT, labelsize=8.5)
ax6.legend(facecolor=PANEL, edgecolor='#2D3148', labelcolor=TEXT, fontsize=8.5)
for sp in ax6.spines.values(): sp.set_color('#2D3148')
ax6.set_ylabel('Score', color=TEXT, fontsize=9)
ax6.set_ylim(0, 1.05)

# ── Row 4: CV All Metrics heatmap ───────────────────────────────
ax7 = fig.add_subplot(3, 1, 3)
ax7.set_facecolor(PANEL)
cv_metrics = {
    'Accuracy'  : cv_results['test_accuracy'],
    'Precision' : cv_results['test_precision_weighted'],
    'Recall'    : cv_results['test_recall_weighted'],
    'F1-Score'  : cv_results['test_f1_weighted'],
}
cv_matrix = np.array(list(cv_metrics.values()))
im2 = ax7.imshow(cv_matrix, cmap='YlGnBu', aspect='auto', vmin=0, vmax=1)
ax7.set_yticks(range(4)); ax7.set_yticklabels(list(cv_metrics.keys()), color=TEXT, fontsize=10)
ax7.set_xticks(range(5)); ax7.set_xticklabels([f'Fold {i+1}' for i in range(5)], color=TEXT, fontsize=9)
ax7.set_title('Cross-Validation Metrics Heatmap (All Folds)', color=TEXT, fontsize=11, pad=8)
for i in range(4):
    for j in range(5):
        ax7.text(j, i, f'{cv_matrix[i, j]:.3f}', ha='center', va='center',
                 color='#111' if cv_matrix[i, j] > 0.6 else TEXT, fontsize=10, fontweight='bold')
ax7.tick_params(colors=TEXT)
for sp in ax7.spines.values(): sp.set_color('#2D3148')

# Mean column
for i, (k, vals) in enumerate(cv_metrics.items()):
    ax7.text(5.35, i, f'μ={vals.mean():.3f}', va='center', color=CYAN,
             fontsize=10, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.90])

dashboard_path = '/mnt/user-data/outputs/dt_dashboard.png'
plt.savefig(dashboard_path, dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()
print(f"\n✅ Dashboard saved → {dashboard_path}")

# ─────────────────────────────────────────────
# 10. SAVE PROBABILITY CSV
# ─────────────────────────────────────────────
prob_df.to_csv('/mnt/user-data/outputs/predicted_probabilities.csv', index=False)
print(f"✅ Probabilities   → /mnt/user-data/outputs/predicted_probabilities.csv")

print(f"\n{'='*65}")
print("  ALL OUTPUTS SAVED SUCCESSFULLY")
print(f"{'='*65}\n")
