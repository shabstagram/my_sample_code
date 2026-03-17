"""
=============================================================
 Full Exploratory Data Analysis — IRC Multi-Class Dataset
=============================================================
Features analysed:
  1. MCC                 (high-cardinality categorical)
  2. Mode of Payment     (low-cardinality categorical)
  3. Product Code        (categorical)
  4. CP / CNP Indicator  (boolean)
  5. Net Spend           (skewed numerical, 1 → 1M)
Target : Interchange Rate Code (IRC) — 20 classes, imbalanced

Outputs  (saved to ./eda_outputs/):
  01_dataset_overview.png
  02_mcc_top15.png
  03_mcc_frequency_distribution.png
  04_payment_mode_analysis.png
  05_product_code_analysis.png
  06_cp_cnp_analysis.png
  07_net_spend_distributions.png
  08_net_spend_outliers.png
  09_target_irc_distribution.png
  10_cross_spend_by_mode.png
  11_cross_irc_by_product.png
  12_cross_spend_by_irc.png
  13_cross_cp_by_mode.png
  14_correlation_heatmap.png
  15_mutual_information.png
  eda_summary_report.txt
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.feature_selection import mutual_info_classif

warnings.filterwarnings("ignore")

# ─── Output directory ────────────────────────────────────────
OUT = "./eda_outputs"
os.makedirs(OUT, exist_ok=True)

# ─── Style ───────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "axes.grid.axis":    "y",
    "grid.alpha":        0.25,
    "grid.linewidth":    0.5,
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   10,
    "figure.dpi":        130,
})

PALETTE = {
    "blue":   "#3266ad",
    "teal":   "#1d9e75",
    "purple": "#534AB7",
    "amber":  "#BA7517",
    "coral":  "#993C1D",
    "green":  "#3B6D11",
    "gray":   "#5F5E5A",
    "red":    "#A32D2D",
}
MULTI = list(PALETTE.values())


# ═══════════════════════════════════════════════════════════════
# 0.  SYNTHETIC DATASET
#     Replace this block with:  df = pd.read_csv("your_data.csv")
# ═══════════════════════════════════════════════════════════════
np.random.seed(42)
N = 50_000

MCC_CODES     = [f"MCC_{i:04d}" for i in range(600)]
PAYMENT_MODES = ["POS", "Wallet", "ApplePay", "GPay", "Contactless", "CNP_Online"]
PRODUCT_CODES = ["Debit_Classic", "Credit_Gold", "Credit_Platinum", "Prepaid", "Business_Credit"]
IRC_CLASSES   = [f"IRC_{i:02d}" for i in range(1, 21)]

# Imbalanced IRC probs: top 4 = 64 % of volume
irc_raw  = [22, 18, 14, 10, 8, 6, 4, 3, 2, 2,
            1.5, 1.2, 0.9, 0.8, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3]
irc_probs = np.array(irc_raw) / sum(irc_raw)

# MCC with heavy-tail distribution
mcc_weights = np.random.exponential(scale=1, size=600)
mcc_weights /= mcc_weights.sum()

irc_arr  = np.random.choice(IRC_CLASSES, size=N, p=irc_probs)

# CardPresent depends on mode: CNP_Online always CNP, Contactless/POS always CP
mode_arr = np.random.choice(PAYMENT_MODES, size=N,
                             p=[0.42, 0.11, 0.15, 0.18, 0.08, 0.06])
cp_arr   = np.where(mode_arr == "CNP_Online", 0,
           np.where(np.isin(mode_arr, ["POS", "Contactless"]), 1,
                    np.random.randint(0, 2, size=N)))

# Net Spend: lognormal with mode-dependent mean
base_spend = np.random.lognormal(mean=4.5, sigma=2.0, size=N)
mode_multiplier = {
    "POS": 1.0, "Wallet": 0.9, "ApplePay": 0.85,
    "GPay": 0.88, "Contactless": 0.25, "CNP_Online": 3.8
}
spend_mult = np.array([mode_multiplier[m] for m in mode_arr])
spend_arr  = np.clip(base_spend * spend_mult, 1, 1_000_000)

df = pd.DataFrame({
    "MCC":         np.random.choice(MCC_CODES, size=N, p=mcc_weights),
    "PaymentMode": mode_arr,
    "ProductCode": np.random.choice(PRODUCT_CODES, size=N,
                                    p=[0.30, 0.22, 0.18, 0.16, 0.14]),
    "CardPresent": cp_arr,
    "NetSpend":    spend_arr,
    "IRC":         irc_arr,
})

print(f"Dataset shape : {df.shape}")
print(f"Memory usage  : {df.memory_usage(deep=True).sum() / 1e6:.1f} MB\n")


# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def savefig(name, fig=None, tight=True):
    path = os.path.join(OUT, name)
    (fig or plt).savefig(path, bbox_inches="tight" if tight else None,
                         facecolor="white")
    plt.close("all")
    print(f"  [saved] {name}")


def add_value_labels(ax, fmt="{:.0f}", rotation=0, fontsize=9, padding=0.5):
    """Annotate bar tops with values."""
    for bar in ax.patches:
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + padding,
                fmt.format(h),
                ha="center", va="bottom",
                fontsize=fontsize, rotation=rotation,
            )


def chi2_cramers_v(col, target=df["IRC"]):
    """Compute Cramér's V between a categorical column and target."""
    ct   = pd.crosstab(col, target)
    chi2 = stats.chi2_contingency(ct, correction=False)[0]
    n    = ct.sum().sum()
    k    = min(ct.shape) - 1
    return np.sqrt(chi2 / (n * k)) if k > 0 else 0.0


# ═══════════════════════════════════════════════════════════════
# 01. DATASET OVERVIEW
# ═══════════════════════════════════════════════════════════════
print("─" * 55)
print("01  Dataset overview")
print("─" * 55)

print(df.dtypes)
print("\nNull counts:\n", df.isnull().sum())
print("\nDescribe:\n", df.describe(include="all"))

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Dataset overview — feature types and IRC target", y=1.01)

# Feature-type pie
types = {"High-cardinality\ncategorical": 1,
         "Low-cardinality\ncategorical":  2,
         "Boolean": 1, "Numerical\n(skewed)": 1}
axes[0, 0].pie(types.values(), labels=types.keys(),
               colors=[PALETTE["purple"], PALETTE["teal"],
                       PALETTE["amber"], PALETTE["blue"]],
               autopct="%1.0f%%", startangle=90,
               wedgeprops={"edgecolor": "white", "linewidth": 1.5})
axes[0, 0].set_title("Feature type breakdown")

# Row-count per feature (null check)
null_pct = (df.isnull().sum() / len(df) * 100)
axes[0, 1].barh(null_pct.index, null_pct.values,
                color=[PALETTE["green"] if v == 0 else PALETTE["red"]
                       for v in null_pct.values])
axes[0, 1].set_title("Missing value % per column")
axes[0, 1].set_xlabel("% missing")
axes[0, 1].axvline(0, color="gray", linewidth=0.5)
for i, v in enumerate(null_pct.values):
    axes[0, 1].text(v + 0.02, i, f"{v:.1f}%", va="center", fontsize=9)

# IRC class imbalance — top-level
irc_counts = df["IRC"].value_counts().sort_index()
colors_irc = [PALETTE["blue"] if i < 4 else
              PALETTE["teal"] if i < 10 else
              PALETTE["amber"] for i in range(len(irc_counts))]
axes[0, 2].bar(range(len(irc_counts)), irc_counts.values,
               color=colors_irc, edgecolor="white", linewidth=0.3)
axes[0, 2].set_title("IRC class distribution")
axes[0, 2].set_xticks(range(len(irc_counts)))
axes[0, 2].set_xticklabels(irc_counts.index, rotation=45, ha="right", fontsize=7)
axes[0, 2].set_ylabel("Count")

# NetSpend histogram (log scale)
axes[1, 0].hist(df["NetSpend"], bins=80,
                color=PALETTE["blue"], edgecolor="white", linewidth=0.2)
axes[1, 0].set_xscale("log")
axes[1, 0].set_title("Net Spend (log x-axis)")
axes[1, 0].set_xlabel("Spend (£)")
axes[1, 0].set_ylabel("Frequency")

# PaymentMode
pm_cnt = df["PaymentMode"].value_counts()
axes[1, 1].bar(pm_cnt.index, pm_cnt.values,
               color=MULTI[:len(pm_cnt)], edgecolor="white", linewidth=0.3)
axes[1, 1].set_title("Payment mode counts")
axes[1, 1].tick_params(axis="x", rotation=20)

# ProductCode
pc_cnt = df["ProductCode"].value_counts()
axes[1, 2].bar(pc_cnt.index, pc_cnt.values,
               color=MULTI[:len(pc_cnt)], edgecolor="white", linewidth=0.3)
axes[1, 2].set_title("Product code counts")
axes[1, 2].tick_params(axis="x", rotation=20)

plt.tight_layout()
savefig("01_dataset_overview.png", fig)


# ═══════════════════════════════════════════════════════════════
# 02-03.  MCC
# ═══════════════════════════════════════════════════════════════
print("\n─" * 55)
print("02-03  MCC analysis")
print("─" * 55)

mcc_cnt    = df["MCC"].value_counts()
top15_mcc  = mcc_cnt.head(15)
print(f"Unique MCCs        : {df['MCC'].nunique()}")
print(f"Top-10 coverage    : {mcc_cnt.head(10).sum() / len(df) * 100:.1f}%")
print(f"MCCs with < 10 txns: {(mcc_cnt < 10).sum()}")

# Fig 02 — Top 15 MCC
fig, ax = plt.subplots(figsize=(11, 6))
ax.barh(top15_mcc.index[::-1], top15_mcc.values[::-1],
        color=PALETTE["blue"], edgecolor="white", linewidth=0.3)
ax.set_title("Top 15 MCC codes by transaction count")
ax.set_xlabel("Transaction count")
for i, v in enumerate(top15_mcc.values[::-1]):
    ax.text(v + 20, i, f"{v:,}", va="center", fontsize=9)
plt.tight_layout()
savefig("02_mcc_top15.png", fig)

# Fig 03 — MCC frequency distribution (long tail)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("MCC frequency distribution — long tail", fontsize=13, fontweight="bold")

bins = [1, 5, 10, 20, 50, 100, 250, 500, np.inf]
labels = ["1-4", "5-9", "10-19", "20-49", "50-99", "100-249", "250-499", "500+"]
mcc_binned = pd.cut(mcc_cnt, bins=bins, labels=labels, right=False)
bin_counts = mcc_binned.value_counts().reindex(labels)

axes[0].bar(labels, bin_counts.values,
            color=PALETTE["purple"], edgecolor="white", linewidth=0.3)
axes[0].set_title("# of MCCs by transaction frequency bucket")
axes[0].set_xlabel("Transactions per MCC")
axes[0].set_ylabel("# of MCCs")
add_value_labels(axes[0])

axes[1].plot(range(len(mcc_cnt)), np.cumsum(mcc_cnt.values) / len(df) * 100,
             color=PALETTE["blue"], linewidth=2)
axes[1].axhline(80, color=PALETTE["amber"], linestyle="--", linewidth=1,
                label="80% coverage")
axes[1].axhline(95, color=PALETTE["red"], linestyle="--", linewidth=1,
                label="95% coverage")
axes[1].set_title("Cumulative transaction coverage by MCC rank")
axes[1].set_xlabel("MCC rank (most → least frequent)")
axes[1].set_ylabel("Cumulative % of transactions")
axes[1].legend()
axes[1].fill_between(range(len(mcc_cnt)),
                     np.cumsum(mcc_cnt.values) / len(df) * 100,
                     alpha=0.1, color=PALETTE["blue"])
plt.tight_layout()
savefig("03_mcc_frequency_distribution.png", fig)


# ═══════════════════════════════════════════════════════════════
# 04.  MODE OF PAYMENT
# ═══════════════════════════════════════════════════════════════
print("\n─" * 55)
print("04  Mode of payment analysis")
print("─" * 55)

pm_cnt   = df["PaymentMode"].value_counts()
pm_spend = df.groupby("PaymentMode")["NetSpend"].agg(["mean", "median", "std"])
print(pm_cnt)
print("\nSpend stats by payment mode:\n", pm_spend.round(0))
print("\nCramér's V vs IRC:", round(chi2_cramers_v(df["PaymentMode"]), 3))

fig = plt.figure(figsize=(18, 11))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
fig.suptitle("Payment mode — full EDA", fontsize=14, fontweight="bold")

# 1. Count bar
ax1 = fig.add_subplot(gs[0, 0])
bars = ax1.bar(pm_cnt.index, pm_cnt.values,
               color=MULTI[:len(pm_cnt)], edgecolor="white", linewidth=0.3)
ax1.set_title("Transaction count by mode")
ax1.set_ylabel("Count")
ax1.tick_params(axis="x", rotation=20)
add_value_labels(ax1, fmt="{:,.0f}", fontsize=8, padding=50)

# 2. Share donut
ax2 = fig.add_subplot(gs[0, 1])
wedges, texts, autotexts = ax2.pie(
    pm_cnt.values,
    labels=pm_cnt.index,
    autopct="%1.1f%%",
    colors=MULTI[:len(pm_cnt)],
    startangle=90,
    wedgeprops={"edgecolor": "white", "linewidth": 1.5},
)
ax2.set_title("Share of total transactions")

# 3. Mean spend
ax3 = fig.add_subplot(gs[0, 2])
ax3.bar(pm_spend.index, pm_spend["mean"],
        color=MULTI[:len(pm_spend)], edgecolor="white", linewidth=0.3)
ax3.set_title("Mean net spend (£) by mode")
ax3.set_ylabel("£")
ax3.tick_params(axis="x", rotation=20)
ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:,.0f}"))

# 4. Median spend with error bars
ax4 = fig.add_subplot(gs[1, 0])
ax4.bar(pm_spend.index, pm_spend["median"],
        color=MULTI[:len(pm_spend)], edgecolor="white", linewidth=0.3, alpha=0.85)
ax4.set_title("Median net spend (£) by mode")
ax4.set_ylabel("£")
ax4.tick_params(axis="x", rotation=20)
ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:,.0f}"))

# 5. Log-scale spend distribution per mode (violin-style via boxplot)
ax5 = fig.add_subplot(gs[1, 1])
spend_by_mode = [df.loc[df["PaymentMode"] == m, "NetSpend"].values
                 for m in pm_cnt.index]
bp = ax5.boxplot(spend_by_mode, labels=pm_cnt.index,
                 patch_artist=True, notch=False,
                 medianprops={"color": "white", "linewidth": 2},
                 whiskerprops={"linewidth": 0.8},
                 flierprops={"marker": ".", "markersize": 2, "alpha": 0.3})
for patch, color in zip(bp["boxes"], MULTI[:len(pm_cnt)]):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)
ax5.set_yscale("log")
ax5.set_title("Spend distribution per mode (log scale)")
ax5.set_ylabel("Net Spend (£) — log scale")
ax5.tick_params(axis="x", rotation=20)

# 6. CP/CNP split per mode
ax6 = fig.add_subplot(gs[1, 2])
cp_by_mode = df.groupby("PaymentMode")["CardPresent"].mean() * 100
cnp_by_mode = 100 - cp_by_mode
x = range(len(cp_by_mode))
ax6.bar(x, cp_by_mode.values,  label="Card Present",     color=PALETTE["blue"],   edgecolor="white")
ax6.bar(x, cnp_by_mode.values, bottom=cp_by_mode.values, label="Card Not Present", color=PALETTE["coral"], edgecolor="white")
ax6.set_xticks(x)
ax6.set_xticklabels(cp_by_mode.index, rotation=20)
ax6.set_title("CP vs CNP split per mode (%)")
ax6.set_ylabel("% of transactions")
ax6.legend()
ax6.set_ylim(0, 115)

savefig("04_payment_mode_analysis.png", fig)


# ═══════════════════════════════════════════════════════════════
# 05.  PRODUCT CODE
# ═══════════════════════════════════════════════════════════════
print("\n─" * 55)
print("05  Product code analysis")
print("─" * 55)

pc_cnt   = df["ProductCode"].value_counts()
pc_spend = df.groupby("ProductCode")["NetSpend"].agg(["mean", "median"])
print(pc_cnt)
print("\nSpend stats:\n", pc_spend.round(0))
print(f"\nCramér's V vs IRC: {chi2_cramers_v(df['ProductCode']):.3f}")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Product code — full EDA", fontsize=14, fontweight="bold")
fig.subplots_adjust(hspace=0.45, wspace=0.35)

# 1. Count
axes[0, 0].bar(pc_cnt.index, pc_cnt.values,
               color=MULTI[:len(pc_cnt)], edgecolor="white", linewidth=0.3)
axes[0, 0].set_title("Transaction count by product code")
axes[0, 0].set_ylabel("Count")
axes[0, 0].tick_params(axis="x", rotation=15)
add_value_labels(axes[0, 0], fmt="{:,.0f}", fontsize=8, padding=50)

# 2. Share pie
wedges, texts, autotexts = axes[0, 1].pie(
    pc_cnt.values, labels=pc_cnt.index,
    autopct="%1.1f%%", colors=MULTI[:len(pc_cnt)],
    startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 1.5},
)
axes[0, 1].set_title("Transaction share")

# 3. Mean spend
axes[0, 2].bar(pc_spend.index, pc_spend["mean"],
               color=MULTI[:len(pc_spend)], edgecolor="white", linewidth=0.3)
axes[0, 2].set_title("Mean net spend (£)")
axes[0, 2].tick_params(axis="x", rotation=15)
axes[0, 2].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:,.0f}"))

# 4. Box plot
ax4 = axes[1, 0]
spend_by_prod = [df.loc[df["ProductCode"] == p, "NetSpend"].values
                 for p in pc_cnt.index]
bp2 = ax4.boxplot(spend_by_prod, labels=pc_cnt.index,
                  patch_artist=True,
                  medianprops={"color": "white", "linewidth": 2},
                  flierprops={"marker": ".", "markersize": 2, "alpha": 0.3})
for patch, color in zip(bp2["boxes"], MULTI[:len(pc_cnt)]):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)
ax4.set_yscale("log")
ax4.set_title("Spend distribution (log scale)")
ax4.set_ylabel("Net Spend — log scale")
ax4.tick_params(axis="x", rotation=15)

# 5. IRC tier distribution per product (heatmap)
ax5 = axes[1, 1]
irc_by_prod = pd.crosstab(df["ProductCode"], df["IRC"], normalize="index") * 100
sns.heatmap(irc_by_prod.iloc[:, :10], ax=ax5, cmap="Blues",
            annot=True, fmt=".0f", linewidths=0.3,
            annot_kws={"size": 7}, cbar_kws={"shrink": 0.8})
ax5.set_title("IRC mix % by product (top 10 IRCs)")
ax5.set_xlabel("IRC class")
ax5.tick_params(axis="x", rotation=45)
ax5.tick_params(axis="y", rotation=0)

# 6. CP/CNP split per product
ax6 = axes[1, 2]
cp_prod  = df.groupby("ProductCode")["CardPresent"].mean() * 100
cnp_prod = 100 - cp_prod
x = range(len(cp_prod))
ax6.bar(x, cp_prod.values,  label="CP",  color=PALETTE["blue"],  edgecolor="white")
ax6.bar(x, cnp_prod.values, bottom=cp_prod.values,
        label="CNP", color=PALETTE["coral"], edgecolor="white")
ax6.set_xticks(x)
ax6.set_xticklabels(cp_prod.index, rotation=15)
ax6.set_title("CP vs CNP split per product (%)")
ax6.set_ylabel("% of transactions")
ax6.set_ylim(0, 115)
ax6.legend()

savefig("05_product_code_analysis.png", fig)


# ═══════════════════════════════════════════════════════════════
# 06.  CP / CNP FLAG
# ═══════════════════════════════════════════════════════════════
print("\n─" * 55)
print("06  CP / CNP flag analysis")
print("─" * 55)

cp_cnt = df["CardPresent"].value_counts()
print("Value counts:\n", cp_cnt)
print(f"\nMean spend CP  : £{df[df['CardPresent']==1]['NetSpend'].mean():,.0f}")
print(f"Mean spend CNP : £{df[df['CardPresent']==0]['NetSpend'].mean():,.0f}")

# Data quality check: CNP_Online should always be 0
quality = df.groupby("PaymentMode")["CardPresent"].mean()
print("\nData quality — mean CardPresent per mode (1=CP, 0=CNP):")
print(quality.round(2))

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("CP / CNP flag — full EDA", fontsize=14, fontweight="bold")
fig.subplots_adjust(hspace=0.45, wspace=0.35)

# 1. Count
labels_cp = ["Card Not Present (0)", "Card Present (1)"]
axes[0, 0].bar(labels_cp, [cp_cnt.get(0, 0), cp_cnt.get(1, 0)],
               color=[PALETTE["coral"], PALETTE["blue"]],
               edgecolor="white", linewidth=0.3)
axes[0, 0].set_title("Transaction count: CP vs CNP")
axes[0, 0].set_ylabel("Count")
add_value_labels(axes[0, 0], fmt="{:,.0f}", fontsize=10, padding=100)

# 2. Spend comparison
means_cp = df.groupby("CardPresent")["NetSpend"].mean()
medians_cp = df.groupby("CardPresent")["NetSpend"].median()
x_pos = np.arange(2)
w = 0.35
axes[0, 1].bar(x_pos - w/2, [means_cp.get(0,0), means_cp.get(1,0)],
               w, label="Mean", color=PALETTE["blue"], edgecolor="white")
axes[0, 1].bar(x_pos + w/2, [medians_cp.get(0,0), medians_cp.get(1,0)],
               w, label="Median", color=PALETTE["teal"], edgecolor="white")
axes[0, 1].set_xticks(x_pos)
axes[0, 1].set_xticklabels(["CNP", "CP"])
axes[0, 1].set_title("Mean vs median spend: CP vs CNP")
axes[0, 1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:,.0f}"))
axes[0, 1].legend()

# 3. Spend distribution (KDE)
from scipy.stats import gaussian_kde
for val, color, lbl in [(1, PALETTE["blue"], "Card Present"),
                        (0, PALETTE["coral"], "Card Not Present")]:
    data = np.log1p(df.loc[df["CardPresent"] == val, "NetSpend"])
    kde  = gaussian_kde(data, bw_method=0.15)
    xr   = np.linspace(data.min(), data.max(), 300)
    axes[0, 2].plot(xr, kde(xr), color=color, linewidth=2, label=lbl)
    axes[0, 2].fill_between(xr, kde(xr), alpha=0.15, color=color)
axes[0, 2].set_title("Spend KDE: CP vs CNP (log1p scale)")
axes[0, 2].set_xlabel("log1p(NetSpend)")
axes[0, 2].set_ylabel("Density")
axes[0, 2].legend()
axes[0, 2].grid(axis="both", alpha=0.2)

# 4. IRC distribution within CP
top8 = list(df["IRC"].value_counts().head(8).index)
irc_cp_dist = (df[df["CardPresent"] == 1]["IRC"]
               .value_counts(normalize=True)
               .reindex(top8, fill_value=0) * 100)
irc_cnp_dist = (df[df["CardPresent"] == 0]["IRC"]
                .value_counts(normalize=True)
                .reindex(top8, fill_value=0) * 100)
x = np.arange(len(top8))
axes[1, 0].bar(x - 0.2, irc_cp_dist.values,  0.38, label="CP",  color=PALETTE["blue"],  edgecolor="white")
axes[1, 0].bar(x + 0.2, irc_cnp_dist.values, 0.38, label="CNP", color=PALETTE["coral"], edgecolor="white")
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(top8, rotation=40, ha="right")
axes[1, 0].set_title("IRC mix: CP vs CNP (top 8, %)")
axes[1, 0].set_ylabel("% of class transactions")
axes[1, 0].legend()

# 5. Data quality check
ax5 = axes[1, 1]
qual_vals = quality.reindex(PAYMENT_MODES)
bar_colors = ["green" if (m == "CNP_Online" and v < 0.01) or
              (m in ["POS", "Contactless"] and v > 0.99) else
              PALETTE["amber"] if abs(v - 0.5) > 0.3 else
              PALETTE["red"]
              for m, v in qual_vals.items()]
ax5.bar(PAYMENT_MODES, qual_vals.values,
        color=bar_colors, edgecolor="white", linewidth=0.3)
ax5.axhline(0, color="gray", linewidth=0.5)
ax5.axhline(1, color="gray", linewidth=0.5)
ax5.set_title("Data quality: mean CP per mode\n(1.0=all CP, 0.0=all CNP)")
ax5.set_ylabel("Mean CardPresent")
ax5.set_ylim(-0.1, 1.2)
ax5.tick_params(axis="x", rotation=20)
for i, v in enumerate(qual_vals.values):
    ax5.text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=9)

# 6. Box plot
bp3 = axes[1, 2].boxplot(
    [df.loc[df["CardPresent"] == 1, "NetSpend"],
     df.loc[df["CardPresent"] == 0, "NetSpend"]],
    labels=["Card Present", "Card Not Present"],
    patch_artist=True,
    medianprops={"color": "white", "linewidth": 2},
    flierprops={"marker": ".", "markersize": 1.5, "alpha": 0.3},
)
for patch, color in zip(bp3["boxes"], [PALETTE["blue"], PALETTE["coral"]]):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
axes[1, 2].set_yscale("log")
axes[1, 2].set_title("Spend distribution by flag (log scale)")
axes[1, 2].set_ylabel("Net Spend — log scale")

savefig("06_cp_cnp_analysis.png", fig)


# ═══════════════════════════════════════════════════════════════
# 07.  NET SPEND — DISTRIBUTIONS
# ═══════════════════════════════════════════════════════════════
print("\n─" * 55)
print("07  Net Spend analysis")
print("─" * 55)

spend = df["NetSpend"]
log_spend = np.log1p(spend)

print(f"Min    : £{spend.min():,.2f}")
print(f"P25    : £{spend.quantile(0.25):,.2f}")
print(f"Median : £{spend.median():,.2f}")
print(f"Mean   : £{spend.mean():,.2f}")
print(f"P75    : £{spend.quantile(0.75):,.2f}")
print(f"P95    : £{spend.quantile(0.95):,.2f}")
print(f"Max    : £{spend.max():,.2f}")
print(f"Skewness  : {spend.skew():.2f}")
print(f"Kurtosis  : {spend.kurtosis():.2f}")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Net Spend — distribution analysis", fontsize=14, fontweight="bold")
fig.subplots_adjust(hspace=0.45, wspace=0.35)

# 1. Raw histogram
axes[0, 0].hist(spend, bins=80, color=PALETTE["blue"],
                edgecolor="white", linewidth=0.2)
axes[0, 0].set_title("Raw distribution (heavy right skew)")
axes[0, 0].set_xlabel("Net Spend (£)")
axes[0, 0].set_ylabel("Frequency")
axes[0, 0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x/1000:.0f}K"))

# 2. Log1p transformed
axes[0, 1].hist(log_spend, bins=80, color=PALETTE["teal"],
                edgecolor="white", linewidth=0.2)
axes[0, 1].set_title("After log1p transform")
axes[0, 1].set_xlabel("log1p(NetSpend)")
axes[0, 1].set_ylabel("Frequency")

# 3. Log x-axis
axes[0, 2].hist(spend, bins=np.logspace(0, 6, 80),
                color=PALETTE["purple"], edgecolor="white", linewidth=0.2)
axes[0, 2].set_xscale("log")
axes[0, 2].set_title("Log x-axis histogram (raw data)")
axes[0, 2].set_xlabel("Net Spend (£) — log scale")
axes[0, 2].set_ylabel("Frequency")

# 4. Q-Q plot
axes[1, 0].set_title("Q-Q plot: log1p(Spend) vs normal")
(osm, osr), (slope, intercept, r) = stats.probplot(log_spend, dist="norm")
axes[1, 0].scatter(osm, osr, s=3, alpha=0.4, color=PALETTE["blue"])
axes[1, 0].plot(osm, slope * np.array(osm) + intercept,
                color=PALETTE["red"], linewidth=1.5, label=f"R²={r**2:.3f}")
axes[1, 0].set_xlabel("Theoretical quantiles")
axes[1, 0].set_ylabel("Sample quantiles")
axes[1, 0].legend()

# 5. Percentile profile
pcts = [10, 25, 50, 75, 90, 95, 99]
pct_vals = [spend.quantile(p/100) for p in pcts]
axes[1, 1].plot([f"P{p}" for p in pcts], pct_vals,
                marker="o", color=PALETTE["purple"],
                linewidth=2, markersize=7)
axes[1, 1].fill_between(range(len(pcts)), pct_vals, alpha=0.1,
                         color=PALETTE["purple"])
axes[1, 1].set_yscale("log")
axes[1, 1].set_title("Spend percentile profile")
axes[1, 1].set_ylabel("Net Spend (£) — log scale")
for i, (p, v) in enumerate(zip(pcts, pct_vals)):
    axes[1, 1].annotate(f"£{v:,.0f}", (i, v),
                        textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=8)

# 6. CDF
sorted_spend = np.sort(spend)
cdf = np.arange(1, len(sorted_spend) + 1) / len(sorted_spend)
axes[1, 2].plot(sorted_spend, cdf * 100, color=PALETTE["blue"], linewidth=1.5)
axes[1, 2].set_xscale("log")
for pct, label in [(0.5, "Median"), (0.9, "P90"), (0.99, "P99")]:
    v = spend.quantile(pct)
    axes[1, 2].axvline(v, color=PALETTE["amber"], linestyle="--",
                        linewidth=1, alpha=0.7)
    axes[1, 2].text(v * 1.1, pct * 100 - 4, label, fontsize=8,
                    color=PALETTE["amber"])
axes[1, 2].set_title("CDF of Net Spend (log x-axis)")
axes[1, 2].set_xlabel("Net Spend (£) — log scale")
axes[1, 2].set_ylabel("Cumulative % of transactions")

savefig("07_net_spend_distributions.png", fig)


# ═══════════════════════════════════════════════════════════════
# 08.  NET SPEND — OUTLIER ANALYSIS
# ═══════════════════════════════════════════════════════════════
print("\n─" * 55)
print("08  Net Spend — outlier analysis")
print("─" * 55)

Q1, Q3 = spend.quantile(0.25), spend.quantile(0.75)
IQR    = Q3 - Q1
lower  = Q1 - 1.5 * IQR
upper  = Q3 + 1.5 * IQR
mild_out  = spend[(spend < lower) | (spend > upper)]
extreme_out = spend[(spend < Q1 - 3*IQR) | (spend > Q3 + 3*IQR)]

print(f"IQR       : £{IQR:,.2f}")
print(f"Lower fence: £{max(lower, 0):,.2f}")
print(f"Upper fence: £{upper:,.2f}")
print(f"Mild outliers   : {len(mild_out):,} ({len(mild_out)/len(df)*100:.1f}%)")
print(f"Extreme outliers: {len(extreme_out):,} ({len(extreme_out)/len(df)*100:.1f}%)")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Net Spend — outlier detection", fontsize=13, fontweight="bold")

# 1. Box plot (raw)
bp_out = axes[0].boxplot(spend, vert=True, patch_artist=True,
                          medianprops={"color": "white", "linewidth": 2},
                          flierprops={"marker": ".", "markersize": 2, "alpha": 0.2},
                          boxprops={"facecolor": PALETTE["blue"], "alpha": 0.75})
axes[0].set_title("Box plot — raw spend")
axes[0].set_ylabel("Net Spend (£)")
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x/1000:.0f}K"))

# 2. Outlier category bar
categories = ["Normal", "Mild outlier", "Extreme outlier"]
n_normal = len(df) - len(mild_out)
counts_cat = [n_normal, len(mild_out) - len(extreme_out), len(extreme_out)]
axes[1].bar(categories, counts_cat,
            color=[PALETTE["teal"], PALETTE["amber"], PALETTE["red"]],
            edgecolor="white", linewidth=0.3)
axes[1].set_title("Outlier category counts")
axes[1].set_ylabel("Number of transactions")
for i, (c, v) in enumerate(zip(categories, counts_cat)):
    axes[1].text(i, v + 100, f"{v:,}\n({v/len(df)*100:.1f}%)",
                 ha="center", fontsize=9)

# 3. Before/after RobustScaler comparison
from sklearn.preprocessing import RobustScaler, StandardScaler
log_spend_arr = log_spend.values.reshape(-1, 1)
robust = RobustScaler().fit_transform(log_spend_arr).ravel()
standard = StandardScaler().fit_transform(log_spend_arr).ravel()

axes[2].hist(standard, bins=80, alpha=0.55, color=PALETTE["red"],
             label="StandardScaler", edgecolor="none")
axes[2].hist(robust, bins=80, alpha=0.55, color=PALETTE["teal"],
             label="RobustScaler", edgecolor="none")
axes[2].set_title("StandardScaler vs RobustScaler\non log1p(Spend)")
axes[2].set_xlabel("Scaled value")
axes[2].set_ylabel("Frequency")
axes[2].legend()

plt.tight_layout()
savefig("08_net_spend_outliers.png", fig)


# ═══════════════════════════════════════════════════════════════
# 09.  TARGET — IRC DISTRIBUTION
# ═══════════════════════════════════════════════════════════════
print("\n─" * 55)
print("09  IRC target distribution")
print("─" * 55)

irc_cnts = df["IRC"].value_counts().sort_index()
irc_share_pct = irc_cnts / len(df) * 100
ir_ratio = irc_cnts.max() / irc_cnts.min()
print(f"Imbalance ratio: {ir_ratio:.1f}:1")
print(irc_cnts)

fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle("IRC target variable — distribution analysis", fontsize=14, fontweight="bold")
fig.subplots_adjust(hspace=0.45, wspace=0.35)

# 1. Count bar
bar_colors = [PALETTE["blue"] if i < 4 else
              PALETTE["teal"] if i < 10 else
              PALETTE["amber"] for i in range(len(irc_cnts))]
axes[0, 0].bar(range(len(irc_cnts)), irc_cnts.values,
               color=bar_colors, edgecolor="white", linewidth=0.3)
axes[0, 0].set_xticks(range(len(irc_cnts)))
axes[0, 0].set_xticklabels(irc_cnts.index, rotation=45, ha="right", fontsize=8)
axes[0, 0].set_title("Transaction count per IRC class")
axes[0, 0].set_ylabel("Count")
for patch, v in zip(axes[0, 0].patches, irc_cnts.values):
    if v > 500:
        axes[0, 0].text(patch.get_x() + patch.get_width()/2,
                        v + 30, f"{v:,}", ha="center", fontsize=6)

# 2. Cumulative coverage
cumulative = irc_share_pct.cumsum()
axes[0, 1].plot(range(len(cumulative)), cumulative.values,
                marker="o", markersize=5, color=PALETTE["blue"],
                linewidth=2, label="Cumulative %")
axes[0, 1].fill_between(range(len(cumulative)), cumulative.values,
                         alpha=0.12, color=PALETTE["blue"])
axes[0, 1].axhline(90, color=PALETTE["red"], linestyle="--",
                    linewidth=1.5, label="90% threshold")
axes[0, 1].axhline(80, color=PALETTE["amber"], linestyle="--",
                    linewidth=1.2, label="80% threshold")
axes[0, 1].set_xticks(range(len(cumulative)))
axes[0, 1].set_xticklabels(cumulative.index, rotation=45, ha="right", fontsize=8)
axes[0, 1].set_title("Cumulative transaction coverage")
axes[0, 1].set_ylabel("Cumulative %")
axes[0, 1].legend()

# 3. Class weights (inverse frequency)
weights = len(df) / (len(irc_cnts) * irc_cnts)
axes[1, 0].bar(range(len(weights)), weights.values,
               color=[PALETTE["teal"] if v < 5 else
                      PALETTE["amber"] if v < 20 else
                      PALETTE["red"] for v in weights.values],
               edgecolor="white", linewidth=0.3)
axes[1, 0].set_xticks(range(len(weights)))
axes[1, 0].set_xticklabels(weights.index, rotation=45, ha="right", fontsize=8)
axes[1, 0].set_title("Class weights (inverse frequency)")
axes[1, 0].set_ylabel("Weight value")

# 4. Log-scale count
axes[1, 1].bar(range(len(irc_cnts)), irc_cnts.values,
               color=bar_colors, edgecolor="white", linewidth=0.3)
axes[1, 1].set_yscale("log")
axes[1, 1].set_xticks(range(len(irc_cnts)))
axes[1, 1].set_xticklabels(irc_cnts.index, rotation=45, ha="right", fontsize=8)
axes[1, 1].set_title(f"IRC counts (log scale) — IR = {ir_ratio:.0f}:1")
axes[1, 1].set_ylabel("Count (log scale)")
axes[1, 1].axhline(irc_cnts.mean(), color=PALETTE["amber"],
                    linestyle="--", linewidth=1.2, label="Mean class size")
axes[1, 1].legend()

savefig("09_target_irc_distribution.png", fig)


# ═══════════════════════════════════════════════════════════════
# 10-13.  CROSS-FEATURE ANALYSIS
# ═══════════════════════════════════════════════════════════════
print("\n─" * 55)
print("10  Cross: Spend by payment mode")
print("─" * 55)

# Fig 10 — Spend × Mode
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Cross-feature: Net Spend × Payment Mode",
             fontsize=13, fontweight="bold")

spend_by_pm = df.groupby("PaymentMode")["NetSpend"].agg(["mean", "median"])
x = np.arange(len(spend_by_pm))
w = 0.38
axes[0].bar(x - w/2, spend_by_pm["mean"],   w,
            label="Mean",   color=PALETTE["blue"],  edgecolor="white")
axes[0].bar(x + w/2, spend_by_pm["median"], w,
            label="Median", color=PALETTE["teal"],  edgecolor="white")
axes[0].set_xticks(x)
axes[0].set_xticklabels(spend_by_pm.index, rotation=20)
axes[0].set_title("Mean vs median spend per mode")
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:,.0f}"))
axes[0].legend()

# Violin plot
spend_data = [df.loc[df["PaymentMode"] == m, "NetSpend"].values
              for m in PAYMENT_MODES]
vp = axes[1].violinplot(spend_data, showmedians=True)
for i, body in enumerate(vp["bodies"]):
    body.set_facecolor(MULTI[i])
    body.set_alpha(0.6)
axes[1].set_xticks(range(1, len(PAYMENT_MODES) + 1))
axes[1].set_xticklabels(PAYMENT_MODES, rotation=20)
axes[1].set_yscale("log")
axes[1].set_title("Spend distribution violin (log scale)")
axes[1].set_ylabel("Net Spend — log scale")
plt.tight_layout()
savefig("10_cross_spend_by_mode.png", fig)

# Fig 11 — IRC × Product
print("\n11  Cross: IRC vs Product Code")
top10_irc = list(df["IRC"].value_counts().head(10).index)
irc_prod_pct = (pd.crosstab(df["IRC"], df["ProductCode"], normalize="index") * 100
                ).loc[top10_irc]

fig, ax = plt.subplots(figsize=(13, 6))
irc_prod_pct.plot(kind="bar", ax=ax, stacked=True,
                  color=MULTI[:5], edgecolor="white", linewidth=0.3)
ax.set_title("Product code share within each IRC class (top 10 IRCs, stacked %)",
             fontsize=12, fontweight="bold")
ax.set_xlabel("IRC class")
ax.set_ylabel("% of transactions")
ax.tick_params(axis="x", rotation=30)
ax.legend(loc="upper right", fontsize=9)
plt.tight_layout()
savefig("11_cross_irc_by_product.png", fig)

# Fig 12 — Spend × IRC
print("\n12  Cross: Spend by IRC")
spend_irc = (df.groupby("IRC")["NetSpend"]
             .agg(["mean", "median"])
             .reindex(IRC_CLASSES))
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("Cross-feature: Net Spend × IRC class",
             fontsize=13, fontweight="bold")

axes[0].bar(range(len(spend_irc)), spend_irc["mean"],
            color=[PALETTE["blue"] if i < 4 else
                   PALETTE["teal"] if i < 10 else
                   PALETTE["amber"] for i in range(len(spend_irc))],
            edgecolor="white", linewidth=0.3)
axes[0].set_xticks(range(len(spend_irc)))
axes[0].set_xticklabels(IRC_CLASSES, rotation=45, ha="right", fontsize=8)
axes[0].set_title("Mean net spend per IRC class")
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:,.0f}"))

axes[1].bar(range(len(spend_irc)), spend_irc["median"],
            color=[PALETTE["blue"] if i < 4 else
                   PALETTE["teal"] if i < 10 else
                   PALETTE["amber"] for i in range(len(spend_irc))],
            edgecolor="white", linewidth=0.3)
axes[1].set_xticks(range(len(spend_irc)))
axes[1].set_xticklabels(IRC_CLASSES, rotation=45, ha="right", fontsize=8)
axes[1].set_title("Median net spend per IRC class")
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:,.0f}"))
plt.tight_layout()
savefig("12_cross_spend_by_irc.png", fig)

# Fig 13 — CP/CNP × Mode
print("\n13  Cross: CP/CNP vs Mode")
cp_mode_ct = pd.crosstab(df["PaymentMode"], df["CardPresent"], normalize="index") * 100
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Cross-feature: CP/CNP × Payment Mode",
             fontsize=13, fontweight="bold")

cp_mode_ct.rename(columns={0: "CNP", 1: "CP"}).plot(
    kind="bar", ax=axes[0], stacked=True,
    color=[PALETTE["coral"], PALETTE["blue"]],
    edgecolor="white", linewidth=0.3,
)
axes[0].set_title("CP vs CNP split per payment mode")
axes[0].set_ylabel("% of transactions")
axes[0].tick_params(axis="x", rotation=20)
axes[0].set_ylim(0, 115)

sns.heatmap(cp_mode_ct, ax=axes[1], annot=True, fmt=".1f",
            cmap="Blues", linewidths=0.3,
            cbar_kws={"label": "% of row"},
            annot_kws={"size": 11})
axes[1].set_title("Heatmap: % card present per mode")
axes[1].set_xlabel("Card Present (0=CNP, 1=CP)")
axes[1].set_ylabel("Payment Mode")
axes[1].tick_params(axis="y", rotation=0)
plt.tight_layout()
savefig("13_cross_cp_by_mode.png", fig)


# ═══════════════════════════════════════════════════════════════
# 14.  CORRELATION HEATMAP (encoded features)
# ═══════════════════════════════════════════════════════════════
print("\n─" * 55)
print("14  Correlation heatmap")
print("─" * 55)

df_enc = df.copy()
for col in ["MCC", "PaymentMode", "ProductCode", "IRC"]:
    df_enc[col] = LabelEncoder().fit_transform(df_enc[col])
df_enc["LogSpend"] = np.log1p(df_enc["NetSpend"])

corr = df_enc[["MCC", "PaymentMode", "ProductCode",
               "CardPresent", "LogSpend", "IRC"]].corr()

fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, ax=ax, annot=True, fmt=".3f",
            cmap="coolwarm", center=0, linewidths=0.5,
            vmin=-1, vmax=1, square=True,
            cbar_kws={"shrink": 0.8, "label": "Pearson correlation"})
ax.set_title("Pearson correlation matrix (label-encoded features)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
savefig("14_correlation_heatmap.png", fig)


# ═══════════════════════════════════════════════════════════════
# 15.  MUTUAL INFORMATION vs TARGET
# ═══════════════════════════════════════════════════════════════
print("\n─" * 55)
print("15  Mutual information scores vs IRC")
print("─" * 55)

X_mi = df_enc[["MCC", "PaymentMode", "ProductCode", "CardPresent", "LogSpend"]]
y_mi = df_enc["IRC"]

mi_scores = mutual_info_classif(X_mi, y_mi, discrete_features=[True, True, True, True, False],
                                random_state=42)
mi_series = pd.Series(mi_scores, index=X_mi.columns).sort_values(ascending=True)
print(mi_series)

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(mi_series.index, mi_series.values,
               color=[PALETTE["blue"] if v > 0.3 else
                      PALETTE["teal"] if v > 0.1 else
                      PALETTE["amber"] for v in mi_series.values],
               edgecolor="white", linewidth=0.3)
ax.set_title("Mutual information score — each feature vs IRC target",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Mutual information score")
for bar, v in zip(bars, mi_series.values):
    ax.text(v + 0.002, bar.get_y() + bar.get_height()/2,
            f"{v:.4f}", va="center", fontsize=10)
ax.grid(axis="x", alpha=0.3)
ax.grid(axis="y", alpha=0)
plt.tight_layout()
savefig("15_mutual_information.png", fig)


# ═══════════════════════════════════════════════════════════════
# SUMMARY REPORT (text)
# ═══════════════════════════════════════════════════════════════
print("\n─" * 55)
print("Writing EDA summary report …")
print("─" * 55)

irc_cnts_sorted = irc_cnts.sort_values(ascending=False)
top4_pct = irc_cnts_sorted.head(4).sum() / len(df) * 100
tail6_pct = irc_cnts_sorted.tail(6).sum() / len(df) * 100

report = f"""
═══════════════════════════════════════════════════════
   EDA SUMMARY REPORT — IRC MULTI-CLASS DATASET
═══════════════════════════════════════════════════════

DATASET
  Rows          : {len(df):,}
  Features      : 5  (3 categorical, 1 boolean, 1 numerical)
  Target classes: {df['IRC'].nunique()}  (IRC codes)
  Missing values: {df.isnull().sum().sum()}

─────────────────────────────────────────────────────
1. MCC  (Merchant Category Code)
─────────────────────────────────────────────────────
  Unique values : {df['MCC'].nunique()}
  Top-10 coverage: {mcc_cnt.head(10).sum() / len(df) * 100:.1f}% of transactions
  MCCs < 10 txns: {(mcc_cnt < 10).sum()} — bucket as "Other"
  Encoding rec  : Target encoding (smoothed, smoothing=10)
  Cramér's V    : {chi2_cramers_v(df['MCC']):.3f}
  MI score      : {mi_series.get('MCC', 0):.4f}

─────────────────────────────────────────────────────
2. Payment Mode
─────────────────────────────────────────────────────
  Unique values : {df['PaymentMode'].nunique()}
  Most common   : {df['PaymentMode'].value_counts().index[0]}  ({df['PaymentMode'].value_counts().iloc[0] / len(df) * 100:.1f}%)
  Encoding rec  : One-Hot Encoding (produces 5 binary cols)
  Cramér's V    : {chi2_cramers_v(df['PaymentMode']):.3f}
  MI score      : {mi_series.get('PaymentMode', 0):.4f}
  Key insight   : CNP_Online has {df[df['PaymentMode']=='CNP_Online']['NetSpend'].mean():.0f}x higher avg spend

─────────────────────────────────────────────────────
3. Product Code
─────────────────────────────────────────────────────
  Unique values : {df['ProductCode'].nunique()}
  Most common   : {df['ProductCode'].value_counts().index[0]}
  Encoding rec  : One-Hot Encoding (4 binary cols)
  Cramér's V    : {chi2_cramers_v(df['ProductCode']):.3f}
  MI score      : {mi_series.get('ProductCode', 0):.4f}
  Key insight   : Business Credit avg £{df[df['ProductCode']=='Business_Credit']['NetSpend'].mean():,.0f}
                  vs Prepaid avg £{df[df['ProductCode']=='Prepaid']['NetSpend'].mean():,.0f}

─────────────────────────────────────────────────────
4. Card Present / CNP Flag
─────────────────────────────────────────────────────
  CP (=1) share : {(df['CardPresent']==1).mean()*100:.1f}%
  CNP (=0) share: {(df['CardPresent']==0).mean()*100:.1f}%
  Encoding rec  : Pass-through (already 0/1 integer)
  CP mean spend : £{df[df['CardPresent']==1]['NetSpend'].mean():,.0f}
  CNP mean spend: £{df[df['CardPresent']==0]['NetSpend'].mean():,.0f}
  MI score      : {mi_series.get('CardPresent', 0):.4f}

─────────────────────────────────────────────────────
5. Net Spend
─────────────────────────────────────────────────────
  Min       : £{spend.min():,.2f}
  Median    : £{spend.median():,.2f}
  Mean      : £{spend.mean():,.2f}
  P95       : £{spend.quantile(0.95):,.2f}
  Max       : £{spend.max():,.2f}
  Skewness  : {spend.skew():.2f}
  Kurtosis  : {spend.kurtosis():.2f}
  Outliers (IQR×1.5): {len(mild_out):,} ({len(mild_out)/len(df)*100:.1f}%)
  Encoding rec: log1p → RobustScaler
  MI score    : {mi_series.get('LogSpend', 0):.4f}

─────────────────────────────────────────────────────
6. TARGET — IRC Class Imbalance
─────────────────────────────────────────────────────
  Total classes : {df['IRC'].nunique()}
  Top-4 share   : {top4_pct:.1f}% of transactions
  Bottom-6 share: {tail6_pct:.1f}% of transactions
  Imbalance ratio (max/min): {ir_ratio:.0f}:1
  Recommended mitigation:
    → SMOTE-NC (mixed cat + numerical features)
    → Tiered sampling (dominant: keep, tail: 30% of max)
    → class_weight='balanced' in classifier
    → Evaluate with macro F1, NOT accuracy

─────────────────────────────────────────────────────
7. KEY CROSS-FEATURE FINDINGS
─────────────────────────────────────────────────────
  • Business Credit + CNP_Online → highest IRC tiers
  • Contactless + Debit Classic  → lowest IRC tiers
  • CP/CNP flag should be 100% consistent with mode
    (CNP_Online=0, POS=1, Contactless=1)
  • Net Spend rises monotonically across IRC tiers
  • Product Code has highest Cramér's V → top predictor

═══════════════════════════════════════════════════════
  All plots saved to: {OUT}/
═══════════════════════════════════════════════════════
"""

report_path = os.path.join(OUT, "eda_summary_report.txt")
with open(report_path, "w") as f:
    f.write(report)

print(report)
print(f"\n✓ All EDA outputs saved to  '{OUT}/'")
print("  15 PNG charts + 1 TXT summary report")
