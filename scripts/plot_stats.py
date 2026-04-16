"""
Plot empirical vs theoretical statistics from data/stats.csv.
Generates grouped bar charts for mean and variance comparison,
and standalone bars for skewness and kurtosis.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("data/stats.csv")

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

assets = df["asset_id"].values
x = np.arange(len(assets))
bar_width = 0.35

# --- Mean comparison ---
ax = axes[0, 0]
ax.bar(x - bar_width / 2, df["mean_empirical"], bar_width, label="Empirical", color="#4C72B0")
ax.bar(x + bar_width / 2, df["mean_theoretical"], bar_width, label="Theoretical", color="#DD8452")
ax.set_title("Mean: Empirical vs Theoretical", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([f"Asset {i}" for i in assets])
ax.legend()
ax.grid(axis="y", alpha=0.3)

# --- Variance comparison ---
ax = axes[0, 1]
ax.bar(x - bar_width / 2, df["variance_empirical"], bar_width, label="Empirical", color="#4C72B0")
ax.bar(x + bar_width / 2, df["variance_theoretical"], bar_width, label="Theoretical", color="#DD8452")
ax.set_title("Variance: Empirical vs Theoretical", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([f"Asset {i}" for i in assets])
ax.legend()
ax.grid(axis="y", alpha=0.3)

# --- Skewness ---
ax = axes[1, 0]
ax.bar(x, df["skewness_empirical"], color="#55A868", width=0.5)
ax.set_title("Skewness (Empirical)", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([f"Asset {i}" for i in assets])
ax.axhline(y=0, color="black", linewidth=0.8)
ax.grid(axis="y", alpha=0.3)

# --- Kurtosis ---
ax = axes[1, 1]
ax.bar(x, df["kurtosis_empirical"], color="#C44E52", width=0.5)
ax.set_title("Excess Kurtosis (Empirical)", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([f"Asset {i}" for i in assets])
ax.axhline(y=0, color="black", linewidth=0.8)
ax.grid(axis="y", alpha=0.3)

fig.suptitle("GBM Terminal Distribution Statistics", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig("data/stats_plot.png", dpi=150)
plt.show()
print("Saved data/stats_plot.png")
