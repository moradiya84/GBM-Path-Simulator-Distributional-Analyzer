"""
Plot strong convergence: log(error) vs log(dt) for Euler and Milstein.
Reads data/convergence.csv and overlays reference slopes.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("data/convergence.csv")

fig, ax = plt.subplots(figsize=(8, 6))

for scheme, group in df.groupby("scheme"):
    group = group.sort_values("dt")
    ax.plot(group["log_dt"], group["log_error"],
            marker="o", linewidth=2, markersize=6, label=scheme)

# Reference slopes
log_dt_range = np.linspace(df["log_dt"].min(), df["log_dt"].max(), 50)

# Slope 0.5 reference (Euler expected)
intercept_euler = df[df["scheme"] == "Euler"]["log_error"].iloc[0] - 0.5 * df[df["scheme"] == "Euler"]["log_dt"].iloc[0]
ax.plot(log_dt_range, 0.5 * log_dt_range + intercept_euler,
        "--", color="gray", alpha=0.6, label="Slope 0.5 (reference)")

# Slope 1.0 reference (Milstein expected)
intercept_milstein = df[df["scheme"] == "Milstein"]["log_error"].iloc[0] - 1.0 * df[df["scheme"] == "Milstein"]["log_dt"].iloc[0]
ax.plot(log_dt_range, 1.0 * log_dt_range + intercept_milstein,
        ":", color="gray", alpha=0.6, label="Slope 1.0 (reference)")

ax.set_xlabel("log(dt)", fontsize=12)
ax.set_ylabel("log(mean error)", fontsize=12)
ax.set_title("Strong Convergence: Euler vs Milstein", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("data/convergence_plot.png", dpi=150)
plt.show()
print("Saved data/convergence_plot.png")
