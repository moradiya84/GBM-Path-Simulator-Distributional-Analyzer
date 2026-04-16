"""
Plot sample GBM paths from data/paths.csv
Generates one subplot per asset showing multiple simulated trajectories.
"""
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/paths.csv")

assets = sorted(df["asset_id"].unique())
paths = sorted(df["path_id"].unique())
num_assets = len(assets)

fig, axes = plt.subplots(1, num_assets, figsize=(6 * num_assets, 5), sharey=False)
if num_assets == 1:
    axes = [axes]

colors = plt.cm.tab10.colors

for idx, asset in enumerate(assets):
    ax = axes[idx]
    asset_df = df[df["asset_id"] == asset]

    for path_id in paths:
        path_df = asset_df[asset_df["path_id"] == path_id]
        ax.plot(path_df["time"], path_df["value"],
                color=colors[path_id % len(colors)],
                alpha=0.7, linewidth=1.2,
                label=f"Path {path_id}")

    ax.set_title(f"Asset {asset}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig.suptitle("Simulated GBM Paths", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig("data/paths_plot.png", dpi=150)
plt.show()
print("Saved data/paths_plot.png")
