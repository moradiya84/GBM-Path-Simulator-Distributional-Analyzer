import json
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Ensure working directory is correctly resolved
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, 'data')

def plot_pnl_distribution():
    print("Loading data for P&L Distribution...")
    try:
        df_pnl = pd.read_csv(os.path.join(data_dir, 'hedging_pnl.csv'))
        with open(os.path.join(data_dir, 'hedging_report.json'), 'r') as f:
            report_data = json.load(f)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return

    plt.figure(figsize=(10, 6))
    
    # Histogram of PnL
    n, bins, patches = plt.hist(df_pnl['terminal_pnl'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    
    mean_pnl = report_data['pnl_summary']['mean_pnl']
    std_pnl = report_data['pnl_summary']['std_pnl']
    
    # Add vertical lines for mean and std
    plt.axvline(mean_pnl, color='red', linestyle='dashed', linewidth=2, label=f'Mean P&L: {mean_pnl:.4f}')
    plt.axvline(mean_pnl + std_pnl, color='green', linestyle='dashed', linewidth=1.5, label=f'+1 Std Dev ({mean_pnl+std_pnl:.4f})')
    plt.axvline(mean_pnl - std_pnl, color='green', linestyle='dashed', linewidth=1.5, label=f'-1 Std Dev ({mean_pnl-std_pnl:.4f})')

    plt.title('Delta Hedging Terminal P&L Distribution (Monte Carlo)', fontsize=14)
    plt.xlabel('Terminal P&L (Cash)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    
    summary_text = (
        f"Paths: {report_data['backtest_settings']['num_paths']}\n"
        f"Steps: {report_data['backtest_settings']['num_rebalancing_steps']}\n"
        f"Option: {report_data['model_inputs']['option_type']}\n"
        f"Implied Vol: {report_data['model_inputs']['implied_volatility']:.2f}\n"
        f"Realized Vol: {report_data['backtest_settings']['realized_vol_sigma']:.2f}"
    )
    plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    
    out_file = os.path.join(data_dir, 'hedging_pnl_distribution.png')
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"Saved P&L Distribution plot to {out_file}")
    plt.close()

def plot_pnl_attribution():
    print("Loading data for P&L Attribution...")
    try:
        df_attr = pd.read_csv(os.path.join(data_dir, 'pnl_attribution.csv'))
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return
        
    # Filter out total_pnl for the waterfall/bar chart
    components = df_attr[df_attr['component'] != 'total_pnl'].copy()
    
    plt.figure(figsize=(10, 6))
    
    # Create simple bar chart for attribution
    colors = ['green' if val > 0 else 'red' for val in components['mean_value']]
    bars = plt.bar(components['component'], components['mean_value'], color=colors, edgecolor='black')
    
    plt.axhline(0, color='black', linewidth=1)
    
    # Add exact labels on top/bottom of bars
    for bar in bars:
        yval = bar.get_height()
        offset = 0.5 if yval > 0 else -1.0
        plt.text(bar.get_x() + bar.get_width()/2, yval + offset, f'{yval:.3f}', ha='center', va='bottom' if yval > 0 else 'top', fontsize=10)

    plt.title('Delta Hedging Expected P&L Attribution (Greek Decomposition)', fontsize=14)
    plt.xlabel('Attribution Component', fontsize=12)
    plt.ylabel('Expected P&L Contribution', fontsize=12)
    plt.xticks(rotation=45)
    
    plt.grid(axis='y', alpha=0.3)
    
    out_file = os.path.join(data_dir, 'hedging_attribution.png')
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"Saved P&L Attribution plot to {out_file}")
    plt.close()

if __name__ == "__main__":
    plot_pnl_distribution()
    plot_pnl_attribution()
