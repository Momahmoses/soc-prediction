"""
Exploratory Data Analysis — Battery SOC Prediction Dataset
Generates visual insights saved to plots/
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

os.makedirs('plots', exist_ok=True)
sns.set_theme(style='whitegrid')

df = pd.read_csv('data/soc_dataset.csv')

print("="*55)
print("  SOC PREDICTION — EXPLORATORY DATA ANALYSIS")
print("="*55)
print(f"\nShape        : {df.shape}")
print(f"\nMissing vals :\n{df.isnull().sum()}")
print(f"\nStats:\n{df.describe().round(2)}")

# ── Plot 1: SOC Distribution ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df['soc_percent'], bins=50, color='steelblue', edgecolor='white', alpha=0.85)
axes[0].set_title('SOC Distribution')
axes[0].set_xlabel('State of Charge (%)')
axes[0].set_ylabel('Count')
axes[0].axvline(df['soc_percent'].mean(), color='crimson', linestyle='--',
                label=f"Mean: {df['soc_percent'].mean():.1f}%")
axes[0].legend()

# SOC categories
bins   = [0, 20, 40, 60, 80, 100]
labels = ['Critical\n(0-20%)', 'Low\n(20-40%)', 'Medium\n(40-60%)',
          'Good\n(60-80%)', 'Full\n(80-100%)']
df['soc_category'] = pd.cut(df['soc_percent'], bins=bins, labels=labels)
counts = df['soc_category'].value_counts().sort_index()
colors = ['crimson','darkorange','gold','yellowgreen','green']
axes[1].bar(counts.index, counts.values, color=colors, alpha=0.85, edgecolor='white')
axes[1].set_title('SOC Category Distribution')
axes[1].set_ylabel('Count')
for bar, val in zip(axes[1].patches, counts.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                 f'{val:,}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('plots/01_soc_distribution.png', dpi=150)
plt.close()

# ── Plot 2: Voltage vs SOC ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
sample = df.sample(2000, random_state=42)
sc = ax.scatter(sample['voltage_v'], sample['soc_percent'],
                c=sample['soc_percent'], cmap='RdYlGn', alpha=0.5, s=10)
plt.colorbar(sc, ax=ax, label='SOC (%)')
ax.set_xlabel('Voltage (V)')
ax.set_ylabel('State of Charge (%)')
ax.set_title('Voltage vs SOC — Strong Correlation')
plt.tight_layout()
plt.savefig('plots/02_voltage_vs_soc.png', dpi=150)
plt.close()

# ── Plot 3: Temperature Effect on SOC ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, col, title in zip(axes,
    ['temperature_c', 'ambient_temp_c'],
    ['Battery Temperature (°C)', 'Ambient Temperature (°C)']):
    sample2 = df.sample(2000, random_state=1)
    ax.scatter(sample2[col], sample2['soc_percent'],
               alpha=0.3, s=8, color='darkorange')
    z = np.polyfit(df[col], df['soc_percent'], 1)
    p = np.poly1d(z)
    xline = np.linspace(df[col].min(), df[col].max(), 100)
    ax.plot(xline, p(xline), 'r-', linewidth=2, label='Trend')
    ax.set_xlabel(title)
    ax.set_ylabel('SOC (%)')
    ax.set_title(f'{title} vs SOC')
    ax.legend()
plt.tight_layout()
plt.savefig('plots/03_temperature_vs_soc.png', dpi=150)
plt.close()

# ── Plot 4: Cycle Count & Degradation ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df['cycle_bin'] = pd.cut(df['cycle_count'], bins=10)
soc_by_cycle = df.groupby('cycle_bin', observed=True)['soc_percent'].mean()
axes[0].plot(range(len(soc_by_cycle)), soc_by_cycle.values, 'b-o', markersize=6)
axes[0].set_title('Average SOC vs Battery Cycle Count')
axes[0].set_xlabel('Cycle Count (binned, low→high)')
axes[0].set_ylabel('Average SOC (%)')
axes[0].set_xticks([])

axes[1].scatter(df.sample(2000, random_state=2)['internal_resistance'],
                df.sample(2000, random_state=2)['soc_percent'],
                alpha=0.3, s=8, color='purple')
axes[1].set_xlabel('Internal Resistance (Ω)')
axes[1].set_ylabel('SOC (%)')
axes[1].set_title('Internal Resistance vs SOC')
plt.tight_layout()
plt.savefig('plots/04_degradation_analysis.png', dpi=150)
plt.close()

# ── Plot 5: Current vs SOC ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
s = df.sample(2000, random_state=3)
sc = ax.scatter(s['current_a'], s['soc_percent'],
                c=s['current_a'], cmap='coolwarm', alpha=0.4, s=10)
plt.colorbar(sc, ax=ax, label='Current (A)')
ax.axvline(0, color='black', linestyle='--', alpha=0.5, label='Idle (0A)')
ax.set_xlabel('Current (A)  [Positive=Charging, Negative=Discharging]')
ax.set_ylabel('SOC (%)')
ax.set_title('Current vs SOC')
ax.legend()
plt.tight_layout()
plt.savefig('plots/05_current_vs_soc.png', dpi=150)
plt.close()

# ── Plot 6: SOC by Battery Capacity ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
for cap in sorted(df['battery_capacity_ah'].unique()):
    data = df[df['battery_capacity_ah'] == cap]['soc_percent']
    ax.hist(data, bins=30, alpha=0.6, label=f'{cap} Ah')
ax.set_title('SOC Distribution by Battery Capacity')
ax.set_xlabel('SOC (%)')
ax.set_ylabel('Count')
ax.legend()
plt.tight_layout()
plt.savefig('plots/06_soc_by_capacity.png', dpi=150)
plt.close()

# ── Plot 7: Correlation Heatmap ────────────────────────────────────────────────
num_cols = ['voltage_v','current_a','temperature_c','internal_resistance',
            'cycle_count','battery_age_days','charge_rate_c','discharge_rate_c',
            'time_since_charge_h','battery_capacity_ah','soc_percent']
fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(df[num_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm',
            center=0, ax=ax, linewidths=0.5)
ax.set_title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('plots/07_correlation_heatmap.png', dpi=150)
plt.close()

# ── Plot 8: Time Since Charge vs SOC ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
s = df.sample(2000, random_state=4)
ax.scatter(s['time_since_charge_h'], s['soc_percent'],
           alpha=0.3, s=8, color='teal')
z = np.polyfit(df['time_since_charge_h'], df['soc_percent'], 1)
p = np.poly1d(z)
xline = np.linspace(0, 48, 100)
ax.plot(xline, p(xline), 'r-', linewidth=2, label='Trend')
ax.set_xlabel('Time Since Last Charge (hours)')
ax.set_ylabel('SOC (%)')
ax.set_title('Self-Discharge: Time Since Charge vs SOC')
ax.legend()
plt.tight_layout()
plt.savefig('plots/08_time_vs_soc.png', dpi=150)
plt.close()

print("\nAll EDA plots saved to plots/")
