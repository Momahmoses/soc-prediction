"""
Battery SOC Prediction — Model Training & Evaluation
This is a REGRESSION task (predicting SOC as a continuous % value).
Trains multiple ML models, evaluates, and saves the best one.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    r2_score, mean_absolute_percentage_error
)

os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# ── Load ───────────────────────────────────────────────────────────────────────
df = pd.read_csv('data/soc_dataset.csv')
df = df.drop(columns=['soc_category'], errors='ignore')

features = [
    'voltage_v', 'current_a', 'temperature_c', 'ambient_temp_c',
    'internal_resistance', 'cycle_count', 'battery_age_days',
    'charge_rate_c', 'discharge_rate_c', 'time_since_charge_h',
    'humidity_pct', 'battery_capacity_ah'
]

X = df[features]
y = df['soc_percent']

# ── Split ──────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── Models ─────────────────────────────────────────────────────────────────────
models = {
    'Linear Regression':   LinearRegression(),
    'Ridge Regression':    Ridge(alpha=1.0),
    'Decision Tree':       DecisionTreeRegressor(max_depth=10, random_state=42),
    'Random Forest':       RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting':   GradientBoostingRegressor(n_estimators=100, random_state=42),
}

results = {}
print("── Model Comparison ──────────────────────────────────────────────────────")
print(f"{'Model':<22} {'MAE':>8} {'RMSE':>8} {'MAPE%':>8} {'R²':>8}")
print("-"*60)

for name, model in models.items():
    use_scaled = name in ['Linear Regression', 'Ridge Regression']
    Xtr = X_train_sc if use_scaled else X_train
    Xte = X_test_sc  if use_scaled else X_test

    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte).clip(0, 100)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    r2   = r2_score(y_test, y_pred)

    results[name] = {
        'model': model, 'scaled': use_scaled,
        'mae': mae, 'rmse': rmse, 'mape': mape, 'r2': r2,
        'y_pred': y_pred
    }
    print(f"{name:<22} {mae:>8.3f} {rmse:>8.3f} {mape:>8.2f} {r2:>8.4f}")

# ── Best model by R² ───────────────────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]['r2'])
best = results[best_name]
print(f"\nBest model: {best_name}")
print(f"  MAE  : {best['mae']:.3f}%")
print(f"  RMSE : {best['rmse']:.3f}%")
print(f"  MAPE : {best['mape']:.2f}%")
print(f"  R²   : {best['r2']:.4f}")

# ── Plot 9: Actual vs Predicted ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scatter
sample_idx = np.random.choice(len(y_test), 1000, replace=False)
y_test_arr = np.array(y_test)
axes[0].scatter(y_test_arr[sample_idx], best['y_pred'][sample_idx],
                alpha=0.4, s=10, color='steelblue')
mn, mx = y_test_arr.min(), y_test_arr.max()
axes[0].plot([mn, mx], [mn, mx], 'r--', linewidth=2, label='Perfect prediction')
axes[0].set_xlabel('Actual SOC (%)')
axes[0].set_ylabel('Predicted SOC (%)')
axes[0].set_title(f'Actual vs Predicted — {best_name}')
axes[0].legend()

# Residuals
residuals = y_test_arr - best['y_pred']
axes[1].hist(residuals, bins=50, color='darkorange', edgecolor='white', alpha=0.85)
axes[1].axvline(0, color='red', linestyle='--')
axes[1].set_xlabel('Residual (Actual - Predicted)')
axes[1].set_ylabel('Count')
axes[1].set_title(f'Residual Distribution (RMSE={best["rmse"]:.2f}%)')
plt.suptitle(f'Model Performance — {best_name}', fontweight='bold')
plt.tight_layout()
plt.savefig('plots/09_actual_vs_predicted.png', dpi=150)
plt.close()

# ── Plot 10: Model Comparison Bar Chart ───────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
names = list(results.keys())
r2s   = [results[n]['r2']  for n in names]
maes  = [results[n]['mae'] for n in names]
colors = ['crimson' if n == best_name else 'steelblue' for n in names]

axes[0].bar(names, r2s, color=colors, alpha=0.85)
axes[0].set_title('R² Score by Model')
axes[0].set_ylabel('R² Score')
axes[0].tick_params(axis='x', rotation=15)
axes[0].set_ylim(0, 1.05)
for i, v in enumerate(r2s):
    axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)

axes[1].bar(names, maes, color=colors, alpha=0.85)
axes[1].set_title('Mean Absolute Error by Model')
axes[1].set_ylabel('MAE (%)')
axes[1].tick_params(axis='x', rotation=15)
for i, v in enumerate(maes):
    axes[1].text(i, v + 0.1, f'{v:.2f}', ha='center', fontsize=9)

plt.suptitle('Model Comparison', fontweight='bold')
plt.tight_layout()
plt.savefig('plots/10_model_comparison.png', dpi=150)
plt.close()

# ── Plot 11: Feature Importance ────────────────────────────────────────────────
if hasattr(best['model'], 'feature_importances_'):
    imp = pd.Series(best['model'].feature_importances_, index=features).sort_values()
    fig, ax = plt.subplots(figsize=(9, 6))
    imp.plot(kind='barh', color='steelblue', ax=ax)
    ax.set_title(f'Feature Importance — {best_name}')
    ax.set_xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('plots/11_feature_importance.png', dpi=150)
    plt.close()

# ── Plot 12: Prediction Error by SOC Range ────────────────────────────────────
bins   = [0, 20, 40, 60, 80, 100]
labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
y_test_s = pd.Series(y_test_arr)
soc_bin  = pd.cut(y_test_s, bins=bins, labels=labels)
mae_by_range = pd.Series(np.abs(residuals)).groupby(soc_bin, observed=True).mean()

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(mae_by_range.index, mae_by_range.values,
              color=['crimson','darkorange','gold','yellowgreen','green'], alpha=0.85)
for bar, val in zip(bars, mae_by_range.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'{val:.2f}%', ha='center', fontsize=10)
ax.set_title('Prediction Error (MAE) by SOC Range')
ax.set_xlabel('SOC Range')
ax.set_ylabel('Mean Absolute Error (%)')
plt.tight_layout()
plt.savefig('plots/12_error_by_soc_range.png', dpi=150)
plt.close()

# ── Save ───────────────────────────────────────────────────────────────────────
joblib.dump(best['model'], 'models/best_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump({'best_model': best_name, 'features': features, 'scaled': best['scaled']},
            'models/metadata.pkl')

print(f"\nModel saved → models/best_model.pkl")
print(f"Plots saved → plots/")
