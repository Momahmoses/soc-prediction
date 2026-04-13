"""
Generates a synthetic battery State of Charge (SOC) dataset.
SOC is the battery charge level expressed as a percentage (0-100%).
Features are based on real-world battery monitoring parameters.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 10000

# ── Battery parameters ─────────────────────────────────────────────────────────
cycle_count          = np.random.randint(0, 1500, N)          # charge/discharge cycles
battery_age_days     = (cycle_count * 0.8 + np.random.normal(0, 20, N)).clip(0, 1500).astype(int)
ambient_temp_c       = np.random.normal(28, 8, N).clip(-10, 50)   # °C
temperature_c        = ambient_temp_c + np.random.normal(5, 3, N)  # battery temp (slightly higher)
voltage_v            = np.random.uniform(3.0, 4.2, N)              # cell voltage (Li-ion range)
current_a            = np.random.normal(0, 15, N).clip(-50, 50)    # charging(+) / discharging(-)
internal_resistance  = (0.05 + cycle_count * 0.00005 +
                         np.random.normal(0, 0.005, N)).clip(0.01, 0.5)   # ohms
charge_rate_c        = np.random.choice([0.2, 0.5, 1.0, 1.5, 2.0], N,
                           p=[0.20, 0.30, 0.30, 0.15, 0.05])
discharge_rate_c     = np.random.choice([0.2, 0.5, 1.0, 1.5, 2.0, 3.0], N,
                           p=[0.15, 0.25, 0.30, 0.15, 0.10, 0.05])
time_since_charge_h  = np.random.exponential(4, N).clip(0, 48)    # hours
humidity_pct         = np.random.normal(60, 15, N).clip(10, 100)
battery_capacity_ah  = np.random.choice([50, 60, 75, 100], N,
                           p=[0.25, 0.35, 0.25, 0.15])             # Ah

# ── SOC calculation (physics-inspired) ────────────────────────────────────────
# Voltage is the strongest predictor of SOC for Li-ion batteries
voltage_norm = (voltage_v - 3.0) / (4.2 - 3.0)   # normalize to 0-1

soc = (
      0.60 * voltage_norm * 100
    + 0.12 * (current_a / 50 + 0.5) * 100          # positive current = charging
    - 0.08 * (time_since_charge_h / 48) * 100
    - 0.05 * (cycle_count / 1500) * 100             # degradation over cycles
    - 0.04 * np.abs((temperature_c - 25) / 25) * 100  # temp deviation hurts
    - 0.03 * (internal_resistance / 0.5) * 100
    + 0.04 * (battery_capacity_ah / 100) * 100
    + np.random.normal(0, 2, N)
).clip(0, 100)

df = pd.DataFrame({
    'voltage_v':             voltage_v.round(4),
    'current_a':             current_a.round(3),
    'temperature_c':         temperature_c.round(2),
    'ambient_temp_c':        ambient_temp_c.round(2),
    'internal_resistance':   internal_resistance.round(5),
    'cycle_count':           cycle_count,
    'battery_age_days':      battery_age_days,
    'charge_rate_c':         charge_rate_c,
    'discharge_rate_c':      discharge_rate_c,
    'time_since_charge_h':   time_since_charge_h.round(3),
    'humidity_pct':          humidity_pct.round(1),
    'battery_capacity_ah':   battery_capacity_ah,
    'soc_percent':           soc.round(2)
})

df.to_csv('data/soc_dataset.csv', index=False)
print(f"Dataset created  : {len(df):,} records")
print(f"SOC Mean         : {soc.mean():.2f}%")
print(f"SOC Std          : {soc.std():.2f}%")
print(f"SOC Range        : {soc.min():.2f}% — {soc.max():.2f}%")
print(df.head())
