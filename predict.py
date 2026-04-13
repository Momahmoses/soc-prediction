"""
Battery SOC Prediction — Interactive CLI Tool
Run: python3 predict.py
"""

import joblib
import numpy as np

model    = joblib.load('models/best_model.pkl')
scaler   = joblib.load('models/scaler.pkl')
metadata = joblib.load('models/metadata.pkl')

def get_input(prompt, type_fn, min_val=None, max_val=None):
    while True:
        try:
            val = type_fn(input(prompt).strip())
            if min_val is not None and val < min_val:
                print(f"  Value must be >= {min_val}")
                continue
            if max_val is not None and val > max_val:
                print(f"  Value must be <= {max_val}")
                continue
            return val
        except ValueError:
            print("  Invalid input, please try again.")

def soc_status(soc):
    if soc <= 20:   return "CRITICAL   — Charge immediately!"
    elif soc <= 40: return "LOW        — Charge soon"
    elif soc <= 60: return "MODERATE   — Acceptable"
    elif soc <= 80: return "GOOD       — Healthy range"
    else:           return "EXCELLENT  — Fully charged"

def battery_health(cycles):
    if cycles < 300:   return "NEW        — Excellent"
    elif cycles < 600: return "GOOD       — Minimal degradation"
    elif cycles < 900: return "FAIR       — Moderate wear"
    elif cycles < 1200:return "AGING      — Notable degradation"
    else:              return "OLD        — Consider replacement"

def predict():
    print("\n" + "="*60)
    print("    BATTERY STATE OF CHARGE (SOC) PREDICTION TOOL")
    print("="*60)
    print("Enter battery readings below:\n")

    voltage     = get_input("Terminal voltage (V)  [Li-ion: 3.0 – 4.2]: ", float, 2.5, 5.0)
    current     = get_input("Current (A)  [+ve=charging, -ve=discharging]: ", float, -100, 100)
    temp        = get_input("Battery temperature (°C): ", float, -20, 80)
    amb_temp    = get_input("Ambient temperature (°C): ", float, -20, 60)
    resistance  = get_input("Internal resistance (Ω)  [new~0.05, old~0.3]: ", float, 0.001, 1.0)
    cycles      = get_input("Number of charge/discharge cycles: ", int, 0, 5000)
    age_days    = get_input("Battery age (days): ", int, 0, 5000)
    charge_rate = get_input("Charge C-rate  (e.g. 0.5, 1.0, 2.0): ", float, 0.1, 5.0)
    disc_rate   = get_input("Discharge C-rate (e.g. 0.5, 1.0, 2.0): ", float, 0.1, 5.0)
    time_charge = get_input("Hours since last full charge: ", float, 0, 200)
    humidity    = get_input("Ambient humidity (%): ", float, 0, 100)
    capacity    = get_input("Battery capacity (Ah): ", float, 1, 500)

    features = np.array([[
        voltage, current, temp, amb_temp, resistance,
        cycles, age_days, charge_rate, disc_rate,
        time_charge, humidity, capacity
    ]])

    if metadata['scaled']:
        features = scaler.transform(features)

    soc = float(model.predict(features)[0].clip(0, 100))

    print("\n" + "="*60)
    print("  BATTERY SOC PREDICTION RESULT")
    print("="*60)
    print(f"  Predicted SOC   : {soc:.1f}%")
    print(f"  Battery Status  : {soc_status(soc)}")
    print(f"  Battery Health  : {battery_health(cycles)}")
    print(f"  Voltage         : {voltage:.3f} V")
    print(f"  Temperature     : {temp:.1f} °C")
    print("="*60)

    print("\n  INSIGHTS:")
    if soc <= 20:
        print("  - Battery critically low — plug in immediately")
    if temp > 45:
        print(f"  - Battery temperature ({temp}°C) is dangerously high!")
        print("    Stop usage and allow it to cool down.")
    elif temp < 0:
        print(f"  - Low temperature ({temp}°C) reduces battery performance.")
    if cycles > 1000:
        print(f"  - High cycle count ({cycles}) — battery capacity may have degraded.")
        print("    Consider replacement if runtime has significantly reduced.")
    if resistance > 0.2:
        print(f"  - High internal resistance ({resistance:.3f}Ω) detected.")
        print("    This indicates battery aging or damage.")
    if current < -30:
        print(f"  - High discharge rate ({abs(current):.1f}A) — reduces battery life.")
    if time_charge > 24 and soc < 40:
        print(f"  - Battery not charged in {time_charge:.0f}h — charge soon.")
    print("="*60)

    again = input("\nPredict another battery? (y/n): ").strip().lower()
    if again == 'y':
        predict()

if __name__ == '__main__':
    predict()
