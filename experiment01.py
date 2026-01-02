# exp1_hjm_construction.py
# One-Factor HJM: Curve Construction & Forward Rate Extraction

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

# -----------------------------
# Load yield data (example CSV)
# Columns: date, 1M, 3M, 1Y, 5Y, 10Y
# -----------------------------
data = pd.read_csv("treasury_yields.csv", parse_dates=["date"])
data = data.dropna()

maturities = np.array([1/12, 3/12, 1, 5, 10])  # in years

def spot_to_forward(yields, maturities):
    """
    Convert spot yields into instantaneous forward rates
    using monotone spline interpolation
    """
    spline = PchipInterpolator(maturities, yields)
    T = np.linspace(maturities.min(), maturities.max(), 200)
    zc = spline(T)

    # Numerical derivative for forward rates
    dz = np.gradient(zc, T)
    forwards = zc + T * dz
    return T, forwards

# Example extraction
example_row = data.iloc[0]
spot = example_row[["1M", "3M", "1Y", "5Y", "10Y"]].values / 100

T, fwd = spot_to_forward(spot, maturities)

print("Forward curve constructed.")
print("Mean forward rate:", np.mean(fwd))
print("Std forward rate:", np.std(fwd))
