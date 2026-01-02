# exp2_hjm_risk_neutral_sim.py
# Risk-Neutral HJM Monte Carlo Simulation

import numpy as np

np.random.seed(42)

# Model parameters
sigma0 = 0.015
beta = 0.625
dt = 1 / 252
n_steps = 252
n_paths = 5000

maturities = np.linspace(0.1, 10, 40)

def sigma(tau):
    return sigma0 * np.exp(-beta * tau)

def hjm_drift(tau):
    return sigma(tau) * (sigma0 / beta) * (1 - np.exp(-beta * tau))

# Initial forward curve
f0 = 0.03 * np.exp(-0.05 * maturities)

paths = np.zeros((n_paths, n_steps, len(maturities)))
paths[:, 0, :] = f0

for i in range(1, n_steps):
    dW = np.random.normal(0, np.sqrt(dt), size=n_paths)
    for j, tau in enumerate(maturities):
        paths[:, i, j] = (
            paths[:, i-1, j]
            + hjm_drift(tau) * dt
            + sigma(tau) * dW
        )

# Diagnostics
mean_path = paths.mean(axis=0)
vol_surface = paths.std(axis=0)

print("Risk-neutral simulation complete.")
print("Short-end RMSE proxy:", np.std(paths[:, -1, 0]))
print("Long-end RMSE proxy:", np.std(paths[:, -1, -1]))
