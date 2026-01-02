# exp3_ross_recovery.py
# Ross Recovery: Physical Measure HJM

import numpy as np

# Empirical mean forward-rate change (example proxy)
empirical_mean_drift = -0.0004

# Market price of risk (constant specification)
lambda_mpr = empirical_mean_drift / 0.015

dt = 1 / 252
n_steps = 252
n_paths = 5000

maturities = np.linspace(0.1, 10, 40)

def sigma(tau):
    return 0.015 * np.exp(-0.625 * tau)

def hjm_drift_physical(tau):
    return (
        sigma(tau) * (0.015 / 0.625) * (1 - np.exp(-0.625 * tau))
        + lambda_mpr * sigma(tau)
    )

paths_P = np.zeros((n_paths, n_steps, len(maturities)))
paths_P[:, 0, :] = 0.03 * np.exp(-0.05 * maturities)

for i in range(1, n_steps):
    dW = np.random.normal(0, np.sqrt(dt), size=n_paths)
    for j, tau in enumerate(maturities):
        paths_P[:, i, j] = (
            paths_P[:, i-1, j]
            + hjm_drift_physical(tau) * dt
            + sigma(tau) * dW
        )

print("Recovered physical-measure simulation complete.")
print("Mean terminal forward rate:", paths_P[:, -1, 0].mean())
