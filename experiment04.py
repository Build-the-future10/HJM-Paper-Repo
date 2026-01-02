# exp4_ml_volatility_hjm.py
# ML-Based Volatility Estimation (HJM-safe)

import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)

# Simple corrective network
class VolatilityNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Softplus()
        )

    def forward(self, x):
        return self.net(x)

model = VolatilityNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Dummy training data
maturity = torch.rand(2000, 1)
realized_vol = torch.rand(2000, 1)
stress_indicator = torch.rand(2000, 1)

X = torch.cat([maturity, realized_vol, stress_indicator], dim=1)
target = 0.01 + 0.02 * stress_indicator

# Training loop
for epoch in range(1000):
    optimizer.zero_grad()
    pred = model(X)
    loss = ((pred - target) ** 2).mean()
    loss.backward()
    optimizer.step()

print("ML volatility training complete.")
print("Sample learned volatility:", model(X[:5]).detach().numpy())
