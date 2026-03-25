"""
Time Series Forecasting with TimesFM
Author: Mahsa Boustani
Course: Time Series Analysis (Tampere University)

Description:
This script implements time series forecasting using the TimesFM deep learning model.
It compares the forecasting performance with a classical ARIMA model using Mean Squared Error (MSE).

Methods:
- Data preprocessing and train/test split
- Zero-shot forecasting using TimesFM
- Performance evaluation (MSE)
- Visualization of forecast vs actual values

Libraries:
- NumPy, Pandas, Matplotlib
- PyTorch, TimesFM
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import timesfm

# Load and split data
df = pd.read_csv("Case_study.csv")
y = df.iloc[:, 44].astype(float).to_numpy()

train = y[:100]
test  = y[100:]

# Initialize TimesFM 2.5 model
torch.set_float32_matmul_precision("high")

model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch"
)

# Set inference configuration
model.compile(
    timesfm.ForecastConfig(
        max_context=1024,
        max_horizon=256,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=False,
        fix_quantile_crossing=True,
    )
)

# Define forecast and MSE function
def timesfm_forecast_and_mse(h: int):
    h_use = min(h, len(test))
    point_fc, quant_fc = model.forecast(
        horizon=h_use,
        inputs=[train],
    )
    pred = point_fc[0]
    mse = np.mean((test[:h_use] - pred) ** 2)
    return h_use, pred, mse

# Calculate MSE for different horizons
h_list = [10, 25, 100]
results = []

for h in h_list:
    h_use, pred, mse = timesfm_forecast_and_mse(h)
    results.append((h_use, mse))
    print(f"TimesFM zero-shot: h={h_use}, MSE={mse:.6f}")

print("\nARIMA(2,1,1) MSEs from R results:")
print("h=10  -> 2.865076")
print("h=25  -> 4.191318")
print("h=100 -> 3.419277")

# Plot forecast vs actual (h=100)
h_plot = 100
h_use, pred, mse = timesfm_forecast_and_mse(h_plot)

t_train = np.arange(1, len(train) + 1)
t_test  = np.arange(len(train) + 1, len(train) + h_use + 1)

plt.figure(figsize=(10, 6))
plt.plot(t_train, train, label="Train", linewidth=1.5)
plt.plot(t_test, test[:h_use], label="Test (actual)", linewidth=1.5)
plt.plot(t_test, pred, label=f"TimesFM forecast (h={h_use})", linewidth=2)

plt.title("TimesFM (2.5) zero-shot forecast vs actual")
plt.xlabel("Time index")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.show()