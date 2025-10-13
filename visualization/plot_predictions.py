"""
Visualization utilities — plots predicted vs. actual traffic speeds
for randomly selected or specified sensors.
"""

import os
import random
import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_predictions(Y_true, Y_pred, sensor_ids=None, num_sensors=3, save_path="outputs/pred_vs_actual.png"):
    """
    Plot predicted vs actual speed curves for selected sensors.
    Args:
        Y_true: torch.Tensor or np.ndarray — shape (N, T, sensors)
        Y_pred: torch.Tensor or np.ndarray — same shape
        sensor_ids: list of sensor indices to plot
        num_sensors: how many random sensors to plot if sensor_ids not given
        save_path: file path to save the plot
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Convert to numpy if tensors
    if torch.is_tensor(Y_true): Y_true = Y_true.cpu().numpy()
    if torch.is_tensor(Y_pred): Y_pred = Y_pred.cpu().numpy()

    n_samples, T, n_sensors = Y_true.shape

    # Pick sensors to plot
    if sensor_ids is None:
        sensor_ids = random.sample(range(n_sensors), num_sensors)

    plt.figure(figsize=(12, 8))
    time_axis = np.arange(T)

    for i, sid in enumerate(sensor_ids):
        plt.subplot(num_sensors, 1, i + 1)
        plt.plot(time_axis, Y_true[0, :, sid], label="Actual", color="black", linewidth=2)
        plt.plot(time_axis, Y_pred[0, :, sid], label="Predicted", color="royalblue", linestyle="--", linewidth=2)
        plt.title(f"Sensor {sid}")
        plt.xlabel("Time steps (5 min intervals)")
        plt.ylabel("Speed (mph)")
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"📈 Saved prediction comparison → {save_path}")
    plt.close()
