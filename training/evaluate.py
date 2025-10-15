"""
Final version — robust to shape mismatches between model outputs and normalization stats.
"""

import torch
import torch.nn as nn
import numpy as np
from math import sqrt


def evaluate_model(model, test_loader, device="cuda", mean=None, std=None):
    model.eval()
    model.to(device)

    criterion = nn.MSELoss()
    mae_fn = nn.L1Loss()

    total_mse, total_mae, count = 0.0, 0.0, 0
    all_preds, all_trues = [], []

    with torch.no_grad():
        for X, Y in test_loader:
            X, Y = X.to(device), Y.to(device)
            preds = model(X)

            mse = criterion(preds, Y).item()
            mae = mae_fn(preds, Y).item()

            total_mse += mse
            total_mae += mae
            count += 1

            all_preds.append(preds.cpu())
            all_trues.append(Y.cpu())

    avg_mse = total_mse / count
    avg_mae = total_mae / count
    rmse = sqrt(avg_mse)

    Y_pred = torch.cat(all_preds)
    Y_true = torch.cat(all_trues)

    # --- Safe denormalization ---
    if mean is not None and std is not None:
        Y_pred_np = Y_pred.numpy()
        Y_true_np = Y_true.numpy()

        num_pred_sensors = Y_pred_np.shape[-1]
        num_stats = len(mean)

        # 🔧 Auto-align mean/std to match model output
        min_len = min(num_pred_sensors, num_stats)

        mean_sensors = np.array(mean[:min_len])
        std_sensors = np.array(std[:min_len])

        # If model predicts extra columns → trim
        Y_pred_np = Y_pred_np[..., :min_len]
        Y_true_np = Y_true_np[..., :min_len]

        # ✅ Now broadcast-safe
        Y_pred_denorm = Y_pred_np * std_sensors + mean_sensors
        Y_true_denorm = Y_true_np * std_sensors + mean_sensors

        denorm_mae = np.mean(np.abs(Y_true_denorm - Y_pred_denorm))
        denorm_rmse = np.sqrt(np.mean((Y_true_denorm - Y_pred_denorm) ** 2))

        print(f"Test (normalized): MAE={avg_mae:.4f}, RMSE={rmse:.4f}")
        print(f"Test (denormalized): MAE={denorm_mae:.3f} mph, RMSE={denorm_rmse:.3f} mph")
        print(f"Features={num_stats}, Predicted sensors={num_pred_sensors}, Using first {min_len} aligned features")

        return denorm_mae, denorm_rmse, Y_pred, Y_true

    else:
        print(f"Test MAE: {avg_mae:.4f}, RMSE: {rmse:.4f} (normalized units)")
        return avg_mae, rmse, Y_pred, Y_true
