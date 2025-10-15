"""
Model evaluation for Hybrid LSTM + TransformerXL.
Computes MAE, RMSE, and returns predictions for visualization.
Now supports denormalized evaluation.
"""

import torch
import torch.nn as nn
import numpy as np
from math import sqrt

def evaluate_model(model, test_loader, device="cuda", mean=None, std=None):
    """
    Evaluates the model on test data.
    
    If mean/std are provided, results are computed in the denormalized scale (mph).
    Otherwise, results are computed in normalized Z-score scale.
    """
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

            # Store for later metric computation
            all_preds.append(preds.cpu())
            all_trues.append(Y.cpu())

            mse = criterion(preds, Y).item()
            mae = mae_fn(preds, Y).item()
            total_mse += mse
            total_mae += mae
            count += 1

    # Combine batches
    Y_pred = torch.cat(all_preds)
    Y_true = torch.cat(all_trues)

    # --- Normalized metrics (Z-score units) ---
    avg_mse = total_mse / count
    avg_mae = total_mae / count
    rmse = sqrt(avg_mse)

    # --- Optional denormalization ---
    if mean is not None and std is not None:
        # Convert mean/std to numpy arrays
        mean = np.array(mean)
        std = np.array(std)

        # Handle tensor shape: (samples, time_steps, sensors)
        Y_pred_np = Y_pred.numpy() * std + mean
        Y_true_np = Y_true.numpy() * std + mean

        denorm_mae = np.mean(np.abs(Y_true_np - Y_pred_np))
        denorm_rmse = np.sqrt(np.mean((Y_true_np - Y_pred_np) ** 2))

        print(f"Test (normalized) → MAE: {avg_mae:.4f}, RMSE: {rmse:.4f}")
        print(f"Test (denormalized) → MAE: {denorm_mae:.3f} mph, RMSE: {denorm_rmse:.3f} mph")

        return denorm_mae, denorm_rmse, Y_pred, Y_true

    else:
        print(f"Test MAE: {avg_mae:.4f}, RMSE: {rmse:.4f} (normalized units)")
        return avg_mae, rmse, Y_pred, Y_true
