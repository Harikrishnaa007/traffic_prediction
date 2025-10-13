"""
Utility for logging experiment metrics and configurations.
Automatically appends run details to outputs/metrics_log.csv
"""

import os
import csv
from datetime import datetime

def log_experiment(
    log_dir="outputs",
    model_name="HybridModel",
    dataset="METR-LA",
    epochs=0,
    lr=0.0,
    patience=0,
    best_val_loss=None,
    test_mae=None,
    test_rmse=None,
    early_stop_epoch=None,
    device="cuda",
    notes=""
):
    """Logs experiment details and results into a CSV file."""

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "metrics_log.csv")

    # Define CSV headers
    headers = [
        "Timestamp",
        "Model",
        "Dataset",
        "Epochs",
        "LR",
        "Patience",
        "Best_Val_Loss",
        "Test_MAE",
        "Test_RMSE",
        "Early_Stop_Epoch",
        "Device",
        "Notes"
    ]

    # Create file if it doesn't exist
    file_exists = os.path.isfile(log_path)

    with open(log_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)

        # Write header only once
        if not file_exists:
            writer.writeheader()

        # Write experiment details
        writer.writerow({
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Model": model_name,
            "Dataset": dataset,
            "Epochs": epochs,
            "LR": lr,
            "Patience": patience,
            "Best_Val_Loss": f"{best_val_loss:.4f}" if best_val_loss is not None else "",
            "Test_MAE": f"{test_mae:.4f}" if test_mae is not None else "",
            "Test_RMSE": f"{test_rmse:.4f}" if test_rmse is not None else "",
            "Early_Stop_Epoch": early_stop_epoch if early_stop_epoch is not None else "",
            "Device": device,
            "Notes": notes
        })

    print(f"🧾 Logged experiment → {log_path}")
