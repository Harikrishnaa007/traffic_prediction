"""
Training loop for Hybrid LSTM + Transformer-XL with Early Stopping,
Configurable Learning Rate Scheduler, Dataset-Specific Saving, and Loss Visualization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os


def train_model(
    model,
    train_loader,
    val_loader,
    epochs=60,
    lr=2e-4,
    device="cpu",
    patience=10,
    dataset_name="default",
    scheduler_type="reduce",   # ✅ can be "reduce", "cosine", or "step"
    plot_curve=False           # ✅ avoid duplicate plotting when called from main.py
):
    """
    Train the model with early stopping and flexible learning rate schedulers.

    Returns:
        model (nn.Module): trained model with best weights
        train_loss_history (list): per-epoch training loss
        val_loss_history (list): per-epoch validation loss
        optimizer (torch.optim.Optimizer): final optimizer (for chaining schedulers)
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # 🔧 Flexible scheduler selection
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        print("📉 Using CosineAnnealingLR scheduler")
    elif scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        print("📉 Using StepLR scheduler")
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
        print("📉 Using ReduceLROnPlateau scheduler (default)")

    best_val_loss = float("inf")
    patience_counter = 0
    train_loss_history, val_loss_history = [], []

    # 🌟 Auto-named model file per dataset
    save_path = f"best_model_{dataset_name.lower().replace('.h5','')}.pth"

    for epoch in range(1, epochs + 1):
        # ---- Training ----
        model.train()
        train_losses = []
        for X, Y in train_loader:
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, Y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # ---- Validation ----
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X, Y in val_loader:
                X, Y = X.to(device), Y.to(device)
                out = model(X)
                loss = criterion(out, Y)
                val_losses.append(loss.item())

        avg_val_loss = np.mean(val_losses)

        # 🔄 Step scheduler
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        else:
            scheduler.step()

        # 📈 Track losses
        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {current_lr:.6f}")

        # ---- Early Stopping ----
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"  ✅ Best model saved at epoch {epoch} → {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⏹️ Early stopping triggered at epoch {epoch}")
                break

    # ---- Reload best weights ----
    model.load_state_dict(torch.load(save_path))

    # 📊 Plot training/validation loss (if standalone)
    if plot_curve:
        plt.figure(figsize=(8, 5))
        plt.plot(train_loss_history, label="Train Loss")
        plt.plot(val_loss_history, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title("Training vs Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return model, train_loss_history, val_loss_history, optimizer
