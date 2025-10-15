"""
🚦 Traffic Prediction Dashboard (Ngrok Version)
----------------------------------------------
Choose dataset (METR-LA / PEMS-BAY), load model, and view predicted vs actual speeds.
Runs in Google Colab using Ngrok.
"""

# --- ✅ FIX IMPORT PATHS for Colab / Ngrok execution ---
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print("✅ Repo root added to sys.path:", sys.path[-1])

# --- 📦 Imports ---
import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from models.hybrid_model import HybridModel
from preprocessing.preprocess import load_and_clean_data, normalize_data
from preprocessing.features import add_time_features
from preprocessing.windowing import create_windowed_dataset


# ===============================
# 🧩 Load model helper
# ===============================
@st.cache_resource
def load_model(model_path, input_dim, output_dim, device="cpu"):
    """Loads the trained Hybrid LSTM + TransformerXL model."""
    model = HybridModel(input_dim=input_dim, output_dim=output_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


# ===============================
# 🧩 Load dataset helper
# ===============================
@st.cache_data
def load_dataset(dataset_path):
    """Loads and preprocesses dataset with normalization + time features."""
    df = load_and_clean_data(dataset_path)
    df_norm, mean, std = normalize_data(df)
    df_time = add_time_features(df_norm)
    return df_time, mean, std


# ===============================
# 🧠 Prediction helper (fixed)
# ===============================
def predict_for_sensor(model, df_time, sensor_id, mean, std, device="cpu"):
    """
    Predicts next few timesteps for a selected sensor.
    Uses the full feature set for model input, extracts only one sensor for plotting.
    """
    # Create windowed dataset using the full feature set
    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = create_windowed_dataset(df_time)

    # Take the latest input window
    X_latest = X_test[-1:].clone().detach().to(device)

    # Run prediction
    with torch.no_grad():
        preds = model(X_latest).cpu().numpy()[0]  # shape: (output_len, num_sensors)

    # 🔧 Ensure target Y matches only sensors, not time features
    num_sensors = preds.shape[1]  # e.g. 207 for METR-LA, 325 for PEMS-BAY
    Y_true = Y_test[-1].numpy()[:, :num_sensors]  # slice only sensor columns

    # 🔧 Match std/mean only to sensors
    sensor_cols = std.index[:num_sensors]
    std_sensors = std[sensor_cols].values
    mean_sensors = mean[sensor_cols].values

    # Denormalize
    preds = preds * std_sensors + mean_sensors
    actual = (Y_true * std_sensors) + mean_sensors

    # Extract only the selected sensor’s predictions
    sensor_idx = list(df_time.columns).index(sensor_id)
    preds_sensor = preds[:, sensor_idx]
    actual_sensor = actual[:, sensor_idx]

    return preds_sensor, actual_sensor




# ===============================
# 🚀 Streamlit UI
# ===============================
def main():
    st.title("🚦 Traffic Prediction Dashboard (Ngrok Edition)")
    st.markdown("### Real-Time Predictions using Hybrid LSTM + TransformerXL")
    st.markdown("---")

    # Sidebar configuration
    dataset_choice = st.sidebar.selectbox(
        "Select Dataset", ["METR-LA", "PEMS-BAY"], index=0
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_map = {
        "METR-LA": {
            "data": "data/metr-la.h5",
            "model": "/content/drive/MyDrive/traffic_prediction/best_model_metrla.pth",
            "input_dim": 215,
            "output_dim": 207,
        },
        "PEMS-BAY": {
            "data": "data/pems-bay.h5",
            "model": "/content/drive/MyDrive/traffic_prediction/best_model_pemsbay.pth",
            "input_dim": 333,
            "output_dim": 333,
        },
    }

    ds_info = dataset_map[dataset_choice]
    st.sidebar.success(f"✅ Loaded {dataset_choice} configuration")

    # Load dataset + model
    with st.spinner("Loading dataset and model..."):
        df_time, mean, std = load_dataset(ds_info["data"])
        model = load_model(
            ds_info["model"],
            ds_info["input_dim"],
            ds_info["output_dim"],
            device
        )

    st.success(f"✅ Model & Dataset ready for {dataset_choice}")
    st.write(f"📊 Dataset shape: {df_time.shape}")

    # Sensor selection
    sensor_id = st.selectbox("Select Sensor ID", df_time.columns)

    # Generate predictions
    preds, actual = predict_for_sensor(model, df_time, sensor_id, mean, std, device)

    # Plot predictions vs actual
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(actual, label="Actual", color="blue")
    ax.plot(preds, label="Predicted", color="red", linestyle="--")
    ax.set_title(f"{dataset_choice} — Sensor {sensor_id}")
    ax.set_xlabel("Future Time Steps (5-min intervals)")
    ax.set_ylabel("Traffic Speed (mph)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)


# ===============================
# 🎯 Entry Point
# ===============================
if __name__ == "__main__":
    main()
