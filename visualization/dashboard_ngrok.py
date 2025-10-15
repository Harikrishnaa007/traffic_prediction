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
    df = load_and_clean_data(dataset_path)
    df_norm, mean, std = normalize_data(df)
    df_time = add_time_features(df_norm)
    return df_time, mean, std

# ===============================
# 🧠 Prediction helper
# ===============================
def predict_for_sensor(model, df_time, sensor_id, mean, std, device="cpu"):
    df_sensor = df_time[[sensor_id]]
    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = create_windowed_dataset(df_sensor)
    X_latest = torch.tensor(X_test[-1:], dtype=torch.float32).to(device)

    with torch.no_grad():
        preds = model(X_latest).cpu().numpy()[0, :, 0]

    preds = preds * std[sensor_id] + mean[sensor_id]
    actual = (Y_test[-1, :, 0] * std[sensor_id]) + mean[sensor_id]
    return preds, actual

# ===============================
# 🚀 Streamlit UI
# ===============================
def main():
    st.title("🚦 Traffic Prediction Dashboard (Ngrok Edition)")
    st.markdown("### Real-Time Predictions using Hybrid LSTM + TransformerXL")

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

    with st.spinner("Loading dataset and model..."):
        df_time, mean, std = load_dataset(ds_info["data"])
        model = load_model(ds_info["model"], ds_info["input_dim"], ds_info["output_dim"], device)

    st.success(f"Model & Dataset ready for {dataset_choice}")
    st.write(f"Dataset shape: {df_time.shape}")

    sensor_id = st.selectbox("Select Sensor ID", df_time.columns)
    preds, actual = predict_for_sensor(model, df_time, sensor_id, mean, std, device)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(actual, label="Actual", color="blue")
    ax.plot(preds, label="Predicted", color="red", linestyle="--")
    ax.set_title(f"{dataset_choice} — Sensor {sensor_id}")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Speed (mph)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

if __name__ == "__main__":
    main()
