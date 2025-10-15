"""
🚦 Traffic Prediction Dashboard (Ngrok Version)
----------------------------------------------
Choose dataset (METR-LA / PEMS-BAY), load model, and view predicted vs actual speeds.
Now includes on-screen MAE/RMSE metrics and debug information.
"""

# --- ✅ Fix imports for Colab execution ---
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print("✅ Repo root added to sys.path:", sys.path[-1])

# --- 📦 Imports ---
import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
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
# 🧠 Prediction helper (with metrics + debug info)
# ===============================
def predict_for_sensor(model, df_time, sensor_id, mean, std, device="cpu"):
    """
    Predicts next few timesteps for a selected sensor.
    Handles both METR-LA and PEMS-BAY automatically.
    Shows metrics and debug info in the Streamlit UI.
    """
    import torch
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    # --- Create windowed dataset ---
    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = create_windowed_dataset(df_time)

    # --- Latest input window ---
    X_latest = X_test[-1:].clone().detach().to(device)

    # --- Run model ---
    with torch.no_grad():
        preds = model(X_latest).cpu().numpy()[0]  # shape: (output_len, total_features)

    total_features = preds.shape[1]
    num_sensors = len(std)  # true sensor count (e.g., 207 or 325)

    # --- True targets for sensors only ---
    Y_true = Y_test[-1].numpy()[:, :num_sensors]

    # --- Normalization stats ---
    sensor_cols = std.index
    std_sensors = std.values
    mean_sensors = mean.values

    # --- Denormalize only sensor outputs ---
    preds_sensors = preds[:, :num_sensors] * std_sensors + mean_sensors
    actual_sensors = (Y_true * std_sensors) + mean_sensors

    # --- Sensor index ---
    sensor_idx = list(sensor_cols).index(sensor_id)

    # --- Selected sensor time series ---
    preds_sensor = preds_sensors[:, sensor_idx]
    actual_sensor = actual_sensors[:, sensor_idx]

    # --- Compute metrics (compatible across sklearn versions) ---
    mae = mean_absolute_error(actual_sensor, preds_sensor)
    mse = mean_squared_error(actual_sensor, preds_sensor)   # returns MSE
    rmse = np.sqrt(mse)                                     # RMSE computed portably

    # --- Display metrics + debug info inside Streamlit ---
    st.markdown(f"""
    ### ⚙️ Prediction Summary
    **Sensor:** `{sensor_id}`  
    **Preds mean/std:** {preds_sensor.mean():.3f} / {preds_sensor.std():.3f}  
    **Actual mean/std:** {actual_sensor.mean():.3f} / {actual_sensor.std():.3f}  
    **Features:** {total_features} **Sensors:** {num_sensors}  
    **MAE:** {mae:.3f} mph **RMSE:** {rmse:.3f} mph
    """)

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

    # Limit selection to actual sensor columns
    sensor_cols = std.index
    sensor_id = st.selectbox("Select Sensor ID", sensor_cols)

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
