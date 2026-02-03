import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from attention import Attention

WINDOW = 10

# Load model + scaler
model = load_model("lstm1.keras",
                   custom_objects={"Attention": Attention})
scaler = joblib.load("scaler1.save")

# Load CSV
df = pd.read_csv("traffic_timeseries.csv")
data = df["vehicle_count"].values.reshape(-1, 1)

# Normalize
data_scaled = scaler.transform(data)

# last 10 timesteps
last_window = data_scaled[-WINDOW:]
last_window = last_window.reshape(1, WINDOW, 1)

# Prediction
pred_scaled = model.predict(last_window)
pred = scaler.inverse_transform(pred_scaled)

print("\nNext predicted vehicle count:", float(pred[0][0]))
