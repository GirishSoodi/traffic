import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Conv1D, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from tensorflow.keras.utils import register_keras_serializable
import pickle
from sklearn.preprocessing import MinMaxScaler


# =========================================
# ✅ Custom Attention Layer (Serializable)
# =========================================
@register_keras_serializable()
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.att_weight = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], 1),
            initializer="normal",
            trainable=True
        )
        self.att_bias = self.add_weight(
            name="att_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True
        )
        super(Attention, self).build(input_shape)

    def call(self, x):
        score = K.tanh(K.dot(x, self.att_weight) + self.att_bias)
        attention_weights = K.softmax(score, axis=1)
        context = attention_weights * x
        context = K.sum(context, axis=1)
        return context

    def get_config(self):
        config = super().get_config()
        return config


# =========================================
# ✅ Build Model
# =========================================
def build_model(time_steps=10, features=1):
    inp = Input(shape=(time_steps, features))
    x = Conv1D(32, kernel_size=3, padding='same', activation='relu')(inp)
    x = LSTM(64, return_sequences=True)(x)
    att = Attention(name="custom_attention")(x)
    out = Dense(1)(att)

    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=Adam(1e-3), loss="mse")
    model.summary()
    return model


# =========================================
# ✅ Load CSV
# =========================================
csv_file = "traffic_timeseries.csv"   # change if needed
df = pd.read_csv(csv_file)

print("\n✅ CSV Loaded!")
print(df.head())

# Column name must match → edit if different
target_col = "vehicle_count"
y_values = df[target_col].values.reshape(-1, 1)


# =========================================
# ✅ Scale Data
# =========================================
scaler = MinMaxScaler()
scaled = scaler.fit_transform(y_values)

# Save scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("✅ scaler.pkl saved")


# =========================================
# ✅ Build dataset sequences
# =========================================
time_steps = 10
X = []
Y = []

for i in range(len(scaled) - time_steps):
    X.append(scaled[i:i + time_steps])
    Y.append(scaled[i + time_steps])

X = np.array(X).reshape(-1, time_steps, 1)
Y = np.array(Y).reshape(-1, 1)

print("\n✅ Dataset created")
print("X shape:", X.shape)
print("Y shape:", Y.shape)


# =========================================
# ✅ Train + Save
# =========================================
model = build_model()
model.fit(X, Y, epochs=50, batch_size=32)

model.save("traffic_lstm_attention.keras")
print("\n✅ Model saved: traffic_lstm_attention.keras")
print("✅ Scaler saved: scaler.pkl")
