import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('solar_data.csv')
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Features
features = ['solar_gen', 'temperature', 'humidity', 'pressure', 'hour', 'month', 'cloud_cover', 'irradiance', 'panel_efficiency']
data = data[features].dropna()

# Normalize
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Sequences - past 7 days = 168 hours
def create_sequences(data, seq_length=168):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length, 0]  # Predict solar_gen
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

X, y = create_sequences(data_scaled)

# Split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, shuffle=False)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)

# Model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1, activation='relu'))  # Non-negative

model.compile(optimizer='adam', loss='mse')
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

model.save('solar_model.h5')

test_loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}')

plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.legend()
plt.savefig('solar_accuracy.png')
plt.close()

# Antigravity note: Higher efficiency in data, model learns it.

print("Solar model trained and saved.")