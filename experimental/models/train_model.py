import pandas as pd
import keras
import numpy as np
import matplotlib.pyplot as plt

train_data = pd.read_csv(r"f1_predictor\data\train_data.csv")
test_data = pd.read_csv(r"f1_predictor\data\test_data.csv")
print
x_train = train_data[['normalized_lap','average_normalized_lap','lap_progress','current_position_norm']]
y_train = train_data['finishing_position']

x_test = test_data[['normalized_lap','average_normalized_lap','lap_progress','current_position_norm']]
y_test = test_data['finishing_position']

print(f"sample of the training data:\n{x_train.columns}")
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(x_train.shape[1],)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='linear')
])
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
loss, mae = model.evaluate(x_test, y_test)
print(f"Test Loss: {loss}, Test MAE: {mae}")



plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()