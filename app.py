from flask import Flask, render_template, jsonify
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import io
import base64
from sklearn.metrics import mean_squared_error

app = Flask(__name__)

# Load dataset
data = pd.read_csv('ethereum_daily_data_2018_2024.csv')
data['time'] = pd.to_datetime(data['time'])
data.set_index('time', inplace=True)

# Load trained model
model = load_model('ethereum_lstm_model.h5')

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# Create sequences
def create_sequences(data, sequence_length):
    sequences, labels = [], []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i + sequence_length])
        labels.append(data[i + sequence_length, 0])
    return np.array(sequences), np.array(labels)

sequence_length = 60
train_size = int(len(scaled_data) * 0.8)
train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

X_train, y_train = create_sequences(train_data, sequence_length)
X_test, y_test = create_sequences(test_data, sequence_length)

# Predict & inverse transform
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# Compute MSE
mse = mean_squared_error(y_test_scaled, predictions)

# Future Predictions (Next 60 Days)
future_days = 60
future_predictions = []
last_60_days = data['Close'][-sequence_length:].values.reshape(-1, 1)
last_60_days_scaled = scaler.transform(last_60_days)

for i in range(future_days):
    X_predict = np.reshape(last_60_days_scaled, (1, sequence_length, 1))
    predicted_price = model.predict(X_predict)
    predicted_price_unscaled = scaler.inverse_transform(predicted_price)

    future_predictions.append(predicted_price_unscaled[0, 0])
    
    new_scaled_data = np.append(last_60_days_scaled[1:], predicted_price, axis=0)
    last_60_days_scaled = new_scaled_data.reshape(-1, 1)

# Create future dates
future_dates = pd.date_range(start=data.index[-1], periods=future_days+1, freq='D')[1:]

# Function to generate graph
def generate_plot():
    plt.figure(figsize=(14, 7))
    
    # Plot actual prices
    plt.plot(data.index[-len(y_test_scaled):], y_test_scaled, label='Actual Ethereum Prices', color='blue', linestyle='solid')

    # Plot predicted prices
    plt.plot(data.index[-len(y_test_scaled):], predictions, label='Predicted Ethereum Prices', color='red', linestyle='solid')

    # Plot future predictions
    plt.plot(future_dates, future_predictions, label='Future Predictions (Next 60 Days)', color='green', linestyle='dashed')

    # Format the graph
    plt.title('Ethereum Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)

    # Convert plot to image
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_url

@app.route('/')
def index():
    plot_url = generate_plot()
    return render_template('index.html', plot_url=plot_url)

if __name__ == '__main__':
    app.run(debug=True)
