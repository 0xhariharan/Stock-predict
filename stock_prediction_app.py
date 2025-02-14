import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Function to generate model path per stock
def get_model_path(ticker):
    return f"models/{ticker}_model.h5"

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Function to fetch stock data
def get_stock_data(ticker):
    stock_data = yf.download(ticker, period="10y", interval="1d")
    return stock_data

# Compute RSI
def compute_rsi(data, window=14):
    diff = data.diff()
    gain = diff.where(diff > 0, 0).rolling(window=window).mean()
    loss = -diff.where(diff < 0, 0).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Compute Bollinger Bands
def compute_bollinger_bands(data, window=20):
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    return rolling_mean + (rolling_std * 2), rolling_mean - (rolling_std * 2)

# Compute MACD
def compute_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data.ewm(span=short_window, adjust=False).mean()
    long_ema = data.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal

# Compute Historical Volatility
def compute_volatility(data, window=30):
    log_returns = np.log(data / data.shift(1))
    return log_returns.rolling(window=window).std() * np.sqrt(window)

# Add technical indicators
def add_technical_indicators(stock_data):
    stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['SMA_200'] = stock_data['Close'].rolling(window=200).mean()
    stock_data['RSI'] = compute_rsi(stock_data['Close'], 14)
    stock_data['Upper_BB'], stock_data['Lower_BB'] = compute_bollinger_bands(stock_data['Close'], 20)
    stock_data['MACD'], stock_data['Signal'] = compute_macd(stock_data['Close'])
    stock_data['Volatility'] = compute_volatility(stock_data['Close'], 30)
    
    stock_data.dropna(inplace=True)
    return stock_data

# Preprocess data
def preprocess_data(stock_data):
    features = stock_data[['Close', 'SMA_50', 'SMA_200', 'RSI', 'Upper_BB', 'Lower_BB', 'MACD', 'Signal', 'Volatility']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(features)

    train_size = int(len(scaled_data) * 0.8)
    train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

    def create_dataset(data, time_step=60):
        X, Y = [], []
        for i in range(len(data) - time_step - 1):
            X.append(data[i:(i + time_step), :-1])
            Y.append(data[i + time_step, 0])
        return np.array(X), np.array(Y)

    X_train, Y_train = create_dataset(train_data)
    X_test, Y_test = create_dataset(test_data)

    return X_train, Y_train, X_test, Y_test, scaler, stock_data

# Build LSTM model
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(100, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Save the trained model
def save_model(model, ticker):
    model.save(get_model_path(ticker))

# Load the saved model
def load_saved_model(ticker):
    model_path = get_model_path(ticker)
    return load_model(model_path) if os.path.exists(model_path) else None

# Train and predict stock price
def train_and_predict(ticker, progress_bar, retrain=False):
    stock_data = get_stock_data(ticker)
    stock_data = add_technical_indicators(stock_data)

    X_train, Y_train, X_test, Y_test, scaler, stock_data = preprocess_data(stock_data)

    model_path = get_model_path(ticker)
    if os.path.exists(model_path) and not retrain:
        st.success(f"✅ Using saved model for {ticker}")
        model = load_saved_model(ticker)
    else:
        st.warning(f"⚡ Training new model for {ticker}, please wait...")
        model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
        model.fit(X_train, Y_train, epochs=30, batch_size=32, verbose=0)
        save_model(model, ticker)
        st.success(f"✅ Model trained and saved for {ticker}")

    predicted_stock_price = model.predict(X_test)
    
    predicted_stock_price = scaler.inverse_transform(
        np.concatenate([predicted_stock_price, np.zeros((predicted_stock_price.shape[0], 8))], axis=1)
    )[:, 0]
    
    Y_test_actual = scaler.inverse_transform(
        np.concatenate([Y_test.reshape(-1, 1), np.zeros((Y_test.shape[0], 8))], axis=1)
    )[:, 0]

    rmse = np.sqrt(mean_squared_error(Y_test_actual, predicted_stock_price))
    r2 = r2_score(Y_test_actual, predicted_stock_price)

    return predicted_stock_price, Y_test_actual, rmse, r2, stock_data

# Function to visualize prediction
def plot_predictions(stock_data, predicted, actual):
    plt.figure(figsize=(12, 6))
    plt.plot(stock_data.index[-len(actual):], actual, label="Actual Prices", color="blue")
    plt.plot(stock_data.index[-len(predicted):], predicted, label="Predicted Prices", color="red")
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.legend()
    st.pyplot(plt)

# Streamlit UI
st.title("📈 Stock Price Prediction")
ticker = st.text_input("Stock Ticker (e.g., TCS.NS, INFY.NS):")
predict_button = st.button("Predict")
retrain_button = st.button("Retrain Model")

if ticker:
    prediction_date = st.date_input("Select Future Date for Prediction", min_value=datetime.date.today())

if ticker and predict_button:
    progress_bar = st.progress(0)
    predicted_prices, actual_prices, rmse, r2, stock_data = train_and_predict(ticker, progress_bar, retrain=False)

    st.write(f"📅 Prediction for {ticker} on {prediction_date}: **₹{predicted_prices[-1]:.2f}**")
    st.write(f"📉 RMSE: {rmse:.2f}")
    st.write(f"📊 R² Score: {r2:.2f}")

    plot_predictions(stock_data, predicted_prices, actual_prices)

if ticker and retrain_button:
    progress_bar = st.progress(0)
    train_and_predict(ticker, progress_bar, retrain=True)
