import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import datetime
import os
import requests
import matplotlib.pyplot as plt
from upstox_api.api import *

# Ensure models directory exists
if not os.path.exists("models"):
    os.makedirs("models")

# Your Upstox API credentials
api_key = "6be46ee7-8e84-4c37-9353-63aa896d09bd"
api_secret = "2eqwmvko5i"
access_token = "eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiIyQ0NYRUIiLCJqdGkiOiI2N2FlY2Q2NmZkNjhlZjVlYWZhYzg2ZjUiLCJpc011bHRpQ2xpZW50IjpmYWxzZSwiaWF0IjoxNzM5NTA5MDk0LCJpc3MiOiJ1ZGFwaS1nYXRld2F5LXNlcnZpY2UiLCJleHAiOjE3Mzk1NzA0MDB9.sMDkSpgfAaQRjFR7vHSspZ557NWVHIQMUm58UdT1KPI"

# Initialize the Upstox API
upstox = Upstox(api_key, api_secret)
upstox.set_access_token(access_token)

# Function to get stock data from Upstox API
def get_stock_data(ticker, interval="1d", period="10y"):
    try:
        quote = upstox.get_live_market_data('NSE', ticker)
        stock_data = {
            'Date': [quote['timestamp']],
            'Open': [quote['open_price']],
            'High': [quote['high_price']],
            'Low': [quote['low_price']],
            'Close': [quote['last_price']],
            'Volume': [quote['quantity_traded']]
        }
        stock_df = pd.DataFrame(stock_data)
        stock_df['Date'] = pd.to_datetime(stock_df['Date'])
        return stock_df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Compute RSI (Relative Strength Index)
def compute_rsi(data, window=14):
    diff = data.diff()
    gain = (diff.where(diff > 0, 0)).rolling(window=window).mean()
    loss = (-diff.where(diff < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Compute Bollinger Bands
def compute_bollinger_bands(data, window=20):
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * 2)
    lower_band = rolling_mean - (rolling_std * 2)
    return upper_band, lower_band

# Compute MACD (Moving Average Convergence Divergence)
def compute_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data.ewm(span=short_window, min_periods=1, adjust=False).mean()
    long_ema = data.ewm(span=long_window, min_periods=1, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, min_periods=1, adjust=False).mean()
    return macd, signal

# Compute Historical Volatility
def compute_volatility(data, window=30):
    log_returns = np.log(data / data.shift(1))
    volatility = log_returns.rolling(window=window).std() * np.sqrt(window)
    return volatility

# Add technical indicators to stock data
def add_technical_indicators(stock_data):
    stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['SMA_200'] = stock_data['Close'].rolling(window=200).mean()
    stock_data['RSI'] = compute_rsi(stock_data['Close'])
    stock_data['Upper_BB'], stock_data['Lower_BB'] = compute_bollinger_bands(stock_data['Close'])
    stock_data['MACD'], stock_data['Signal'] = compute_macd(stock_data['Close'])
    stock_data['Volatility'] = compute_volatility(stock_data['Close'])
    stock_data.dropna(inplace=True)
    return stock_data

# Preprocess data
def preprocess_data(stock_data):
    features = stock_data[['Close', 'SMA_50', 'SMA_200', 'RSI', 'Upper_BB', 'Lower_BB', 'MACD', 'Signal', 'Volatility']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(features)

    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]

    def create_dataset(data, time_step=60):
        X, Y = [], []
        for i in range(len(data) - time_step - 1):
            X.append(data[i:(i + time_step), :-1])
            Y.append(data[i + time_step, 0])
        return np.array(X), np.array(Y)

    X_train, Y_train = create_dataset(train_data)
    X_test, Y_test = create_dataset(test_data)

    return X_train, Y_train, X_test, Y_test, scaler

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

# Save and load model functions
def save_model(model, ticker):
    model.save(f"models/{ticker}_model.h5")

def load_model(ticker):
    return tf.keras.models.load_model(f"models/{ticker}_model.h5")

# Train or load the model
def train_and_predict(ticker, progress_bar, retrain=False):
    model_path = f"models/{ticker}_model.h5"

    if retrain or not os.path.exists(model_path):
        stock_data = get_stock_data(ticker)
        stock_data = add_technical_indicators(stock_data)
        X_train, Y_train, X_test, Y_test, scaler = preprocess_data(stock_data)

        model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
        model.fit(X_train, Y_train, epochs=40, batch_size=64, verbose=0)

        save_model(model, ticker)
    else:
        model = load_model(ticker)

    stock_data = get_stock_data(ticker)
    stock_data = add_technical_indicators(stock_data)
    X_train, Y_train, X_test, Y_test, scaler = preprocess_data(stock_data)

    predicted_stock_price = model.predict(X_test)

    predicted_stock_price = scaler.inverse_transform(
        np.concatenate([predicted_stock_price, np.zeros((predicted_stock_price.shape[0], 8))], axis=1)
    )[:, 0]
    Y_test_actual = scaler.inverse_transform(
        np.concatenate([Y_test.reshape(-1, 1), np.zeros((Y_test.shape[0], 8))], axis=1)
    )[:, 0]

    rmse = np.sqrt(mean_squared_error(Y_test_actual, predicted_stock_price))
    r2 = r2_score(Y_test_actual, predicted_stock_price)

    return predicted_stock_price[-1], rmse, r2

# Get next day prediction
def get_next_day_prediction(ticker, date, retrain=False):
    with st.spinner('Training model... Please wait'):
        progress_bar = st.progress(0)
        final_prediction, rmse, r2 = train_and_predict(ticker, progress_bar, retrain)

    next_day_date = (datetime.datetime.strptime(date, "%Y-%m-%d") + datetime.timedelta(days=1)).strftime('%Y-%m-%d')

    st.write(f"**Prediction for {ticker} on {next_day_date}:** ₹{final_prediction:.2f}")
    st.write(f"**RMSE:** {rmse:.2f}")
    st.write(f"**R² Score:** {r2:.2f}")

# Streamlit UI
st.title("📈 Stock Price Prediction")
st.write("Enter the stock ticker to predict its next-day price.")

ticker = st.text_input("Stock Ticker (e.g., TCS.NS):")
date = st.date_input("Select Prediction Date", min_value=datetime.date.today())

predict_button = st.button("Predict")
retrain_button = st.button("Retrain Model")

if ticker and predict_button:
    get_next_day_prediction(ticker, str(date))

if retrain_button:
    get_next_day_prediction(ticker, str(date), retrain=True)

# Plot actual vs predicted prices
st.subheader("📊 Predicted vs Actual Prices")
if ticker:
    st.write("Visualizing the model's accuracy...")
    stock_data = get_stock_data(ticker)
    stock_data = add_technical_indicators(stock_data)

    _, _, X_test, Y_test, scaler = preprocess_data(stock_data)
    model = load_model(ticker)
    predicted_stock_price = model.predict(X_test)

    predicted_stock_price = scaler.inverse_transform(
        np.concatenate([predicted_stock_price, np.zeros((predicted_stock_price.shape[0], 8))], axis=1)
    )[:, 0]

    plt.figure(figsize=(10, 5))
    plt.plot(predicted_stock_price, label="Predicted")
    plt.plot(Y_test, label="Actual")
    plt.legend()
    st.pyplot(plt)
