import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout

MODEL_PATH = "saved_model.h5"  # File path to save the trained model

# Function to fetch stock data
def get_stock_data(ticker):
    stock_data = yf.download(ticker, period="10y", interval="1d")  
    return stock_data

# Function to compute RSI (Relative Strength Index)
def compute_rsi(data, window=14):
    diff = data.diff()
    gain = (diff.where(diff > 0, 0)).rolling(window=window).mean()
    loss = (-diff.where(diff < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Function to compute Bollinger Bands
def compute_bollinger_bands(data, window=20):
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * 2)
    lower_band = rolling_mean - (rolling_std * 2)
    return upper_band, lower_band

# Function to compute MACD
def compute_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data.ewm(span=short_window, adjust=False).mean()
    long_ema = data.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal

# Function to compute Historical Volatility
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
    
    stock_data.dropna(inplace=True)  # Remove NaN values
    return stock_data

# Preprocess data for training
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

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("Insufficient data for training. Try a different stock.")

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

# Save the trained model
def save_model(model):
    model.save(MODEL_PATH)

# Load the saved model
def load_saved_model():
    if os.path.exists(MODEL_PATH):
        return load_model(MODEL_PATH)
    return None

# Train and predict stock price
def train_and_predict(ticker, progress_bar, retrain=False):
    try:
        stock_data = get_stock_data(ticker)
        stock_data = add_technical_indicators(stock_data)

        X_train, Y_train, X_test, Y_test, scaler = preprocess_data(stock_data)

        # Check if we already have a trained model
        if os.path.exists(MODEL_PATH) and not retrain:
            st.success("✅ Using saved model to predict")
            model = load_saved_model()
        else:
            st.warning("⚡ Training new model, this may take time...")
            model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
            model.fit(X_train, Y_train, epochs=30, batch_size=32, verbose=0)
            save_model(model)  # Save model after training
            st.success("✅ Model trained and saved!")

        predicted_stock_price = model.predict(X_test)

        # Reverse scaling
        predicted_stock_price = scaler.inverse_transform(
            np.concatenate([predicted_stock_price, np.zeros((predicted_stock_price.shape[0], 8))], axis=1)
        )[:, 0]
        
        Y_test_actual = scaler.inverse_transform(
            np.concatenate([Y_test.reshape(-1, 1), np.zeros((Y_test.shape[0], 8))], axis=1)
        )[:, 0]

        rmse = np.sqrt(mean_squared_error(Y_test_actual, predicted_stock_price))
        r2 = r2_score(Y_test_actual, predicted_stock_price)

        final_prediction = predicted_stock_price[-1]
        progress_bar.progress(100)

        return final_prediction, rmse, r2

    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None, None, None

# Function to display prediction result
def get_next_day_prediction(ticker, retrain=False):
    with st.spinner('Processing... Please wait'):
        progress_bar = st.progress(0)
        final_prediction, rmse, r2 = train_and_predict(ticker, progress_bar, retrain)

        if final_prediction:
            next_day_date = (datetime.datetime.now() + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
            st.write(f"**Prediction for {ticker} on {next_day_date}: ₹{final_prediction:.2f}**")
            st.write(f"**RMSE:** {rmse:.2f}")
            st.write(f"**R² Score:** {r2:.2f}")
        else:
            st.write("⚠️ Unable to generate prediction. Try a different stock or check the data.")

# Streamlit UI
st.title("📈 Stock Price Prediction")
st.write("Enter the stock ticker symbol to predict its next day price.")

ticker = st.text_input("Stock Ticker (e.g., TCS.NS, INFY.NS):")
predict_button = st.button("Predict")
retrain_button = st.button("Retrain Model")

if ticker and predict_button:
    get_next_day_prediction(ticker, retrain=False)

if ticker and retrain_button:
    get_next_day_prediction(ticker, retrain=True)
