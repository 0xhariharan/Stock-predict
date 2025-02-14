import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import datetime

# Function to load stock data (10 years for better trend capture)
def get_stock_data(ticker):
    stock_data = yf.download(ticker, period="10y", interval="1d")
    return stock_data

# Function to compute RSI
def compute_rsi(data, window=14):
    diff = data.diff()
    gain = diff.where(diff > 0, 0).rolling(window=window).mean()
    loss = -diff.where(diff < 0, 0).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Function to compute Bollinger Bands
def compute_bollinger_bands(data, window=20):
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    return rolling_mean + (rolling_std * 2), rolling_mean - (rolling_std * 2)

# Function to compute MACD
def compute_macd(data):
    short_ema = data.ewm(span=12, adjust=False).mean()
    long_ema = data.ewm(span=26, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

# Function to compute ATR (Average True Range for volatility)
def compute_atr(stock_data, window=14):
    high_low = stock_data['High'] - stock_data['Low']
    high_close = abs(stock_data['High'] - stock_data['Close'].shift())
    low_close = abs(stock_data['Low'] - stock_data['Close'].shift())

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    return atr

# Function to add technical indicators
def add_technical_indicators(stock_data):
    stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['RSI'] = compute_rsi(stock_data['Close'])
    stock_data['Upper_BB'], stock_data['Lower_BB'] = compute_bollinger_bands(stock_data['Close'])
    stock_data['MACD'], stock_data['Signal'] = compute_macd(stock_data['Close'])
    stock_data['ATR'] = compute_atr(stock_data)  # Adding ATR

    stock_data.dropna(inplace=True)
    return stock_data

# Function to preprocess stock data
def preprocess_data(stock_data):
    features = stock_data[['Close', 'SMA_50', 'RSI', 'Upper_BB', 'Lower_BB', 'MACD', 'Signal', 'ATR']].values

    scalers = {}  # Store individual scalers for each feature
    scaled_data = np.zeros_like(features)

    for i in range(features.shape[1]):
        scalers[i] = MinMaxScaler(feature_range=(0, 1))
        scaled_data[:, i] = scalers[i].fit_transform(features[:, i].reshape(-1, 1)).flatten()

    train_size = int(len(scaled_data) * 0.9)  # Train-test split (90-10)
    train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

    def create_dataset(data, time_step=50):  # Increased from 30 to 50 days
        X, Y = [], []
        for i in range(len(data) - time_step - 1):
            X.append(data[i:(i + time_step), :])  # Using all features
            Y.append(data[i + time_step, 0])  # Predicting Close price
        return np.array(X), np.array(Y)

    X_train, Y_train = create_dataset(train_data)
    X_test, Y_test = create_dataset(test_data)

    return X_train, Y_train, X_test, Y_test, scalers

# Function to build a stronger LSTM model
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(80, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(80, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to train and predict
def train_and_predict(ticker, progress_bar):
    stock_data = get_stock_data(ticker)
    stock_data = add_technical_indicators(stock_data)

    X_train, Y_train, X_test, Y_test, scalers = preprocess_data(stock_data)

    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, Y_train, epochs=40, batch_size=32, verbose=0)  # Increased to 40 epochs

    predicted_stock_price = model.predict(X_test)

    # Inverse transform only Close price
    predicted_stock_price = scalers[0].inverse_transform(predicted_stock_price)
    Y_test_actual = scalers[0].inverse_transform(Y_test.reshape(-1, 1))

    rmse = np.sqrt(mean_squared_error(Y_test_actual, predicted_stock_price))
    r2 = r2_score(Y_test_actual, predicted_stock_price)

    progress_bar.progress(1)

    return predicted_stock_price[-1][0], rmse, r2

# Function to get next day prediction
def get_next_day_prediction(ticker):
    with st.spinner('Training model... Please wait'):
        progress_bar = st.progress(0)
        final_prediction, rmse, r2 = train_and_predict(ticker, progress_bar)
        
    next_day_date = (datetime.datetime.now() + datetime.timedelta(days=1)).strftime('%Y-%m-%d')

    st.write(f"**Prediction for {ticker} on {next_day_date}: ₹{final_prediction:.2f}**")
    st.write(f"**RMSE:** {rmse:.2f}")
    st.write(f"**R²:** {r2:.2f}")

# Streamlit UI
st.title("Stock Price Prediction")
st.write("Enter the stock ticker symbol to predict its next day price:")

ticker = st.text_input("Stock Ticker (e.g., TCS.NS, RELIANCE.NS):")
predict_button = st.button("Predict")

if ticker and predict_button:
    get_next_day_prediction(ticker)
