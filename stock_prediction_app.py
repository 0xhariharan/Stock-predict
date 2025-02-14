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
import matplotlib.pyplot as plt
import os

# Function to load the data
def get_stock_data(ticker, interval='1d'):
    stock_data = yf.download(ticker, period="10y", interval=interval)  # 10 years of data, can be adjusted
    return stock_data

# Adding technical indicators to improve model's learning
def add_technical_indicators(stock_data):
    stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['SMA_200'] = stock_data['Close'].rolling(window=200).mean()

    stock_data['RSI'] = compute_rsi(stock_data['Close'], 14)
    stock_data['Upper_BB'], stock_data['Lower_BB'] = compute_bollinger_bands(stock_data['Close'], window=20)
    stock_data['MACD'], stock_data['Signal'] = compute_macd(stock_data['Close'])
    stock_data['Volatility'] = compute_volatility(stock_data['Close'], window=30)

    stock_data.dropna(inplace=True)
    return stock_data

# Function to compute RSI (Relative Strength Index)
def compute_rsi(data, window):
    diff = data.diff()
    gain = (diff.where(diff > 0, 0)).rolling(window=window).mean()
    loss = (-diff.where(diff < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to compute Bollinger Bands
def compute_bollinger_bands(data, window=20):
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * 2)
    lower_band = rolling_mean - (rolling_std * 2)
    return upper_band, lower_band

# Function to compute MACD (Moving Average Convergence Divergence)
def compute_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data.ewm(span=short_window, min_periods=1, adjust=False).mean()
    long_ema = data.ewm(span=long_window, min_periods=1, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, min_periods=1, adjust=False).mean()
    return macd, signal

# Function to compute Historical Volatility
def compute_volatility(data, window=30):
    log_returns = np.log(data / data.shift(1))
    volatility = log_returns.rolling(window=window).std() * np.sqrt(window)
    return volatility

# Function to preprocess the stock data
def preprocess_data(stock_data):
    # Ensure there is no missing data
    stock_data = stock_data.dropna(subset=['Close', 'SMA_50', 'SMA_200', 'RSI', 'Upper_BB', 'Lower_BB', 'MACD', 'Signal', 'Volatility'])
    
    if stock_data.empty:
        raise ValueError("Stock data is empty after removing missing values.")

    features = stock_data[['Close', 'SMA_50', 'SMA_200', 'RSI', 'Upper_BB', 'Lower_BB', 'MACD', 'Signal', 'Volatility']].values
    
    # Check if there is any missing data in the features array
    if np.any(np.isnan(features)):
        raise ValueError("Feature data contains NaN values.")
    
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

    # Ensure X_train and X_test are 3D arrays for LSTM input
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))

    return X_train, Y_train, X_test, Y_test, scaler

# Function to build the LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to save the model
def save_model(model, ticker):
    model.save(f"models/{ticker}_model.h5")

# Function to load the model
def load_model(ticker):
    return tf.keras.models.load_model(f"models/{ticker}_model.h5")

# Function to train the model and get predictions
def train_and_predict(ticker, progress_bar, retrain=False, interval='1d'):
    if retrain or not os.path.exists(f"models/{ticker}_model.h5"):
        stock_data = get_stock_data(ticker, interval)
        stock_data = add_technical_indicators(stock_data)

        X_train, Y_train, X_test, Y_test, scaler = preprocess_data(stock_data)

        model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
        model.fit(X_train, Y_train, epochs=40, batch_size=64, verbose=0)

        save_model(model, ticker)

    else:
        model = load_model(ticker)

    stock_data = get_stock_data(ticker, interval)
    stock_data = add_technical_indicators(stock_data)
    X_train, Y_train, X_test, Y_test, scaler = preprocess_data(stock_data)

    predicted_stock_price = model.predict(X_test)

    predicted_stock_price = scaler.inverse_transform(np.concatenate([predicted_stock_price, np.zeros((predicted_stock_price.shape[0], 8))], axis=1))[:, 0]
    Y_test_actual = scaler.inverse_transform(np.concatenate([Y_test.reshape(-1, 1), np.zeros((Y_test.shape[0], 8))], axis=1))[:, 0]

    rmse = np.sqrt(mean_squared_error(Y_test_actual, predicted_stock_price))
    r2 = r2_score(Y_test_actual, predicted_stock_price)

    predictions = predicted_stock_price[-1]
    
    return predictions, rmse, r2

# Function to get the next day prediction
def get_next_day_prediction(ticker, date, retrain=False, interval='1d'):
    with st.spinner('Training model... Please wait'):
        progress_bar = st.progress(0)
        final_prediction, rmse, r2 = train_and_predict(ticker, progress_bar, retrain, interval)

    next_day_date = (datetime.datetime.strptime(date, "%Y-%m-%d") + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    
    st.write(f"Prediction for {ticker} on {next_day_date}: ₹{final_prediction:.2f}")
    st.write(f"RMSE: {rmse:.2f}")
    st.write(f"R²: {r2:.2f}")

# Streamlit UI
st.title("Stock Price Prediction")
st.write("Enter the stock ticker symbol to predict its next day price:")

ticker = st.text_input("Stock Ticker (e.g., TCS.NS):")
predict_button = st.button("Predict")
date = st.date_input("Select Date to Predict", min_value=datetime.date.today())
interval = st.selectbox("Select Data Interval", options=["1d", "5m", "15m"])

retrain_button = st.button("Retrain Model")

if ticker and predict_button:
    get_next_day_prediction(ticker, str(date), retrain=False, interval=interval)

if retrain_button:
    get_next_day_prediction(ticker, str(date), retrain=True, interval=interval)

st.write("### Stock Price Prediction Visualization")
st.subheader("Predicted vs Actual Prices")
if ticker:
    stock_data = get_stock_data(ticker, interval)
    stock_data = add_technical_indicators(stock_data)
    
    X_train, Y_train, X_test, Y_test, scaler = preprocess_data(stock_data)
    
    model = load_model(ticker)
    predicted_stock_price = model.predict(X_test)
    
    predicted_stock_price = scaler.inverse_transform(np.concatenate([predicted_stock_price, np.zeros((predicted_stock_price.shape[0], 8))], axis=1))[:, 0]
    Y_test_actual = scaler.inverse_transform(np.concatenate([Y_test.reshape(-1, 1), np.zeros((Y_test.shape[0], 8))], axis=1))[:, 0]
    
    fig, ax = plt.subplots()
    ax.plot(Y_test_actual, color='blue', label='Actual Price')
    ax.plot(predicted_stock_price, color='red', label='Predicted Price')
    ax.set_title(f'{ticker} Price Prediction')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.legend()
    
    st.pyplot(fig)
