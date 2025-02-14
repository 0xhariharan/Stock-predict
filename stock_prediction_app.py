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

# Function to load the data
def get_stock_data(ticker):
    stock_data = yf.download(ticker, period="10y", interval="1d")  # Increased to 10 years
    return stock_data

# Adding technical indicators to improve model's learning
def add_technical_indicators(stock_data):
    # 50-day and 200-day moving averages
    stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()  
    stock_data['SMA_200'] = stock_data['Close'].rolling(window=200).mean()  

    # Relative Strength Index (RSI)
    stock_data['RSI'] = compute_rsi(stock_data['Close'], 14)  
    # Bollinger Bands
    stock_data['Upper_BB'], stock_data['Lower_BB'] = compute_bollinger_bands(stock_data['Close'], window=20)
    # Moving Average Convergence Divergence (MACD)
    stock_data['MACD'], stock_data['Signal'] = compute_macd(stock_data['Close'])
    # Historical Volatility
    stock_data['Volatility'] = compute_volatility(stock_data['Close'], window=30)

    stock_data.dropna(inplace=True)  # Remove NaN values generated by rolling functions
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
    upper_band = rolling_mean + (rolling_std * 2)  # Upper band
    lower_band = rolling_mean - (rolling_std * 2)  # Lower band
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
    volatility = log_returns.rolling(window=window).std() * np.sqrt(window)  # Annualized volatility
    return volatility

# Function to preprocess the stock data
def preprocess_data(stock_data):
    features = stock_data[['Close', 'SMA_50', 'SMA_200', 'RSI', 'Upper_BB', 'Lower_BB', 'MACD', 'Signal', 'Volatility']].values
    
    # Scale data
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
    
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))
    
    return X_train, Y_train, X_test, Y_test, scaler

# Function to build the LSTM model with dropout and increased LSTM units
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))  
    model.add(LSTM(units=100, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to train the model and get predictions
def train_and_predict(ticker, progress_bar):
    stock_data = get_stock_data(ticker)
    stock_data = add_technical_indicators(stock_data)
    
    X_train, Y_train, X_test, Y_test, scaler = preprocess_data(stock_data)
    
    predictions = []
    rmse_scores = []
    r2_scores = []
    
    for i in range(10):
        model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
        model.fit(X_train, Y_train, epochs=50, batch_size=32, verbose=0)
        
        predicted_stock_price = model.predict(X_test)
        
        predicted_stock_price = scaler.inverse_transform(np.concatenate([predicted_stock_price, np.zeros((predicted_stock_price.shape[0], 8))], axis=1))[:, 0]
        Y_test_actual = scaler.inverse_transform(np.concatenate([Y_test.reshape(-1, 1), np.zeros((Y_test.shape[0], 8))], axis=1))[:, 0]
        
        predictions.append(predicted_stock_price[-1])
        rmse = np.sqrt(mean_squared_error(Y_test_actual, predicted_stock_price))
        rmse_scores.append(rmse)
        r2 = r2_score(Y_test_actual, predicted_stock_price)
        r2_scores.append(r2)
        
        # Update progress bar
        progress_bar.progress((i + 1) / 10)

    final_prediction = np.mean(predictions)
    average_rmse = np.mean(rmse_scores)
    average_r2 = np.mean(r2_scores)
    
    return final_prediction, average_rmse, average_r2

# Function to get the next day prediction
def get_next_day_prediction(ticker):
    with st.spinner('Training model... Please wait'):
        progress_bar = st.progress(0)
        final_prediction, average_rmse, average_r2 = train_and_predict(ticker, progress_bar)
        
    next_day_date = (datetime.datetime.now() + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    
    st.write(f"Prediction for {ticker} on {next_day_date}: {final_prediction}")
    st.write(f"RMSE: {average_rmse}")
    st.write(f"R²: {average_r2}")

# Streamlit UI
st.title("Stock Price Prediction")
st.write("Enter the stock ticker symbol to predict its next day price:")

ticker = st.text_input("Stock Ticker (e.g., AAPL):")
predict_button = st.button("Predict")

if ticker and predict_button:
    get_next_day_prediction(ticker)
