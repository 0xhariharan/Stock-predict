import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Function to calculate technical indicators
def calculate_indicators(stock_data):
    # Calculating Simple Moving Averages (SMA)
    stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['SMA_200'] = stock_data['Close'].rolling(window=200).mean()

    # Relative Strength Index (RSI)
    delta = stock_data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    stock_data['RSI'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    stock_data['Upper_BB'] = stock_data['SMA_50'] + (stock_data['Close'].rolling(window=50).std() * 2)
    stock_data['Lower_BB'] = stock_data['SMA_50'] - (stock_data['Close'].rolling(window=50).std() * 2)

    # MACD (Moving Average Convergence Divergence)
    stock_data['EMA_12'] = stock_data['Close'].ewm(span=12, adjust=False).mean()
    stock_data['EMA_26'] = stock_data['Close'].ewm(span=26, adjust=False).mean()
    stock_data['MACD'] = stock_data['EMA_12'] - stock_data['EMA_26']
    stock_data['Signal'] = stock_data['MACD'].ewm(span=9, adjust=False).mean()

    # Volatility (Standard deviation)
    stock_data['Volatility'] = stock_data['Close'].rolling(window=14).std()

    # Drop rows with NaN values after indicator calculations
    stock_data = stock_data.dropna()

    return stock_data

# Function to fetch stock data and calculate indicators
def fetch_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date, interval="1d")
    if stock_data.empty:
        raise ValueError(f"No data found for {ticker} from {start_date} to {end_date}.")
    stock_data = calculate_indicators(stock_data)
    return stock_data

# Preprocessing data for LSTM
def preprocess_data(stock_data):
    stock_data = stock_data.dropna(subset=['Close', 'SMA_50', 'SMA_200', 'RSI', 'Upper_BB', 'Lower_BB', 'MACD', 'Signal', 'Volatility'])
    features = stock_data[['Close', 'SMA_50', 'SMA_200', 'RSI', 'Upper_BB', 'Lower_BB', 'MACD', 'Signal', 'Volatility']]
    target = stock_data['Close']
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(features)
    X = []
    y = []
    for i in range(60, len(stock_data)):
        X.append(scaled_features[i-60:i])
        y.append(target[i])
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y, scaler

# Function to create and train the LSTM model
def create_model(X_train):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to train and predict stock prices
def train_and_predict(ticker, start_date, end_date):
    stock_data = fetch_data(ticker, start_date, end_date)
    
    X, y, scaler = preprocess_data(stock_data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = create_model(X_train)
    model.fit(X_train, y_train, epochs=10, batch_size=32)
    
    # Predict on the test data
    predictions = model.predict(X_test)
    
    # Inverse scale predictions and actual values
    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    return predictions, rmse, r2

# Example usage
if __name__ == "__main__":
    ticker = 'AAPL'  # Change to the stock ticker of your choice
    start_date = '2020-01-01'
    end_date = '2023-12-31'
    
    predictions, rmse, r2 = train_and_predict(ticker, start_date, end_date)
    print(f"Predictions: {predictions}")
    print(f"RMSE: {rmse}")
    print(f"R²: {r2}")
