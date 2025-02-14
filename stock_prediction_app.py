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

    # Volatility
    stock_data['Volatility'] = stock_data['Close'].rolling(window=50).std()

    # Drop rows with missing values (e.g., first few rows due to rolling calculations)
    stock_data.dropna(inplace=True)
    
    return stock_data

# Fetch historical stock data
def fetch_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data = calculate_indicators(stock_data)
    return stock_data

# Preprocess the data for LSTM model
def preprocess_data(stock_data):
    features = stock_data[['Close', 'SMA_50', 'SMA_200', 'RSI', 'Upper_BB', 'Lower_BB', 'MACD', 'Signal', 'Volatility']]
    target = stock_data['Close']

    scaler = MinMaxScaler(feature_range=(0, 1))
    features_scaled = scaler.fit_transform(features)

    # Split data into training and testing sets
    X = features_scaled
    Y = target.values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

    return X_train, Y_train, X_test, Y_test, scaler

# Create LSTM model
def create_model(X_train):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train the model and make predictions
def train_and_predict(ticker, start_date, end_date):
    stock_data = fetch_data(ticker, start_date, end_date)
    X_train, Y_train, X_test, Y_test, scaler = preprocess_data(stock_data)

    # Reshape X_train for LSTM input
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    model = create_model(X_train)
    model.fit(X_train, Y_train, epochs=10, batch_size=32)

    # Predict the stock prices
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    Y_test_rescaled = scaler.inverse_transform(Y_test.reshape(-1, 1))

    # Calculate RMSE and R-squared
    rmse = np.sqrt(mean_squared_error(Y_test_rescaled, predictions))
    r2 = r2_score(Y_test_rescaled, predictions)

    return predictions, rmse, r2

# Example usage
if __name__ == '__main__':
    ticker = 'TCS.NS'
    start_date = '2020-01-01'
    end_date = '2025-01-01'

    predictions, rmse, r2 = train_and_predict(ticker, start_date, end_date)

    print(f"Predicted stock prices: {predictions}")
    print(f"RMSE: {rmse}")
    print(f"R²: {r2}")
