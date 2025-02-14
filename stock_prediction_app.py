import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import datetime

# Function to fetch stock data from Yahoo Finance
def fetch_stock_data(ticker, start_date, end_date, interval):
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    return data

# Function to calculate technical indicators
def calculate_indicators(stock_data):
    # Simple Moving Averages (SMA)
    stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['SMA_200'] = stock_data['Close'].rolling(window=200).mean()

    # Relative Strength Index (RSI)
    delta = stock_data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    stock_data['RSI'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    stock_data['Upper_BB'] = stock_data['Close'].rolling(window=20).mean() + (stock_data['Close'].rolling(window=20).std() * 2)
    stock_data['Lower_BB'] = stock_data['Close'].rolling(window=20).mean() - (stock_data['Close'].rolling(window=20).std() * 2)

    # MACD
    stock_data['EMA_12'] = stock_data['Close'].ewm(span=12, adjust=False).mean()
    stock_data['EMA_26'] = stock_data['Close'].ewm(span=26, adjust=False).mean()
    stock_data['MACD'] = stock_data['EMA_12'] - stock_data['EMA_26']
    stock_data['Signal'] = stock_data['MACD'].ewm(span=9, adjust=False).mean()

    # Volatility (Standard Deviation of Returns)
    stock_data['Returns'] = stock_data['Close'].pct_change()
    stock_data['Volatility'] = stock_data['Returns'].rolling(window=21).std()

    return stock_data

# Function to preprocess data for training
def preprocess_data(stock_data):
    stock_data = stock_data.dropna(subset=['Close', 'SMA_50', 'SMA_200', 'RSI', 'Upper_BB', 'Lower_BB', 'MACD', 'Signal', 'Volatility'])
    features = stock_data[['Close', 'SMA_50', 'SMA_200', 'RSI', 'Upper_BB', 'Lower_BB', 'MACD', 'Signal', 'Volatility']]
    target = stock_data['Close']

    # Scale features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(features)
    X = []
    Y = []

    # Create sequences of data for LSTM model
    for i in range(60, len(scaled_features)):
        X.append(scaled_features[i-60:i])
        Y.append(target[i])

    X = np.array(X)
    Y = np.array(Y)
    
    # Train-test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)
    
    return X_train, Y_train, X_test, Y_test, scaler

# Function to create and train the LSTM model
def create_model(X_train):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to train and predict using the model
def train_and_predict(ticker, interval):
    # Fetch stock data
    end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime('%Y-%m-%d')
    stock_data = fetch_stock_data(ticker, start_date, end_date, interval)

    # Calculate technical indicators
    stock_data = calculate_indicators(stock_data)

    # Preprocess data for model training
    X_train, Y_train, X_test, Y_test, scaler = preprocess_data(stock_data)

    # Create the LSTM model
    model = create_model(X_train)

    # Train the model
    model.fit(X_train, Y_train, epochs=5, batch_size=32)

    # Predict on test data
    predictions = model.predict(X_test)

    # Calculate RMSE and R2 score
    rmse = np.sqrt(mean_squared_error(Y_test, predictions))
    r2 = r2_score(Y_test, predictions)

    # Inverse transform the predicted prices
    predictions = scaler.inverse_transform(predictions)
    Y_test = scaler.inverse_transform(Y_test.reshape(-1, 1))

    # Plot predictions vs true values
    plt.figure(figsize=(10, 6))
    plt.plot(Y_test, color='blue', label='Actual Stock Price')
    plt.plot(predictions, color='red', label='Predicted Stock Price')
    plt.title(f'{ticker} Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

    return predictions[-1], rmse, r2

# Function to get the next day's prediction
def get_next_day_prediction(ticker, interval):
    # Fetch stock data
    end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime('%Y-%m-%d')
    stock_data = fetch_stock_data(ticker, start_date, end_date, interval)

    # Calculate technical indicators
    stock_data = calculate_indicators(stock_data)

    # Preprocess data for model training
    X_train, Y_train, X_test, Y_test, scaler = preprocess_data(stock_data)

    # Load trained model and predict
    model = create_model(X_train)
    model.fit(X_train, Y_train, epochs=5, batch_size=32)
    prediction = model.predict(X_test)

    # Inverse transform to get the real stock price
    prediction = scaler.inverse_transform(prediction)

    return prediction[-1][0]

# Example usage
if __name__ == '__main__':
    ticker = 'TCS.NS'  # Change to your desired stock ticker
    interval = '5m'  # Change to '15m' for 15-minute interval
    prediction, rmse, r2 = train_and_predict(ticker, interval)
    print(f"Predicted stock price: {prediction}")
    print(f"RMSE: {rmse}")
    print(f"R²: {r2}")
