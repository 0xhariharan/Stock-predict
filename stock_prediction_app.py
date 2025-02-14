pip install talib-binary
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import yfinance as yf
import talib
from keras.models import Sequential
from keras.layers import LSTM, Dense
from datetime import datetime, timedelta
import time

# Function to fetch stock data from Yahoo Finance
def fetch_data(ticker, interval='5m', lookback=1000):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback)
    data = yf.download(ticker, interval=interval, start=start_date, end=end_date)
    return data

# Function to compute technical indicators (SMA, RSI, MACD, Bollinger Bands, etc.)
def add_technical_indicators(df):
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['RSI'] = talib.RSI(df['Close'].values, timeperiod=14)
    df['Upper_BB'], df['Middle_BB'], df['Lower_BB'] = talib.BBANDS(df['Close'].values, timeperiod=20)
    df['MACD'], df['Signal'], _ = talib.MACD(df['Close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
    df['Volatility'] = df['Close'].rolling(window=20).std()
    df = df.dropna(subset=['SMA_50', 'SMA_200', 'RSI', 'Upper_BB', 'Lower_BB', 'MACD', 'Signal', 'Volatility'])
    return df

# Function to preprocess the stock data for LSTM model
def preprocess_data(df):
    features = df[['Close', 'SMA_50', 'SMA_200', 'RSI', 'Upper_BB', 'Lower_BB', 'MACD', 'Signal', 'Volatility']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(features)
    
    X = []
    Y = []
    for i in range(60, len(df)):
        X.append(scaled_data[i-60:i])
        Y.append(scaled_data[i, 0])
    
    X = np.array(X)
    Y = np.array(Y)
    
    return X, Y, scaler

# Function to create and train an LSTM model
def create_model(X_train):
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(units=100, return_sequences=False))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to load model if it exists
def load_model(ticker):
    model_path = f"models/{ticker}_model.h5"
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        raise FileNotFoundError(f"Model file for {ticker} not found at {model_path}.")

# Function to save the trained model
def save_model(model, ticker):
    model.save(f"models/{ticker}_model.h5")

# Function to train and predict the stock price
def train_and_predict(ticker, retrain=False, interval='5m'):
    stock_data = fetch_data(ticker, interval)
    stock_data = add_technical_indicators(stock_data)
    
    X_train, Y_train, scaler = preprocess_data(stock_data)
    
    if retrain:
        model = create_model(X_train)
        model.fit(X_train, Y_train, epochs=10, batch_size=32)
        save_model(model, ticker)
    else:
        model = load_model(ticker)
    
    X_test = X_train[-1:]
    predicted_stock_price = model.predict(X_test)
    predicted_price = scaler.inverse_transform(np.column_stack((predicted_stock_price, np.zeros((predicted_stock_price.shape[0], X_train.shape[2]-1)))))[:,0]
    
    rmse = np.sqrt(mean_squared_error(Y_train[-1:], predicted_stock_price))
    r2 = r2_score(Y_train[-1:], predicted_stock_price)
    
    return predicted_price[0], rmse, r2

# Function to get the next day's prediction for a given stock ticker and date
def get_next_day_prediction(ticker, date, retrain=False, interval='5m'):
    print(f"Prediction for {ticker} on {date}:")
    prediction, rmse, r2 = train_and_predict(ticker, retrain, interval)
    print(f"Predicted Price: ₹{prediction:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.2f}")

# Main block to run predictions for specific stock tickers
if __name__ == '__main__':
    ticker = 'TCS.NS'
    date = datetime.now().strftime("%Y-%m-%d")
    interval = '5m'  # Change to '15m' for 15-minute intervals
    
    get_next_day_prediction(ticker, date, retrain=False, interval=interval)
