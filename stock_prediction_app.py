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

# Function to load stock data
def get_stock_data(ticker):
    stock_data = yf.download(ticker, period="10y", interval="1d")  
    return stock_data

# Function to compute volume-based indicators
def add_volume_indicators(stock_data):
    stock_data["OBV"] = (np.sign(stock_data["Close"].diff()) * stock_data["Volume"]).cumsum()
    stock_data["VWAP"] = (stock_data["Close"] * stock_data["Volume"]).cumsum() / stock_data["Volume"].cumsum()
    stock_data["Volume_EMA"] = stock_data["Volume"].ewm(span=14, adjust=False).mean()
    stock_data.dropna(inplace=True)
    return stock_data

# Function to preprocess data
def preprocess_data(stock_data):
    features = stock_data[['Close', 'Volume', 'OBV', 'VWAP', 'Volume_EMA']].values
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

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2]))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2]))

    return X_train, Y_train, X_test, Y_test, scaler

# Function to build the LSTM model
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(units=100, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=100, return_sequences=False),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to save and load model
def save_model(model, ticker):
    model.save(f"models/{ticker}_model.h5")

def load_model(ticker):
    return tf.keras.models.load_model(f"models/{ticker}_model.h5")

# Function to train the model and make predictions
def train_and_predict(ticker, progress_bar, retrain=False):
    if retrain or not os.path.exists(f"models/{ticker}_model.h5"):
        stock_data = get_stock_data(ticker)
        stock_data = add_volume_indicators(stock_data)
        X_train, Y_train, X_test, Y_test, scaler = preprocess_data(stock_data)

        model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
        model.fit(X_train, Y_train, epochs=40, batch_size=64, verbose=0)

        save_model(model, ticker)
    else:
        model = load_model(ticker)

    stock_data = get_stock_data(ticker)
    stock_data = add_volume_indicators(stock_data)
    X_train, Y_train, X_test, Y_test, scaler = preprocess_data(stock_data)

    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = scaler.inverse_transform(
        np.concatenate([predicted_stock_price, np.zeros((predicted_stock_price.shape[0], 4))], axis=1))[:, 0]
    Y_test_actual = scaler.inverse_transform(
        np.concatenate([Y_test.reshape(-1, 1), np.zeros((Y_test.shape[0], 4))], axis=1))[:, 0]

    rmse = np.sqrt(mean_squared_error(Y_test_actual, predicted_stock_price))
    r2 = r2_score(Y_test_actual, predicted_stock_price)

    return predicted_stock_price[-1], rmse, r2

# Function to get next day prediction
def get_next_day_prediction(ticker, date, retrain=False):
    with st.spinner("Training model... Please wait"):
        progress_bar = st.progress(0)
        final_prediction, rmse, r2 = train_and_predict(ticker, progress_bar, retrain)

    next_day_date = (datetime.datetime.strptime(date, "%Y-%m-%d") + datetime.timedelta(days=1)).strftime('%Y-%m-%d')

    st.write(f"Prediction for {ticker} on {next_day_date}: ₹{final_prediction:.2f}")
    st.write(f"RMSE: {rmse:.2f}")
    st.write(f"R²: {r2:.2f}")

# Streamlit UI
st.title("Stock Price Prediction with Volume Indicators")
st.write("Enter the stock ticker symbol to predict its next-day price:")

ticker = st.text_input("Stock Ticker (e.g., TCS.NS):")
date = st.date_input("Select Date to Predict", min_value=datetime.date.today())

predict_button = st.button("Predict")
retrain_button = st.button("Retrain Model")

if ticker and predict_button:
    get_next_day_prediction(ticker, str(date))

if retrain_button:
    get_next_day_prediction(ticker, str(date), retrain=True)

# Visualization
st.write("### Predicted vs Actual Stock Prices")
if ticker:
    stock_data = get_stock_data(ticker)
    stock_data = add_volume_indicators(stock_data)

    X_train, Y_train, X_test, Y_test, scaler = preprocess_data(stock_data)
    
    model = load_model(ticker)
    predicted_stock_price = model.predict(X_test)
    
    predicted_stock_price = scaler.inverse_transform(
        np.concatenate([predicted_stock_price, np.zeros((predicted_stock_price.shape[0], 4))], axis=1))[:, 0]
    Y_test_actual = scaler.inverse_transform(
        np.concatenate([Y_test.reshape(-1, 1), np.zeros((Y_test.shape[0], 4))], axis=1))[:, 0]
    
    fig, ax = plt.subplots()
    ax.plot(Y_test_actual, color="blue", label="Actual Price")
    ax.plot(predicted_stock_price, color="red", label="Predicted Price")
    ax.set_title(f"{ticker} Price Prediction")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.legend()
    
    st.pyplot(fig)
