import requests
import os
import datetime
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import streamlit as st
import matplotlib.pyplot as plt

# Replace these with your actual API key, secret, and redirect URI
API_KEY = 'your_api_key'
API_SECRET = 'your_api_secret'
REDIRECT_URI = 'your_redirect_uri'

# Function to get the access token from Upstox API using OAuth2
def get_access_token():
    # Step 1: Redirect the user to the authorization URL
    auth_url = f'https://api.upstox.com/v2/login/authorization/dialog?response_type=code&client_id={API_KEY}&redirect_uri={REDIRECT_URI}'
    print(f'Please go to this URL and authorize: {auth_url}')

    # Step 2: Get the authorization code from the redirect URL
    authorization_code = input('Enter the authorization code from the redirect URL: ')

    # Step 3: Exchange the authorization code for an access token
    token_url = 'https://api.upstox.com/v2/login/authorization/token'
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    data = {
        'code': authorization_code,
        'client_id': API_KEY,
        'client_secret': API_SECRET,
        'redirect_uri': REDIRECT_URI,
        'grant_type': 'authorization_code'
    }

    response = requests.post(token_url, headers=headers, data=data)
    response_data = response.json()

    if 'access_token' in response_data:
        access_token = response_data['access_token']
        print(f'Access Token: {access_token}')
        return access_token
    else:
        print('Error getting access token:', response_data)
        return None

# Function to get stock data from Upstox API
def get_stock_data(ticker, access_token):
    url = f'https://api.upstox.com/v2/market/quote'
    headers = {'Authorization': f'Bearer {access_token}'}
    params = {'symbol': ticker}
    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code == 200:
        data = response.json()
        stock_data = {
            'Date': [data['timestamp']],
            'Open': [data['open_price']],
            'High': [data['high_price']],
            'Low': [data['low_price']],
            'Close': [data['last_price']],
            'Volume': [data['quantity_traded']]
        }
        stock_df = pd.DataFrame(stock_data)
        stock_df['Date'] = pd.to_datetime(stock_df['Date'])
        return stock_df
    else:
        print(f"Error fetching stock data: {response.status_code}")
        return None

# Function to preprocess the stock data and prepare features
def preprocess_data(stock_data):
    features = stock_data[['Close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(features)

    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]

    def create_dataset(data, time_step=60):
        X, Y = [], []
        for i in range(len(data) - time_step - 1):
            X.append(data[i:(i + time_step), 0])
            Y.append(data[i + time_step, 0])
        return np.array(X), np.array(Y)

    X_train, Y_train = create_dataset(train_data)
    X_test, Y_test = create_dataset(test_data)

    return X_train, Y_train, X_test, Y_test, scaler

# Function to build the LSTM model
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

# Function to save the trained model
def save_model(model, ticker):
    model.save(f"models/{ticker}_model.h5")

# Function to load the trained model
def load_model(ticker):
    return tf.keras.models.load_model(f"models/{ticker}_model.h5")

# Main prediction function
def train_and_predict(ticker, access_token, retrain=False):
    model_path = f"models/{ticker}_model.h5"

    if retrain or not os.path.exists(model_path):
        stock_data = get_stock_data(ticker, access_token)
        if stock_data is None:
            return None, None, None
        
        X_train, Y_train, X_test, Y_test, scaler = preprocess_data(stock_data)

        model = build_lstm_model((X_train.shape[1], 1))
        model.fit(X_train, Y_train, epochs=40, batch_size=64, verbose=0)

        save_model(model, ticker)
    else:
        model = load_model(ticker)

    stock_data = get_stock_data(ticker, access_token)
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

# Streamlit UI
st.title("📈 Stock Price Prediction")
st.write("Enter the stock ticker to predict its next-day price.")

ticker = st.text_input("Stock Ticker (e.g., TCS.NS):")
date = st.date_input("Select Prediction Date", min_value=datetime.date.today())

predict_button = st.button("Predict")
retrain_button = st.button("Retrain Model")

if ticker:
    # Get access token once and store it
    access_token = get_access_token()

    if access_token:
        if predict_button:
            st.spinner('Predicting next-day stock price...')
            final_prediction, rmse, r2 = train_and_predict(ticker, access_token)
            st.write(f"**Prediction for {ticker} on {date}:** ₹{final_prediction:.2f}")
            st.write(f"**RMSE:** {rmse:.2f}")
            st.write(f"**R² Score:** {r2:.2f}")

        if retrain_button:
            st.spinner('Retraining model...')
            final_prediction, rmse, r2 = train_and_predict(ticker, access_token, retrain=True)
            st.write(f"**Prediction for {ticker} on {date}:** ₹{final_prediction:.2f}")
            st.write(f"**RMSE:** {rmse:.2f}")
            st.write(f"**R² Score:** {r2:.2f}")

        # Plot actual vs predicted prices
        st.subheader("📊 Predicted vs Actual Prices")
        if ticker:
            st.write("Visualizing the model's accuracy...")
            stock_data = get_stock_data(ticker, access_token)

            X_train, Y_train, X_test, Y_test, scaler = preprocess_data(stock_data)
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
