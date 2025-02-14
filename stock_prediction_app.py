import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt


# Fetch stock data for a specific ticker symbol
def fetch_data(ticker):
    stock_data = yf.download(ticker, period="1y", interval="1d")
    return stock_data


# Calculate indicators without TA-Lib
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

    # Volatility
    stock_data['Volatility'] = stock_data['Close'].rolling(window=14).std()

    # Drop NaN values
    stock_data = stock_data.dropna(subset=['Close', 'SMA_50', 'SMA_200', 'RSI', 'Upper_BB', 'Lower_BB', 'MACD', 'Signal', 'Volatility'])

    return stock_data


# Split the data for training and testing
def train_predict_model(stock_data):
    X = stock_data[['SMA_50', 'SMA_200', 'RSI', 'Upper_BB', 'Lower_BB', 'MACD', 'Signal', 'Volatility']]
    y = stock_data['Close']  # Predicting the closing price

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the model
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = model.predict(X_test_scaled)

    # Calculate RMSE
    rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
    print(f"RMSE: {rmse}")
    
    return model


# Predict next day's closing price
def predict_next_day(model, stock_data):
    last_data = stock_data.iloc[-1][['SMA_50', 'SMA_200', 'RSI', 'Upper_BB', 'Lower_BB', 'MACD', 'Signal', 'Volatility']].values.reshape(1, -1)
    predicted_price = model.predict(last_data)
    print(f"Predicted Next Day Closing Price: {predicted_price[0]}")


# Visualize actual vs predicted stock prices
def plot_predictions(stock_data, y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.plot(stock_data.index[-len(y_test):], y_test, label="Actual Prices")
    plt.plot(stock_data.index[-len(y_test):], y_pred, label="Predicted Prices")
    plt.title('Actual vs Predicted Stock Prices')
    plt.xlabel('Date')
    plt.ylabel('Price (INR)')
    plt.legend()
    plt.show()


# Main execution flow
if __name__ == "__main__":
    ticker = "TCS.NS"  # You can change this to any stock ticker symbol
    data = fetch_data(ticker)  # Fetch data for the stock
    
    # Calculate technical indicators
    stock_data = calculate_indicators(data)

    # Train the model and get predictions
    model = train_predict_model(stock_data)

    # Predict the next day's closing price
    predict_next_day(model, stock_data)

    # Visualize actual vs predicted stock prices
    y_test = stock_data.iloc[-int(0.2 * len(stock_data)):]['Close']
    y_pred = model.predict(StandardScaler().fit_transform(stock_data[['SMA_50', 'SMA_200', 'RSI', 'Upper_BB', 'Lower_BB', 'MACD', 'Signal', 'Volatility']].iloc[-int(0.2 * len(stock_data)):]))
    plot_predictions(stock_data, y_test, y_pred)
