# stock_prediction.py

from upstox_api.api import *
import datetime
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk

# Download Vader Sentiment lexicon
nltk.download('vader_lexicon')

# Upstox API credentials (replace with your actual credentials securely)
API_KEY = '6be46ee7-8e84-4c37-9353-63aa896d09bd'
API_SECRET = '2eqwmvko5i'
ACCESS_TOKEN = 'eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiIyQ0NYRUIiLCJqdGkiOiI2N2FlY2Q2NmZkNjhlZjVlYWZhYzg2ZjUiLCJpc011bHRpQ2xpZW50IjpmYWxzZSwiaWF0IjoxNzM5NTA5MDk0LCJpc3MiOiJ1ZGFwaS1nYXRld2F5LXNlcnZpY2UiLCJleHAiOjE3Mzk1NzA0MDB9.sMDkSpgfAaQRjFR7vHSspZ557NWVHIQMUm58UdT1KPI'
 # Use your actual Access Token

# Set up Upstox API
upstox = Upstox(api_key, api_secret)
upstox.set_access_token(access_token)

# Fetch historical data (example for TCS stock)
symbol = 'TCS'
start_date = datetime.datetime(2023, 1, 1)
end_date = datetime.datetime.now()

data = upstox.get_ohlc(upstox.get_instruments('NSE')[0], 
                       from_date=start_date, to_date=end_date, 
                       interval=Interval.Minute_5)
df = pd.DataFrame(data)

# Technical Indicator Calculations
# 1. Simple Moving Average (SMA)
df['SMA'] = df['close'].rolling(window=14).mean()

# 2. Exponential Moving Average (EMA)
df['EMA'] = df['close'].ewm(span=14, adjust=False).mean()

# 3. Relative Strength Index (RSI)
delta = df['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

# 4. Moving Average Convergence Divergence (MACD)
df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
df['MACD'] = df['EMA_12'] - df['EMA_26']
df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

# 5. Bollinger Bands
df['Bollinger_middle'] = df['close'].rolling(window=20).mean()
df['Bollinger_upper'] = df['Bollinger_middle'] + (df['close'].rolling(window=20).std() * 2)
df['Bollinger_lower'] = df['Bollinger_middle'] - (df['close'].rolling(window=20).std() * 2)

# Sentiment Analysis
# Example: Pull stock market news headlines and analyze sentiment
news = ["Stock market sees positive growth today", "Bearish trend in tech stocks"]
analyzer = SentimentIntensityAnalyzer()
sentiment_scores = [analyzer.polarity_scores(headline)['compound'] for headline in news]
df['Sentiment'] = sum(sentiment_scores) / len(sentiment_scores)  # Average sentiment score

# Prepare features and target
df['Target'] = df['close'].shift(-1)  # Predict next close price
df.dropna(inplace=True)  # Remove missing values

features = ['SMA', 'EMA', 'RSI', 'MACD', 'MACD_signal', 'Bollinger_upper', 'Bollinger_middle', 'Bollinger_lower', 'Sentiment']
X = df[features]
y = df['Target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model using XGBoost
model = xgb.XGBRegressor(objective='reg:squarederror')
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Squared Error: {rmse}")

# Visualize the results
plt.plot(y_test.values, label='Actual Prices')
plt.plot(predictions, label='Predicted Prices')
plt.legend()
plt.show()
