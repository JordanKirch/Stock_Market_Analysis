import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import sqlite3
import datetime
from sklearn.linear_model import LinearRegression

# Connect to a local SQLite database to store the stock market data
conn = sqlite3.connect('stock_data.db')

# Use yfinance to get historical stock market data for a particular stock
msft = yf.Ticker('MSFT')
msft_data = msft.history(period='max')

# Write the data to a table in the database
msft_data.to_sql('msft', conn, if_exists='replace')

# Use SQL to wrangle the data (example: filter out missing values)
df = pd.read_sql_query('SELECT DISTINCT * FROM msft WHERE Close > 0', conn)

# Remove timestamp from Date column
df['Date'] = df['Date'].str[:10]

# Convert Date column to datetime format
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

# Perform data analysis using Python libraries like pandas, numpy, and matplotlib
# Question 1 What is the trend of the stock price over time

mean_price = np.mean(df['Close'])
# plot of entire historical data
plt.figure(figsize=[20,20])
plt.subplot(1,2,1)
plt.plot(df['Date'], df['Close'])
plt.title('Microsoft Stock Price Over Time')
plt.xlabel('Date (Year)')
plt.ylabel('Stock Price ($)')

# Overall trend is positive as the stock price increases over time with the highest price at $339

# Function used to annotate the Max y value
def annot_max(x,y, ax=None):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text= "x={}, y={:.3f}".format(xmax, ymax)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.5,0.96), **kw)


annot_max(df['Date'],df['Close'])


# Question 2 How does the trading volume affect the stock price?

plt.subplot(2,2,2)
plt.scatter(df['Volume'], df['Close'])
plt.title('Relationship between Trading Volume and Stock Price')
plt.xlabel('Trading Volume (Share)')
plt.ylabel('Stock Price ($)')

# Most points are at the bottom left. drawing the conclusion that there is a negative relationship between trading volume and stock price

#Question 3 What is the volatility of the stock price?

# Filter the dates in df that only includes data from last 20 days
today = pd.to_datetime(datetime.date.today())
last2Weeks = pd.to_datetime(today - datetime.timedelta(days=20))
filtered_dates = df[df['Date'] >= last2Weeks]['Date']
filtered_close = df[df['Date'] >= last2Weeks]['Close']

plt.subplot(2,2,4)
window_size = 2
rolling_mean = filtered_close.rolling(window=window_size).mean()
rolling_std = filtered_close.rolling(window=window_size).std()

plt.plot(filtered_dates, filtered_close, label='Actual')
plt.plot(filtered_dates, rolling_mean, label='Moving Average')
plt.fill_between(filtered_dates, rolling_mean - rolling_std, rolling_mean + rolling_std, alpha=0.2, label='Standard Deviation')
plt.title('Microsoft Stock Volatility Over the Last 20 days')
plt.xlabel('Date (Year-Month-Day)')
plt.ylabel('Stock Price ($)')
plt.legend()

# Using Bollinger Bands the Actual price has stayed within the standard deviation proving that the volatility is low 

plt.suptitle("Stock Analysis Of MSFT")
plt.show()

# Use Python to build a predictive model
# TODO learn scikit-learn


# X = np.array(df['Date'])
# Y = np.array(df['Close'])

# model = LinearRegression()
# model.fit(X, Y)


# # Use the model to make predictions
# next_day = df.index[-1] + pd.DateOffset(1)
# prediction = model.predict([[next_day]])[0]
# print(f'The predicted stock price for {next_day.date()} is {prediction:.2f}')

# # Use the model to make predictions
# last_day = pd.to_datetime(df.index[-1], format="%Y-%m-%d")
# next_day = last_day + pd.Timedelta(days=1)
# # Convert next_day to a Unix timestamp
# next_day_unix = next_day.timestamp()
# prediction = model.predict([[next_day_unix]])[0]
# print(f'The predicted stock price for {next_day.date()} is {prediction:.2f}')
