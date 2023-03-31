import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import sqlite3

# Connect to a local SQLite database to store the stock market data
conn = sqlite3.connect('stock_data.db')

# Use yfinance to get historical stock market data for a particular stock
msft = yf.Ticker('MSFT')
msft_data = msft.history(period='max')

# Write the data to a table in the database
msft_data.to_sql('msft', conn, if_exists='replace')

# Use SQL to wrangle the data (example: filter out missing values)
df = pd.read_sql_query('SELECT * FROM msft WHERE Close > 0', conn)

# Perform data analysis using Python libraries like pandas, numpy, and matplotlib
mean_price = np.mean(df['Close'])
plt.plot(df['Date'], df['Close'])
plt.title('Microsoft Stock Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

# Use Python to build a predictive model
# TODO learn scikit-learn
from sklearn.linear_model import LinearRegression

X = np.array(df.index).reshape(-1, 1)
y = np.array(df['Close'])

model = LinearRegression()
model.fit(X, y)

# Use the model to make predictions
last_day = pd.to_datetime(df.index[-1], format="%Y-%m-%d")
next_day = last_day + pd.Timedelta(days=1)
# Convert next_day to a Unix timestamp
next_day_unix = next_day.timestamp()
prediction = model.predict([[next_day_unix]])[0]
print(f'The predicted stock price for {next_day.date()} is {prediction:.2f}')
