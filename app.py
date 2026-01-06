import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from keras.models import load_model
import datetime as dt
from sklearn.preprocessing import MinMaxScaler


# Load the trained model

model = load_model('Stock Predictions Model.keras')

st.header('Stock Market Predictor')


# User inputs

stock = st.text_input('Enter Stock Symbol', 'GOOG')

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input('Start Date', value=dt.date(2012, 1, 1))
with col2:
    end_date = st.date_input('End Date', value=dt.date.today())

# Download stock data

data = yf.download(stock, start=start_date, end=end_date, auto_adjust=True)

st.subheader(f'Stock Data for {stock}')
st.write(data.tail(10))  # Show last 10 rows


# Calculate Moving Averages

ma_50 = data.Close.rolling(50).mean()
ma_100 = data.Close.rolling(100).mean()
ma_200 = data.Close.rolling(200).mean()


# Plot 1: Price vs MA50

st.subheader('Price vs MA50')
fig1 = plt.figure(figsize=(10,6))
plt.plot(data.Close, 'g', label='Close Price')
plt.plot(ma_50, 'r', label='MA50')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig1)


# Plot 2: Price vs MA50 vs MA100

st.subheader('Price vs MA50 vs MA100')
fig2 = plt.figure(figsize=(10,6))
plt.plot(data.Close, 'g', label='Close Price')
plt.plot(ma_50, 'r', label='MA50')
plt.plot(ma_100, 'b', label='MA100')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)


# Plot 3: Price vs MA100 vs MA200

st.subheader('Price vs MA100 vs MA200')
fig3 = plt.figure(figsize=(10,6))
plt.plot(data.Close, 'g', label='Close Price')
plt.plot(ma_100, 'r', label='MA100')
plt.plot(ma_200, 'b', label='MA200')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig3)


# Prepare data for prediction

data_train = pd.DataFrame(data.Close[0:int(len(data)*0.8)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.8):])

scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(data_train)

past_100_days = data_train.tail(100)
final_df = pd.concat([past_100_days, data_test], ignore_index=True)

input_data = scaler.transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test, y_test = np.array(x_test), np.array(y_test)


# Make predictions

y_predicted = model.predict(x_test)

# Inverse scaling
scale_factor = 1/scaler.scale_[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor


# Plot 4: Original vs Predicted Price

st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(10,6))
plt.plot(y_test, 'g', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig4)
