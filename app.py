import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st

start = '2010-01-01'
end = '2020-12-01'

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
df = yf.download(user_input, start= start, end = end)

#Describing data
st.subheader('Data from 2010-2020')
st.write(df.describe())

st.subheader('Closing price vs Time chart')
ma100= df.Close.rolling(100).mean()
ma200= df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing price vs Time chart with 100MA and 200MA')
ma100= df.Close.rolling(100).mean()
ma200= df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)


training_data = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
testing_data = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
training_data_ppp = scaler.fit_transform(training_data)

x_train = []
y_train = []

for i in range(100,training_data_ppp.shape[0]):
    x_train.append(training_data_ppp[i-100:i])
    y_train.append(training_data_ppp[i,0])

x_train, y_train = np.array(x_train),np.array(y_train)

#Loading Model

model = load_model('keras_model.h5')

#Testing phase

past100_days = training_data.tail(100)
final_df = pd.concat([past100_days, testing_data], ignore_index=True)

input_data = scaler.fit_transform(final_df)

x_test=[]
y_test=[]

for i in range (100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test,y_test=np.array(x_test),np.array(y_test)

y_predicted = model.predict(x_test)

scale = scaler.scale_
scale_fact=1/scale
y_predicted= y_predicted* scale_fact
y_test= y_test* scale_fact

plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
