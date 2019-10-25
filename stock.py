import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
#this library is for normalize
from sklearn.preprocessing import MinMaxScaler
#this keras library is for build module LSTM
#LSTM is for time sequential
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout


dataset_train = pd.read_csv('Google_Stock_Price_Train.csv') # load data
training_set = dataset_train.iloc[:, 1:2].values

#here to normalize the scaling
sc = MinMaxScaler(feature_range  =(0, 1))
training_set_scaled = sc.fit_transform(training_set)

print(training_set_scaled)

#here 60 is the date i predict
x_train = []  #the date before 60 days
y_train = []  #the predict date
for i in range(60, 1258): #1258 is the totals for training
    x_train.append(training_set_scaled[i-60:i,0])
    y_train.append(training_set_scaled[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train) #trans to numpy array to input RNN

#x_train is two dimension, and reshape to three dimension
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1], 1))
#stock prices, timesteps indicators

#Initialisin the RNN
regressor = Sequential()
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))
regressor.add(Dense(units = 1))

#compiling
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
#starting to train
regressor.fit(x_train, y_train, epochs = 100, batch_size = 32)



##to predict
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)#Feature Scaling

x_test = []
for i in range(60, 80):
    x_test.append(inputs[i-60:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_stock_price = regressor.predict(x_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

 
print(real_stock_price)
print(predicted_stock_price)


plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')  
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')  
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

