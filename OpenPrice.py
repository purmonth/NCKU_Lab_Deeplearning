import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 2:3].values

print(training_set)

sc = MinMaxScaler(feature_range=(0,1))
print(sc)
training_set_scaled = sc.fit_transform(training_set)

print(training_set_scaled)


x_train = []
y_train = []
for i in range(100, 1000)
    x_train.append(training_set_scaled[i-100:i,0])
