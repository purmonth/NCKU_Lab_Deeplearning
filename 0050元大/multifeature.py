import numpy
import tensorflow as tf 
import datetime
from pandas import read_csv
import math
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

look_back = 5
scaler = MinMaxScaler(feature_range=(0,1))

def create_dataset(dataset, look_back = 5):
    dataX, dataY = [],[]
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back),0]
        dataX.append(a)
        dataY.append(dataset[i + look_back,0])
    return numpy.array(dataX), numpy.array(dataY)


def ratio(a,b):
    if(a>0 and b>0):
        return 1
    if(a<0 and b<0):
        return 1
    else: 
        return 0

dataset_train = read_csv('0050_2004_AllData-date.csv')
Raw_dataset_train = dataset_train.values
dataset_train =Raw_dataset_train.astype('float32')
print(dataset_train)