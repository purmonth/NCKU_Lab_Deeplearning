import numpy 
import tensorflow as tf
import datetime
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

numpy.random.seed(0)

look_back = 3


#look back how many days
scaler = MinMaxScaler(feature_range=(0, 1))

def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)

def ratio(a,b):
	if(a>0 and b > 0):
	  	return 1
	if(a<0 and b < 0):
		return 1
	else:
		return 0


dataset_train = read_csv('0050_2004_close-date.csv',usecols=[0],engine='python')
Raw_dataset_train = dataset_train.values
dataset_train = Raw_dataset_train.astype('float32')
dataset_train = scaler.fit_transform(dataset_train)
train_size = len(dataset_train)
#create numpy data and trainX 2 dimension  trainY 1 dimension
trainX, trainY = create_dataset(dataset_train, look_back)
#reshape input to be 3 dimension[samples, time steps, feature]
print(trainY.shape)
print(trainX.shape)
trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1],1))
print(trainX.shape)

dataset_test = read_csv('0050_2019_close-date.csv',usecols=[0],engine='python')
Raw_dataset_test = dataset_test.values
dataset_test = Raw_dataset_test.astype('float32')
dataset_test = scaler.fit_transform(dataset_test)
test_size = len(dataset_test)
testX, testY = create_dataset(dataset_test, look_back)
print(testY.shape)
print(testX.shape)
testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))
print(testX.shape)
