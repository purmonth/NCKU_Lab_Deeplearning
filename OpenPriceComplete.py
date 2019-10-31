import numpy
import matplotlib.pyplot as plt 
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
%matplotlib inline

look_back = 5
#look back how many days
scaler = MinMaxScaler(feature_range=(0, 1))

def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)

dataset_train = read_csv('Google_Stock_Price_Train.csv',usecols=[1],engine='python')
Raw_dataset_train = dataset_train.values
dataset_train = Raw_dataset_train.astype('float32')
dataset_train = scaler.fit_transform(dataset_train)
train_size = len(dataset_train)
#create numpy data and trainX 2 dimension  trainY 1 dimension
trainX, trainY = create_dataset(dataset_train, look_back)
#reshape input to be 3 dimension[samples, time steps, feature]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))


dataset_test = read_csv('Google_Stock_Price_Test.csv',usecols=[1],engine='python')
Raw_dataset_test = dataset_test.values
dataset_test = Raw_dataset_test.astype('float32')
dataset_test = scaler.fit_transform(dataset_test)
test_size = len(dataset_test)
testX, testY = create_dataset(dataset_test, look_back)
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

#create and fit the LSTM network
model = Sequential()
model.add(LSTM(units = 50, return_sequences = True, input_shape = (1,look_back)))
model.add(Dropout(0.5))
model.add(LSTM(units = 50))
model.add(Dropout(0.5))

model.add(Dense(units = 1))

model.compile(loss='mean_squared_error',optimizer = 'adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

#make the prediction
trainPredict = model.predict(trainX)
testPredict =  model.predict(testX)

# invert predictions
# all these four datas are 4 dimesion
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
print("Raw_dataset_train")
print(Raw_dataset_train)
print("Raw_dataset_test")
print(Raw_dataset_test)
print("trainX & trainY")
print(trainX)
print(trainY)
print("testX & testY")
print(testX)
print(testY)
print("trainPredict")
print(trainPredict)
print("testPredict")
print(testPredict)


# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
plt.plot(scaler.inverse_transform(dataset_test), color = 'green', label = 'Google Stock Price Test')
plt.plot(testPredict, color = 'purple', label = 'Predicted Test Google Stock Price')
plt.plot(scaler.inverse_transform(dataset_train), color = 'red', label = 'Google Stock Price Train')
plt.plot(trainPredict, color = 'blue', label = 'Predicted Train Google Stock Price')
plt.legend()
plt.show()


