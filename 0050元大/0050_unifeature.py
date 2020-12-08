import numpy 
import tensorflow as tf
import datetime
##import matplotlib.pyplot as plt 
from pandas import read_csv
import math
from keras.models import Sequential, save_model, load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

numpy.random.seed(0)
Total = 1000000
Original = Total
sum = 0



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

dataset_train = read_csv('0050_2004_open-nodate.csv',usecols=[0],engine='python')
Raw_dataset_train = dataset_train.values
dataset_train = Raw_dataset_train.astype('float32')
dataset_train = scaler.fit_transform(dataset_train)
train_size = len(dataset_train)
#create numpy data and trainX 2 dimension  trainY 1 dimension
trainX, trainY = create_dataset(dataset_train, look_back)
#reshape input to be 3 dimension[samples, time steps, feature]
print(trainX.shape)
trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1],1))


dataset_test = read_csv('0050_2019_open-nodate.csv',usecols=[0],engine='python')
Raw_dataset_test = dataset_test.values
dataset_test = Raw_dataset_test.astype('float32')
dataset_test = scaler.fit_transform(dataset_test)
test_size = len(dataset_test)
testX, testY = create_dataset(dataset_test, look_back)
testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))

#create and fit the LSTM network
model = Sequential()
model.add(LSTM(units = 50, return_sequences = True, input_shape = (look_back,1)))
model.add(Dropout(0.5))
model.add(LSTM(units = 50))
model.add(Dropout(0.5))
model.add(Dense(units = 1))
model.compile(loss='mean_squared_error',optimizer = 'adam')

# Set callback functions to early stop training and save the best model so far
callbacks = [EarlyStopping(monitor='val_loss', patience=20),
             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)
			 ]

model.fit(	trainX,
			trainY, 
			epochs=100,
			batch_size=100, 
			verbose=2,
			validation_data=(testX, testY),
			callbacks=callbacks
			)

#make the prediction
trainPredict = model.predict(trainX)
testPredict =  model.predict(testX)
print(trainＹ.shape)
print(trainX.shape)
print(testY.shape)
print(testX.shape)
# invert predictions
# all these four datas are 4 dimesion
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])



# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

def ratio(a,b):
    if(a>0 and b>0):
        return 1
    if(a<0 and b<0):
        return 1
    else: 
        return 0


BSRightRatio = 0
DifRightRatio = 0

for i in range(len(testPredict)):
	difference = (testPredict[i][0]-Raw_dataset_test[i+look_back][0])
	##如果 明天預測-今天 跟 明天實際-今天 的正負號一樣
	##print(ratio(testPredict[i][0]-Raw_dataset_test[i+5][0],Raw_dataset_test[i+6][0]-Raw_dataset_test[i+5][0]))
	BSRightRatio += ratio(testPredict[i][0]-Raw_dataset_test[i+look_back][0],Raw_dataset_test[i+look_back+1][0]-Raw_dataset_test[i+look_back][0])
	DifRightRatio += abs((testPredict[i][0]-Raw_dataset_test[i+look_back][0])/Raw_dataset_test[i+look_back][0])
	##如果開盤價小於預測買股票  大於則賣
	##buy
	if(difference < 0):
		if(Total >= 1000*Raw_dataset_test[i+look_back+1][0]):
			Total -=  1000*Raw_dataset_test[i+look_back+1][0]
			##print(Total)
			sum += 1
	##sold
	if(difference > 0):
		if(sum > 0):
			Total += 1000*Raw_dataset_test[i+look_back+1][0]
			sum -= 1

BSRightRatio /= len(testPredict)
DifRightRatio /= len(testPredict)
print("\nThis is sign accuracy : ", round(BSRightRatio,2))
print("This is Different price error rate: ", round(DifRightRatio,3))
if(i == len(testPredict)-1):
	if(sum > 0):
		Total += 1000*(Raw_dataset_test[i+look_back][0] * (sum))
		sum = 0

print("If we invest ",Original)
print("We earn",Total-Original)
print("This is ",round(100*(Total-Original)/Original,2),"%")

'''
plt.plot(scaler.inverse_transform(dataset_test), color = 'green', label = 'Google Stock Price Test')
plt.plot(testPredict, color = 'purple', label = 'Predicted Test Google Stock Price')
plt.plot(scaler.inverse_transform(dataset_train), color = 'red', label = 'Google Stock Price Train')
plt.plot(trainPredict, color = 'blue', label = 'Predicted Train Google Stock Price')
plt.legend()
plt.show()

'''
