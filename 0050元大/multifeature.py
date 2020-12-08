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

numpy.random.seed(0)
look_back = 5
Total = 1000000
Original = Total
sum = 0



scaler = MinMaxScaler(feature_range=(0,1))

def createX_dataset(dataset, look_back):
    dataX = []
    for i in range(len(dataset)-look_back-1):
        #0~5:"adj_close","close","high","low","open","volume"
        a = dataset[i:(i+look_back),:]
        dataX.append(a)
    return numpy.array(dataX)

def createY_dataset(dataset, look_back):
    dataY = []
    for i in range(len(dataset)-look_back-1):
        #0~5:"adj_close","close","high","low","open","volume"
        a = dataset[i:(i+look_back),0]
        dataY.append(dataset[i + look_back,3])
    return numpy.array(dataY)


def ratio(a,b):
    if(a>0 and b>0):
        return 1
    if(a<0 and b<0):
        return 1
    else: 
        return 0

dataset_train = read_csv('0050_2004_AllData-nodate.csv',engine='python')
Raw_dataset_train = dataset_train.values
dateset_train = Raw_dataset_train.astype('float32')
#data transform from real value to 0~1
dataset_train = scaler.fit_transform(dataset_train)
#data array shape (3691*6) 3691 days and 6 feature
#"adj_close","close","high","low","open","volume"
#array len also 3691 (from 2004/01~2018/12)
train_size = len(dataset_train)
#[[0.55002833 0.32465986 0.32852976 0.33468082 0.32909245 0.05510401]
# [0.55988669 0.33945578 0.33213244 0.33804952 0.33095844 0.04069354]
# [0.55716714 0.33503401 0.33196089 0.34546067 0.34537744 0.02622473]
# ...
# [0.83342776 0.74914966 0.75810602 0.75745326 0.76166243 0.06098174]
# [0.84985836 0.77465986 0.77268828 0.77429678 0.77523325 0.06605305]
# [0.85552408 0.78231293 0.77869274 0.78019202 0.78032231 0.02969516]]
trainX = createX_dataset(dataset_train, look_back)
trainY = createY_dataset(dataset_train, look_back)
maxvalue =  48.30 / trainY[0]
print(maxvalue)
print(trainＹ.shape)
print(trainX.shape)
trainX = numpy.reshape(trainX, (trainX.shape[0], look_back, trainX.shape[2]))
print(trainX.shape)


dataset_test = read_csv('0050_2019_AllData-nodate.csv',engine='python')
Raw_dataset_test = dataset_test.values
dataset_test = Raw_dataset_test.astype('float32')
dataset_test = scaler.fit_transform(dataset_test)
test_size = len(dataset_test)
testX = createX_dataset(dataset_test, look_back)
testY = createY_dataset(dataset_test, look_back)
print(testY.shape)
print(testX.shape)
testX = numpy.reshape(testX, (testX.shape[0], look_back, testX.shape[2]))
print(testX.shape)



model = Sequential()
model.add(LSTM(units = 50, return_sequences = True, input_shape = (look_back,testX.shape[2])))
model.add(Dropout(0.5))
model.add(LSTM(units = 50))
model.add(Dropout(0.5))
model.add(Dense(units = 1))
model.compile(loss='mean_squared_error',optimizer = 'adam')


callbacks = [EarlyStopping(monitor='val_loss', patience=20)
			 ]

model.fit(	trainX,
			trainY, 
			epochs=200,
			batch_size=100, 
			verbose=2,
			validation_data=(testX, testY),
			callbacks=callbacks
			)




model.save('lstm.hdf5')
model = load_model('lstm.hdf5')

#make the prediction
#label adj_close price
trainPredict = model.predict(trainＸ)
#label adj_close price
testPredict =  model.predict(testＸ)

trainPredict *= maxvalue
testPredict *= maxvalue


DifRightRatio = 0
BSRightRatio = 0

for i in range(len(testPredict)):
	difference = (testPredict[i][0]-Raw_dataset_test[i+look_back][0])
	##如果 明天預測-今天 跟 明天實際-今天 的正負號一樣
	##print(ratio(testPredict[i][0]-Raw_dataset_test[i+5][0],Raw_dataset_test[i+6][0]-Raw_dataset_test[i+5][0]))

	BSRightRatio += ratio(testPredict[i][0]-Raw_dataset_test[i+look_back][0],Raw_dataset_test[i+look_back+1][0]-Raw_dataset_test[i+look_back][0])
	DifRightRatio += abs((testPredict[i][0]-Raw_dataset_test[i+look_back][0])/Raw_dataset_test[i+look_back][0])
	##如果開盤價小於預測買股票  大於則賣
	##buy
	if(difference < 0):
		if(Total >= 1000*Raw_dataset_test[i+look_back][0]):
			Total -=  1000*Raw_dataset_test[i+look_back][0]
			##print(Total)
			sum += 1
	##sold
	if(difference > 0):
		if(sum > 0):
			Total += 1000*Raw_dataset_test[i+look_back][0]
			sum -= 1

print(testPredict)
print(Raw_dataset_test)

BSRightRatio /= len(testPredict)
DifRightRatio /= len(testPredict)
DifRightRatio = 1 - DifRightRatio
print("\nThis is Right Ratio: ", round(BSRightRatio,2))
print("This is Different Right Ratio: ", round(DifRightRatio,3))

if(i == len(testPredict)-1):
	if(sum > 0):
		Total += 1000*(Raw_dataset_test[i+look_back][0] * (sum))
		sum = 0

print("If we invest ",Original)
print("We earn",Total-Original)
print("This is ",round(100*(Total-Original)/Original,2),"%")