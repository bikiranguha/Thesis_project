#use keras as the rnn
# rnn are useful for using time series data since they have some memory functions
import numpy as np
import pandas as pd
import os
currentdir = os.getcwd()
# get data and organize it
#loc="C:\\Data\\COE.csv"
loc = currentdir + '/' + 'COE.csv'
temp=pd.read_csv(loc)
data=temp.drop(temp.columns[[0,1]],axis=1)
y=data["COE$"]
x=data.drop(data.columns[[0,4]],axis=1) # drop column 0 and column 4 (COE$ and Open?)
x=x.apply(np.log) # apply log
x=pd.concat([x,data["Open?"]],axis=1)

# to view x
#print x.head()
#print x.tail()
####
# scale x (inputs) and y (target)
from sklearn import preprocessing
scaler_x=preprocessing.MinMaxScaler(feature_range=(0,1))
x=np.array(x).reshape((len(x),4))
x=scaler_x.fit_transform(x)


scaler_y=preprocessing.MinMaxScaler(feature_range=(0,1))
y=np.array(y).reshape((len(y),1))
y=np.log(y)
y=scaler_y.fit_transform(y)

# partition the data into test (5%) and training set (95%)
end = len(x) - 1
learn_end = int(end*0.954)
x_train = x[0:learn_end-1,]
x_test = x[learn_end:end-1,]
y_train = y[1:learn_end]
y_test = y[learn_end+1:end]
x_train = x_train.reshape(x_train.shape + (1,)) # (number of samples, number of features, time step)
x_test = x_test.reshape(x_test.shape + (1,)) # (number of samples, number of features, time step)

# import keras libraries
from keras.models import Sequential # sequential model, allows stacking of layers
from keras.optimizers import SGD # stochastic gradient descent optimizer
from keras.layers.core import Dense, Activation # imports activation fns and a dense layer (fully connected NN layer)
from keras.layers.recurrent import SimpleRNN # imports a fully connected RNN

# determine model structure
seed = 2016
np.random.seed(seed)
fit1 = Sequential() # the function where the model is stored
fit1.add(SimpleRNN(units=8, activation='tanh',input_shape=(4,1))) # hidden layers
fit1.add(Dense(units=1,activation='linear')) # output layer (COE forecast)

# choosing momentum 

# momentum is defined as the fraction of the previous weight update added to this update
# here a large momentum is combined with a small learning rate
sgd = SGD(lr=0.0001, momentum=0.95,nesterov=True) 
												  
fit1.compile(loss='mean_squared_error',optimizer=sgd)

# fitting the model
fit1.fit(x_train,y_train,batch_size=10,nb_epoch=700)
score_train = fit1.evaluate(x_train,y_train,batch_size = 10)
print "in train MSE = ", round(score_train,6)
score_test = fit1.evaluate(x_test,y_test,batch_size=10)
print "in test MSE = ", round(score_test,6)


# see the predictions
pred1 = fit1.predict(x_test)
pred1 = scaler_y.inverse_transform(np.array(pred1).reshape(len(pred1),1))

pred1 = np.exp(pred1)
print np.rint(pred1)



