# Useful literature:
# The simple RNN chapter of the Time Series Forecasting book
# dimensionality: https://stackoverflow.com/questions/47272351/understanding-simplernn-process
# class weights: https://datascience.stackexchange.com/questions/13490/how-to-set-class-weights-for-imbalanced-classes-in-keras
import pickle
from getROCFn import getROC
import numpy as np
from sklearn.model_selection import train_test_split

############## Functions
def load_obj(name ):
    # load pickle object
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
###################

# get the voltage templates for class 0 and class 1

# Load the voltage data
print 'Loading the voltage data from the object file...'
VoltageDataDict = load_obj('VoltageData') # this voltage data dictionary has been generated from the script 'TS3phSimN_2FaultDataSave.py', see it for key format


# crop the data till after fault clearance
print 'Formatting the data to be used by the LR model....'
tme = VoltageDataDict['time']
timestep = tme[1] - tme[0]
#ind_fault_clearance = int(1.31/timestep) #  the fault is cleared at this time 
ind_fault_clearance = int(0.31/timestep)  + 1 #  the fault is cleared at this time 
ind_fc_1s = int(1.31/timestep)  + 1 # one sec after the fault is cleared
ind_line1_outage = int(0.1/timestep)  + 5 # time when line 1 is outaged (added 5 time steps to make sure the voltage settles to the new value)
samplevCropped = VoltageDataDict[VoltageDataDict.keys()[0]][ind_fault_clearance:]


# get the input features and the classifications

# make an array of zeros where each row is a sample (cropped) voltage
# in each row, contains all the voltage info after the fault clearance
croppedVArray = np.zeros((len(VoltageDataDict)-1,samplevCropped.shape[0])) 


dvdtTarget = np.zeros(len(VoltageDataDict)-1) # the target vector for dvdt classification


k= 0 # row number of croppedVArray
for key in VoltageDataDict:
    if key == 'time':
        continue
    voltage = VoltageDataDict[key]
    dv_dt =  getROC(voltage,tme)
    croppedV = voltage[ind_fault_clearance:]
    croppedVArray[k] = croppedV
    steadyV = voltage[-100:] # the final 100 samples of the voltage
    dv_dtSteady = dv_dt[-100:]

    # classify instability according to the rate of change of voltage
    highdvdtList = [steadyV[j] for j in range(steadyV.shape[0]) if dv_dtSteady[j] > 0.05] # based only on dv_dt thresholds
    if len(highdvdtList) > 10:
        dvdtTarget[k] = 1.0

    k+=1


# construct a simple RNN model

from keras.models import Sequential
from keras.layers.recurrent import SimpleRNN
from keras.layers.core import Dense, Activation








# define inputs
print 'Partitioning test/train data'
x = croppedVArray[:,:60]
x = np.array(x).reshape((x.shape[0],x.shape[1],1))
y = dvdtTarget
y = np.array(y).reshape((len(y),1))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)

# constructing the model
print 'Constructing model and evaluating...'
batch_size = 100 # if you want to train in batches
model = Sequential()
# output dim refers to the number of delay nodes
model.add(SimpleRNN(units=60,activation="tanh",input_shape=(60,1))) # units: output dim of the SRNN, does not have to be equal to number of features
#model.add(SimpleRNN(60, activation='relu', batch_input_shape=(batch_size, x.shape[1], 1))) # using batches 
model.add(Dense(units=1,activation='linear'))


model.compile(loss = 'mean_squared_error',optimizer='sgd')
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#model.fit(x_train, y_train, epochs=10,class_weight = {0:3,1:100}, batch_size = batch_size)
#model.fit(x_train, y_train, epochs=10,class_weight = {0:3,1:100})
model.fit(x_train, y_train, epochs=10)
score_train = model.evaluate(x_train,y_train)
print score_train


# Things to do:
# Get the average accuracy, confusion matrix
# Find out if class weights have any effects
# Try different optimization algorithms
# Test LSTM neural networks
# This link looks interesting: https://machinelearningmastery.com/indoor-movement-time-series-classification-with-machine-learning-algorithms/
