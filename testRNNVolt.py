# Script to apply RNN to classify abnormal voltage

print 'Importing modules...'
import pickle
import numpy as np
# importing evaluation metrics
from sklearn.metrics import accuracy_score, confusion_matrix
# for splitting the data
from sklearn.model_selection import train_test_split
import os
from getROCFn import getROC
from getBusDataFn import getBusData

# keras modules required
from keras.models import Sequential
from keras.layers.recurrent import SimpleRNN
from keras.layers.core import Dense, Activation


#############
# Functions
def load_obj(name ):
    # load pickle object
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def trainSRNN(x_train, y_train, timeSteps, classes, ep, cwdict):
    # train and return a Recurrent Neural Network 
    model = Sequential()
    # output dim refers to the number of delay nodes
    model.add(SimpleRNN(units=timeSteps,activation="tanh",input_shape=(timeSteps,1))) # units: output dim of the SRNN, does not have to be equal to number of features
    #model.add(SimpleRNN(60, activation='relu', batch_input_shape=(batch_size, x.shape[1], 1))) # using batches 
    model.add(Dense(units=classes,activation='linear'))


    model.compile(loss = 'mean_squared_error',optimizer='sgd', metrics = ['accuracy'])
    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    #model.fit(x_train, y_train, epochs=10,class_weight = {0:3,1:100}, batch_size = batch_size)
    model.fit(x_train, y_train,epochs = ep, class_weight = cwdict,verbose = 2)
    #model.fit(x_train, y_train, epochs=10, verbose = 0)
    return model
#######################


# Load the voltage data
print 'Loading the voltage data from the object file...'
VoltageDataDict = load_obj('VoltageData') # this voltage data dictionary has been generated from the script 'TS3phSimN_2FaultDataSave.py', see it for key format

###############
# crop the data till after fault clearance
print 'Formatting the data to be used by the RNN model....'
tme = VoltageDataDict['time']
timestep = tme[1] - tme[0]
#ind_fault_clearance = int(1.31/timestep) #  the fault is cleared at this time 
ind_fault_clearance = int(0.31/timestep)  + 1 #  the fault is cleared at this time 
ind_fc_1s = int(1.31/timestep)  + 1 # one sec after the fault is cleared
ind_line1_outage = int(0.1/timestep)  + 5 # time when line 1 is outaged (added 5 time steps to make sure the voltage settles to the new value)
samplevCropped = VoltageDataDict[VoltageDataDict.keys()[0]][ind_fault_clearance:]



# get the input features and the classifications
croppedVArray = np.zeros((len(VoltageDataDict)-1,samplevCropped.shape[0])) # make an array of zeros where each row is a sample (cropped) voltage
TargetVec = np.zeros(len(VoltageDataDict)-1) # the target vector for dvdt classification

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

    # see if the voltage is within the thresholds
    abnormalVList = [steadyV[j] for j in range(steadyV.shape[0]) if (steadyV[j] < 0.95 or steadyV[j] > 1.1) and dv_dt[j] < 0.01]
    if len(abnormalVList) > 10:
        TargetVec[k] = 1.0

    k+=1
######################

###################
# get the performance from just using the raw input voltage data
print 'Training the RNN model for voltage abnormalities using raw time series data...'

fpList = []
fnList = []
accuracy = []
print 'Average performance evaluation'
y = TargetVec
y = np.array(y).reshape((len(y),1))

loop = 1
noOfTrials = 1
for i in range(noOfTrials):
    
    print 'Loop {} out of {}'.format(loop,noOfTrials)
    x = croppedVArray[:,:60]
    x = np.array(x).reshape((x.shape[0],x.shape[1],1))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)

    cwdict = {0:0.03, 1: 1.0}
    model =  trainSRNN(x_train, y_train, 60, 1, 50, cwdict)

    scores = model.evaluate(x_test, y_test, verbose=0)
    y_pred = model.predict(x_test)
    y_pred  = (y_pred>0.5)

    cm = confusion_matrix(y_test, y_pred)
    fpList.append(cm[0][1])
    fnList.append(cm[1][0])
    accuracy.append(scores[1]*100)
    loop +=1

# get the means
mean_accuracy = np.mean(accuracy)
mean_fp = np.mean(fpList)
mean_fn = np.mean(fnList)
print 'Average accuracy: {}'.format(mean_accuracy)
print 'Average false positives: {}'.format(mean_fp)
print 'Average false negatives: {}'.format(mean_fn)
##################

