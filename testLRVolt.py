# Script to rigorously test the LR classifier on the voltage oscillation data

print 'Importing modules...'
import pickle
import numpy as np
# importing evaluation metrics
from sklearn.metrics import accuracy_score, confusion_matrix
# Logistic Regression
from sklearn import linear_model as lm
# for splitting the data
from sklearn.model_selection import train_test_split
import os
from getROCFn import getROC
from getBusDataFn import getBusData


#############
# Functions
def load_obj(name ):
    # load pickle object
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def testMLModel(x,y, testSize, classWeightDict,noOfTrials):
    # train the ML a number of times with randomly selected training and test sets
    # return the average number of false positives and false negatives

    fpList = []
    fnList = []
    accuracyList = []
    y = np.array(y).reshape(-1)
    for i in range(noOfTrials):
       
        print 'Trial {} out of {}'.format(i+1,noOfTrials)
        # partition the data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = testSize)


        # train LR
        model_LR = lm.LogisticRegression(C=1e5,class_weight=classWeightDict)
        model_LR.fit(x_train, y_train)
        y_pred_LR = model_LR.predict(x_test)
        cm = confusion_matrix(y_test, y_pred_LR)
        fnList.append(cm[0][1]) # false alarm
        fpList.append(cm[1][0]) # event undetected
        accuracyList.append(accuracy_score(y_test,y_pred_LR)*100)

    avg_fp = np.mean(fpList)
    avg_fn = np.mean(fnList)
    avg_accuracy = np.mean(accuracyList)
    return avg_fp, avg_fn, avg_accuracy
#######################


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



# get the performance from just using the raw input voltage data
print 'Training the LR model for voltage abnormalities using raw time series data...'
x = croppedVArray[:,:60] # the first 60 timesteps of the voltage array after line clearance
y = TargetVec
testSize = 0.25
classWeightDict = {0:0.03,1:1}
avg_fp, avg_fn, avg_accuracy = testMLModel(x,y, testSize, classWeightDict,100)
print 'Average accuracy: {}'.format(avg_accuracy)
print 'Average fp: {}'.format(avg_fp)
print 'Average fn: {}'.format(avg_fn)
