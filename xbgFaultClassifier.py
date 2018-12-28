#!/usr/bin/python

from __future__ import division

import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier


# data organization part
# here the classifier only outputs the fault type using the voltage data of all the buses at the time of fault
# the input data is also shifted by half the time window
print('Importing modules')
import csv
#import numpy as np
#from sklearn.model_selection import train_test_split, cross_val_predict, cross_validate, cross_val_score, KFold
from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score, confusion_matrix
#from avgFilterFn import avgFilter # to emulate filtered data
import matplotlib.pyplot as plt
#import matplotlib.patches as patches
from getBusDataFn import getBusData
import random

refRaw = 'savnw.raw'
BusDataDict = getBusData(refRaw)


class Signals(object):
    def __init__(self):
        self.SignalDict = {}



print('Reading the csv files')
vFileName = 'fault3ph/vData3phLI.csv' # csv file containing voltage data (different types of fault)
tFileName = 'fault3ph/tData3ph.csv' # csv file containing the time data
eventKeyFile = 'fault3ph/eventIDFileLI.txt'
#vFile = open(vFileName,'rb') # python 2
vFile = open(vFileName,'r')
#tFile = open(tFileName,'rb') # python 2
tFile = open(tFileName,'r')
readerV=csv.reader(vFile,quoting=csv.QUOTE_NONNUMERIC) # so that the entries are converted to floats
readerT = csv.reader(tFile,quoting=csv.QUOTE_NONNUMERIC)
tme = [row for idx, row in enumerate(readerT) if idx==0][0]

###
timeWindow = 10
start = 0.1
startind = min([idx for idx,val in enumerate(tme) if val >= start])
endind = startind + timeWindow



# make an organized dictionary
# read the event file
print('Organizing the csv data...')
eventList = []
with open(eventKeyFile,'r') as f:
    fileLines = f.read().split('\n')
    for line in fileLines[1:]:
        if line == '':
            continue
        eventList.append(line.strip())


EventDict = {}
SignalDict = {}
for idx, row in enumerate(readerV):
    eventKey = eventList[idx]
    eventKeyWords = eventKey.split('/')
    PL = eventKeyWords[0][1:].strip()
    faultbus = eventKeyWords[1][1:].strip()
    currentbus = eventKeyWords[2][1:].strip()
    faulttype = eventKeyWords[3].strip()
    phase = eventKeyWords[4].strip()
    faultZ = eventKeyWords[5].strip()
    eventID = 'R{}/F{}/{}/{}'.format(PL,faultbus,faulttype,faultZ)
    if eventID not in EventDict:
        EventDict[eventID] = Signals()
    EventDict[eventID].SignalDict['B{}/{}'.format(currentbus,phase)] = row

# arrange all the fault event data into rows sequentially

AllEventList = []
targetList = []
eventList = []
for event in EventDict:
    s = EventDict[event].SignalDict
    # get the event class
    eventKeyWords = event.split('/')
    faulttype = eventKeyWords[2].strip()
    if faulttype == 'ABCG':

        #for i in range(timeWindow/2):
        for i in range(timeWindow-1): 
            eventList.append(event)  
            targetList.append(0)
            allSignals = []
            for b in BusDataDict:
                valA = s['B{}/A'.format(b)][startind-i:endind-i]
                valB = s['B{}/B'.format(b)][startind-i:endind-i]
                valC = s['B{}/C'.format(b)][startind-i:endind-i]
                for v in valA:
                    allSignals.append(v)
                    
                for v in valB:
                    allSignals.append(v)

                for v in valC:
                    allSignals.append(v)

            AllEventList.append(allSignals)



    elif faulttype == 'AG':
        # get SLG A data

        #for i in range(timeWindow/2):
        for i in range(timeWindow-1):
            eventList.append(event) 
            targetList.append(1)
            allSignals = []
            for b in BusDataDict:
                valA = s['B{}/A'.format(b)][startind-i:endind-i]
                valB = s['B{}/B'.format(b)][startind-i:endind-i]
                valC = s['B{}/C'.format(b)][startind-i:endind-i]
                for v in valA:
                    allSignals.append(v)
                    
                for v in valB:
                    allSignals.append(v)

                for v in valC:
                    allSignals.append(v)

            AllEventList.append(allSignals)



        # get SLG B data
        #for i in range(timeWindow/2):
        for i in range(timeWindow-1):
            eventList.append(event.replace('AG','BG')) 
            targetList.append(2)
            allSignals = []
            for b in BusDataDict:
                valA = s['B{}/A'.format(b)][startind-i:endind-i]
                valB = s['B{}/B'.format(b)][startind-i:endind-i]
                valC = s['B{}/C'.format(b)][startind-i:endind-i]
                for v in valB:
                    allSignals.append(v)
                    
                for v in valA:
                    allSignals.append(v)

                for v in valC:
                    allSignals.append(v)

            AllEventList.append(allSignals)  


        # get SLG C data
        #for i in range(timeWindow/2):
        for i in range(timeWindow-1):
            eventList.append(event.replace('AG','CG')) 
            targetList.append(3)
            allSignals = []
            for b in BusDataDict:
                valA = s['B{}/A'.format(b)][startind-i:endind-i]
                valB = s['B{}/B'.format(b)][startind-i:endind-i]
                valC = s['B{}/C'.format(b)][startind-i:endind-i]
                for v in valC:
                    allSignals.append(v)
                    
                for v in valB:
                    allSignals.append(v)

                for v in valA:
                    allSignals.append(v)

            AllEventList.append(allSignals)  

AllEventArray = np.array(AllEventList)
targetArray = np.array(targetList)


x = AllEventArray
y = targetArray

train_X, test_X, train_Y, test_Y = train_test_split(x, y, test_size = 0.25)


# classifier part
xg_train = xgb.DMatrix(train_X, label=train_Y)
xg_test = xgb.DMatrix(test_X, label=test_Y)
# setup parameters for xgboost
param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softmax'
# scale weight of positive examples
param['eta'] = 0.1
param['max_depth'] = 6
param['silent'] = 1
param['nthread'] = 4
param['num_class'] = 6

watchlist = [(xg_train, 'train'), (xg_test, 'test')]
num_round = 5
bst = xgb.train(param, xg_train, num_round, watchlist)
# get prediction
pred = bst.predict(xg_test)
error_rate = np.sum(pred != test_Y) / test_Y.shape[0]
print('Test error using softmax = {}'.format(error_rate))

# do the same thing again, but output probabilities
param['objective'] = 'multi:softprob'
bst = xgb.train(param, xg_train, num_round, watchlist)
# Note: this convention has been changed since xgboost-unity
# get prediction, this is in 1D array, need reshape to (ndata, nclass)
pred_prob = bst.predict(xg_test).reshape(test_Y.shape[0], 6)
pred_label = np.argmax(pred_prob, axis=1)
error_rate = np.sum(pred_label != test_Y) / test_Y.shape[0]
print('Test error using softprob = {}'.format(error_rate))