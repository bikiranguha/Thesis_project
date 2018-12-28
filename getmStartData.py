# script to get the motor start data from matlab csv file
print 'Importing modules'
import csv
import numpy as np
#from sklearn.model_selection import train_test_split, cross_val_predict, cross_validate, cross_val_score, KFold, StratifiedKFold
#from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
#from avgFilterFn import avgFilter # to emulate filtered data
import matplotlib.pyplot as plt
import random
from avgFilterFn import avgFilter # to emulate filtered data





### getting the motor start data
startTime = 1.0
numSamples = 40
shiftRange = 5
numcyc = 6
std = 0.001
vFileName = 'mStartVpu.csv' # csv file containing voltage data (different types of fault)
tFileName = 'mStartTime.csv' # csv file containing the time data
#eventKeyFile = 'fault3ph/eventIDFileLI.txt'
vFile = open(vFileName,'rb')
tFile = open(tFileName,'rb')
readerV=csv.reader(vFile,quoting=csv.QUOTE_NONNUMERIC) # so that the entries are converted to floats
readerT = csv.reader(tFile,quoting=csv.QUOTE_NONNUMERIC)
#tme = [row for idx, row in enumerate(readerT) if idx==0][0]

tDict = {}
vDict = {}

for idx, row in enumerate(readerT):
    tDict[idx] = row

for idx, row in enumerate(readerV):
    vDict[idx] = row

croppedV = []
for idx in tDict:
    t = tDict[idx]
    startInd =  min([ind for ind, val in enumerate(t) if val >=startTime])
    endInd = startInd + numSamples
    v = vDict[idx]

    # add noise and smoothing
    # pass the input through a 6 cycle average filter
    v = avgFilter(v,numcyc)
    # add some noise to the outputs
    v = np.array(v) + np.random.normal(0,std,len(v)) # normal noise with standard deviation of std


    for ts in range(shiftRange):
        lst = []
        for rep in range(3):
            for val in v[startInd-ts:endInd-ts]:
                lst.append(val)
        croppedV.append(lst)

mStartArray = np.array(croppedV)

"""
for i in range(5):
    plt.plot(mStartArray[i])
plt.show()
"""
mStartTarget = np.array([6]*mStartArray.shape[0])

vFile.close()
tFile.close()

############