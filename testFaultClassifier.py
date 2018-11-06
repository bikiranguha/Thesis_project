# script to prepare the fault data and test using various classifiers
import csv
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_predict, cross_validate, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix

vFileName = 'fault3ph/vData3ph.csv' # csv file containing voltage data (different types of fault)
tFileName = 'fault3ph/tData3ph.csv' # csv file containing the time data
eventKeyFile = 'fault3ph/eventIDFile.txt'
vFile = open(vFileName,'rb')
tFile = open(tFileName,'rb')
readerV=csv.reader(vFile,quoting=csv.QUOTE_NONNUMERIC) # so that the entries are converted to floats
readerT = csv.reader(tFile,quoting=csv.QUOTE_NONNUMERIC)
tme = [row for idx, row in enumerate(readerT) if idx==0][0]


# read the event file
eventList = []
with open(eventKeyFile,'r') as f:
    fileLines = f.read().split('\n')
    for line in fileLines[1:]:
        if line == '':
            continue
        eventList.append(line.strip())

# get the indices for fault on and fault off
faultontime = 0.1
faultofftime = 0.2
faultonind = min([idx for idx,val in enumerate(tme) if val >= faultontime])
faultoffind = min([idx for idx,val in enumerate(tme) if val >= faultofftime])


listPHA = []
listPHB = []
listPHC = []

listSteadyA = []
listSteadyB = []
listSteadyC = []

condensedEventKey = []
condensedset = set()
eventTarget = []
for idx, row in enumerate(readerV):
    eventKey = eventList[idx]
    eventKeyWords = eventKey.split('/')
    faultbus = eventKeyWords[0][1:].strip()
    currentbus = eventKeyWords[1][1:].strip()
    faulttype = eventKeyWords[2].strip()
    phase = eventKeyWords[-1].strip()

    if phase == 'A':
        listPHA.append(row[faultonind:faultoffind])
        listSteadyA.append(row[:faultonind])

    if phase == 'B':
        listPHB.append(row[faultonind:faultoffind])
        listSteadyB.append(row[:faultonind])

    if phase == 'C':
        listPHC.append(row[faultonind:faultoffind])
        listSteadyC.append(row[:faultonind])

    condensedKey = 'F{}/B{}/{}'.format(faultbus,currentbus,faulttype)
    if condensedKey not in condensedset:
        condensedEventKey.append(condensedKey)
        condensedset.add(condensedKey)

        if faulttype == 'ABCG' and currentbus == faultbus: # 3 phase fault at current bus
            eventTarget.append(1)
        elif faulttype == 'AG' and currentbus == faultbus: # single phase fault at phase A
            eventTarget.append(2)
        else: # fault at some other bus
            eventTarget.append(3)


arrayPHA = np.array(listPHA)
arrayPHB = np.array(listPHB)
arrayPHC = np.array(listPHC)


arraySteadyA = np.array(listSteadyA)
arraySteadyB = np.array(listSteadyB)
arraySteadyC = np.array(listSteadyC)

eventArray = np.concatenate((arrayPHA,arrayPHB,arrayPHC),axis=1)
steadyArray = np.concatenate((arraySteadyA,arraySteadyB,arraySteadyC),axis=1)
steadyArray = steadyArray[:,:eventArray.shape[1]] # crop columns to match that of eventArray
fullArray = np.concatenate((steadyArray,eventArray),axis=0) # append all the steady state cases at the top


steadyTarget = np.zeros(steadyArray.shape[0])
eventTargetArray = np.array(eventTarget)
fullTargetArray = np.concatenate((steadyTarget,eventTargetArray))

#np.savetxt(vName, fullArray, delimiter=",")

# get a list of all event keys
allKeys = ['steady']*steadyArray.shape[0]
for key in condensedEventKey:
    allKeys.append(key)

# close files
vFile.close()
tFile.close()


# evaluate SVM classifier
from sklearn.svm import SVC
x = fullArray
y = fullTargetArray.astype(int)
svm_model_linear = SVC(kernel = 'linear', C = 1)

y_pred  =  cross_val_predict(svm_model_linear, x, y, cv=5)
accuracy = accuracy_score(y,y_pred)*100
print 'Accuracy: {}'.format(accuracy)
#conf_mat = confusion_matrix(y, y_pred)
#print accuracy
#print conf_mat

#results = cross_validate(svm_model_linear, x, y, cv=)
#print results['test_score'] 


# get the indices where y_pred != y
wrongInd = []
for i in range(y.shape[0]):
    if y_pred[i] != y[i]:
        wrongInd.append(i)


for i in wrongInd:
    print '{},{},{}'.format(allKeys[i],y[i],y_pred[i])