# here the classifier only outputs the fault type using the voltage data of all the buses at the time of fault
print 'Importing modules'
import csv
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_predict, cross_validate, cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from avgFilterFn import avgFilter # to emulate filtered data
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from getBusDataFn import getBusData

refRaw = 'savnw.raw'
BusDataDict = getBusData(refRaw)


class Signals(object):
    def __init__(self):
        self.SignalDict = {}



print 'Reading the csv files'

##files containg seq data
vFileName = 'fault3ph/vData3phLISeq.csv' # csv file containing voltage data (different types of fault)
tFileName = 'fault3ph/tData3phLISeq.csv' # csv file containing the time data
eventKeyFile = 'fault3ph/eventIDFileLISeq.txt'
##

"""
## files not containing seq data
vFileName = 'fault3ph/vData3phLI.csv' # csv file containing voltage data (different types of fault)
tFileName = 'fault3ph/tData3ph.csv' # csv file containing the time data
eventKeyFile = 'fault3ph/eventIDFileLI.txt'
"""





vFile = open(vFileName,'rb')
tFile = open(tFileName,'rb')
readerV=csv.reader(vFile,quoting=csv.QUOTE_NONNUMERIC) # so that the entries are converted to floats
readerT = csv.reader(tFile,quoting=csv.QUOTE_NONNUMERIC)
tme = [row for idx, row in enumerate(readerT) if idx==0][0]

###
std = 0.001
start = 0.1
timesteps = 20
startind = min([idx for idx,val in enumerate(tme) if val >= start])
endind = startind + timesteps



# make an organized dictionary
# read the event file
print 'Organizing the csv data...'
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

    # implement 6 cycle filter and smoothing
    row = avgFilter(row,6)
    row = np.array(row) + np.random.normal(0,std,len(row)) # normal noise with standard deviation of std
    if eventID not in EventDict:
        EventDict[eventID] = Signals()
    EventDict[eventID].SignalDict['B{}/{}'.format(currentbus,phase)] = row

# arrange all the fault event data into rows sequentially

AllEventList = []
targetList = []
for event in EventDict:
    s = EventDict[event].SignalDict
    # get the event class
    eventKeyWords = event.split('/')
    faulttype = eventKeyWords[2].strip()
    if faulttype == 'ABCG':
        targetList.append(0)
        allSignals = []
        for b in BusDataDict:
            valA = s['B{}/A'.format(b)][startind:endind]
            valB = s['B{}/B'.format(b)][startind:endind]
            valC = s['B{}/C'.format(b)][startind:endind]
            for v in valA:
                allSignals.append(v)
                
            for v in valB:
                allSignals.append(v)

            for v in valC:
                allSignals.append(v)

        AllEventList.append(allSignals)



    elif faulttype == 'AG':
        # get SLG A data
        targetList.append(1)
        allSignals = []
        for b in BusDataDict:
            valA = s['B{}/A'.format(b)][startind:endind]
            valB = s['B{}/B'.format(b)][startind:endind]
            valC = s['B{}/C'.format(b)][startind:endind]
            for v in valA:
                allSignals.append(v)
                
            for v in valB:
                allSignals.append(v)

            for v in valC:
                allSignals.append(v)

        AllEventList.append(allSignals)



        # get SLG B data
        targetList.append(2)
        allSignals = []
        for b in BusDataDict:
            valA = s['B{}/A'.format(b)][startind:endind]
            valB = s['B{}/B'.format(b)][startind:endind]
            valC = s['B{}/C'.format(b)][startind:endind]
            for v in valB:
                allSignals.append(v)
                
            for v in valA:
                allSignals.append(v)

            for v in valC:
                allSignals.append(v)

        AllEventList.append(allSignals)  


        # get SLG C data
        targetList.append(3)
        allSignals = []
        for b in BusDataDict:
            valA = s['B{}/A'.format(b)][startind:endind]
            valB = s['B{}/B'.format(b)][startind:endind]
            valC = s['B{}/C'.format(b)][startind:endind]
            for v in valC:
                allSignals.append(v)
                
            for v in valB:
                allSignals.append(v)

            for v in valA:
                allSignals.append(v)

        AllEventList.append(allSignals)  

AllEventArray = np.array(AllEventList)
targetArray = np.array(targetList)

##### evaluate SVM classifier
print 'Evaluating SVM'
from sklearn.svm import SVC
x = AllEventArray
#x = fullArrayFil
y = targetArray.astype(int)

svm_model_linear = SVC(kernel = 'linear', C = 1)
#svm_model_linear = SVC(kernel = 'rbf', C = 1) # is not as good as linear


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
"""
### quick test
svm_model_linear.fit(x_train, y_train)
y_pred = svm_model_linear.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)*100
print ('Accuracy: {}'.format(accuracy))
conf_mat = confusion_matrix(y_test, y_pred)
print (conf_mat)
####
"""


### extensive test
kfold = StratifiedKFold(n_splits=3, shuffle=True) # preserve the distribution of classes in folds
cvscores = []
for train, test in kfold.split(x_train, y_train):
    # this part measures performance using the folds (the test set here is essentially a cross-validation set)
    model = SVC(kernel = 'linear', C = 1)
    model.fit(x_train[train], y_train[train])

  
    
    # evaluate the model
    y_pred = model.predict(x_train[test])
    #y_pred_prob = model.predict_proba(x_train[test])
    
    # evaluate predictions
    accuracy = accuracy_score(y_train[test], y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    #print(classification_report(y_train[test], y_pred))
    cvscores.append(accuracy * 100)

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

# Evaluate on test data
y_test_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_test_pred)*100
print ('Accuracy: {}'.format(accuracy))
conf_mat = confusion_matrix(y_test, y_test_pred)
print (conf_mat)
###


###########
#############
