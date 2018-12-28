# here the classifier only outputs the fault type using the voltage data of all the buses at the time of fault
# the input data is also shifted by half the time window
print 'Importing modules'
import csv
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_predict, cross_validate, cross_val_score, KFold
from sklearn.metrics import accuracy_score, confusion_matrix
from avgFilterFn import avgFilter # to emulate filtered data
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from getBusDataFn import getBusData
import random

refRaw = 'savnw.raw'
BusDataDict = getBusData(refRaw)


class Signals(object):
    def __init__(self):
        self.SignalDict = {}



print 'Reading the csv files'
vFileName = 'fault3ph/vData3phLI.csv' # csv file containing voltage data (different types of fault)
tFileName = 'fault3ph/tData3ph.csv' # csv file containing the time data
eventKeyFile = 'fault3ph/eventIDFileLI.txt'
vFile = open(vFileName,'rb')
tFile = open(tFileName,'rb')
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

"""
###### 
#Illustrating the time shift
for i in range(9):
    plt.plot(AllEventArray[i,:9])

    #print eventList[i]

plt.title('Voltage shifting illustration')
plt.xlabel('Timestep')
plt.ylabel('V (pu)')
plt.grid()
plt.show()
####
"""




##### evaluate SVM classifier
print 'Evaluating the classifier'
from sklearn.svm import SVC
x = AllEventArray
#x = fullArrayFil
y = targetArray.astype(int)

svm_model_linear = SVC(kernel = 'linear', C = 1)
#svm_model_linear = SVC(kernel = 'rbf', C = 1) # is not as good as linear

cv = KFold(n_splits=10)
y_pred  =  cross_val_predict(svm_model_linear, x, y, cv=cv)
accuracy = accuracy_score(y,y_pred)*100

print 'Accuracy: {}'.format(accuracy)
conf_mat = confusion_matrix(y, y_pred)
print conf_mat

scores = cross_val_score(svm_model_linear, x, y, cv=cv)
print("Accuracy: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))
#####







### build an algorithm which ranks the proximity of the PMU voltage to the fault
### according to the voltage drop of the sample


#### pick a random sample
ind = random.choice(range(x.shape[0]))
random_sample = x[ind]
event = eventList[ind]
print event
# reshape into an array where each row represents a three phase voltage sample
# over the time window
sampleArray = random_sample.reshape(-1,30)
######


"""
###### test a certain event
event = 'R100/F151/ABCG/4.6e-02'
print event
eventInd = eventList.index(event)
sample = x[eventInd]
# reshape into an array where each row represents a three phase voltage sample
# over the time window
sampleArray = sample.reshape(-1,30)
####
"""




# build a dictionary of the minimum voltage of each row
minVDict = {}
for i in range(sampleArray.shape[0]):
    minVDict[i] = sampleArray[i].min()

print 'Bus ID: Min voltage (sorted)'
for ind, value in sorted(minVDict.iteritems(), key=lambda (k,v): v): 
    print '{}:{}'.format(BusDataDict.keys()[ind],value)

######################
###########


