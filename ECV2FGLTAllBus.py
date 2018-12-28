# Event (fault, gen outage, line outage, tf outage) classification using three phase voltage and angles of all buses together
# added noise and smoothing
# limited number (5) of PMUs available
# no time shifts applied

print 'Importing modules'
import csv
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_predict, cross_validate, cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from avgFilterFn import avgFilter # to emulate filtered data
import matplotlib.pyplot as plt
import random
from getBusDataFn import getBusData


class Signals(object):
    def __init__(self):
        self.SignalDict = {}

#### 

def loopFn(event, sV,sA, faulttype):
    # function which reads the fault voltage and angle signals 
    # and put them side by list in the AllEventLists


    if faulttype == 'ABCG':
        targetList.append(0)
        eventTracker.append(event)
        allSignalsV = []
        allSignalsA = []
        for b in PMUBuses:
            valA = sV['B{}/A'.format(b)][startind:endind]
            valB = sV['B{}/B'.format(b)][startind:endind]
            valC = sV['B{}/C'.format(b)][startind:endind]

            valAA = sA['B{}/A'.format(b)][startind:endind]
            valBA = sA['B{}/B'.format(b)][startind:endind]
            valCA = sA['B{}/C'.format(b)][startind:endind]

            # voltage
            for v in valA:
                allSignalsV.append(v)
                
            for v in valB:
                allSignalsV.append(v)

            for v in valC:
                allSignalsV.append(v)

            # angle
            for v in valAA:
                allSignalsA.append(v)
                
            for v in valBA:
                allSignalsA.append(v)

            for v in valCA:
                allSignalsA.append(v)


        AllEventListV.append(allSignalsV)
        AllEventListA.append(allSignalsA)
    

    elif faulttype == 'AG':

        targetList.append(1)
        eventTracker.append(event)
        allSignalsV = []
        allSignalsA = []
        for b in PMUBuses:
            valA = sV['B{}/A'.format(b)][startind:endind]
            valB = sV['B{}/B'.format(b)][startind:endind]
            valC = sV['B{}/C'.format(b)][startind:endind]

            valAA = sA['B{}/A'.format(b)][startind:endind]
            valBA = sA['B{}/B'.format(b)][startind:endind]
            valCA = sA['B{}/C'.format(b)][startind:endind]

            # voltage
            for v in valA:
                allSignalsV.append(v)
                
            for v in valB:
                allSignalsV.append(v)

            for v in valC:
                allSignalsV.append(v)

            # angle
            for v in valAA:
                allSignalsA.append(v)
                
            for v in valBA:
                allSignalsA.append(v)

            for v in valCA:
                allSignalsA.append(v)

        AllEventListV.append(allSignalsV)
        AllEventListA.append(allSignalsA)

        # SLG B
        targetList.append(2)
        eventTracker.append(event.replace('AG','BG'))
        allSignalsV = []
        allSignalsA = []
        for b in PMUBuses:
            valA = sV['B{}/A'.format(b)][startind:endind]
            valB = sV['B{}/B'.format(b)][startind:endind]
            valC = sV['B{}/C'.format(b)][startind:endind]

            valAA = sA['B{}/A'.format(b)][startind:endind]
            valBA = sA['B{}/B'.format(b)][startind:endind]
            valCA = sA['B{}/C'.format(b)][startind:endind]

            # voltage
            for v in valB:
                allSignalsV.append(v)
                
            for v in valA:
                allSignalsV.append(v)

            for v in valC:
                allSignalsV.append(v)

            # angle
            for v in valBA:
                allSignalsA.append(v)
                
            for v in valAA:
                allSignalsA.append(v)

            for v in valCA:
                allSignalsA.append(v)
                
        AllEventListV.append(allSignalsV)
        AllEventListA.append(allSignalsA)



        # SLG C
        targetList.append(3)
        eventTracker.append(event.replace('AG','CG'))
        allSignalsV = []
        allSignalsA = []
        for b in PMUBuses:
            valA = sV['B{}/A'.format(b)][startind:endind]
            valB = sV['B{}/B'.format(b)][startind:endind]
            valC = sV['B{}/C'.format(b)][startind:endind]

            valAA = sA['B{}/A'.format(b)][startind:endind]
            valBA = sA['B{}/B'.format(b)][startind:endind]
            valCA = sA['B{}/C'.format(b)][startind:endind]

            # voltage
            for v in valC:
                allSignalsV.append(v)
                
            for v in valB:
                allSignalsV.append(v)

            for v in valA:
                allSignalsV.append(v)

            # angle
            for v in valCA:
                allSignalsA.append(v)
                
            for v in valBA:
                allSignalsA.append(v)

            for v in valAA:
                allSignalsA.append(v)
                
        AllEventListV.append(allSignalsV)
        AllEventListA.append(allSignalsA)

#######





# Function to get any type of outage data
def  outageFn(dataFileName,tFileName,eventFileName, classInd, eventPrefix):
    # use this function to get any sort of data from outage events
    EventDict = {}
    # file objects
    eventList = []
    with open(eventFileName,'r') as f:
        fileLines = f.read().split('\n')
        for line in fileLines[1:]:
            if line == '':
                continue
            eventList.append(line.strip())


    dataFileName = open(dataFileName, 'rb') # 'wb' needed to avoid blank space in between lines
    tFileName = open(tFileName, 'rb')


    dataReader = csv.reader(dataFileName,quoting=csv.QUOTE_NONNUMERIC) # 'wb' needed to avoid blank space in between lines
    tReader = csv.reader(tFileName,quoting=csv.QUOTE_NONNUMERIC)

    tme = [row for idx, row in enumerate(tReader) if idx==0][0]



    # make a dictionary where each key corresponds to all measurements available for the event
    for idx, row in enumerate(dataReader):
        eventKey = eventList[idx]
        eventKeyWords = eventKey.split('/')
        PL = eventKeyWords[0][1:].strip()
        stuffout = eventKeyWords[1][1:].strip()
        currentbus = eventKeyWords[2][1:].strip()

        if currentbus in PMUBuses:
            # add some noise to the outputs
            row = np.array(row) + np.random.normal(0,std,len(row)) # normal noise with standard deviation of std
            # pass the input through a 6 cycle average filter
            row = avgFilter(row,numcyc)

            eventID = 'R{}/{}{}'.format(PL,eventPrefix,stuffout)
            if eventID not in EventDict:
                #eventTracker.append(eventID)
                #eventSet.add(eventID)
                EventDict[eventID] = []

            for bs in range(3): # simulate three phase
                for v in row[startind:endind]:
                    #lineoutvdata.append(v)
                    EventDict[eventID].append(v)

    # now make an array out of the dictionary keys
    EventList = []
    for event in EventDict:
        EventList.append(EventDict[event])
        eventTracker.append(event)

    EventArray = np.array(EventList)


    targetArray = np.array([classInd]*(EventArray.shape[0]))
    return EventArray, targetArray


####











##### get a random set of buses where to place the PMUs
refRaw = 'savnw.raw'
BusDataDict = getBusData(refRaw)
tapbuslist = ['3007','204']
buslist = [bus for bus in BusDataDict.keys() if bus not in tapbuslist]
numBuses = len(buslist)

PMUBuses = random.sample(buslist,5)
print('Selected PMU buses: {}'.format(PMUBuses))
####




##### 
# get the fault data
eventTracker = []
### voltage
print('Reading fault data')
vFileName = 'fault3ph/vData3phLI.csv' # csv file containing voltage data (different types of fault)
aFileName = 'fault3ph/aData3phLI.csv' # csv file containing angle data (different types of fault)
tFileName = 'fault3ph/tData3ph.csv' # csv file containing the time data
eventKeyFile = 'fault3ph/eventIDFileLI.txt'
vFile = open(vFileName,'rb')
aFile = open(aFileName,'rb')
tFile = open(tFileName,'rb')
readerV=csv.reader(vFile,quoting=csv.QUOTE_NONNUMERIC) # so that the entries are converted to floats
readerA = csv.reader(aFile,quoting=csv.QUOTE_NONNUMERIC) # so that the entries are converted to floats
readerT = csv.reader(tFile,quoting=csv.QUOTE_NONNUMERIC)
tme = [row for idx, row in enumerate(readerT) if idx==0][0]


# make an organized dictionary
# read the event file
#print 'Organizing the csv data...'
eventList = []
with open(eventKeyFile,'r') as f:
    fileLines = f.read().split('\n')
    for line in fileLines[1:]:
        if line == '':
            continue
        eventList.append(line.strip())

# get the indices for fault on and fault off
faultontime = 0.1
timesteps = 40
numcyc = 6
shiftRange = 5
std = 0.001 # standard deviation of noise
startind = min([idx for idx,val in enumerate(tme) if val >= faultontime])
endind = startind + timesteps


EventDictV = {}
EventDictA = {}
SignalDict = {}
# get voltage
for idx, row in enumerate(readerV):
    eventKey = eventList[idx]
    eventKeyWords = eventKey.split('/')
    PL = eventKeyWords[0][1:].strip()
    faultbus = eventKeyWords[1][1:].strip()
    currentbus = eventKeyWords[2][1:].strip()
    faulttype = eventKeyWords[3].strip()
    phase = eventKeyWords[4].strip()
    faultZ = eventKeyWords[5].strip()


    if currentbus in PMUBuses:

        # add some noise to the outputs
        row = np.array(row) + np.random.normal(0,std,len(row)) # normal noise with standard deviation of std
        # pass the input through a 6 cycle average filter
        row = avgFilter(row,numcyc)

        eventID = 'R{}/F{}/{}/{}'.format(PL,faultbus,faulttype,faultZ)
        if eventID not in EventDictV:
            EventDictV[eventID] = Signals()
        EventDictV[eventID].SignalDict['B{}/{}'.format(currentbus,phase)] = row

# get angle
for idx, row in enumerate(readerA):
    eventKey = eventList[idx]
    eventKeyWords = eventKey.split('/')
    PL = eventKeyWords[0][1:].strip()
    faultbus = eventKeyWords[1][1:].strip()
    currentbus = eventKeyWords[2][1:].strip()
    faulttype = eventKeyWords[3].strip()
    phase = eventKeyWords[4].strip()
    faultZ = eventKeyWords[5].strip()

    if currentbus in PMUBuses:

        # add some noise to the outputs
        row = np.array(row) + np.random.normal(0,std,len(row)) # normal noise with standard deviation of std
        # pass the input through a 6 cycle average filter
        row = avgFilter(row,numcyc)


        eventID = 'R{}/F{}/{}/{}'.format(PL,faultbus,faulttype,faultZ)
        if eventID not in EventDictA:
            EventDictA[eventID] = Signals()
        EventDictA[eventID].SignalDict['B{}/{}'.format(currentbus,phase)] = row



# arrange all the fault event data into rows sequentially

AllEventListV = []
AllEventListA = []
targetList = []
for event in EventDictV:
    sV = EventDictV[event].SignalDict
    sA = EventDictA[event].SignalDict
    # get the event class
    eventKeyWords = event.split('/')
    faulttype = eventKeyWords[2].strip()
    if faulttype == 'ABCG':
        #targetList.append(0)
        #eventTracker.append(event)
        #loopFn(sV,sA)
        loopFn(event, sV,sA, 'ABCG')


    elif faulttype == 'AG':
        loopFn(event, sV,sA, 'AG')
        # get SLG A data
        #targetList.append(1)
        #eventTracker.append(event)
        #loopFn(sV,sA)




        # get SLG B data
        #targetList.append(2)
        #eventTracker.append(event.replace('AG','BG'))
        #loopFn(sV,sA)



        # get SLG C data
        #targetList.append(3)
        #eventTracker.append(event.replace('AG','CG'))
        #loopFn(sV,sA)
 


AllEventListV = np.array(AllEventListV)
AllEventListA = np.array(AllEventListA)


FaultEventArray = np.concatenate((AllEventListV, AllEventListA),axis = 1)
faulttargetArray = np.array(targetList)


###########



#### now complete the modification of all the other types of events

##############
# generator outages
print('Reading gen out data...')
genOutDir = 'GenOut'
vFilePath = '{}/vGenOut.csv'.format(genOutDir)
aFilePath = '{}/aGenOut.csv'.format(genOutDir)
eventFilePath = '{}/eventGenOut.txt'.format(genOutDir)
timeDataFilePath = '{}/tGenOut.csv'.format(genOutDir)
genoutArrayV, genoutTarget =  outageFn(vFilePath,timeDataFilePath,eventFilePath, 4, 'G')
genoutArrayA, _ =  outageFn(aFilePath,timeDataFilePath,eventFilePath, 4, 'G')

genoutArray = np.concatenate((genoutArrayV,genoutArrayA),axis = 1)

###########


##############
# line  outages
print('Reading line out data...')
# get the generator outage data
lineOutDir = 'LineOut'

vFilePath = '{}/vLineOut.csv'.format(lineOutDir)
aFilePath = '{}/aLineOut.csv'.format(lineOutDir)
eventFilePath = '{}/eventLineOut.txt'.format(lineOutDir)
timeDataFilePath = '{}/tLineOut.csv'.format(lineOutDir)
lineOutArrayV, lineOutTarget =  outageFn(vFilePath,timeDataFilePath,eventFilePath, 5, 'L')
lineOutArrayA, _ =  outageFn(aFilePath,timeDataFilePath,eventFilePath, 5, 'L')
lineOutArray = np.concatenate((lineOutArrayV,lineOutArrayA),axis = 1)
###########



##############
# transformer  outages
print('Reading tf out data...')
# get the generator outage data
TFOutDir = 'TFOut'

vFilePath = '{}/vTFOut.csv'.format(TFOutDir)
aFilePath = '{}/aTFOut.csv'.format(TFOutDir)
eventFilePath = '{}/eventTFOut.txt'.format(TFOutDir)
timeDataFilePath = '{}/tTFOut.csv'.format(TFOutDir)
TFOutArrayV, TFOutTarget =  outageFn(vFilePath,timeDataFilePath,eventFilePath, 6, 'T')
TFOutArrayA, _ =  outageFn(aFilePath,timeDataFilePath,eventFilePath, 6, 'T')
TFOutArray = np.concatenate((TFOutArrayV,TFOutArrayA),axis = 1)

###########




# concatenate the fault, genout, lineout and tfout array
fullArray = np.concatenate((FaultEventArray, genoutArray,lineOutArray,TFOutArray),axis=0) # vertical stacks: 3 ph data, 1 ph data 
fullTargetArray = np.concatenate((faulttargetArray,genoutTarget,lineOutTarget,TFOutTarget))



# plt.plot(fullArray[-1][:120])
# plt.grid()
# plt.show()


##### evaluate SVM classifier
print 'Evaluating SVM'
from sklearn.svm import SVC
x = fullArray
print('Shape of input array: {}'.format(x.shape))
#x = fullArrayFil
y = fullTargetArray.astype(int)

svm_model_linear = SVC(kernel = 'linear', C = 1)
#svm_model_linear = SVC(kernel = 'rbf', C = 1) # is not as good as linear


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

# ### quick test
# svm_model_linear.fit(x_train, y_train)
# y_pred = svm_model_linear.predict(x_test)
# accuracy = accuracy_score(y_test, y_pred)*100
# print ('Accuracy: {}'.format(accuracy))
# conf_mat = confusion_matrix(y_test, y_pred)
# print (conf_mat)
# ####



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

