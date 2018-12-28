# test an event classifier where the events to be classified are: (balanced and unbalanced) faults, generator and line outages (grouped together)

# here the classifier only outputs the fault type using the voltage data of all the buses at the time of fault
print 'Importing modules'
import csv
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_predict, cross_validate, cross_val_score, KFold
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


#####################
# organize the fault data
eventTracker = []
print 'Reading the csv files for fault data...'
vFileName = 'fault3ph/vData3phLI.csv' # csv file containing voltage data (different types of fault)
tFileName = 'fault3ph/tData3ph.csv' # csv file containing the time data
eventKeyFile = 'fault3ph/eventIDFileLI.txt'
vFile = open(vFileName,'rb')
tFile = open(tFileName,'rb')
readerV=csv.reader(vFile,quoting=csv.QUOTE_NONNUMERIC) # so that the entries are converted to floats
readerT = csv.reader(tFile,quoting=csv.QUOTE_NONNUMERIC)
tme = [row for idx, row in enumerate(readerT) if idx==0][0]

###
start = 0.1
startind = min([idx for idx,val in enumerate(tme) if val >= start])
endind = startind + 10
#endind = startind + 20



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

    # ignore all faults with relatively high impedance
    if float(faultZ) > 0.5e-2:
        continue
    eventID = 'R{}/F{}/{}/{}'.format(PL,faultbus,faulttype,faultZ)
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
        eventTracker.append(event)
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
        eventTracker.append(event)
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
        eventTracker.append(event.replace('AG','BG'))
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
        eventTracker.append(event.replace('AG','CG'))
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

FaultEventArray = np.array(AllEventList)
faulttargetArray = np.array(targetList)
#############






#####
eventSet = set() # to keep track of the event ids
N_1EventDict = {}

##############
# generator outages
genoutvdata = []
# get the generator outage data
genOutDir = 'GenOut'

vFilePath = '{}/vGenOut.csv'.format(genOutDir)
aFilePath = '{}/aGenOut.csv'.format(genOutDir)
fFilePath = '{}/fGenOut.csv'.format(genOutDir)
eventFilePath = '{}/eventGenOut.txt'.format(genOutDir)
timeDataFilePath = '{}/t.csv'.format(genOutDir)



# file objects
eventList = []
with open(eventFilePath,'r') as f:
    fileLines = f.read().split('\n')
    for line in fileLines[1:]:
        if line == '':
            continue
        eventList.append(line.strip())


vFilePath = open(vFilePath, 'rb') # 'wb' needed to avoid blank space in between lines
aFilePath = open(aFilePath, 'rb')
fFilePath = open(fFilePath, 'rb')
timeDataFilePath = open(timeDataFilePath, 'rb')


vReader = csv.reader(vFilePath,quoting=csv.QUOTE_NONNUMERIC) # 'wb' needed to avoid blank space in between lines
aReader = csv.reader(aFilePath,quoting=csv.QUOTE_NONNUMERIC)
fReader = csv.reader(fFilePath,quoting=csv.QUOTE_NONNUMERIC)
tReader = csv.reader(timeDataFilePath,quoting=csv.QUOTE_NONNUMERIC)


tme = [row for idx, row in enumerate(tReader) if idx==0][0]

# get the voltage data (and simulate balanced three phase)
for idx, row in enumerate(vReader):
    eventKey = eventList[idx]
    eventKeyWords = eventKey.split('/')
    PL = eventKeyWords[0][1:].strip()
    genbus = eventKeyWords[1][1:].strip()
    currentbus = eventKeyWords[2][1:].strip()
    eventID = 'R{}/G{}'.format(PL,genbus)
    if eventID not in N_1EventDict:
        N_1EventDict[eventID] = []
        #eventTracker.append(eventID)
        #eventSet.add(eventID)

    for bs in range(3): # simulate three phase
        for v in row[startind:endind]:
            #genoutvdata.append(v)
            N_1EventDict[eventID].append(v)

#genoutArray = np.array(genoutvdata)

# close all files
vFilePath.close()
aFilePath.close()
fFilePath.close()
timeDataFilePath.close()
###########





####### 
# line outage data
lineoutvdata = []
lineOutDir = 'LineOut'

vFilePath = '{}/vLineOut.csv'.format(lineOutDir)
aFilePath = '{}/aLineOut.csv'.format(lineOutDir)
fFilePath = '{}/fLineOut.csv'.format(lineOutDir)
eventFilePath = '{}/eventLineOut.txt'.format(lineOutDir)
timeDataFilePath = '{}/t.csv'.format(lineOutDir)

# file objects
eventList = []
with open(eventFilePath,'r') as f:
    fileLines = f.read().split('\n')
    for line in fileLines[1:]:
        if line == '':
            continue
        eventList.append(line.strip())


vFilePath = open(vFilePath, 'rb') # 'wb' needed to avoid blank space in between lines
aFilePath = open(aFilePath, 'rb')
fFilePath = open(fFilePath, 'rb')
timeDataFilePath = open(timeDataFilePath, 'rb')


vReader = csv.reader(vFilePath,quoting=csv.QUOTE_NONNUMERIC) # 'wb' needed to avoid blank space in between lines
aReader = csv.reader(aFilePath,quoting=csv.QUOTE_NONNUMERIC)
fReader = csv.reader(fFilePath,quoting=csv.QUOTE_NONNUMERIC)
tReader = csv.reader(timeDataFilePath,quoting=csv.QUOTE_NONNUMERIC)


tme = [row for idx, row in enumerate(tReader) if idx==0][0]

# examine the voltage data
vDict = {}

for idx, row in enumerate(vReader):
    eventKey = eventList[idx]
    eventKeyWords = eventKey.split('/')
    PL = eventKeyWords[0][1:].strip()
    line = eventKeyWords[1][1:].strip()
    currentbus = eventKeyWords[2][1:].strip()
    eventID = 'R{}/L{}'.format(PL,line)
    if eventID not in N_1EventDict:
        #eventTracker.append(eventID)
        #eventSet.add(eventID)
        N_1EventDict[eventID] = []

    for bs in range(3): # simulate three phase
        for v in row[startind:endind]:
            #lineoutvdata.append(v)
            N_1EventDict[eventID].append(v)

#lineoutArray = np.array(lineoutvdata).reshape(-1,)





# close all files
vFilePath.close()
aFilePath.close()
fFilePath.close()
timeDataFilePath.close()
######################

N_1Events =[]
for event in N_1EventDict:
    N_1Events.append(N_1EventDict[event])
    eventTracker.append(event)

N_1EventArray = np.array(N_1Events)


N_1targetArray = np.array([4]*(N_1EventArray.shape[0]))

#N_1targetArray = np.array([4]*(genoutArray.shape[0] + lineoutArray.shape[0]))

# stack the arrays
fullArray = np.concatenate((FaultEventArray, N_1EventArray),axis=0) # vertical stacks: faults, gen out, line out
fullTargetArray = np.concatenate((faulttargetArray,N_1targetArray))


with open('EventClassifierLog.txt','w') as f:
    for event in eventTracker:
        f.write(event)
        f.write('\n')


"""
print 'Evaluating the classifier'
##### evaluate SVM classifier
from sklearn.svm import SVC
x = fullArray
#x = fullArrayFil
y = fullTargetArray.astype(int)

#svm_model_linear = SVC(kernel = 'linear', C = 1)
svm_model_linear = SVC(kernel = 'rbf', C = 1) # is not as good as linear

cv = KFold( n_splits=10)
y_pred  =  cross_val_predict(svm_model_linear, x, y, cv=cv)
accuracy = accuracy_score(y,y_pred)*100

print 'Accuracy: {}'.format(accuracy)
conf_mat = confusion_matrix(y, y_pred)
print accuracy
print conf_mat

scores = cross_val_score(svm_model_linear, x, y, cv=cv)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
"""



"""
# plot any event
eventID  = 'R100/F3003/ABCG/8.0e-03'
eventInd = eventTracker.index(eventID)
sampleData = fullArray[eventInd]
plt.plot(sampleData)
plt.title(eventID)
plt.grid()
plt.show()
"""