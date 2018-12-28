# script to test multidimensional modeling on fault voltage drop data
print 'Importing modules'
import csv
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_predict, cross_validate, cross_val_score, KFold
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from getBusDataFn import getBusData

refRaw = 'savnw.raw'
BusDataDict = getBusData(refRaw)


class Signals(object):
    def __init__(self):
        self.SignalDict = {}



print 'Reading the csv files'
vFileName = 'G:/My Drive/My PhD research/Running TS3ph/fault3ph/vData3phLI.csv' # csv file containing voltage data (different types of fault)
tFileName = 'G:/My Drive/My PhD research/Running TS3ph/fault3ph/tData3ph.csv' # csv file containing the time data
eventKeyFile = 'G:/My Drive/My PhD research/Running TS3ph/fault3ph/eventIDFileLI.txt'
vFile = open(vFileName,'rb')
tFile = open(tFileName,'rb')
readerV=csv.reader(vFile,quoting=csv.QUOTE_NONNUMERIC) # so that the entries are converted to floats
readerT = csv.reader(tFile,quoting=csv.QUOTE_NONNUMERIC)
tme = [row for idx, row in enumerate(readerT) if idx==0][0]

###
start = 0.1
startind = min([idx for idx,val in enumerate(tme) if val >= start]) - 5 # to get prefault voltage
endind = startind + 10



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
        eventList.append(event)
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
        eventList.append(event)
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
        eventList.append(event.replace('AG','BG'))
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
        eventList.append(event.replace('AG','CG'))
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



# get the fault impedances of events where the voltage doesnt drop much
for ind, event in enumerate(eventList):
    faultZ = event.split('/')
    data = AllEventArray[ind]
    minV = data.min()
    if minV >0.5:
        print event


"""
## get the phase A voltage plot for a fault which has high impedance
event = 'R106/F3005/ABCG/2.7e-02'
print event
eventInd = eventList.index(event)
x=AllEventArray
sample = x[eventInd]
# reshape into an array where each row represents a three phase voltage sample
# over the time window
sampleArray = sample.reshape(-1,30)
for i in range(sampleArray.shape[0]):
    v = sampleArray[i][:10]
    plt.plot(v)

plt.title('Phase A voltage of all buses')
plt.xlabel('Samples')
plt.ylabel('V (pu)')
plt.grid()
plt.show()
"""





