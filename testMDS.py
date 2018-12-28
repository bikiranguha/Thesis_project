# script to test multidimensional modeling on fault voltage drop data
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
start = 0.1
startind = min([idx for idx,val in enumerate(tme) if val >= start]) - 5 # to get prefault voltage
endind = startind + 10



# make an organized dictionary
# read the event file
print 'Organizing the csv data...'
tapbuslist = ['3007','204']
BusList = [bus for bus in BusDataDict.keys() if bus not in tapbuslist]
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
        for b in BusList:
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
        for b in BusList:
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
        for b in BusList:
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
        for b in BusList:
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


"""
## compare voltage drop differences for a case
faultbus1 = '151'
faultbus2 = '152'
event1 = 'R100/F{}/ABCG/1.0e-6'.format(faultbus1)
event2 = 'R100/F{}/ABCG/1.0e-6'.format(faultbus2)
event1Ind = eventList.index(event1)
event2Ind = eventList.index(event2)

bus1ind = BusList.index(faultbus1)
bus2ind = BusList.index(faultbus2)

x=AllEventArray



# get the voltage at bus 152 when fault at bus 151
sample1 = x[event1Ind]
sample1array = sample1.reshape(-1,30)
bus1fbus2data = sample1array[bus2ind]

# get the voltage at bus 152 when fault at bus 151
sample2 = x[event2Ind]
sample2array = sample2.reshape(-1,30)
bus2fbus1data = sample2array[bus1ind]


# plot the data
plt.plot(bus1fbus2data,label = 'f151b152')
plt.plot(bus2fbus1data,label = 'f152b151')
plt.grid()
plt.title('Comparison of voltage drop at two buses with fault exchange')
plt.legend()
plt.show()
##
"""



## test MDS on multiple events stacked in different rows
x=AllEventArray
tapbuslist = ['3007','204']
BusList = [bus for bus in BusDataDict.keys() if bus not in tapbuslist]

# build a distance array
numBuses = len(BusList)
distArray = np.zeros((numBuses,numBuses))


### getting the metric differences
for ind, faultbus in enumerate(BusList):
    if faultbus in tapbuslist:
        continue
    event = 'R100/F{}/ABCG/1.0e-6'.format(faultbus)
    eventInd = eventList.index(event)
    sample = x[eventInd]
    # reshape into an array where each row represents a three phase voltage sample
    # over the time window
    sampleArray = sample.reshape(-1,30)

    


    # get a min v dict
    minVDict = {}
    for i in range(sampleArray.shape[0]):
        minVDict[i] = sampleArray[i].min()

    # get a prefault dict
    prefaultDict = {}
    for i in range(sampleArray.shape[0]):
        prefaultDict[i] = sampleArray[i][0]


    # get a vdrop dict
    vdropDict = {}
    for i in range(sampleArray.shape[0]):
        prefaulti = prefaultDict[i]
        minvi = minVDict[i]
        vdropi = abs((prefaulti-minvi)/prefaulti*100)
        vdropDict[i] = vdropi


    for j in range(numBuses): # populate only the current row
        prefaulti = prefaultDict[ind]
        prefaultj = prefaultDict[j]

        minvi = minVDict[ind]
        minvj = minVDict[j]

        vdropi = abs((prefaulti-minvi)/prefaulti*100)
        vdropj = abs((prefaultj-minvj)/prefaultj*100)

        diff = abs(vdropi-vdropj)
        distArray[ind,j] = diff
    
        # hack to make distArray symmetric
        #if j > ind:
        #    distArray[ind,j] = diff
        #    distArray[j,ind] = diff
###


# make the array symmetric by replacing with averages of the off-diagonal mirror points

for i in range(distArray.shape[0]):
    for j in range(distArray.shape[1]):
        avg = (distArray[i,j] + distArray[j,i])/2
        distArray[i,j] = avg
        distArray[j,i] = avg









"""
### getting a rank matrix (higher the rank, closer the distance)

for ind, faultbus in enumerate(BusList):
    if faultbus in tapbuslist:
        continue
    event = 'R100/F{}/ABCG/1.0e-6'.format(faultbus)
    eventInd = eventList.index(event)
    sample = x[eventInd]
    # reshape into an array where each row represents a three phase voltage sample
    # over the time window
    sampleArray = sample.reshape(-1,30)

    


    # get a min v dict
    minVDict = {}
    for i in range(sampleArray.shape[0]):
        minVDict[i] = sampleArray[i].min()

    # get a prefault dict
    prefaultDict = {}
    for i in range(sampleArray.shape[0]):
        prefaultDict[i] = sampleArray[i][0]


    # get a vdrop dict
    vdropDict = {}
    for i in range(sampleArray.shape[0]):
        prefaulti = prefaultDict[i]
        minvi = minVDict[i]
        vdropi = abs((prefaulti-minvi)/prefaulti*100)
        vdropDict[i] = vdropi


    diffList = []
    for j in range(numBuses): # populate only the current row
        prefaulti = prefaultDict[ind]
        prefaultj = prefaultDict[j]

        minvi = minVDict[ind]
        minvj = minVDict[j]

        vdropi = abs((prefaulti-minvi)/prefaulti*100)
        vdropj = abs((prefaultj-minvj)/prefaultj*100)

        diff = abs(vdropi-vdropj)
        diffList.append(diff)

    diffListSorted = sorted(diffList)

    rank = []
    for diff in diffList:
        r = diffListSorted.index(diff)
        rank.append(r)
    #print diffList
    #print rank
    distArray[ind,:] = np.array(rank)

###
"""





### export distArray to a pandas csv file
import pandas as pd

# get a dict of the of the bus indices
indexDict = {}
for ind, val in enumerate(BusList):
    indexDict[ind] = val

df = pd.DataFrame(data = distArray, columns=BusList)
df.rename(index=indexDict, inplace=True)


df.to_csv('dist4.csv')

###


####
# feed the distance array to a MDS algorithm and get the co-ordinates
from sklearn import manifold
mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9,
                   dissimilarity="precomputed", n_jobs=1)
pos = mds.fit(distArray).embedding_
for i in range(numBuses):
    bus = BusDataDict.keys()[i]
    plt.scatter(pos[i, 0], pos[i, 1])
    plt.text(pos[i, 0], pos[i, 1],s=bus)

plt.grid()
#plt.title('Co-ordinate map with symmetric hack')
#plt.title('Co-ordinate map with average substitution')
plt.title('Distance: Voltage drop difference')
plt.show()
##



"""
## test MDS on one event
event = 'R100/F152/ABCG/1.0e-6'
print event
eventInd = eventList.index(event)
x=AllEventArray
sample = x[eventInd]
# reshape into an array where each row represents a three phase voltage sample
# over the time window
sampleArray = sample.reshape(-1,30)




# get a min v dict
minVDict = {}
for i in range(sampleArray.shape[0]):
    minVDict[i] = sampleArray[i].min()

# get a prefault dict
prefaultDict = {}
for i in range(sampleArray.shape[0]):
    prefaultDict[i] = sampleArray[i][0]


# get a vdrop dict
vdropDict = {}
for i in range(sampleArray.shape[0]):
    prefaulti = prefaultDict[i]
    minvi = minVDict[i]
    vdropi = abs((prefaulti-minvi)/prefaulti*100)
    vdropDict[i] = vdropi



for ind, value in sorted(vdropDict.iteritems(), key=lambda (k,v): v, reverse = True): 
    print '{}:{}'.format(BusDataDict.keys()[ind],value)


# build a distance array
numBuses = len(BusDataDict.keys())
distArray = np.zeros((numBuses,numBuses))




for i in range(numBuses):
    for j in range(numBuses):
        prefaulti = prefaultDict[i]
        prefaultj = prefaultDict[j]

        minvi = minVDict[i]
        minvj = minVDict[j]

        vdropi = abs((prefaulti-minvi)/prefaulti*100)
        vdropj = abs((prefaultj-minvj)/prefaultj*100)

        diff = abs(vdropi-vdropj)
        distArray[i,j] = diff


### export distArray to a pandas csv file
import pandas as pd

# get a dict of the of the bus indices
indexDict = {}
for ind, val in enumerate(BusDataDict.keys()):
    indexDict[ind] = val

df = pd.DataFrame(data = distArray, columns=BusDataDict.keys())
df.rename(index=indexDict, inplace=True)


df.to_csv('distTmp.csv')
####

# feed the distance array to a MDS algorithm and get the co-ordinates
from sklearn import manifold
mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9,
                   dissimilarity="precomputed", n_jobs=1)
pos = mds.fit(distArray).embedding_
for i in range(numBuses):
    bus = BusDataDict.keys()[i]
    plt.scatter(pos[i, 0], pos[i, 1])
    plt.text(pos[i, 0], pos[i, 1],s=bus)

plt.grid()
plt.show()
##
"""

