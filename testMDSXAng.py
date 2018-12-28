# get electrical map using branch impedance
# then try and use angle data to see if the distance matches the electrical distance
print 'Importing modules'
import csv
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from getBusDataFn import getBusData
import pandas as pd
from getNeighboursAtCertainDepthFn import getNeighboursDepthN
refRaw = 'savnw.raw'
BusDataDict = getBusData(refRaw)


class Signals(object):
    def __init__(self):
        self.SignalDict = {}


tapbuslist = ['3007','204']
buslist = [bus for bus in BusDataDict.keys() if bus not in tapbuslist]
numBuses = len(buslist)









"""
### generate an array of electrical distance using reactance values
print('Generating electrical distance array...')
elecDistArray = np.zeros((len(buslist),len(buslist)))
#buslist = BusDataDict.keys()
numBuses = len(buslist)
for i in range(numBuses):
    for j in range(numBuses):
        busi = buslist[i]
        busj = buslist[j]
        
        if i == j:
            elecDistArray[i,j] == 0.0
            continue

        depth = 1
        neighbours = getNeighboursDepthN(busi,refRaw,depth)
        while busj not in neighbours[busi].toBus:
            depth +=1
            neighbours = getNeighboursDepthN(busi,refRaw,depth)
        busjind = neighbours[busi].toBus.index(busj)
        X = neighbours[busi].X[busjind]
        elecDistArray[i,j] = X
        elecDistArray[j,i] = X

# make the array symmetric by replacing the off-diagonal elements by their average
for i in range(elecDistArray.shape[0]):
    for j in range(elecDistArray.shape[1]):
        avg = (elecDistArray[i,j] + elecDistArray[j,i])/2
        elecDistArray[i,j] = avg
        elecDistArray[j,i] = avg

### export distArray to a pandas csv file
import pandas as pd

# get a dict of the of the bus indices
indexDict = {}
for ind, val in enumerate(buslist):
    indexDict[ind] = val

df = pd.DataFrame(data = elecDistArray, columns=buslist)
df.rename(index=indexDict, inplace=True)


df.to_csv('elecdistsavnw.csv')
"""




### read array from csv file
print('Reading electrical distance array from csv file')
df = pd.read_csv('elecdistsavnw.csv')

# get a dict of the of the bus indices
indexDict = {}
for ind, val in enumerate(buslist):
    indexDict[ind] = val

df.rename(index=indexDict, inplace=True)

# get the array from the dataframe
elecDistArray = np.zeros((len(buslist),len(buslist)))
numBuses = len(buslist)
for i in range(numBuses):
    for j in range(numBuses):
        busi = buslist[i]
        busj = buslist[j]
        elecDistArray[i,j] = df[busi][busj]
###


# use MDS ro get the electrical distance map
print('Using MDS to get you the electrical map')
from sklearn import manifold
mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9,
                   dissimilarity="precomputed", n_jobs=1)
pos = mds.fit(elecDistArray).embedding_
for i in range(numBuses):
    bus = buslist[i]
    plt.scatter(pos[i, 0], pos[i, 1])
    plt.text(pos[i, 0], pos[i, 1],s=bus)

plt.grid()
#plt.title('Co-ordinate map with symmetric hack')
plt.title('Electrical Distance Map')
plt.show()
plt.close()

####


######################
# using the generator outage sensitivity
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


start = 0.1
faultind  = min([idx for idx,val in enumerate(tme) if val >= start])
startind = faultind - 5 # to get prefault voltage
endind = startind + 10


EventDict = {}
# get the voltage data (and simulate balanced three phase)
for idx, row in enumerate(vReader):
    eventKey = eventList[idx]
    EventDict[eventKey] = row[startind:endind]
    #eventKeyWords = eventKey.split('/')
    #PL = eventKeyWords[0][1:].strip()
    #genbus = eventKeyWords[1][1:].strip()
    #currentbus = eventKeyWords[2][1:].strip()
    #eventID = 'R{}/G{}'.format(PL,genbus)
    

# close all files
vFilePath.close()
aFilePath.close()
fFilePath.close()
timeDataFilePath.close()


genbuslist = []
for bus in BusDataDict:
    bustype = BusDataDict[bus].type
    if bustype == '2' or bustype == '3':
        genbuslist.append(bus)

# make a vector of voltage deviations for each bus wrt each generator outage

voltdevdict ={}

for bus in buslist:
    voltdevdict[bus] = []

    for gen in genbuslist:

        event = 'R100/G{}/B{}'.format(gen,bus)
        try:
            vData = np.array(EventDict[event])
        except: # gen outage caused convergence issues, so not recorded
            continue
        precontv = vData[0]
        postconminv = vData.min()
        delv = abs(precontv - postconminv)
        voltdevdict[bus].append(delv)

# build a distance array using delv diff

vsendistarray = np.zeros((len(buslist),len(buslist)))

for i, busi in enumerate(buslist):
    for j, busj in enumerate(buslist):
        delvi = np.array(voltdevdict[busi])
        delvj = np.array(voltdevdict[busj])
        d = LA.norm(delvi - delvj)
        vsendistarray[i,j] = d

### export distArray to a pandas csv file
import pandas as pd

# get a dict of the of the bus indices
indexDict = {}
for ind, val in enumerate(buslist):
    indexDict[ind] = val

df = pd.DataFrame(data = vsendistarray, columns=buslist)
df.rename(index=indexDict, inplace=True)


df.to_csv('vsentdistsavnw.csv')


# use MDS ro get the electrical distance map
print('Using MDS to get you the electrical map')
from sklearn import manifold
mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9,
                   dissimilarity="precomputed", n_jobs=1)
pos = mds.fit(vsendistarray).embedding_
for i in range(numBuses):
    bus = buslist[i]
    plt.scatter(pos[i, 0], pos[i, 1])
    plt.text(pos[i, 0], pos[i, 1],s=bus)

plt.grid()
#plt.title('Co-ordinate map with symmetric hack')
plt.title('Voltage sensitivity Distance Map')
plt.show()
plt.close()















#################








#############
# Getting the angular change during a fault

print 'Reading the csv files having the angle data'
aFileName = 'fault3ph/aData3phLI.csv' # csv file containing angle data (for all the different load levels as well as different fault Z)
tFileName = 'fault3ph/tData3ph.csv' # csv file containing the time data
eventKeyFile = 'fault3ph/eventIDFileLI.txt'
aFile = open(aFileName,'rb')
tFile = open(tFileName,'rb')
readerA=csv.reader(aFile,quoting=csv.QUOTE_NONNUMERIC) # so that the entries are converted to floats
readerT = csv.reader(tFile,quoting=csv.QUOTE_NONNUMERIC)
tme = [row for idx, row in enumerate(readerT) if idx==0][0]

###
start = 0.1
faultind  = min([idx for idx,val in enumerate(tme) if val >= start])
startind = faultind - 5 # to get prefault voltage
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



# get all the angle data

EventDict = {}
SignalDict = {}
for idx, row in enumerate(readerA):
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
        for b in buslist:
            valA = s['B{}/A'.format(b)][startind:endind]

            for a in valA:
                allSignals.append(a)
                

        AllEventList.append(allSignals)



    elif faulttype == 'AG':
        # get SLG A data
        eventList.append(event)
        targetList.append(1)
        allSignals = []
        for b in buslist:
            valA = s['B{}/A'.format(b)][startind:endind]

            for a in valA:
                allSignals.append(a)
                


        AllEventList.append(allSignals)





AllEventArray = np.array(AllEventList)



## test MDS on multiple events stacked in different rows
x=AllEventArray


event = 'R100/F153/ABCG/1.0e-6'
eventInd = eventList.index(event)
sample = x[eventInd]
# reshape into an array where each row represents a three phase voltage sample
# over the time window
sampleArray = sample.reshape(-1,10)

for i in range(10):
    plt.plot(sampleArray[i],label = buslist[i])

plt.grid()
plt.legend()
plt.show()


# build a distance array
numBuses = len(buslist)
anglDistArray = np.zeros((numBuses,numBuses))



### getting the metric differences
for ind, faultbus in enumerate(buslist):

    event = 'R100/F{}/ABCG/1.0e-6'.format(faultbus)
    eventInd = eventList.index(event)
    sample = x[eventInd]
    # reshape into an array where each row represents a three phase voltage sample
    # over the time window
    sampleArray = sample.reshape(-1,10)

    
    #maxchangeDict = {}
    #for i in range(sampleArray.shape[0]):
    #    roc = abs(np.gradient(sampleArray[i])) # abs rate of change
    #    maxchangeDict[i] = roc.max()

    valAfterFault = {}
    for i in range(sampleArray.shape[0]):
        valAfterFault[i] = sampleArray[i][-1] # sample just when fault happens


    # get a prefault angle dict
    prefaultDict = {}
    for i in range(sampleArray.shape[0]):
        prefaultDict[i] = sampleArray[i][0]


    # get an angle change dict
    aDropDict = {}
    for i in range(sampleArray.shape[0]):
        prefaulti = prefaultDict[i]
        #maxchangei = maxchangeDict[i]
        valfi = valAfterFault[i]
        #adropi = abs(prefaulti-maxchangei)
        adropi = abs(prefaulti-valfi)
        aDropDict[i] = adropi


    
    # array based on difference in angle change after fault
    for j in range(numBuses): # populate only the current row
        changediff = abs(aDropDict[ind]-aDropDict[j])
        anglDistArray[ind,j] = changediff
    

    

###


# make the array symmetric by replacing with averages of the off-diagonal mirror points

for i in range(anglDistArray.shape[0]):
    for j in range(anglDistArray.shape[1]):
        avg = (anglDistArray[i,j] + anglDistArray[j,i])/2
        anglDistArray[i,j] = avg
        anglDistArray[j,i] = avg


### export distArray to a pandas csv file
import pandas as pd

# get a dict of the of the bus indices
indexDict = {}
for ind, val in enumerate(buslist):
    indexDict[ind] = val

df = pd.DataFrame(data = anglDistArray, columns=buslist)
df.rename(index=indexDict, inplace=True)


df.to_csv('angldistsavnw.csv')


# use MDS ro get the electrical distance map
print('Using MDS to get you the electrical map')
from sklearn import manifold
mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9,
                   dissimilarity="precomputed", n_jobs=1)
pos = mds.fit(anglDistArray).embedding_
for i in range(numBuses):
    bus = buslist[i]
    plt.scatter(pos[i, 0], pos[i, 1])
    plt.text(pos[i, 0], pos[i, 1],s=bus)

plt.grid()
#plt.title('Co-ordinate map with symmetric hack')
plt.title('Distance: Angular change difference')
plt.show()
plt.close()
######################3




























