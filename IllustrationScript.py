# script to illustrate the various event plots we currently have
print 'Importing modules'
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from getBusDataFn import getBusData
import random

refRaw = 'savnw.raw'
BusDataDict = getBusData(refRaw)


class Signals(object):
    def __init__(self):
        self.SignalDict = {}


###
def getEventDict(dataFileName, tFileName, eventKeyFile,eventtype):
    # function to return an organized  event dict from the given filedata

    #print 'Reading the csv files'
    #vFileName = 'fault3ph/Long/vData3phLI.csv' # csv file containing 5.0 second voltage data (different types of fault)
    #vFileName = 'fault3ph/vData3phLI.csv' # csv file containing 0.5 second voltage data (different types of fault)
    #tFileName = 'fault3ph/tData3ph.csv' # csv file containing the time data
    #eventKeyFile = 'fault3ph/eventIDFileLI.txt'
    dataFile = open(dataFileName,'rb')
    tFile = open(tFileName,'rb')
    readerV=csv.reader(dataFile,quoting=csv.QUOTE_NONNUMERIC) # so that the entries are converted to floats
    readerT = csv.reader(tFile,quoting=csv.QUOTE_NONNUMERIC)
    tme = [row for idx, row in enumerate(readerT) if idx==0][0]



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
    SimpleEventDict = {}
    for idx, row in enumerate(readerV):
        eventKey = eventList[idx]
        SimpleEventDict[eventKey] = row
        eventKeyWords = eventKey.split('/')
        PL = eventKeyWords[0][1:].strip()
        stuffout = eventKeyWords[1][1:].strip()
        currentbus = eventKeyWords[2][1:].strip()

        eventID = 'R{}/{}{}'.format(PL,eventtype,stuffout)
        if eventID not in EventDict:
            EventDict[eventID] = Signals()
        EventDict[eventID].SignalDict['B{}'.format(currentbus)] = row
    dataFile.close()
    tFile.close()

    return EventDict, tme
############




#### get the generator outage data
genOutDir = 'GenOut'
vFilePath = '{}/vGenOut.csv'.format(genOutDir)
aFilePath = '{}/aGenOut.csv'.format(genOutDir)
fFilePath = '{}/fGenOut.csv'.format(genOutDir)
eventFilePath = '{}/eventGenOut.txt'.format(genOutDir)
timeDataFilePath = '{}/tGenOut.csv'.format(genOutDir)



#GenOutEventDict,tme = getEventDict(vFilePath, timeDataFilePath, eventFilePath,'G') # v
GenOutEventDict,tme = getEventDict(aFilePath, timeDataFilePath, eventFilePath,'G') # a
# plot all bus voltage for a certain generator outage
genoutevent = 'R100/G3018'
SignalDict = GenOutEventDict[genoutevent].SignalDict

plotdir = 'GenOutPlots'
k=1
for bus in SignalDict:
    plt.plot(tme, SignalDict[bus])
    ttl = '{}/{}'.format(genoutevent,bus)
    plt.title(ttl)
    plt.ylabel('V(pu)')
    plt.xlabel('Time')
    plt.grid()
    figname = '{}/Plot{}'.format(plotdir,k)
    plt.savefig('{}.png'.format(figname))
    plt.close()
    k+=1
####



"""
#### get the line outage data
LineOutDir = 'LineOut'
vFilePath = '{}/vLineOut.csv'.format(LineOutDir)
aFilePath = '{}/aLineOut.csv'.format(LineOutDir)
fFilePath = '{}/fLineOut.csv'.format(LineOutDir)
eventFilePath = '{}/eventLineOut.txt'.format(LineOutDir)
timeDataFilePath = '{}/tLineOut.csv'.format(LineOutDir)



#LineOutEventDict,tme = getEventDict(vFilePath, timeDataFilePath, eventFilePath,'L') # v
LineOutEventDict,tme = getEventDict(aFilePath, timeDataFilePath, eventFilePath,'L') # a 
# plot all bus voltage for a certain generator outage
lineoutevent = 'R100/L154,3008,1'
SignalDict = LineOutEventDict[lineoutevent].SignalDict

plotdir = 'LineOutPlots'
k=1
for bus in SignalDict:
    plt.plot(tme, SignalDict[bus])
    ttl = '{}/{}'.format(lineoutevent,bus)
    plt.title(ttl)
    plt.ylabel('V(pu)')
    plt.xlabel('Time')
    plt.grid()
    figname = '{}/Plot{}'.format(plotdir,k)
    plt.savefig('{}.png'.format(figname))
    plt.close()
    k+=1
####
"""





"""
#### get the tf outage data
TFOutDir = 'TFOut'
vFilePath = '{}/vTFOut.csv'.format(TFOutDir)
aFilePath = '{}/aTFOut.csv'.format(TFOutDir)
fFilePath = '{}/fTFOut.csv'.format(TFOutDir)
eventFilePath = '{}/eventTFOut.txt'.format(TFOutDir)
timeDataFilePath = '{}/tTFOut.csv'.format(TFOutDir)



#TFOutEventDict,tme = getEventDict(vFilePath, timeDataFilePath, eventFilePath,'TF') # v
TFOutEventDict,tme = getEventDict(aFilePath, timeDataFilePath, eventFilePath,'TF') # a
# plot all bus voltage for a certain generator outage
TFOutevent = "R100/TFF  3008,  3018,     0,'1 '"
SignalDict = TFOutEventDict[TFOutevent].SignalDict

plotdir = 'TFOutPlots'
k=1
for bus in SignalDict:
    plt.plot(tme, SignalDict[bus])
    ttl = '{}/{}'.format(TFOutevent,bus)
    plt.title(ttl)
    plt.ylabel('V(pu)')
    plt.xlabel('Time')
    plt.grid()
    figname = '{}/Plot{}'.format(plotdir,k)
    plt.savefig('{}.png'.format(figname))
    plt.close()
    k+=1
####
"""


# plot all bus voltage for a certain line outage (near the generator)





# plot all bus voltage for a certain transformer outage (near the generator)























"""
##### Illustration of fault data


print 'Reading the csv files'
#vFileName = 'fault3ph/Long/vData3phLI.csv' # csv file containing 5.0 second voltage data (different types of fault)
vFileName = 'fault3ph/vData3phLI.csv' # csv file containing 0.5 second voltage data (different types of fault)
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
SimpleEventDict = {}
for idx, row in enumerate(readerV):
    eventKey = eventList[idx]
    SimpleEventDict[eventKey] = row
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

eventkey0 = EventDict.keys()[0]
signalkey0 = EventDict[eventkey0].SignalDict.keys()[0]

v = EventDict[eventkey0].SignalDict[signalkey0]

print(len(v[startind:]))
######
"""

"""
#### find out the min dip in the fault dataset
timeWindow = 10
start = 0.1
startind = min([idx for idx,val in enumerate(tme) if val >= start])
endind = startind + timeWindow

minVDict = {}

for key in SimpleEventDict:
    eventKeyWords = key.split('/')
    phase = eventKeyWords[4].strip()
    if phase == 'A':
        val = np.array(SimpleEventDict[key][startind:endind])
        minVDict[key] = val.min()


count = 0
out = []
for key, value in sorted(minVDict.iteritems(), key=lambda (k,v): v, reverse = True):
    out.append('{}:{}'.format(key,value))
    count +=1
    if value < 0.7:
        break

with open('tmp.txt','w') as f:
    for l in out:
        f.write(l)
        f.write('\n')
###
"""

"""
### high impedance faults
minDipKeys = ['R102/F3003/B151/AG/A/4.0e-03','R104/F151/B3003/AG/A/1.5e-02','R102/F3005/B102/ABCG/A/2.7e-02']

for key in minDipKeys:
    plt.plot(tme,SimpleEventDict[key])

plt.title('Not so much dips during faults')
plt.xlabel('Time (s)')
plt.ylabel('Normalized voltage')
plt.legend()
plt.ylim(0,1.2)
plt.grid()
plt.show()
###
"""







# plot some events
"""
# three phase fault
eventID = 'R100/F151/ABCG/1.0e-6'
busIDA = 'B151/A'
busIDB = 'B151/B'
busIDC = 'B151/C'

plt.plot(tme,EventDict[eventID].SignalDict[busIDA],label = 'Phase A')
plt.plot(tme,EventDict[eventID].SignalDict[busIDB],label = 'Phase B')
plt.plot(tme,EventDict[eventID].SignalDict[busIDC],label = 'Phase C')
plt.title('Three phase fault (Class 0)')
plt.xlabel('Time (s)')
plt.ylabel('Normalized voltage')
plt.legend()
plt.grid()
plt.show()
plt.close()

# SLG A fault
eventID = 'R100/F151/AG/1.0e-6'
busIDA = 'B151/A'
busIDB = 'B151/B'
busIDC = 'B151/C'

plt.plot(tme,EventDict[eventID].SignalDict[busIDA],label = 'Phase A')
plt.plot(tme,EventDict[eventID].SignalDict[busIDB],label = 'Phase B')
plt.plot(tme,EventDict[eventID].SignalDict[busIDC],label = 'Phase C')
plt.title('Phase A fault (Class 1)')
plt.xlabel('Time (s)')
plt.ylabel('Normalized voltage')
plt.legend()
plt.grid()
plt.show()
plt.close()
###############



### ilustration of change of voltage drop with change in location
location = ['151', '201', '3004']
eventID = 'R100/F151/ABCG/1.0e-6'

for bus in location:
    busIDA = 'B{}/A'.format(bus)
    plt.plot(tme,EventDict[eventID].SignalDict[busIDA],label = bus)
plt.title('Variation of voltage dip with location (for a single phase)')
plt.xlabel('Time (s)')
plt.ylabel('Normalized voltage')
plt.legend()
plt.grid()
plt.show()
plt.close()
######


### ilustration of change of voltage drop with change in fault impedance
zlist = ['1.0e-6','7.5e-03','1.5e-02']

busIDA = 'B151/A'
for z in zlist:
    eventID = 'R100/F151/ABCG/{}'.format(z)
    plt.plot(tme,EventDict[eventID].SignalDict[busIDA],label = z)
plt.title('Variation of voltage dip with fault impedance')
plt.xlabel('Time (s)')
plt.ylabel('Normalized voltage')
plt.legend()
plt.grid()
plt.show()
plt.close()

"""













"""
######### generator outages
# get the files
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


EventDict = {}
SignalDict = {}
for idx, row in enumerate(vReader):
    eventKey = eventList[idx]
    EventDict[eventKey] = row




#### find out the max dip in the gen out dataset
timeWindow = 10
start = 0.1
startind = min([idx for idx,val in enumerate(tme) if val >= start])
endind = startind + timeWindow

minV = []

for key in EventDict:
    val = np.array(EventDict[key][startind:endind])
    minV.append(val.min())

minVvec = np.array(minV)
maxVDropIndex = minVvec.argmin()
event = EventDict.keys()[maxVDropIndex]
print(event)
###
"""


"""
#### illustrate different gen outages
#key = 'R100/G211/B151'
#key = 'R100/G3011/B3011'
#key = 'R100/G101/B101'

#keyList = ['R100/G211/B151', 'R100/G3011/B3011', 'R100/G101/B101']
keyList = ['R106/G211/B151', 'R106/G3011/B3011', 'R106/G101/B101','R106/G3011/B154']

for key in keyList:
    plt.plot(tme,EventDict[key])
plt.title('Some gen outage samples')
plt.xlabel('Time (s)')
plt.ylabel('Normalized voltage')
plt.ylim(0,1.2)
plt.xlim(0,0.5)
plt.legend()
plt.grid()
plt.show()
plt.close()
##########
"""











"""
#### visualizing all the voltage data together
# examine the voltage data
vDict = {}

for idx, row in enumerate(vReader):
    eventKey = eventList[idx]
    eventKeyWords = eventKey.split('/')

    genbus = eventKeyWords[0][1:].strip()
    currentbus = eventKeyWords[1][1:].strip()
    eventID = 'G{}'.format(genbus)
    if eventID not in vDict:
            vDict[eventID] = []
    
    for f in row:
        vDict[eventID].append(f)





key = 'G101'
arr = np.array(vDict[key]).reshape(23,-1)


plt.plot(arr[0])
plt.grid()
plt.show()
#####



# close all files
vFilePath.close()
aFilePath.close()
fFilePath.close()
timeDataFilePath.close()
"""