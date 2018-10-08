# load the object which contains all the angle info for the N-2 fault studies
import pickle
import matplotlib.pyplot as plt
from getROCFn import getROC # function to get the rate of change of the angle
import time
import numpy as np
from getBusDataFn import getBusData


class AngleEventOrg():
    def __init__(self):
        self.AngDict = {}


# Functions
def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

#####

# load the angle object
start = time.time()
AngleDataDict = load_obj('AngleData')
end = time.time()

print 'Time to load the data in seconds: ', end - start



"""
EventDict = {}
# Organize the angle data by event
for key in AngleDataDict:
    if key == 'time':
        continue
    angle = AngleDataDict[key]
    Event = key.split('/')[0].strip()
    Bus = key.split('/')[1].strip()
    if Event not in EventDict:
        EventDict[Event] = AngleEventOrg()
    EventDict[Event].AngDict[Bus] = angle
##############

# get the set of generators in the raw file
raw = 'savnw.raw'
rawBusDataDict = getBusData(raw)
genSet = set()
for bus in rawBusDataDict:
    BusType = rawBusDataDict[bus].type 
    if BusType == '2' or BusType == '3':
        genSet.add(bus)

print genSet
"""
tme = AngleDataDict['time']
timestep = abs(tme[1] - tme[0])
ind_fault_clearance = int(0.31/timestep)  + 1 # fault cleared


"""
# plot the angle deviations for an event which causes huge oscillations
#searchedKey = '151,201,1;151,152,1;F151'
searchedKey = '203,205,2;3007,3008,1;F3008'

plotNo = 1
for key in AngleDataDict:
    if key.startswith(searchedKey):
        angle = AngleDataDict[key]
        dAngle_dt = getROC(angle,tme)
        plt.plot(tme, dAngle_dt)
        #plt.ylim(-6000,6000)
        #plt.ylim(-100,100)
        titleStr = 'Angular velo:' + key
        #plt.xlim(9,10)
        plt.title(titleStr)
        figName = 'plot' + str(plotNo) +  '.png'
        plt.savefig(figName)
        #plt.show()
        plt.close()
        plotNo +=1
"""



# analyze the steady state angular velocities for the oscillating and non-oscillating cases
OscillationClassFile = 'Casedvdt.txt' # file which has all the cases classified

with open(OscillationClassFile,'r') as f:
    fileLines = f.read().split('\n')

Class0StartIndex = fileLines.index('Class 0:') + 1
Class0EndIndex = fileLines.index('Class 1:')
Class1StartIndex = Class0EndIndex + 1


class0keys = []
class1keys = []
# get the class 0 events
for i in range(Class0StartIndex, Class0EndIndex):
    line = fileLines[i]
    if line == '':
        continue
    class0keys.append(line)

# get the class 1 events
for i in range(Class1StartIndex, len(fileLines)):
    line = fileLines[i]
    if line == '':
        continue
    class1keys.append(line)


start = time.time()
averageClass0 = [] # list of average dv_dt in steady state (last 100 samples) for class 0
initialClass0 = [] # list of average dv_dt just after the fault clearance for class 0
for key in class0keys:
    angle = AngleDataDict[key]
    dAngle_dt = getROC(angle,tme)
    initialdAngle_dt = dAngle_dt[ind_fault_clearance:ind_fault_clearance+100]   
    steadydAngle_dt = dAngle_dt[-100:]
    averageSteadydAngle_dt = np.mean(steadydAngle_dt)
    initialdAngle_dtAvg = np.mean(initialdAngle_dt)
    initialClass0.append(initialdAngle_dtAvg)
    averageClass0.append(averageSteadydAngle_dt)

averageClass1 = [] # list of average dv_dt in steady state (last 100 samples) for class 1
initialClass1 = [] # list of average dv_dt just after the fault clearance for class 1
for key in class1keys:
    angle = AngleDataDict[key]
    dAngle_dt = getROC(angle,tme) 
    initialdAngle_dt = dAngle_dt[ind_fault_clearance:ind_fault_clearance+100]   
    steadydAngle_dt = dAngle_dt[-100:]
    averageSteadydAngle_dt = np.mean(steadydAngle_dt)
    initialdAngle_dtAvg = np.mean(initialdAngle_dt)
    initialClass1.append(initialdAngle_dtAvg)
    averageClass1.append(averageSteadydAngle_dt)
end = time.time()

print 'Time to analyze the angle data: ', end - start




# histograms of the steady state average dv_dt for class 0 and 1
plt.hist(averageClass0, bins='auto',label='Class 0')  
plt.hist(averageClass1, bins='auto',label='Class 1') 
plt.show()


"""
# histograms of the initial (after fault clearance) average dv_dt for class 0 and 1
plt.hist(initialClass0, bins='auto',label='Class 0')  
plt.hist(initialClass1, bins='auto',label='Class 1') 
plt.legend()
plt.show()
"""



