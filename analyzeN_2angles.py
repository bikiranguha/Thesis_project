# can do the following analysis
#   get a set of events where generator angle separation occurs
#   plot all the gen angles of any given event
#   analyze the steady state angular velocities for the oscillating and non-oscillating cases (voltage)
#   outputs a text file which classifies the angular data according to their rate of change





# load the object which contains all the angle info for the N-2 fault studies
import pickle
import matplotlib.pyplot as plt
from getROCFn import getROC # function to get the rate of change of the angle
import time
import numpy as np
from getBusDataFn import getBusData
import random


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

tme = AngleDataDict['time']
timestep = abs(tme[1] - tme[0])
ind_fault_clearance = int(0.31/timestep)  + 1 # fault cleared


EventDict = {}
# Organize the angle data by event
for key in AngleDataDict:
    if key == 'time':
        continue
    angle = AngleDataDict[key]
    event = key.split('/')[0].strip()
    Bus = key.split('/')[1].strip()
    if event not in EventDict:
        EventDict[event] = AngleEventOrg()
    EventDict[event].AngDict[Bus] = angle
##############

# get the set of generators in the raw file
raw = 'savnw.raw'
rawBusDataDict = getBusData(raw)
genSet = set()
for bus in rawBusDataDict:
    BusType = rawBusDataDict[bus].type 
    if BusType == '2' or BusType == '3':
        genSet.add(bus)
    if BusType == '3':
        swingBus = bus

#print genSet
"""
####### NOTE: THIS METHOD CANNOT DETECT SEPARATION WHEN ALL ANGLES 
# see which events cause generator angle separation
angSepEvents = []
for event in EventDict:
    AngDict = EventDict[event].AngDict
    dAngSigns = []
    for bus in AngDict:
        if bus in genSet:
            angle = AngDict[bus]
            dAngDt = getROC(angle,tme,absolute=False)
            dAngDtSteadyMean = np.mean(dAngDt[-100:])
            dAngSigns.append(dAngDtSteadyMean)
    all_pos = all(val > 0 for val in dAngSigns)
    if all_pos == True: # all gen angle deviations are positive
        continue
    all_neg = all(val < 0 for val in dAngSigns) 
    if all_neg == True: # all gen angle deviations are negative
        continue

    all_zero = all(val == 0.0 for val in dAngSigns) 
    if all_zero == True: # # all gen angle deviations are zero
        continue
    angSepEvents.append(event)
######################## 
"""

"""
##### generator generator angle plots for events
# plot all the generator angles for a given event
event1 = '151,152,1;151,201,1;F151' # event where angle separates
event2 = '151,152,1;154,3008,1;F154' # event where no angle separation occurs, but all gen angle constantly go upwards
event3 = '153,3006,1;3007,3008,1;F3008' # event where angles are all stable


# angle separation
AngDict = EventDict[event1].AngDict
for bus in AngDict:
    if bus in genSet:
        angle = AngDict[bus]
        plt.plot(tme, angle, label = bus)
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Angle (degrees)')
plt.title('Angle separation')
plt.grid()
plt.savefig('AngleSep.png')
plt.close()
#plt.show()


# angle instability but no separation
AngDict = EventDict[event2].AngDict
for bus in AngDict:
    if bus in genSet:
        angle = AngDict[bus]
        plt.plot(tme, angle, label = bus)
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Angle (degrees)')
plt.title('Angular instability but no separation')
plt.grid()
plt.savefig('AngleUnStable.png')
plt.close()

# angle stability
AngDict = EventDict[event3].AngDict
for bus in AngDict:
    if bus in genSet:
        angle = AngDict[bus]
        plt.plot(tme, angle, label = bus)
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Angle (degrees)')
plt.title('Angular stability')
plt.grid()
plt.savefig('AngleStable.png')
plt.close()
"""


# plot all the generator angles for any event
#event = '151,152,2;151,201,1;F201'
#event = '151,152,1;151,201,1;F201'
event = '151,152,1;151,201,1;F151'
AngDict = EventDict[event].AngDict
for bus in AngDict:
    if bus in genSet:
        angle = AngDict[bus]
        plt.plot(tme, angle, label = bus)
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Angle (degrees)')
plt.title(event + ' Gen Angles')
plt.grid()
plt.savefig('test3.png')
plt.close()

#########################






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


"""
########### analyze the steady state angular velocities for the oscillating and non-oscillating cases
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
####################
"""




"""
# histograms of the steady state average dv_dt for class 0 and 1
plt.hist(averageClass0, bins='auto',label='Class 0')  
plt.hist(averageClass1, bins='auto',label='Class 1') 
plt.show()
"""

"""
# histograms of the initial (after fault clearance) average dv_dt for class 0 and 1
plt.hist(initialClass0, bins='auto',label='Class 0')  
plt.hist(initialClass1, bins='auto',label='Class 1') 
plt.legend()
plt.show()
"""


"""
############ divide the simulations into angle stability (class 0: stable and class 1: unstable) by looking 
########### at the rate of change of the angle (relative to the swing bus angle) during steady state
angleDevMeans = []
angleDevThreshold = 1.0
angleDevClass0 = [] # cases where the bus angle does not deviate wrt swing bus angle
angleDevClass1 = [] # cases where the bus angle deviates wrt swing bus angle
tapBus = ['204', '3007']
eventSet = set()
for event in EventDict:
    AngDict = EventDict[event].AngDict
    refBusAngle = AngDict[swingBus]
    for bus in AngDict:
        if bus != swingBus and bus not in tapBus:
            angle = AngDict[bus]
            relAngle = refBusAngle - angle # numpy array of the angle difference to the swing bus
            ddtrelAngle = getROC(relAngle,tme)
            ddtrelAngSS = ddtrelAngle[-100:] # steady state angle deviation wrt the swing bus
            meanddtrelAngSS = np.mean(ddtrelAngSS) 
            signalKey =  event + '/' + bus
            if meanddtrelAngSS > angleDevThreshold: # the bus angle deviates wrt the swing bus angle
                angleDevClass1.append(signalKey)
                if event not in eventSet:
                    print event
                    eventSet.add(event)
            else: # the bus angle does not deviate wrt the swing bus angle
                angleDevClass0.append(signalKey)






###############################
"""

"""
# plots of the distributions of the angle deviation means
plt.hist(angleDevMeans, bins='auto') 
#plt.legend()
plt.show()
"""

"""
# plots of some of the class examples
class0Samples = random.sample(angleDevClass0,10)
for case in class0Samples:
    plt.plot(tme,AngleDataDict[case])

plt.show()
plt.close()

class1Samples = random.sample(angleDevClass1,10)
for case in class1Samples:
    plt.plot(tme,AngleDataDict[case])

plt.show()
##############
"""

"""
# output the cases and the classes
with open( 'AngleCases.txt','w') as f:
    f.write('Class 0:')
    f.write('\n')

    for case in angleDevClass0:
        f.write(case)
        f.write('\n')

    f.write('Class 1:')
    f.write('\n')


    for case in angleDevClass1:
        f.write(case)
        f.write('\n')
#########################
"""







