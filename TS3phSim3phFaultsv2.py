# run balanced and unbalanced faults for all the buses in the raw file and save them to different text and csv files

# external stuff
from getBusDataFn import getBusData
from runSimFn3ph import runSim  # function to run simulation and get the 3 ph voltage and angle data
import matplotlib.pyplot as plt
import os
import numpy as np
import csv
#import pickle

# create plot directory
currentdir = os.getcwd()
obj_dir = currentdir +  '/fault3ph'
if not os.path.isdir(obj_dir):
    os.mkdir(obj_dir)
########



# files
raw = 'test_cases/savnw/savnw_sol.raw'
eventIDFile = obj_dir + '/' + 'eventIDFile.txt'
voltFileName = obj_dir + '/' + 'vData3ph.csv'
anglFileName = obj_dir + '/' + 'aData3ph.csv'
timeDataFileName = obj_dir + '/' + 'tData3ph.csv'
TS3phLogFile = 'fault3ph.log'
#outputFile = 'VoltageData.txt' # contains organized voltage data after the fault is cleared
# lists all the N-2 contingencies which cause topology inconsistencies
#topology_inconsistency_file = 'topology_inconsistency_cases_savnw.txt'

# variables

rawBusDataDict = getBusData(raw)






# put faults at all buses and write the data to files

# file objects
vFile = open(voltFileName, 'wb') # 'wb' needed to avoid blank space in between lines
aFile = open(anglFileName, 'wb')
tFile = open(timeDataFileName, 'wb')

eventFileObj = open(eventIDFile, 'w')
eventHeader = 'FaultBus/Bus/FaultType/Phase'
eventFileObj.write(eventHeader)
eventFileObj.write('\n')


writerObjV = csv.writer(vFile)
writerObjA = csv.writer(aFile)
timeObj = csv.writer(tFile)




"""
#######
# test using a fault bus
FaultBus = '151'
# test case
event1Flag = '-event01'
event1Param = '0.1,FAULTON,ABCG,' + FaultBus + ',,,,1.0e-6,1.0e-6,1.0e-6,0.0,0.0,0.0'

event2Flag = '-event02'
event2Param = '0.2,FAULTOFF,ABCG,' + FaultBus + ',,,,,,,,,'

exitFlag = '-event03'
exitParam = '0.5,EXIT,,,,,,,,,,,'
faultType = 'ABCG'

EventList = [event1Flag, event1Param, event2Flag, event2Param, exitFlag, exitParam]
tme = runSim(raw,EventList,TS3phLogFile,FaultBus, faultType, eventFileObj, writerObjV, writerObjA)

# single phase fault in phase A
event1Flag = '-event01'
event1Param = '0.1,FAULTON,AG,' + FaultBus + ',,,,1.0e-6,1.0e-6,1.0e-6,0.0,0.0,0.0'

event2Flag = '-event02'
event2Param = '0.2,FAULTOFF,AG,' + FaultBus + ',,,,,,,,,'

exitFlag = '-event03'
exitParam = '0.5,EXIT,,,,,,,,,,,'
faultType = 'AG'

EventList = [event1Flag, event1Param, event2Flag, event2Param, exitFlag, exitParam]
tme = runSim(raw,EventList,TS3phLogFile,FaultBus, faultType, eventFileObj, writerObjV, writerObjA)
print type(tme)
############
"""


##### Loop through all the buses in the system
sim = 0
tapBusList = ['204', '3007']
for FaultBus in rawBusDataDict:

    # no need to apply faults at tap buses
    if FaultBus in tapBusList:
        continue
    # 3ph fault
    event1Flag = '-event01'
    event1Param = '0.1,FAULTON,ABCG,' + FaultBus + ',,,,1.0e-6,1.0e-6,1.0e-6,0.0,0.0,0.0'

    event2Flag = '-event02'
    event2Param = '0.2,FAULTOFF,ABCG,' + FaultBus + ',,,,,,,,,'

    exitFlag = '-event03'
    exitParam = '0.5,EXIT,,,,,,,,,,,'
    faultType = 'ABCG'

    EventList = [event1Flag, event1Param, event2Flag, event2Param, exitFlag, exitParam]
    tme  = runSim(raw,EventList,TS3phLogFile,FaultBus, faultType, eventFileObj, writerObjV, writerObjA)

    # single phase fault in phase A
    event1Flag = '-event01'
    event1Param = '0.1,FAULTON,AG,' + FaultBus + ',,,,1.0e-6,1.0e-6,1.0e-6,0.0,0.0,0.0'

    event2Flag = '-event02'
    event2Param = '0.2,FAULTOFF,AG,' + FaultBus + ',,,,,,,,,'

    exitFlag = '-event03'
    exitParam = '0.5,EXIT,,,,,,,,,,,'
    faultType = 'AG'

    EventList = [event1Flag, event1Param, event2Flag, event2Param, exitFlag, exitParam]
    tme = runSim(raw,EventList,TS3phLogFile,FaultBus, faultType, eventFileObj, writerObjV, writerObjA)
    #print type(tme)
    sim +=1
    print 'Total events run so far: ', sim

###########

# close files
vFile.close()
aFile.close()
eventFileObj.close()

# save the time data (needed only once)
timeObj.writerow(tme)
tFile.close()





