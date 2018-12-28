# run balanced and unbalanced faults for all the buses in the raw file and save them to different text and csv files
# runs the simulations on multiple raw files with scaled loads
# besides very low fault impedance, it also simulates fault impedance =  line/2 and  fault impedance = line impedance

# external stuff
from getBusDataFn import getBusData
from runSimFn3phLI import runSim  # function to run simulation and get the 3 ph voltage and angle data
from generateNeighbourImpedanceData import getBranchTFData # returns dictionary of all branch data with impedances
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
"""
#raw = 'test_cases/savnw/savnw_sol.raw'
eventIDFile = obj_dir + '/' + 'eventIDFileLI.txt'
voltFileName = obj_dir + '/' + 'vData3phLI.csv'
anglFileName = obj_dir + '/' + 'aData3phLI.csv'
timeDataFileName = obj_dir + '/' + 'tData3phLI.csv'
"""
eventIDFile = obj_dir + '/' + 'eventIDFileTest.txt'
voltFileName = obj_dir + '/' + 'vData3phTest.csv'
anglFileName = obj_dir + '/' + 'aData3phTest.csv'
timeDataFileName = obj_dir + '/' + 'tData3phTest.csv'




TS3phLogFile = 'fault3ph.log'
#outputFile = 'VoltageData.txt' # contains organized voltage data after the fault is cleared
# lists all the N-2 contingencies which cause topology inconsistencies
#topology_inconsistency_file = 'topology_inconsistency_cases_savnw.txt'

# variables
refRaw = 'test_cases/savnw/savnw_sol.raw'
rawBusDataDict = getBusData(refRaw)
BranchNeighbourDict = getBranchTFData(refRaw)

# get the list of raw files to be considered for the simulation
direc = './test_cases/savnw/'
fileList = os.listdir(direc)
RawFileList = []
for file in fileList:
    if file.endswith('.raw') and 'savnw_conz' in file:
        #print file
        RawFileList.append(file)



# put faults at all buses and write the data to files

# file objects
vFile = open(voltFileName, 'wb') # 'wb' needed to avoid blank space in between lines
aFile = open(anglFileName, 'wb')
tFile = open(timeDataFileName, 'wb')

eventFileObj = open(eventIDFile, 'w')
eventHeader = 'LoadPercent/FaultBus/Bus/FaultType/Phase/FaultZ'
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

"""
##### Loop through all the buses in the system
sim = 0
tapBusList = ['204', '3007']

for raw in RawFileList:

    rawFileName = raw.replace('.raw','')
    rawFlag = '-ts_raw_dir'
    rawParam =  'test_cases/savnw/{}'.format(raw)
    print 'Reading raw file: {}'.format(raw)
    rawPath = rawParam

    for FaultBus in rawBusDataDict:



        # no need to apply faults at tap buses
        if FaultBus in tapBusList:
            continue

        # organize all the line impedances
        BranchNeighbours = BranchNeighbourDict[FaultBus].toBus
        IsBranchList = BranchNeighbourDict[FaultBus].IsBranch
        ZList = BranchNeighbourDict[FaultBus].Z
        ZFaultSim = ['1.0e-6'] # list of fault impedances to simulate 
        for a, val in enumerate(BranchNeighbours):
            IsBranch = IsBranchList[a]
            if IsBranch != 1:
                continue
            
            Z = ZList[a]
            ZHalf =  Z/2
            Zstr = '%.1e' % Z
            ZstrHalf = '%.1e' %ZHalf
            ZFaultSim.append(Zstr)
            ZFaultSim.append(ZstrHalf)

        ######
        for faultZ in ZFaultSim:
            # 3ph fault
            #faultZ = '1.0e-6'
            event1Flag = '-event01'
            event1Param = '0.1,FAULTON,ABCG,{},,,,{},{},{},0.0,0.0,0.0'.format(FaultBus,faultZ,faultZ,faultZ)

            event2Flag = '-event02'
            event2Param = '0.2,FAULTOFF,ABCG,{},,,,,,,,,'.format(FaultBus)

            exitFlag = '-event03'
            exitParam = '5.0,EXIT,,,,,,,,,,,'
            faultType = 'ABCG'

            EventList = [rawFlag, rawParam, event1Flag, event1Param, event2Flag, event2Param, exitFlag, exitParam]
            tme  = runSim(rawPath,EventList,TS3phLogFile,FaultBus, faultType, eventFileObj, writerObjV, writerObjA,rawFileName,faultZ)
            sim +=1



            # single phase fault in phase A
            #faultZ = '1.0e-6'
            event1Flag = '-event01'
            event1Param = '0.1,FAULTON,AG,{},,,,{},{},{},0.0,0.0,0.0'.format(FaultBus,faultZ,faultZ,faultZ)

            event2Flag = '-event02'
            event2Param = '0.2,FAULTOFF,AG,{},,,,,,,,,'.format(FaultBus)

            exitFlag = '-event03'
            exitParam = '5.0,EXIT,,,,,,,,,,,'
            faultType = 'AG'

            EventList = [rawFlag, rawParam, event1Flag, event1Param, event2Flag, event2Param, exitFlag, exitParam]
            tme = runSim(rawPath,EventList,TS3phLogFile,FaultBus, faultType, eventFileObj, writerObjV, writerObjA,rawFileName, faultZ)
            sim +=1
        ##########







        #print type(tme)
        #sim +=1
        print 'Total events run so far: ', sim

###########
"""
# close files
vFile.close()
aFile.close()
eventFileObj.close()

# save the time data (needed only once)
timeObj.writerow(tme)
tFile.close()





