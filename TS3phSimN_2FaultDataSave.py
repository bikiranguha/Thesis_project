# run all the N-2 contingencies (with faults in between) (except for ones which cause topology issues) and saves voltage data after fault clearance with IDs for the event
# and the bus number. The cropped time data is also saved

# external stuff
from getBusDataFn import getBusData
from runSimFn import runSim  # function to run simulation and plot the results
import matplotlib.pyplot as plt
import os
import numpy as np
import pickle

# create plot directory
currentdir = os.getcwd()
obj_dir = currentdir +  '/obj'
if not os.path.isdir(obj_dir):
    os.mkdir(obj_dir)
########



# Functions
def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
####
# files
raw = 'test_cases/savnw/savnw_sol.raw'
outputFile = 'VoltageData.txt' # contains organized voltage data after the fault is cleared
# lists all the N-2 contingencies which cause topology inconsistencies
topology_inconsistency_file = 'topology_inconsistency_cases_savnw.txt'

# variables
HVBusSet = set()
HVLineSet = set()
rawBusDataDict = getBusData(raw)
topology_inconsistent_set = set()
#LVReportDict = {}
VoltageDataDict = {}
AngleDataDict = {}


# constants
LVThreshold = 0.90
dv_dt_threshold = 0.1


# generate the HV bus set
for Bus in rawBusDataDict:
    BusVolt = float(rawBusDataDict[Bus].NominalVolt)
    BusType = rawBusDataDict[Bus].type
    if BusVolt >= 34.5:  # no need to consider type 4 buses since they are already filtered out in the get Bus Data function
        HVBusSet.add(Bus)


# get the N-2 events which cause topology inconsistencies
with open(topology_inconsistency_file, 'r') as f:
    fileLines = f.read().split('\n')
    for line in fileLines:
        if line == '':
            continue
        topology_inconsistent_set.add(line.strip())


# read the raw file and get the HV line set
with open(raw, 'r') as f:
    fileLines = f.read().split('\n')
branchStartIndex = fileLines.index(
    '0 / END OF GENERATOR DATA, BEGIN BRANCH DATA') + 1
branchEndIndex = fileLines.index(
    '0 / END OF BRANCH DATA, BEGIN TRANSFORMER DATA')
# extract all the HV lines
for i in range(branchStartIndex, branchEndIndex):
    line = fileLines[i]
    words = line.split(',')
    Bus1 = words[0].strip()
    Bus2 = words[1].strip()
    cktID = words[2].strip("'").strip()
    status = words[13].strip()
    if Bus1 in HVBusSet and Bus2 in HVBusSet and status != '0':
        key = Bus1 + ',' + Bus2 + ',' + cktID
        HVLineSet.add(key)

# total N-2 contingencies to be carried out
# totalSims = len(HVLineSet)**2 - len(topology_inconsistent_set)
# totalSims = 274 # found from status update

totalSims = 0
# nested loops to count how many events will be run
for line1 in list(HVLineSet):
    for line2 in list(HVLineSet):
        # the lines cannot be the same
        if line1 == line2:
            continue
        # part to ensure there is no duplication of events
        currentSet = line1+';'+line2
        currentSetReverse = line2 + ';' + line1
        # if case causes topology inconsistencies, continue
        if currentSet in topology_inconsistent_set or currentSetReverse in topology_inconsistent_set:
            continue
        totalSims += 2


"""
# temporary part for debugging
HVLineSet = set()
HVLineSet.add('151,152,2')
HVLineSet.add('153,154,1')
"""

# run nested loops to see if there are any abnormal low voltages
simCount = 0  # to keep track of how many simulations are already done
for line1 in list(HVLineSet):
    for line2 in list(HVLineSet):
        # stability_indicator = 1
        # Bus_issues = [] # list of buses where issues (low voltage or high dv_dt) are reported
        # the lines cannot be the same
        if line1 == line2:
            continue
        # part to ensure there is no duplication of events
        currentSet = line1+';'+line2
        currentSetReverse = line2 + ';' + line1
        # if case causes topology inconsistencies, continue
        if currentSet in topology_inconsistent_set or currentSetReverse in topology_inconsistent_set:
            continue


        line1Elements = line1.split(',')
        line2Elements = line2.split(',')

        # Line 1 params
        L1Bus1 = line1Elements[0]
        L1Bus2 = line1Elements[1]
        L1cktID = line1Elements[2]

        # Line 2 params
        L2Bus1 = line2Elements[0]
        L2Bus2 = line2Elements[1]
        L2cktID = line2Elements[2]

        # generate the event
        # one line out then a fault
        # list of buses where faults will be applied
        FaultBusList = [L2Bus1, L2Bus2]
        for FaultBus in FaultBusList:  # simulate faults on each side of the 2nd line
            event1Flag = '-event01'
            event1Param = '0.1,OUT,LINE,' + L1Bus1 + ',' + L1Bus2 + ',,' + L1cktID + ',7,,,,,'

            event2Flag = '-event02'
            event2Param = '0.2,FAULTON,ABCG,' + FaultBus + ',,,,1.0e-6,1.0e-6,1.0e-6,0.0,0.0,0.0'

            event3Flag = '-event03'
            event3Param = '0.3,FAULTOFF,ABCG,' + FaultBus + ',,,,,,,,,'

            event4Flag = '-event04'
            event4Param = '0.31,OUT,LINE,' + L2Bus1 + ',' + L2Bus2 + ',,' + L2cktID + ',7,,,,,'

            exitFlag = '-event05'
            exitParam = '10,EXIT,,,,,,,,,,,'
            EventList = [event1Flag, event1Param, event2Flag, event2Param,event3Flag, event3Param, event4Flag, event4Param, exitFlag, exitParam]
            Results = runSim(raw, EventList, 'TS3phEvent.log')
            currentEvent = currentSet + ';' + 'F' + FaultBus
            # print 'Current event: ' + currentEvent

            time = list(Results['time'])
            # get the time index when its greater than 1 sec (after the 2nd line out)
            ind_fault_clearance = [ind for ind, t in enumerate(time) if t >= 0.31][0]
            # print time[ind_fault_clearance]

            # extract all the voltage data from each event
            for key in Results:
                if key == 'time':
                    continue

                vMag = Results[key].mag
                angle = Results[key].ang
                #croppedvMag = vMag[ind_fault_clearance:] # only get the voltage data after fault clearance
                dictKey = currentEvent + '/' + str(key) # event ID + bus number
                #vMagStrList = [] # each element is a string of the voltage data

                #for v in croppedvMag:
                #	vMagStrList.append(str(v))

                #vMagDataStr = ','.join(vMagStrList) # the whole vMag data is now a string with each entry separated by commas
                #VoltageDataDict[dictKey] = croppedvMag
                VoltageDataDict[dictKey] = vMag # get the whole voltage data
                AngleDataDict[dictKey] = angle






            # status update       
            simCount+=1
            print 'Simulations done:' + str(simCount) + ' out of ' + str(totalSims)


# get the time string
#timeStrList = []
#for ele in time: # time is a list of floating point time data
#	timeStrList.append(str(ele))
#timeStr = ','.join(timeStrList)

VoltageDataDict['time'] = time
AngleDataDict['time'] = time

# save the voltage dictionary
save_obj(VoltageDataDict,'VoltageData')
save_obj(AngleDataDict,'AngleData')
"""
with open(outputFile,'w') as f:
	for key in VoltageDataDict:
		outputString = key + '/' + VoltageDataDict[key]
	f.write(outputString)
	f.write('\n')
"""


