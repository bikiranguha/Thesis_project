# runs N-2 contingencies on all high voltage lines in the raw file, given that there are no topology inconsistencies
# Then the results are sorted with the least voltage first (i mean the average of the depth 2 neighbours of the branch ends)

from getMultDepthNeighboursFn import getMultDepthNeighbours # given a bus, get its neighbours within a certain depth
from generateNeighboursFn import getNeighbours
from getBusDataFn import getBusData
import numpy as np
raw = 'savnw.raw'
NeighbourDict = getNeighbours(raw)
rawBusDataDict = getBusData(raw)
AverageVoltageDict = {} # key: line outaged, value: the mean of the voltage in the depth 2 neighbours (includes the buses at either end of line)
HVBusSet = set()
HVLineSet = set()
BusVInfoDict = {}
#BusNDict = getMultDepthNeighbours('151',NeighbourDict,5)



class BusVBehaviour(object):
    def __init__(self):
        self.VpuDict = {} # key: event id, value: vpu

# generate the HV bus set
for Bus in rawBusDataDict:
    BusVolt = float(rawBusDataDict[Bus].NominalVolt)
    BusType = rawBusDataDict[Bus].type
    if BusVolt >= 34.5: # no need to consider type 4 buses since they are already filtered out in the get Bus Data function
        HVBusSet.add(Bus)

# initialize the bus voltage behaviour class for all the buses in the raw file
for Bus in rawBusDataDict:
    BusVInfoDict[Bus] = BusVBehaviour()


with open(raw,'r') as f:
    fileLines = f.read().split('\n')
branchStartIndex = fileLines.index('0 / END OF GENERATOR DATA, BEGIN BRANCH DATA') + 1
branchEndIndex = fileLines.index('0 / END OF BRANCH DATA, BEGIN TRANSFORMER DATA')

# extract all the HV lines
for i in range(branchStartIndex,branchEndIndex):
    line = fileLines[i]
    words = line.split(',')
    Bus1 = words[0].strip()
    Bus2 = words[1].strip()
    cktID = words[2].strip("'").strip()
    status = words[13].strip()
    if Bus1 in HVBusSet and Bus2 in HVBusSet and status != '0':
        key = Bus1 + ',' + Bus2 + ',' + cktID
        HVLineSet.add(key)



# the section where we need PSSE
import sys,os
# The following 2 lines need to be added before redirect and psspy
sys.path.append(r"C:\Program Files (x86)\PTI\PSSE33\PSSBIN")
os.environ['PATH'] = (r"C:\Program Files (x86)\PTI\PSSE33\PSSBIN;"
                       + os.environ['PATH'])
###########################################

import redirect
import psspy


psse_log = 'tmp.log'
simulation_count = 0
nonConvCount = 0
disconnectDict = 0
settings = {
# use the same raw data in PSS/E and TS3ph #####################################
    'filename':raw, #use the same raw data in PSS/E and TS3ph
################################################################################
    'dyr_file':'',
    'out_file':'output2.out',
    'pf_options':[
        0,  #disable taps
        0,  #disable area exchange
        0,  #disable phase-shift
        0,  #disable dc-tap
        0,  #disable switched shunts
        0,  #do not flat start
        0,  #apply var limits immediately
        0,  #disable non-div solution
    ]
}

redirect.psse2py()
psspy.psseinit(buses=80000)
psspy.report_output(2,psse_log,[0,0])
psspy.progress_output(2,psse_log,[0,0])
psspy.alert_output(2,psse_log,[0,0])
psspy.prompt_output(2,psse_log,[0,0])
_i=psspy.getdefaultint()
_f=psspy.getdefaultreal()
_s=psspy.getdefaultchar()
print "\n Reading raw file:",settings['filename']
#ierr = psspy.read(0, settings['filename'])

# get the bus list
#ierr = psspy.read(0, settings['filename'])
#ierr, buslist = psspy.abusint(-1, 1, 'NUMBER') # getting bus numbers from raw file
#buslist=buslist[0]

# loop to simulate all line outages
SimulationDoneSet = set()
for line1 in list(HVLineSet):
    for line2 in list(HVLineSet):
        # part to ensure there is no duplication of events
        currentSet = line1+';'+line2
        if currentSet in SimulationDoneSet:
            continue
        else:
            currentSetReverse = line2+';'+line1
            SimulationDoneSet.add(currentSet)
            SimulationDoneSet.add(currentSetReverse)

        # event key
        key = line1 + ';' + line2

        line1Elements = line1.split(',')
        line2Elements = line2.split(',')

        # Line 1 params
        L1Bus1 = int(line1Elements[0])
        L1Bus2 = int(line1Elements[1])
        L1cktID = line1Elements[2]

        # Line 2 params
        L2Bus1 = int(line2Elements[0])
        L2Bus2 = int(line2Elements[1])
        L2cktID = line2Elements[2]


        # read the original raw file and try to solve power flow
        ierr = psspy.read(0, settings['filename'])
        ierr = psspy.branch_chng(L1Bus1,L1Bus2,L1cktID,[0,_i,_i,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f]) # disconnect branch 1
        ierr = psspy.branch_chng(L2Bus1,L2Bus2,L2cktID,[0,_i,_i,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f]) # disconnect branch 2
        ierr = psspy.fnsl(settings['pf_options']) #solve power flow

        # check for topology inconsistencies
        converge =  psspy.solved()
        if converge != 0: # topology inconsistencies, skip
            if converge == 1:
                string = 'Iteration limit exceeded: ' + key
                print string
            elif converge == 2:
                string = 'Blown up: ' + key
                print string
            elif converge == 5:
                string = 'Singular Jacobian matrix or voltage of 0.0 detected: ' + key
                print string
            else:
                print 'Converge flag = ' + str(converge) + ',' + 'Event: ' + key
            continue


        ierr, buslist = psspy.abusint(-1, 1, 'NUMBER') # getting bus numbers from raw file
        buslist=buslist[0]
        ierr,vpulist = psspy.abusreal(-1,1,'PU')
        vpulist = vpulist[0]

        # gather the voltage data in a dictionary after the event is applied
        VoltageDict = {} # key: Bus, value: per unit voltage after the event has been applied and power flow solved
        for i in range(len(buslist)):
            Bus = buslist[i]
            vpu = float(vpulist[i])
            VoltageDict[str(Bus)] = vpu
            # save the voltage alongwith event info for each bus
            BusVInfoDict[str(Bus)].VpuDict[key] = vpu # can be used for analysis later on

        # get all the important buses to analyze (within depth 2 of the either end of the branch)
        L1Bus1Depth2Dict = getMultDepthNeighbours(str(L1Bus1),NeighbourDict,2)
        L1Bus2Depth2Dict = getMultDepthNeighbours(str(L1Bus2),NeighbourDict,2)
        L1B1Depth2N = L1Bus1Depth2Dict[str(L1Bus1)] # the buses (at the end of the line outaged) are included in these sets
        L1B2Depth2N = L1Bus2Depth2Dict[str(L1Bus2)]

        #for line 2
        L2Bus1Depth2Dict = getMultDepthNeighbours(str(L2Bus1),NeighbourDict,2)
        L2Bus2Depth2Dict = getMultDepthNeighbours(str(L2Bus2),NeighbourDict,2)
        L2B1Depth2N = L2Bus1Depth2Dict[str(L2Bus1)] # the buses (at the end of the line outaged) are included in these sets
        L2B2Depth2N = L2Bus2Depth2Dict[str(L2Bus2)]

        ImpVoltageList = []
        for Bus in VoltageDict:
            if Bus in L1B1Depth2N or Bus in L1B2Depth2N or Bus in L2B1Depth2N or Bus in L2B2Depth2N:
                ImpVoltageList.append(VoltageDict[Bus])

        VMean = np.mean(ImpVoltageList)
        
        AverageVoltageDict[key] = VMean



outputLines = []
for key, value in sorted(AverageVoltageDict.iteritems(), key=lambda (k,v): v, reverse = False): # ascending order
    currentLine =  key + ':' + str(value)
    outputLines.append(currentLine)

with open('N_2VMeanSorted.txt','w') as f:
    for line in outputLines:
        f.write(line)
        f.write('\n')

import delete_all_pyc