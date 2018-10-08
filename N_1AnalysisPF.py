# runs N-1 contingencies on all high voltage lines in the raw file
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
#BusNDict = getMultDepthNeighbours('151',NeighbourDict,5)



# generate the HV bus set
for Bus in rawBusDataDict:
    BusVolt = float(rawBusDataDict[Bus].NominalVolt)
    BusType = rawBusDataDict[Bus].type
    if BusVolt >= 34.5: # no need to consider type 4 buses since they are already filtered out in the get Bus Data function
        HVBusSet.add(Bus)

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


# read raw file and get bus list
#ierr = psspy.read(0, settings['filename'])
ierr, buslist = psspy.abusint(-1, 1, 'NUMBER') # getting bus numbers from raw file
buslist=buslist[0]

# loop to simulate all line outages
for line in list(HVLineSet):
    
    lineElements = line.split(',')
    Bus1 = int(lineElements[0])
    Bus2 = int(lineElements[1])
    cktID = lineElements[2]


    # read the original raw file and try to solve power flow
    ierr = psspy.read(0, settings['filename'])
    ierr = psspy.branch_chng(Bus1,Bus2,cktID,[0,_i,_i,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f]) # disconnect branch 1
    #ierr = psspy.branch_chng(Bus2,toBus2,cktID2,[0,_i,_i,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f]) # disconnect branch 2
    ierr = psspy.fnsl(settings['pf_options']) # solve power flow

    # check for topology inconsistencies
    converge =  psspy.solved()
    if converge != 0: # topology inconsistencies, skip
        continue

    # get the buses and voltages from psse itself
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

    # get all the important buses to analyze (within depth 2 of the either end of the branch)
    Bus1Depth2Dict = getMultDepthNeighbours(str(Bus1),NeighbourDict,2)
    Bus2Depth2Dict = getMultDepthNeighbours(str(Bus2),NeighbourDict,2)
    B1Depth2N = Bus1Depth2Dict[str(Bus1)] # the buses (at the end of the line outaged) are included in these sets
    B2Depth2N = Bus2Depth2Dict[str(Bus2)]


    ImpVoltageList = []
    for Bus in VoltageDict:
        if Bus in B1Depth2N or Bus in B2Depth2N:
            ImpVoltageList.append(VoltageDict[Bus])

    VMean = np.mean(ImpVoltageList)
    AverageVoltageDict[line] = VMean


outputLines = []
for key, value in sorted(AverageVoltageDict.iteritems(), key=lambda (k,v): v, reverse = False): # ascending order
    currentLine =  key + ':' + str(value)
    outputLines.append(currentLine)

with open('N_1VMeanSorted.txt','w') as f:
    for line in outputLines:
        f.write(line)
        f.write('\n')

import delete_all_pyc