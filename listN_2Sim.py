# output a list of all the events to be simulated to a text file, from the double branch outage cases
import os
from getBusDataFn import getBusData
from N_2Inputs import raw,  eventListFile, OKN_2
#raw = 'savnw_conp.raw'

BusDataDict = getBusData(raw)
# # get the list of raw files to be considered for the simulation
# fileList = os.listdir('.')
# RawFileList = []
# for file in fileList:
#     if file.endswith('.raw') and 'savnw_conp' in file:
#         #print file
#         RawFileList.append(file)

outputLines = []

# get the branch outages which do not cause topology inconsistency
N_2BranchOutageList = []
with open(OKN_2, 'r') as f:
    fileLines = f.read().split('\n')
    for line in fileLines:
        if line == '':
            continue
        N_2BranchOutageList.append(line.strip())


for N_2 in N_2BranchOutageList:
    N_2Words = N_2.split(';')
    line1 = N_2Words[0]
    line2 = N_2Words[1]


    line1Elements = line1.split(',')
    line2Elements = line2.split(',')

    # Line 1 params
    L1Bus1 = int(line1Elements[0])
    L1Bus2 = int(line1Elements[1])

    # Line 2 params
    L2Bus1 = int(line2Elements[0])
    L2Bus2 = int(line2Elements[1])



    FaultBusList = [L2Bus1, L2Bus2] # apply faults at both buses

    for FaultBus in FaultBusList:
        eventStr = line1 + ';' + line2 + '/F' + str(FaultBus)
        outputLines.append(eventStr)
        #print eventStr

with open(eventListFile,'w') as f:
    for line in outputLines:
        f.write(line)
        f.write('\n')


# topology_inconsistency_file = 'topology_inconsistency_cases_savnw.txt'

# # get the N-2 events which cause topology inconsistencies
# topology_inconsistent_set = set()
# with open(topology_inconsistency_file, 'r') as f:
#     fileLines = f.read().split('\n')
#     for line in fileLines:
#         if line == '':
#             continue
#         topology_inconsistent_set.add(line.strip())

# # read the raw file and get the HV line set
# HVLineSet = set()
# with open(raw, 'r') as f:
#     fileLines = f.read().split('\n')
# branchStartIndex = fileLines.index(
#     '0 / END OF GENERATOR DATA, BEGIN BRANCH DATA') + 1
# branchEndIndex = fileLines.index(
#     '0 / END OF BRANCH DATA, BEGIN TRANSFORMER DATA')
# # extract all the HV lines
# for i in range(branchStartIndex, branchEndIndex):
#     line = fileLines[i]
#     words = line.split(',')
#     Bus1 = words[0].strip()
#     Bus2 = words[1].strip()
#     cktID = words[2].strip("'").strip()
#     status = words[13].strip()
#     if Bus1 in HVBusSet and Bus2 in HVBusSet and status != '0':
#         key = Bus1 + ',' + Bus2 + ',' + cktID
#         HVLineSet.add(key)






# # generate the events
# outputLines = []
# for rawFile in RawFileList:

#     # get the percentage loading from the raw file name
#     if rawFile == 'savnw_conp.raw':
#         PL = '100'
#     else:
#         rawFileName = rawFile.replace('.raw','')
#         PL = rawFileName[-3:]


#     # run nested loops to see if there are any abnormal low voltages

#     croppedHVLineSet = list(HVLineSet)

#     for line1 in croppedHVLineSet:
#         for line2 in croppedHVLineSet:
#             # stability_indicator = 1
#             # Bus_issues = [] # list of buses where issues (low voltage or high dv_dt) are reported
#             # the lines cannot be the same
#             if line1 == line2:
#                 continue
#             # part to ensure there is no duplication of events
#             currentSet = line1+';'+line2
#             currentSetReverse = line2 + ';' + line1
#             # if case causes topology inconsistencies, continue
#             if currentSet in topology_inconsistent_set or currentSetReverse in topology_inconsistent_set:
#                 continue


#             line1Elements = line1.split(',')
#             line2Elements = line2.split(',')

#             # Line 1 params
#             L1Bus1 = int(line1Elements[0])
#             L1Bus2 = int(line1Elements[1])

#             # Line 2 params
#             L2Bus1 = int(line2Elements[0])
#             L2Bus2 = int(line2Elements[1])



#             FaultBusList = [L2Bus1, L2Bus2] # apply faults at both buses

#             for FaultBus in FaultBusList:
#                 eventStr = PL + '/' +  line1 + ';' + line2 + '/F' + str(FaultBus)
#                 outputLines.append(eventStr)
#                 #print eventStr

# with open(eventListFile,'w') as f:
#     for line in outputLines:
#         f.write(line)
#         f.write('\n')