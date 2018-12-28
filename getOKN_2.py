# on a given raw file, get all the N-2 HV line outage combos possible
# test these combos on a power flow and see which ones do not lead to any topology inconsistencies
# List these cases out, so that we can do some TS on these
import sys,os
# The following 2 lines need to be added before redirect and psspy
sys.path.append(r"C:\Program Files (x86)\PTI\PSSE33\PSSBIN")
os.environ['PATH'] = (r"C:\Program Files (x86)\PTI\PSSE33\PSSBIN;"
                       + os.environ['PATH'])
###########################################

# get the required files
from N_2Inputs import raw,OKN_2 as outputfile

import redirect
import psspy
from getBusDataFn import getBusData

#raw = 'pf_ornl0823conz.raw'
psse_log = 'pf_ornl0823conz.log'
#outputfile = 'OKN_2pfornl.txt' # double HV line outages which do not cause island
outputfile2 = 'NotOKN_2pfornl.txt' # double HV line outages which  cause island
BusDataDict = getBusData(raw)
nonIslandEvents = []
IslandEvents = []

######
# read the raw file and get the HV line set

# generate the HV bus set
HVBusSet = set()
for Bus in BusDataDict:
    BusVolt = float(BusDataDict[Bus].NominalVolt)
    BusType = BusDataDict[Bus].type
    if BusVolt >= 34.5:  # no need to consider type 4 buses since they are already filtered out in the get Bus Data function
        HVBusSet.add(Bus)


HVLineSet = set()
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


HVLineList = list(HVLineSet)
#################


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
ierr = psspy.read(0, settings['filename'])


# # get a list of all the non-transformer branches from psse
# ierr, brnchs = psspy.abrnint(_i,_i,_i,_i,_i,['FROMNUMBER','TONUMBER']) # page 1789 of API book
# ierr, carray = psspy.abrnchar(_i,_i,_i,_i,_i, ['ID']) # get the character ids (page 1798 of API book)
# fromBusList = brnchs[0]
# toBusList = brnchs[1]
# cktIDList = carray[0]

alreadyOutagedSet = set()

for i in range(len(HVLineList)):

    line1 = HVLineList[i]
    line1Elements = line1.split(',')
    fromBus1 = line1Elements[0]
    toBus1 = line1Elements[1]
    cktID1 = line1Elements[2]


    # fromBus1 = fromBusList[i]

    # toBus1 = toBusList[i]
    # cktID1 = cktIDList[i].strip("'").strip()


    
    for j in range(len(HVLineList)):

        if i == j:
            continue

        line2 = HVLineList[j]
        line2Elements = line2.split(',')
        fromBus2 = line2Elements[0]
        toBus2 = line2Elements[1]
        cktID2 = line2Elements[2]



        # fromBus2 = fromBusList[j]

        # toBus2 = toBusList[j]
        # cktID2 = cktIDList[j].strip("'").strip()     


        # branch1ID = str(fromBus1) + ',' +  str(toBus1) + ',' + cktID1
        # branch2ID = str(fromBus2) + ',' +  str(toBus2) + ',' + cktID2

        # read the original raw file and try to solve power flow
        ierr = psspy.read(0, settings['filename'])
        ierr = psspy.branch_chng(int(fromBus1),int(toBus1),cktID1,[0,_i,_i,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f]) # disconnect branch 1
        ierr = psspy.branch_chng(int(fromBus2),int(toBus2),cktID2,[0,_i,_i,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f]) # disconnect branch 2
        ierr = psspy.fnsl(settings['pf_options'])
        converge =  psspy.solved()
        if converge != 9:
            string =  line1 + ';' + line2
            nonIslandEvents.append(string)
        else: # island
            string =  line1 + ';' + line2
            IslandEvents.append(string)


with open(outputfile,'w') as f:
    for line in nonIslandEvents:
        f.write(line)
        f.write('\n')

with open(outputfile2,'w') as f:
    for line in IslandEvents:
        f.write(line)
        f.write('\n')