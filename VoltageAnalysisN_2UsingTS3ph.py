# get a list of all the HV buses in the raw file
# Get a list of all N-2 HV line outages which do not cause topology inconsistencies
# simulate all such N-2 contingencies on each of the HV lines and organize the average voltages, lowest first
import platform
currentOS = platform.system()
from runSimFn import runSim
import numpy as np

# Functions and classes
def convertFileLinux(file,currentOS):
	# function to convert file from crlf to lf (if needed)
	if currentOS == 'Linux':
		text = open(file, 'rb').read().replace('\r\n', '\n')
		open(file, 'wb').write(text)

class OutageData(object):
	def __init__(self,Line1,Line2):
		self.Line1 = Line1
		self.Line2 = Line2
#####################
InputFile = 'InputFile.txt' # file containing the raw data and the event data
branchN_2OutageListFile = 'OKDoubleBranchOutages.txt' # lists all the double branch outages which do not cause any topology inconsistencies
HVBusSet = set()
HVLineSet = set()
N_2OutageDict = {}
# convert input file crlf to lf (needed for linux)
convertFileLinux(InputFile,currentOS)

from getBusDataFn import getBusData
# get the raw file
with open(InputFile,'r') as f:
	fileLines = f.read().split('\n')
	for line in fileLines:
		if 'rawfile' in line:
			words = line.split('=')
			rawPath = words[1].strip()
			break

# convert raw file crlf to lf (needed for linux)
convertFileLinux(rawPath,currentOS)
convertFileLinux(branchN_2OutageListFile,rawPath)


rawBusDataDict = getBusData(rawPath)
# generate the HV bus set
for Bus in rawBusDataDict:
	BusVolt = float(rawBusDataDict[Bus].NominalVolt)
	BusType = rawBusDataDict[Bus].type
	if BusVolt >= 34.5: # no need to consider type 4 buses since they are already filtered out in the get Bus Data function
		HVBusSet.add(Bus)


with open(rawPath,'r') as f:
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
		keyReverse = Bus2 + ',' + Bus1 + ',' + cktID
		HVLineSet.add(key)
		HVLineSet.add(keyReverse)



eventCount = 1
with open(branchN_2OutageListFile,'r') as f:
	fileLines = f.read().split('\n')
	for line in fileLines:
		if ';' not in line:
			continue
		words = line.split(';')
		Line1 = words[0].strip()
		Line2 = words[1].strip()

		if Line1 in HVLineSet and Line2 in HVLineSet:
			N_2OutageDict[eventCount] = OutageData(Line1,Line2)
		eventCount +=1
			



AverageVMagDict = {} # key: lines outaged, value: avg voltage over all HV buses in the 10 cycle time after branch outage

# run simulations, get the average values over all the buses in 10 cycles
for eventCount in N_2OutageDict:
	Line1 = N_2OutageDict[eventCount].Line1
	Line2 = N_2OutageDict[eventCount].Line2
	
	line1Elements = Line1.split(',')
	L1Bus1 = line1Elements[0]
	L1Bus2 = line1Elements[1]
	L1cktID = line1Elements[2]

	line2Elements = Line2.split(',')
	L2Bus1 = line2Elements[0]
	L2Bus2 = line2Elements[1]
	L2cktID = line2Elements[2]


	event1Flag = '-event01'
	event1Param = '0.1,OUT,LINE,' + L1Bus1 + ',' + L1Bus2 + ',,' + L1cktID + ',7,,,,,'

	event2Flag = '-event02'
	event2Param = '0.1,OUT,LINE,' + L2Bus1 + ',' + L2Bus2 + ',,' + L2cktID + ',7,,,,,'

	exitFlag = '-event03'
	exitParam = '0.2,EXIT,,,,,,,,,,,'
	EventList = [event1Flag,event1Param,event2Flag,event2Param,exitFlag,exitParam]
	#print EventList
	Results = runSim(rawPath,EventList,'LineOut' + str(eventCount) + '.log')
	time = Results['time']
	LineOutTime = min([i for i in time if i > 0.1])
	timeList = list(time)
	EventStartIndex = timeList.index(LineOutTime)
	#print EventStartIndex


	AverageVmagList = []
	for key in Results:
		if key == 'time' or str(key) not in HVBusSet:
			continue
		VmagAvg = np.mean(Results[key].mag[EventStartIndex:EventStartIndex+11])	#extract the voltage for 10 cycles after the event
		#print VmagAvg
		AverageVmagList.append(VmagAvg) #  list of averages of each bus over the 10 cycles
		#CroppedVMagDict[key] = Vmag

	OverallMean = np.mean(AverageVmagList) # overall average (of all the bus)
	#print line + ':' + str(OverallMean)
	key = Line1 + ';' + Line2
	AverageVMagDict[key] = OverallMean


outputLines = []
# sort the average voltage values in ascending order
for key, value in sorted(AverageVMagDict.iteritems(), key=lambda (k,v): v, reverse = False): # ascending order
	string = key + ':\t\t\t' + str(value)
	outputLines.append(string)

# output
with open('AverageVoltageN_2.log','w') as f:
	f.write('Lines out:		Average voltage')
	f.write('\n')
	for line in outputLines:
		f.write(line)
		f.write('\n')


