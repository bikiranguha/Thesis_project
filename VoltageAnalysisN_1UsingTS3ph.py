# get a list of all the HV buses in the raw file
# simulate N-1 contingencies on each of the HV lines and organize the data
# extract the average voltage of the HV bus set over the 10 cycles after the line outage, organize according to line outaged
# Sort the data, lowest voltage first
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


#####################
InputFile = 'InputFile.txt' # file containing the raw data and the event data
HVBusSet = set()
HVLineSet = set()
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
		HVLineSet.add(key)


AverageVMagDict = {} # key: line outaged, value: avg voltage over all HV buses in the 10 cycle time after branch outage

# run simulations, get the average values over all the buses in 10 cycles
for line in list(HVLineSet):
	
	lineElements = line.split(',')
	Bus1 = lineElements[0]
	Bus2 = lineElements[1]
	cktID = lineElements[2]
	event1Flag = '-event01'
	event1Param = '0.1,OUT,LINE,' + Bus1 + ',' + Bus2 + ',,' + cktID + ',7,,,,,'
	exitFlag = '-event02'
	exitParam = '0.2,EXIT,,,,,,,,,,,'
	EventList = [event1Flag,event1Param,exitFlag,exitParam]
	Results = runSim(rawPath,EventList,'TS3phLineOut.log')
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
	AverageVMagDict[line] = OverallMean


outputLines = []
# sort the average voltage values in ascending order
for key, value in sorted(AverageVMagDict.iteritems(), key=lambda (k,v): v, reverse = False): # ascending order
	string = key + ':\t\t\t' + str(value)
	outputLines.append(string)

# output
with open('AverageVoltageN_1.log','w') as f:
	f.write('Line out:		Average voltage')
	f.write('\n')
	for line in outputLines:
		f.write(line)
		f.write('\n')


