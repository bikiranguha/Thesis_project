# get the branch and tf flows from the bus report in descending order
from analyzeBusReportFnv2 import BusReport 
from getBusDataFn import getBusData
import math
rawfile = 'savnw_dy_sol_0905.raw'
BusFlowReport = 'BusReportsavnw_dy_sol_0905.txt'
ReportDict = BusReport(BusFlowReport, rawfile)
rawBusDataDict = getBusData(rawfile)
HVBusSet =set()
FlowDict = {}
for Bus in rawBusDataDict:
	BusVolt = float(rawBusDataDict[Bus].NominalVolt)
	BusType = rawBusDataDict[Bus].type
	if BusVolt >= 34.5: # no need to consider type 4 buses since they are already filtered out in the get Bus Data function
		HVBusSet.add(Bus)

# generate the HV bus set
for Bus in rawBusDataDict:
	BusVolt = float(rawBusDataDict[Bus].NominalVolt)
	BusType = rawBusDataDict[Bus].type
	if BusVolt >= 34.5: # no need to consider type 4 buses since they are already filtered out in the get Bus Data function
		HVBusSet.add(Bus)


for Bus in ReportDict:
	if Bus not in HVBusSet:
		continue
	neighbours = ReportDict[Bus].toBusList
	for i in range(len(neighbours)):
		nBus = neighbours[i]
		if nBus not in HVBusSet:
			continue
		cktID = ReportDict[Bus].cktID[i]
		MW = ReportDict[Bus].MWList[i]
		MVAR = ReportDict[Bus].MVARList[i]
		MVA = math.sqrt(MW**2 + MVAR**2)
		key = Bus + ',' + nBus + ',' + cktID
		FlowDict[key] = MVA
		#print key + ':' + str(MVA)
 
outputLines = []
for key, value in sorted(FlowDict.iteritems(), key=lambda (k,v): v, reverse = True): # ascending order
	string = key + ':\t\t\t' + str(value)
	outputLines.append(string)

# output
with open('FlowsDescending.txt','w') as f:
	f.write('Branch:		MVA Flow')
	f.write('\n')
	for line in outputLines:
		f.write(line)
		f.write('\n')	

