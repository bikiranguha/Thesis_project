"""
Function which generates a bus flow report of comed buses
"""


def BusReport(flowReportFile,Raw):

	from getBusDataFn import getBusData

	BusDataDict = getBusData(Raw)
	ComedPlusBoundarySet = set()

	flowDict = {}
	#FromBusLines = []
	#ToBusLines = []
	class flowReport(object):
		def __init__(self):
			self.toBusList = []
			self.MWList = []
			self.MVARList = []
			self.MVAList = []
			self.cktID = []




	"""
	with open(Raw,'r') as f:
		filecontent = f.read()
		fileLines = filecontent.split('\n')

	branchStartIndex = fileLines.index('0 / END OF GENERATOR DATA, BEGIN BRANCH DATA') + 1
	branchEndIndex = fileLines.index('0 / END OF BRANCH DATA, BEGIN TRANSFORMER DATA')


	for i in range(branchStartIndex, branchEndIndex):
		line = fileLines[i]
		words = line.split(',')
		Bus1 = words[0].strip()
		Bus2 = words[1].strip()	
		try:
			Bus1Area = BusDataDict[Bus1].area 
			Bus2Area = BusDataDict[Bus2].area 
		except: # for buses '243083' and '638082'
			continue

		if Bus1Area == '222' and Bus2Area == '222':
			ComedPlusBoundarySet.add(Bus1)
			ComedPlusBoundarySet.add(Bus2)
		if Bus1Area == '222' and Bus2Area != '222':
			ComedPlusBoundarySet.add(Bus1)
			ComedPlusBoundarySet.add(Bus2)

		if Bus1Area != '222' and Bus2Area == '222':
			ComedPlusBoundarySet.add(Bus1)
			ComedPlusBoundarySet.add(Bus2)

	for Bus in BusDataDict:
		area = BusDataDict[Bus].area
		if area == '222':
			ComedPlusBoundarySet.add(Bus)
	"""



	with open(flowReportFile,'r') as f:
		filecontent = f.read()
		fileLines = filecontent.split('\n')

	indices = [i for i, line in enumerate(fileLines) if line.startswith('BUS')]

	for i in indices:
		#print i
		line = fileLines[i]
		FromBus = line[4:10].strip()
		"""
		if FromBus not in ComedPlusBoundarySet:
			continue
		"""
		flowDict[FromBus] = flowReport()
		i+=2
		line = fileLines[i]
		while  not 'M I S M A T C H' in line:
			if 'RATING' in line:
				break
			
			if 'GENERATION' in line or 'LOAD' in line or 'SHUNT' in line:
				i+=1
				line = fileLines[i]
				continue

			toBus = line[4:10].strip()
			MW=float(line[34:42].strip())
			MVAR=float(line[42:50].strip())
			cktID = line[31:34]
			#print toBus
			flowDict[FromBus].toBusList.append(toBus)
			flowDict[FromBus].MWList.append(MW)
			flowDict[FromBus].MVARList.append(MVAR)
			flowDict[FromBus].cktID.append(cktID)
			#ToBusLines.append(toBus)

			i+=1
			if i >=len(fileLines):
				break
			line = fileLines[i]
	return flowDict


"""
with open('tmp.txt','w') as f:
	for Bus in ToBusLines:
		f.write(Bus)
		f.write('\n')
"""
if __name__ == '__main__':

	flowReportFile = 'BusReportsRawCropped_0723.txt'
	Raw = 'RawCropped_0723v2.raw'
	flowDict = BusReport(flowReportFile,Raw)