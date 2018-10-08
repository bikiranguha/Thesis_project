"""
Function to get the load data with raw file name as input
"""

def getLoadData(Raw):

	LoadDataDict = {}
	class LoadData(object):
		def __init__(self):
			self.constP = 0.0 # constant power MW
			self.constQ = 0.0 # constant power MVAR
			self.constIP = 0.0 # constant current MW
			self.constIQ = 0.0 # constant current MVAR
			self.constZP = 0.0 # constant impedance MW
			self.constZQ = 0.0 # constant current MVAR


	with open(Raw, 'r') as f:
		filecontent = f.read()
		fileLines = filecontent.split('\n')

	loadStartIndex = fileLines.index('0 / END OF BUS DATA, BEGIN LOAD DATA') + 1
	loadEndIndex = fileLines.index('0 / END OF LOAD DATA, BEGIN FIXED SHUNT DATA')

	for i in range(loadStartIndex, loadEndIndex):
		line = fileLines[i]
		    
		words = line.split(',')		
		Bus = words[0].strip()
		constP = float(words[5].strip())
		constQ = float(words[6].strip())
		constIP = float(words[7].strip())
		constIQ = float(words[8].strip())
		constZP = float(words[9].strip())
		constZQ = float(words[10].strip())
		if Bus not in LoadDataDict:
			LoadDataDict[Bus] = LoadData()

		LoadDataDict[Bus].constP += constP
		LoadDataDict[Bus].constQ += constQ


		LoadDataDict[Bus].constIP += constIP
		LoadDataDict[Bus].constIQ += constIQ

		LoadDataDict[Bus].constZP += constZP
		LoadDataDict[Bus].constZQ += constZQ
	return LoadDataDict

if __name__ == '__main__':
	raw = 'savnw.raw'
	LoadDataDict = getLoadData(raw)
			