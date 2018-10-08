"""
Script which returns NeighbourDict whose values are a set of neighbours
"""

def getNeighbours(Raw):

	NeighbourDict = {} # Dictionary of branch and tf connections
	def BusAppend(Bus,NeighbourBus,NeighbourDict):
		if Bus not in NeighbourDict.keys():
			NeighbourDict[Bus] = set()
		NeighbourDict[Bus].add(NeighbourBus)


	with open(Raw, 'r') as f:
		filecontent = f.read()
		fileLines = filecontent.split('\n')


		# get branch data
		branchStartIndex = fileLines.index('0 / END OF GENERATOR DATA, BEGIN BRANCH DATA') + 1
		branchEndIndex = fileLines.index('0 / END OF BRANCH DATA, BEGIN TRANSFORMER DATA')

		for i in range(branchStartIndex, branchEndIndex):
			line = fileLines[i]
			words = line.split(',')
			Bus1 = words[0].strip()
			Bus2 = words[1].strip()
			status = words[13].strip()

			if status == '1':
				BusAppend(Bus1,Bus2,NeighbourDict)
				BusAppend(Bus2,Bus1,NeighbourDict)


		# get tf data
		tfStartIndex = fileLines.index('0 / END OF BRANCH DATA, BEGIN TRANSFORMER DATA') + 1
		tfEndIndex = fileLines.index('0 / END OF TRANSFORMER DATA, BEGIN AREA DATA')
		i = tfStartIndex
		while i < tfEndIndex:
			line = fileLines[i]
			words = line.split(',')
			Bus1 = words[0].strip()
			Bus2 = words[1].strip()
			Bus3 = words[2].strip()
			status  = words[11].strip()

			if status == '1':

				if Bus3 == '0': # two winder
					BusAppend(Bus1,Bus2,NeighbourDict)
					BusAppend(Bus2,Bus1,NeighbourDict)
					i+=4



				else: # three winder
					BusAppend(Bus1,Bus2,NeighbourDict)
					BusAppend(Bus1,Bus3,NeighbourDict)
					BusAppend(Bus2,Bus1,NeighbourDict)
					BusAppend(Bus2,Bus3,NeighbourDict)
					BusAppend(Bus3,Bus1,NeighbourDict)
					BusAppend(Bus3,Bus2,NeighbourDict)
					i+=5
			
			else: # tf disconnected
				if Bus3 == '0':
					i+=4
				else:
					i+=5

	return NeighbourDict


