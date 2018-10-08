"""
Find path from starting bus to end bus
"""

# Data import from other files
from generateNeighboursFn import getNeighbours
from Queue import Queue



def getPath(Raw,startBus,endBus):
	# Function to generate the list of paths


	NeighbourDict = getNeighbours(Raw) # key: any bus in the raw file, value: set of all neighbours (line and tf)

	# Use CAPENeighbourDict and BFS to find path from one bus to another. Use the concept given in getNeighboursAtCertainDepthFn
	PathDict = {}
	explored = set()
	#startBus = raw_input('Enter start bus: ')
	#endBus = raw_input('Enter end bus: ')



	frontier = Queue(maxsize=0)
	frontier.put(startBus)

	while not frontier.empty():
		currentBus = frontier.get()
		frontier.task_done()
		if currentBus == endBus:
			break

		NeighBourList = list(NeighbourDict[currentBus])


		explored.add(currentBus)

		for neighbour in NeighBourList:
			if neighbour in explored:
				continue

			if currentBus in PathDict.keys():
				PathDict[neighbour] = PathDict[currentBus] + '->' + neighbour
			else: # if currentBus is the start bus
				PathDict[neighbour] = currentBus + '->' + neighbour

			frontier.put(neighbour)

	Path = PathDict[endBus]
	
	return Path


if __name__ == '__main__':
	raw = 'savnw_dy_sol_0905.raw'
	#print getPath(raw,'101','3008')
	print getPath(raw,'151','3004')
