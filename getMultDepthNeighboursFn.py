# Function to return multi-depth neighbours of any bus given the depth 1 neighbour dictionary and depth
from Queue import Queue
def getMultDepthNeighbours(Bus,NeighbourDict,depth):
	DepthNNeighbourDict = {}

	explored = set()
	frontier = Queue(maxsize=0)
	frontier.put(Bus)
	DepthDict  = {}
	DepthDict[Bus]  = 0
	# BFS search
	while not frontier.empty():
		currentBus =  frontier.get()
		frontier.task_done()
		NeighbourList = list(NeighbourDict[currentBus])
		if currentBus not in explored and currentBus!= Bus:
			explored.add(currentBus)

		currentBusDepth = DepthDict[currentBus]
		neighbourDepth = currentBusDepth + 1
		if neighbourDepth > depth:
			continue
		for neighbour in NeighbourList:
			if neighbour not in explored and neighbour != Bus:
				frontier.put(neighbour)
				DepthDict[neighbour] = neighbourDepth

	DepthNNeighbourDict[Bus] = explored
	return DepthNNeighbourDict
