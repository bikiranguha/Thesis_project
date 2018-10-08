"""
Generates a dictionary which organizes all the branch info within a certain specified depth
"""


from generateNeighboursDataFn import generateNeighbours
from Queue import Queue



def getNeighboursDepthN(OriginBus,Raw,maxDepth):
	MultDepthBranchDataDict = {} # the output dict of this fn
	explored = set() # used for the BFS algorithm


	class multDepthBranchData(object):
		# to store all the branch data
		def __init__(self):
			self.toBus = []
			self.R = []
			self.X = []
			self.Z = []
			self.depth = [] # topology depth
			self.Path = [] # Path from Origin to current neighbour



	_,DepthOneBranchDataDict = generateNeighbours(Raw) # DepthOneBranchDataDict contains all the required info

	MultDepthBranchDataDict[OriginBus] = multDepthBranchData()



	# BFS algorithm to extract and organize all the data
	frontier = Queue(maxsize=0)
	frontier.put(OriginBus)

	while not frontier.empty():
		currentBus =  frontier.get()
		frontier.task_done()
		NeighbourList = list(DepthOneBranchDataDict[currentBus].toBus)
		if currentBus not in explored:
			explored.add(currentBus)

		#### initialization
		if currentBus == OriginBus:
			# get the details from the Depth One Branch Dictionary
			MultDepthBranchDataDict[OriginBus].toBus = DepthOneBranchDataDict[OriginBus].toBus
			MultDepthBranchDataDict[OriginBus].R = DepthOneBranchDataDict[OriginBus].R
			MultDepthBranchDataDict[OriginBus].X = DepthOneBranchDataDict[OriginBus].X
			#MultDepthBranchDataDict[OriginBus].cktID = DepthOneBranchDataDict[OriginBus].cktID
			MultDepthBranchDataDict[OriginBus].Z = DepthOneBranchDataDict[OriginBus].Z

			# make lists of size equal to no. of branch neighbours of the origin bus
			for n in range(len(NeighbourList)):
				MultDepthBranchDataDict[OriginBus].depth.append(1)
				MultDepthBranchDataDict[OriginBus].Path.append('')

			# generate path strings of depth 1 neighbours
			for neighbour in NeighbourList:
				PathStr = currentBus + '->' + neighbour
				ind = MultDepthBranchDataDict[OriginBus].toBus.index(neighbour)
				MultDepthBranchDataDict[OriginBus].Path[ind] = PathStr
				frontier.put(neighbour)

			continue
 		###############


 		# scan all neighbours of current bus and extract all data if the neighbour is within maxDepth
		for neighbour in NeighbourList:
			if neighbour not in explored:
				ParentInd = MultDepthBranchDataDict[OriginBus].toBus.index(currentBus)
				# generate neighbour depth
				ParentDepth = MultDepthBranchDataDict[OriginBus].depth[ParentInd]
				neighbourDepth = ParentDepth + 1
				# generate neighbour path from origin
				ParentPathStr = MultDepthBranchDataDict[OriginBus].Path[ParentInd]
				neighbourPathStr = ParentPathStr + '->' + neighbour

				
				neighbourInd = DepthOneBranchDataDict[currentBus].toBus.index(neighbour)
				# get total R from origin to current neighbour
				neighbourR = DepthOneBranchDataDict[currentBus].R[neighbourInd]
				ParentR = MultDepthBranchDataDict[OriginBus].R[ParentInd]
				totalR = neighbourR + ParentR

				# get total X from origin to current neighbour
				neighbourX = DepthOneBranchDataDict[currentBus].X[neighbourInd]
				ParentX = MultDepthBranchDataDict[OriginBus].X[ParentInd]
				totalX = neighbourX + ParentX

				# get total Z from origin to current neighbour
				neighbourZ = DepthOneBranchDataDict[currentBus].Z[neighbourInd]
				ParentZ = MultDepthBranchDataDict[OriginBus].Z[ParentInd]
				totalZ = neighbourZ + ParentZ

				# get cktID
				#neighbourcktID = DepthOneBranchDataDict[currentBus].cktID[neighbourInd]


				if neighbourDepth <= maxDepth and neighbourPathStr not in MultDepthBranchDataDict[OriginBus].Path : # dont add if path has already been included
					# append all data
					MultDepthBranchDataDict[OriginBus].toBus.append(neighbour)
					MultDepthBranchDataDict[OriginBus].depth.append(neighbourDepth)
					MultDepthBranchDataDict[OriginBus].Path.append(neighbourPathStr)
					#MultDepthBranchDataDict[OriginBus].cktID.append(neighbourcktID)
					MultDepthBranchDataDict[OriginBus].R.append(totalR)
					MultDepthBranchDataDict[OriginBus].X.append(totalX)
					MultDepthBranchDataDict[OriginBus].Z.append(totalZ)
					frontier.put(neighbour)




	return MultDepthBranchDataDict




if __name__ == '__main__':
	Raw = 'Raw0509.raw'
	nDepthDict = getNeighboursDepthN('750221',Raw,5)
	for key in nDepthDict.keys():
		print nDepthDict[key].toBus
		print nDepthDict[key].X
		print nDepthDict[key].Path
		print nDepthDict[key].depth