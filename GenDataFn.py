# function to generate a dictionary of gen data


def getGenData(Raw):

	GenDataDict = {} # key: genBus, value: all relevant gen data
	class GenData(object):
		def __init__(self):
			self.ID = ''
			self.Pgen = ''
			self.Qgen = ''
			self.Vset = ''
			self.MBase = ''
			self.R = ''
			self.X = ''
			self.status = ''

	with open(Raw, 'r') as f:
		filecontent = f.read()
		fileLines = filecontent.split('\n')
		GenStartIndex = fileLines.index('0 / END OF FIXED SHUNT DATA, BEGIN GENERATOR DATA') + 1
		GenEndIndex = fileLines.index('0 / END OF GENERATOR DATA, BEGIN BRANCH DATA')

		fileLines=fileLines[GenStartIndex:GenEndIndex] # crop fileLines to just include gen data

		for line in fileLines:
			words=line.split(',')
			# get gen data
			genbus=words[0].strip()
			ID = words[1].strip("'").strip()
			Pgen = words[2].strip()
			Qgen = words[3].strip()
			Vset = words[6].strip()
			MBase = words[8].strip()
			R = words[9].strip()
			X = words[10].strip()
			status=words[14].strip()
			# input gen data to dict
			GenDataDict[genbus] = GenData()
			GenDataDict[genbus].ID = ID
			GenDataDict[genbus].Pgen = Pgen
			GenDataDict[genbus].Qgen = Qgen
			GenDataDict[genbus].Vset = Vset
			GenDataDict[genbus].MBase = MBase
			GenDataDict[genbus].R = R
			GenDataDict[genbus].X = X
			GenDataDict[genbus].status = status
	return GenDataDict

