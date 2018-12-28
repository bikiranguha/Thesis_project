"""
Organizes Branch or TF impedance data within a  proper class structure
Note: CAN only be used for Raw files where there are no 3 winders
"""




def getBranchTFData(Raw):

	# generates a structure which stores all the branch and tf impedance info
	import math
	import sys


	BranchTFDataDict = {}

	class BranchTFData(object):
		def __init__(self):
			self.toBus = []
			self.R = []
			self.X = []
			self.Z = []
			self.IsBranch = []

	def parallel(Z1,Z2):
		# calculate parallel impedance
		if Z1 == 0.0:
			Zp = Z2
		elif Z2 == 0.0:
			Zp = Z1
		else:
			Zp = (Z1*Z2)/(Z1+Z2)
		return Zp

	def generateProperImpedance(CZ,R,X,Sbase,Base12):
		# Sbase: System base, Base12: tf base
		# if CZ == '1', no need to change

		if CZ == '2':
			Base = Base12 # tf base
			R = R/Base12*Sbase
			X = X/Base12*Sbase

		elif CZ == '3':
			# R is the power loss, X is the branch impedance magnitude
			Base = Base12
			R = ((R/10**6)*Sbase/(Base**2))
			Z = X*Sbase/Base
			X = math.sqrt(Z**2 - R**2)

		return R,X






	def getBranchTFInfo(Bus1,Bus2,R,X,BranchTFDataDict,IsBranch):
		# generate branch impedance info structure
		Z = math.sqrt(R**2 + X**2)
		if Bus1 not in BranchTFDataDict.keys():
			BranchTFDataDict[Bus1] = BranchTFData()
		if Bus2 not in BranchTFDataDict[Bus1].toBus:
			BranchTFDataDict[Bus1].toBus.append(Bus2)
			BranchTFDataDict[Bus1].R.append(R)
			BranchTFDataDict[Bus1].X.append(X)
			BranchTFDataDict[Bus1].Z.append(Z)
			BranchTFDataDict[Bus1].IsBranch.append(IsBranch)
		else: # Bus 2 already had another branch connection to Bus 1
			#print Bus1 + ',' + Bus2
			ind = BranchTFDataDict[Bus1].toBus.index(Bus2)
			OldR = BranchTFDataDict[Bus1].R[ind]
			OldX = BranchTFDataDict[Bus1].X[ind]
			OldZ = BranchTFDataDict[Bus1].Z[ind]
			Rp = parallel(R,OldR)
			Xp = parallel(X,OldX)
			Zp = parallel(Z,OldZ)
			BranchTFDataDict[Bus1].R[ind] = Rp
			BranchTFDataDict[Bus1].X[ind] = Xp
			BranchTFDataDict[Bus1].Z[ind] = Zp
			


	##############

	with open(Raw, 'r') as f:
		filecontent = f.read()
		fileLines = filecontent.split('\n')






	#print 'List of branches or tf which constitute parallel connections between two buses:'
	# extract branch impedance data
	branchStartIndex = fileLines.index('0 / END OF GENERATOR DATA, BEGIN BRANCH DATA') + 1
	branchEndIndex = fileLines.index('0 / END OF BRANCH DATA, BEGIN TRANSFORMER DATA')

	for i in range(branchStartIndex, branchEndIndex):
		line = fileLines[i]
		words = line.split(',')
		Bus1 = words[0].strip()
		Bus2 = words[1].strip()
		status = words[-5].strip()


		if status == '1':
			R = float(words[3].strip())
			X = float(words[4].strip())
			getBranchTFInfo(Bus1,Bus2,R,X,BranchTFDataDict,1)
			getBranchTFInfo(Bus2,Bus1,R,X,BranchTFDataDict,1)


	# extract tf impedance data
	tfStartIndex = fileLines.index('0 / END OF BRANCH DATA, BEGIN TRANSFORMER DATA') + 1
	tfEndIndex = fileLines.index('0 / END OF TRANSFORMER DATA, BEGIN AREA DATA')
	i = tfStartIndex
	while i < tfEndIndex:
		line = fileLines[i]
		words = line.split(',')
		Bus1 = words[0].strip()
		Bus2 = words[1].strip()
		Bus3 = words[2].strip()
		try:
			status = words[11].strip()
		except:
			sys.exit("This function can only be used on a raw file where there are no 3 winder tf.")
		CZ = words[5].strip()


		if status == '1': # get tf data
			i+=1
			line = fileLines[i]
			words = line.split(',')
			R = float(words[0].strip())
			X = float(words[1].strip())
			Base12 = float(words[2].strip())
			Sbase = 100.0 # system base
			R,X = generateProperImpedance(CZ,R,X,Sbase,Base12)


			getBranchTFInfo(Bus1,Bus2,R,X,BranchTFDataDict,0)
			getBranchTFInfo(Bus2,Bus1,R,X,BranchTFDataDict,0)
			i+=3
		else: # tf is off
			i+=4


	return BranchTFDataDict

if __name__ == "__main__":
	R = 5.19310E+5
	X = 8.99000E-2
	Base12 = 420.00
	Sbase = 100.0
	CZ = '3'
	R,X = generateProperImpedance(CZ,R,X,Sbase,Base12)
	print R 
	print X
