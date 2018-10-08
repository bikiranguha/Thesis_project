InputFile = 'InputFile.txt'
TS3phEventList = []
RealWorldEventList = []
with open(InputFile,'r') as f:
	fileLines = f.read().split('\n')
	for line in fileLines:
		if 'rawfile' in line:
			words = line.split('=')
			rawfile = words[1].strip()
			break
print rawfile

TS3phEventStartIndex = fileLines.index("TS3ph event:") + 1
TS3phEventEndIndex = fileLines.index("Real world event:")
RealEventStartIndex = fileLines.index("Real world event:") + 1

# get the events for the TS3ph sim
for i in range(TS3phEventStartIndex,TS3phEventEndIndex):
	line = fileLines[i]
	words = line.split()
	flag = words[0].strip()
	param = words[1].strip()
	TS3phEventList.append(flag)
	TS3phEventList.append(param)

# get the events for the real-world sim
for i in range(RealEventStartIndex,len(fileLines)):
	line = fileLines[i]
	if line == '':
		continue
	words = line.split()
	flag = words[0].strip()
	param = words[1].strip()
	RealWorldEventList.append(flag)
	RealWorldEventList.append(param)
