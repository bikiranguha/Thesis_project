# script to process the input data

InputFile = 'InputFile.txt' # file containing the raw data and the event data
# convert crlf to lf (needed for linux)
import platform
currentOS = platform.system()
if currentOS == 'Linux':
	text = open(InputFile, 'rb').read().replace('\r\n', '\n')
	open(InputFile, 'wb').write(text)



EventDescriptionDict = {} # key: event index starting from 0, value: event description string without the time

TS3phEventList = [] # list of events  in the TS3ph simulations
RealWorldEventList = [] # list of events in the real world simulations
# get the raw file
with open(InputFile,'r') as f:
	fileLines = f.read().split('\n')
	for line in fileLines:
		if 'rawfile' in line:
			words = line.split('=')
			rawPath = words[1].strip()
			break

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
eventIndex = 0
for i in range(RealEventStartIndex,len(fileLines)):
	line = fileLines[i]
	if line == '':
		continue
	words = line.split()
	flag = words[0].strip()
	param = words[1].strip()
	EventDescriptionWords = param.split(',')
	EventDescriptionStr = ','.join(EventDescriptionWords[1:])
	#print EventDescriptionStr
	RealWorldEventList.append(flag)
	RealWorldEventList.append(param)
	EventDescriptionDict[eventIndex] = EventDescriptionStr
	eventIndex +=1
