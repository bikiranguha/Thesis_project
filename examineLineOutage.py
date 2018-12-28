# see the effect of generator outage
import csv
import matplotlib.pyplot as plt
import numpy as np

# get the files
lineOutDir = 'LineOut'

vFilePath = '{}/vLineOut.csv'.format(lineOutDir)
aFilePath = '{}/aLineOut.csv'.format(lineOutDir)
fFilePath = '{}/fLineOut.csv'.format(lineOutDir)
eventFilePath = '{}/eventLineOut.txt'.format(lineOutDir)
timeDataFilePath = '{}/t.csv'.format(lineOutDir)



# file objects
eventList = []
with open(eventFilePath,'r') as f:
    fileLines = f.read().split('\n')
    for line in fileLines[1:]:
        if line == '':
            continue
        eventList.append(line.strip())


vFilePath = open(vFilePath, 'rb') # 'wb' needed to avoid blank space in between lines
aFilePath = open(aFilePath, 'rb')
fFilePath = open(fFilePath, 'rb')
timeDataFilePath = open(timeDataFilePath, 'rb')


vReader = csv.reader(vFilePath,quoting=csv.QUOTE_NONNUMERIC) # 'wb' needed to avoid blank space in between lines
aReader = csv.reader(aFilePath,quoting=csv.QUOTE_NONNUMERIC)
fReader = csv.reader(fFilePath,quoting=csv.QUOTE_NONNUMERIC)
tReader = csv.reader(timeDataFilePath,quoting=csv.QUOTE_NONNUMERIC)


tme = [row for idx, row in enumerate(tReader) if idx==0][0]

"""
# examine the frequency data
freqDict = {}

for idx, row in enumerate(fReader):
    eventKey = eventList[idx]
    eventKeyWords = eventKey.split('/')

    genbus = eventKeyWords[0][1:].strip()
    currentbus = eventKeyWords[1][1:].strip()
    eventID = 'G{}'.format(genbus)
    if eventID not in freqDict:
            freqDict[eventID] = []
    
    for f in row:
        freqDict[eventID].append(f)





key = 'G3018'
arr = 60*(1+np.array(freqDict[key]).reshape(23,-1))

#tme = 
for i in range(arr.shape[0]):
    plt.plot(arr[i])
plt.ylim(55,65)
plt.grid()
plt.show()

"""
# examine the voltage data
vDict = {}

for idx, row in enumerate(vReader):
    eventKey = eventList[idx]
    eventKeyWords = eventKey.split('/')
    PL = eventKeyWords[0][1:].strip()
    line = eventKeyWords[1][1:].strip()
    currentbus = eventKeyWords[1][1:].strip()
    eventID = 'R{}/L{}'.format(PL,line)
    if eventID not in vDict:
            vDict[eventID] = []
    
    for f in row:
        vDict[eventID].append(f)





#key = 'L151,152,1'
key = 'R100/L203,205,1'
arr = np.array(vDict[key]).reshape(-1,len(tme))


#for i in range(arr.shape[0]):
#    plt.plot(tme,arr[i])


plt.plot(tme,arr[0])
plt.ylim(0.8,1.2)
plt.grid()
plt.show()









# close all files
vFilePath.close()
aFilePath.close()
fFilePath.close()
timeDataFilePath.close()