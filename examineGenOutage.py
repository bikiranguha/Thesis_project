# see the effect of generator outage

# get the files
genOutDir = 'GenOut'

vFilePath = '{}/vGenOut.csv'.format(genOutDir)
aFilePath = '{}/aGenOut.csv'.format(genOutDir)
fFilePath = '{}/fGenOut.csv'.format(genOutDir)
eventFilePath = '{}/eventGenOut.txt'.format(genOutDir)

# file objects



eventList = []
with open(eventFilePath,'r') as f:
    fileLines = f.read().split('\n')
    for line in fileLines[1:]:
        if line == '':
            continue
        eventList.append(line.strip())

vFile = open(vFilePath, 'rb') # 'wb' needed to avoid blank space in between lines
aFile = open(aFilePath, 'rb')
fFile = open(fFilePath, 'rb')
tFile = open(timeDataFilePath, 'rb')



