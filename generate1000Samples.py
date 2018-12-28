import random
import csv
import matplotlib.pyplot as plt
import time
import numpy as np
######
# # some random testing
# key = 'R106/F151/B151/AG/A/1.0e-6'
# eventKeyFile = 'eventIDFileLI.txt'
# eventList =  []
# eventListInd = []
# with open(eventKeyFile,'r') as f:
#     fileLines = f.read().split('\n')

#     for line in fileLines[1:]:
#         eventList.append(line)

# keyInd = eventList.index(key)


# #print(keyInd)



# print('Sampling fault voltage data')
# vFileName = 'vData3phLI.csv' # csv file containing voltage data (different types of fault)
# vFile = open(vFileName,'rb')

# readerV=csv.reader(vFile,quoting=csv.QUOTE_NONNUMERIC) # so that the entries are converted to floats
# for idx, row in enumerate(readerV):
#     if idx == keyInd:
#         val = row
#         break


# plt.plot(val)
# plt.show()
##########



###########
# eventKeyFile = 'eventIDFileLI.txt'

# # read the event file
# eventList =  []
# eventListInd = []
# with open(eventKeyFile,'r') as f:
#     fileLines = f.read().split('\n')
#     fileLines =  fileLines[1:]
#     for ind, line in enumerate(fileLines):
#         line = fileLines[ind]
#         if line == '':
#             continue
#         eventKey = line
#         eventKeyWords = eventKey.split('/')
#         PL = eventKeyWords[0][1:].strip()
#         faultbus = eventKeyWords[1][1:].strip()
#         currentbus = eventKeyWords[2][1:].strip()
#         faulttype = eventKeyWords[3].strip()
#         phase = eventKeyWords[4].strip()
#         faultZ = eventKeyWords[5].strip()
#         eventList.append(line)
#         if phase == 'A':
#             eventListInd.append(ind)

# # take 1000 sample indices from eventListInd
# # these indices will be extracted from the csv data
# sampleIndices =  random.sample(eventListInd,1000)
# sampleIndices = sorted(sampleIndices)


# start = time.time()
# print('Sampling fault voltage data')
# vFileName = 'vData3phLI.csv' # csv file containing voltage data (different types of fault)
# #vFileName = 'G:/My Drive/My PhD research/Running TS3ph/fault3ph/vData3phLI.csv'
# #aFileName = 'fault3ph/aData3phLI.csv' # csv file containing angle data (different types of fault)

# vFile = open(vFileName,'rb')
# #aFile = open(aFileName,'rb')

# readerV=csv.reader(vFile,quoting=csv.QUOTE_NONNUMERIC) # so that the entries are converted to floats
# #readerA = csv.reader(aFile,quoting=csv.QUOTE_NONNUMERIC) # so that the entries are converted to floats

# sampleV = [row for idx, row in enumerate(readerV) if idx in sampleIndices]
# end = time.time()

# print('Getting the sample took {} seconds'.format(end-start))

# # write the 1000 samples to a new csv file
# sampledVFile = 'vData3phLISamples.csv'



# sampledVFile = open(sampledVFile, 'wb')
# writerObjVTFOut = csv.writer(sampledVFile)

# for sample in sampleV:
#     writerObjVTFOut.writerow(sample)




# sampledVFile.close()


# # write the corresponding event ids to a new event log file
# with open('sampleVEventID.txt','w') as f:
#     f.write('LoadPercent/FaultBus/Bus/FaultType/Phase/FaultZ')
#     f.write('\n')
#     for sampleInd in sampleIndices:
#         f.write(eventList[sampleInd])
#         f.write('\n')

############



########
# read the data
vFileName = 'vData3phLISamples.csv' # csv file containing voltage data (different types of fault)
eventKeyFile = 'sampleVEventID.txt'

# read the event file
eventList = []
with open(eventKeyFile,'r') as f:
    fileLines = f.read().split('\n')
    for line in fileLines[1:]:
        if line == '':
            continue
        eventList.append(line.strip())


EventDict = {}
vFile = open(vFileName,'rb')
readerV=csv.reader(vFile,quoting=csv.QUOTE_NONNUMERIC) # so that the entries are converted to floats
minList = []
for idx, row in enumerate(readerV):
    minval = np.array(row).min()
    minList.append(minval)
    eventKey = eventList[idx]
    EventDict[eventKey] = row


key = 'R104/F151/B151/ABCG/A/1.0e-6'
plt.plot(EventDict[key])
plt.grid()
plt.show()



# plt.hist(minList,bins = 'auto')
# plt.show()
#############


