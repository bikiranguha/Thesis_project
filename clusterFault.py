# used to show that the clustering algorithm can clearly distinguish between normal operation and faults
import csv
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
import scipy.cluster.hierarchy as hcluster
vFileName = 'G:/My Drive/My PhD research/Running TS3ph/fault3ph/vData3phLI.csv' # csv file containing voltage data (different types of fault)
tFileName = 'G:/My Drive/My PhD research/Running TS3ph/fault3ph/tData3ph.csv' # csv file containing the time data
eventKeyFile = 'G:/My Drive/My PhD research/Running TS3ph/fault3ph/eventIDFileLI.txt'


vFile = open(vFileName,'rb')
tFile = open(tFileName,'rb')
readerV=csv.reader(vFile,quoting=csv.QUOTE_NONNUMERIC) # so that the entries are converted to floats
readerT = csv.reader(tFile,quoting=csv.QUOTE_NONNUMERIC)
tme = [row for idx, row in enumerate(readerT) if idx==0][0]
#interestingrow = [row for idx, row in enumerate(reader) if idx ==1]


# read the event file
eventList = []
with open(eventKeyFile,'r') as f:
    fileLines = f.read().split('\n')
    for line in fileLines[1:]:
        if line == '':
            continue
        eventList.append(line.strip())


# visualize 3 ph at the fault bus
#event = 'R106/F3018/B3018/ABCG'
event = 'R105/F3001/B3018/AG'
#faultZ = '4.0e-03'
faultZ = '1.0e-6'
eventKeyA = '{}/A/{}'.format(event,faultZ)
eventIndA = eventList.index(eventKeyA)

eventKeyB = '{}/B/{}'.format(event,faultZ)
eventIndB = eventList.index(eventKeyB)


eventKeyC = '{}/C/{}'.format(event,faultZ)
eventIndC = eventList.index(eventKeyC)

interestingInd = []
interestingInd.append(eventIndA)
interestingInd.append(eventIndB)
interestingInd.append(eventIndC)
#print eventInd
interestingrows = [row for idx, row in enumerate(readerV) if idx in interestingInd]
interestingrowA = interestingrows[0]
interestingrowB = interestingrows[1]
interestingrowC = interestingrows[2]


#interestingrowArray = np.array(interestingrowA)
interestingrowArray = np.array(interestingrowB)
sampleVArray = np.array(interestingrowArray[:60]).reshape(-1,10) # 30 features
#thresh = 0.1 # works for faulted phase, not for unfaulted phase
thresh = 0.05

clusters = hcluster.fclusterdata(sampleVArray, thresh, criterion="distance")
x = []
for cl in range(clusters.shape[0]):
    for i in range(sampleVArray.shape[1]):
        x.append(clusters[cl])

f, (ax1, ax2) = plt.subplots(2, 1)
ax1.set_title('Illustration of clustering algorithm')
ax1.set_ylabel('V (pu)')
ax2.set_ylabel('Cluster No.')
ax2.set_xlabel('Sample No.')
ax2.set_ylim(0,5)
ax1.plot(interestingrowArray)
ax2.plot(x)
ax1.grid(True)
ax2.grid(True)
plt.show()
"""
# histograms to get an idea of the threshold choice
distance = []
for i in range(sampleVArray.shape[0]-1):
    distance.append(LA.norm(sampleVArray[i]-sampleVArray[i+1]))

#plt.hist(distance,bins = 'auto')
#plt.show()
#dArray = np.array(distance)
#print dArray.mean()
#print dArray.std()
####
"""
