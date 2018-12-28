# used to show the operation of the classifier in cases where mutliple events occur in a timeframe
import csv
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
import scipy.cluster.hierarchy as hcluster
vFileName = 'vTest.csv' # csv file containing voltage data (different types of fault)
aFileName = 'aTest.csv'
tFileName = 'timeTest.csv'
eventKeyFile = 'eventKeysTest.csv'


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
    for line in fileLines:
        if line == '':
            continue
        eventList.append(line.strip())


# get the event key
#event = 'R106/F3018/B3018/ABCG'
#event = '151,201,1;201,202,1/F201/3005'
#event = '151,201,1;201,202,1/F201/201'

# reshape into rows with 60 samples each
timeWindow = 60
thresh = 0.15 # chosen from histogram
eventQueries = ['151,201,1;201,202,1/F201/201', '151,201,1;201,202,1/F201/3005']


eventdict = {}

for idx, row in enumerate(readerV):

    key = eventList[idx]
    eventdict[key] = row







k = 1
for event in eventQueries:
    v = np.array(eventdict[event])

    vArr = np.array(v[:-(v.shape[0]%timeWindow)]).reshape(-1,timeWindow)

    
    #####
    # histograms to get an idea of the threshold choice
    distance = []
    for i in range(vArr.shape[0]-1):
        distance.append(LA.norm(vArr[i]-vArr[i+1]))

    plt.hist(distance,bins = 'auto')
    plt.grid()
    #plt.show()
    plt.title(event)
    plt.savefig('dist{}.png'.format(k))
    plt.close()


    #dArray = np.array(distance)
    #print dArray.mean()
    #print dArray.std()
    ####
    





    clusters = hcluster.fclusterdata(vArr, thresh, criterion="distance")
    x = []
    for cl in range(clusters.shape[0]):
        for i in range(vArr.shape[1]):
            x.append(clusters[cl])

    f, (ax1, ax2) = plt.subplots(2, 1)
    ax1.set_title('Illustration of clustering algorithm: {}'.format(event))
    ax1.set_ylabel('V (pu)')
    ax2.set_ylabel('Cluster No.')
    ax2.set_xlabel('Sample No.')
    ax2.set_ylim(0,10)
    ax1.plot(v)
    ax2.plot(x)
    ax1.grid(True)
    ax2.grid(True)
    figName = 'cluster{}.png'.format(k)
    plt.savefig(figName)
    plt.close()
    k+=1
    #plt.show()



"""
#####
# histograms to get an idea of the threshold choice
distance = []
for i in range(vArr.shape[0]-1):
    distance.append(LA.norm(vArr[i]-vArr[i+1]))

plt.hist(distance,bins = 'auto')
plt.grid()
plt.show()


#dArray = np.array(distance)
#print dArray.mean()
#print dArray.std()
####
"""


"""
#interestingrowArray = np.array(interestingrowA)
interestingrowArray = np.array(interestingrows)
sampleVArray = np.array(interestingrowArray[:60]).reshape(-1,10) # 30 features
#thresh = 0.1 # works for faulted phase, not for unfaulted phase
thresh = 0.05

clusters = hcluster.fclusterdata(sampleVArray, thresh, criterion="distance")
x = []
for cl in range(clusters.shape[0]):
    for i in range(sampleVArray.shape[1]):
        x.append(clusters[cl])
"""
vFile.close()
tFile.close()