# trying to get clusters in the PMU frequency data

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#import scipy.cluster.hierarchy as hcluster
from numpy import linalg as LA
#from sklearn import preprocessing
csv_file = '120103,010000000,UT,Austin,3378,Phasor.csv'
df = pd.read_csv(csv_file)
dttme = list(df.Timestamp)
freqList = []
freqList.append(df['Z_UT_3378_AO[079]_Value'])
freqList.append(df['Z_UT_3378_AO[085]_Value'])
freqList.append(df['Z_UT_3378_AO[087]_Value'])
freqList.append(df['Z_UT_3378_AO[090]_Value'])
freqList.append(df['Z_UT_3378_AO[091]_Value'])
freqList.append(df['Z_UT_3378_AO[091]_Value'])
freqList.append(df['Z_UT_3378_AO[092]_Value'])
def getSec(dtTmeString):
    splt = dtTmeString.split()
    tme = splt[1]
    tmesplt = tme.strip().split(':')
    hour = float(tmesplt[0])
    mnt = float(tmesplt[1])
    sec = float(tmesplt[2])
    totsec = hour*3600+ mnt*60 + sec
    #return hour, mnt, sec  
    return totsec 

# convert the datetime object to seconds
tmeSec = []

startSec =  getSec(dttme[0])

for t in dttme:
    currentSec = getSec(t)
    relSec = currentSec - startSec
    tmeSec.append(relSec)


"""
#############
# visualizing the frequency data

# plot all the available frequencies
for freq in freqList:
    plt.plot(tmeSec, freq)

plt.xlabel('Time (s)')
plt.ylabel('Freq (Hz)')
plt.title('Frequency at UT_3378[079]')
plt.grid()
plt.show()
##############
"""

"""
# get the distribution of ROCOF over a time window of 10 seconds
ROCOF = np.gradient(freqList[0])
plt.plot(tmeSec,ROCOF)
plt.xlabel('Time (s)')
plt.ylabel('Freq (Hz)/s')
plt.title('dFrequency/ds at UT_3378[079]')
plt.grid()
plt.show()


ROCOFArray = np.array(ROCOF).reshape(-1,300)

meanROCOFList = []
for i in range(ROCOFArray.shape[0]):
    meanROCOF = ROCOFArray[i].mean()
    meanROCOFList.append(meanROCOF)

#plt.hist(meanROCOFList,bins = 'auto')
#plt.grid()
#plt.show()
####
"""


"""
####
# trying k-means
from sklearn.cluster import KMeans
wcss = []
freq = freqList[0]
freqArray = np.array(freq).reshape(-1,300)
#ROCOF = np.gradient(freq)

X = freqArray


# Using the elbow method to find the optimal number of clusters
for i in range(1,11): # to get wcss for 1 to 10 clusters
    # init: method to initialize the centroids, n_init = Number of times the k-mean algorithm run with different centroid seeds
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0) 
    kmeans.fit(X)
    wcss.append(kmeans.inertia_) # kmeans.inertia_ : computes wcss
 
plt.plot(range(1,11),wcss)
plt.title('Elbow Method')
plt.xlabel('No. of clusters')
plt.ylabel('WCSS')
plt.grid()
plt.show()



# Applying k means with optimal (5) clusters
kmeans = KMeans(n_clusters = 2, init = 'k-means++', max_iter = 300, n_init = 10) 
y_kmeans = kmeans.fit_predict(X)


out = []
for cl in range(y_kmeans.shape[0]):
    for i in range(freqArray.shape[1]):
        out.append(y_kmeans[cl])

f, (ax1, ax2) = plt.subplots(2, 1)
ax1.set_title('Illustration of clustering algorithm')
ax1.set_ylabel('freq (hz)')
ax2.set_ylabel('Cluster No.')
ax2.set_xlabel('Sample No.')
#ax2.set_ylim(0,5)
ax1.plot(freq)
ax2.plot(out)
ax1.grid(True)
ax2.grid(True)
plt.show()
######
"""


#######
# trying out hierarchical clustering

# Using the dendrogram to find the optimal number of clusters on the spectrum data
import scipy.cluster.hierarchy as sch
freq = freqList[0]
freqArray = np.array(freq).reshape(-1,300)
#ROCOF = np.gradient(freq)
X = freqArray

"""
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('freq data')
plt.ylabel('Euclidean distances')
plt.show()
"""

# Fitting Hierarchical Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
X = freqArray
hc = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)


# getting the predictions in a way so that we can plot
out = []
for cl in range(y_hc.shape[0]):
    for i in range(freqArray.shape[1]):
        out.append(y_hc[cl])

# Visualizing the performance
f, (ax1, ax2) = plt.subplots(2, 1)
ax1.set_title('Illustration of clustering algorithm')
ax1.set_ylabel('Freq (Hz)')
ax2.set_ylabel('Cluster No.')
ax2.set_xlabel('Sample No.')
#ax2.set_ylim(0,5)
ax1.plot(freq)
ax2.plot(out)
ax1.grid(True)
ax2.grid(True)
plt.show()


#####

