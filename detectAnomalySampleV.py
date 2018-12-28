import csv
import matplotlib.pyplot as plt 
import numpy as np
from EventDetectPCAFn import eventdetectPCA
dataFileName = 'sampleVAnomalyDetection.csv'
dataFile = open(dataFileName,'rb')
reader=csv.reader(dataFile,quoting=csv.QUOTE_NONNUMERIC) # so that the entries are converted to floats
v = [row for idx, row in enumerate(reader) if idx==0][0]
v  = np.array(v)
t = np.array(range(v.shape[0]))/30.0


# plt.plot(t,v)
# plt.grid()
# plt.show()




# test hierarchical clustering

#######
# trying out hierarchical clustering (agglomerative)

# Using the dendrogram to find the optimal number of clusters on the spectrum data
import scipy.cluster.hierarchy as sch
vArray = np.array(v).reshape(-1,300)
X = vArray


# dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
# plt.title('Dendrogram')
# plt.xlabel('freq data')
# plt.ylabel('Euclidean distances')
# plt.show()


# Fitting Hierarchical Clustering by just specifying the the number of clusters
from sklearn.cluster import AgglomerativeClustering

X = vArray
hc = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)
out = []
for cl in range(y_hc.shape[0]):
    for i in range(vArray.shape[1]):
        out.append(y_hc[cl])


# Visualizing the performance
f, (ax1, ax2) = plt.subplots(2, 1)
ax1.set_title('Illustration of hierarchical clustering using Agglomerative Clustering')
ax1.set_ylabel('Voltage (pu)')
ax2.set_ylabel('Cluster Output.')
ax2.set_xlabel('t(s)')
#ax2.set_ylim(0,5)
ax1.plot(t,v)
ax2.plot(t,out)
ax1.grid(True)
ax2.grid(True)
plt.show()






# Hierarchical clustering using distance threshold
threshold = 0.1
clusters = sch.fclusterdata(X, threshold, criterion="distance")
# getting the predictions in a way so that we can plot
out = []
for cl in range(clusters.shape[0]):
    for i in range(vArray.shape[1]):
        out.append(clusters[cl])

# Visualizing the performance
f, (ax1, ax2) = plt.subplots(2, 1)
ax1.set_title('Illustration of hierarchical clustering using threshold')
ax1.set_ylabel('Voltage (pu)')
ax2.set_ylabel('Cluster Output.')
ax2.set_xlabel('t(s)')
#ax2.set_ylim(0,5)
ax1.plot(t,v)
ax2.plot(t,out)
ax1.grid(True)
ax2.grid(True)
plt.show()
#####



### using PCA
# startTime = 0.0
# endTime = 1900.0


# # define a steady state window, to get steady state predictions of the voltage
# startSample = int(startTime*30)
# endSample = int(endTime*30)
# errorThreshold = 0.01

# yPredSteady, abnormalTimeInd =  eventdetectPCA(t,v,startSample,endSample,errorThreshold)



# detectionSignal = np.zeros(t.shape[0])

# for i in range(detectionSignal.shape[0]):
#     if t[i] in abnormalTimeInd:
#         detectionSignal[i] = 1.0

# # Visualizing the performance
# f, (ax1, ax2) = plt.subplots(2, 1)
# ax1.set_title('Illustration of PCA')
# ax1.set_ylabel('Voltage (pu)')
# ax2.set_ylabel('PCA Output')
# ax2.set_xlabel('t (s)')
# #ax2.set_ylim(0,5)
# ax1.plot(t,v)
# ax2.plot(t,detectionSignal)
# ax1.grid(True)
# ax2.grid(True)
# plt.show()
##########