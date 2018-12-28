# trying to cluster all the spikes in the input as a single cluster
# tried two different clustering techniques: K-Means and Hierarchical clustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import linalg as LA
csv_file = '120103,010000000,UT,Austin,3378,Phasor.csv'
df = pd.read_csv(csv_file)


# voltage angle from two different points in power system
angle1 = np.array(df['UT 3 phase_VALPM_Angle'])
angle2 = np.array(df['McDonald 1P_V1LPM_Angle'])


relangle = abs(np.unwrap(angle1) - np.unwrap(angle2)) # relative angle
drelangle = abs(np.gradient(relangle)) 
# this is the input time series vector (gradient of angle differences)
X = drelangle.reshape(-1,30) 



#####################################
#####################################
# evaluate k means clustering
# Applying k means with 2 clusters
from sklearn.cluster import KMeans
# reshaping the vector into an array where each row has 30 samples

kmeans = KMeans(n_clusters = 2, init = 'k-means++', max_iter = 300, n_init = 10) 
y_kmeans = kmeans.fit_predict(X)

# generating the cluster output for each row of X
out = []
for cl in range(y_kmeans.shape[0]):
    for i in range(X.shape[1]):
        out.append(y_kmeans[cl])


f, (ax1, ax2) = plt.subplots(2, 1)
ax1.set_title('Illustration of KMeans')
ax1.set_ylabel('V (pu)')
ax2.set_ylabel('Cluster No.')
ax2.set_xlabel('Sample No.')
#ax2.set_ylim(0,5)
ax1.plot(drelangle)
ax2.plot(out)
ax1.grid(True)
ax2.grid(True)
plt.show()


# Observation: K-Means only detects some of the spikes, how to improve it so that it detects all of the spikes (or at least more than this)
#########################
########################



#########################
#########################
# evaluating hierarchical clustering
import scipy.cluster.hierarchy as hcluster

"""
####
# histograms to determine threshold choice
distance = [] # distance between two successive samples
for i in range(X.shape[0]-1):
    distance.append(LA.norm(X[i]-X[i+1]))

plt.hist(distance,bins = 'auto')
plt.show()
#dArray = np.array(distance)
#print dArray.mean()
#print dArray.std()
####
"""



# getting clusters
thresh = 0.5
clusters = hcluster.fclusterdata(X, thresh, criterion="distance")


# visualizing the output
out = []
for cl in range(clusters.shape[0]):
    for i in range(X.shape[1]):
        out.append(clusters[cl])

f, (ax1, ax2) = plt.subplots(2, 1)
ax1.set_title('Illustration of hierarchical clustering')
ax1.set_ylabel('Signal')
ax2.set_ylabel('Cluster No.')
ax2.set_xlabel('Sample No.')
#ax2.set_ylim(0,5)
ax1.plot(drelangle)
ax2.plot(out)
ax1.grid(True)
ax2.grid(True)
plt.show()

# Observation: Much better than K Means in detecting abnormality but has so many different clusters for the spikes 
#  (i would ideally want there to be only two clusters)
######################
######################

