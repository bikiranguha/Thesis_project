# trying to get clusters of the angle difference data using hierarchical clustering

# so far, it can cluster 10 second time windows using hierarchical clustering
# the threshold has to be carefully chosen though
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as hcluster
from numpy import linalg as LA
from sklearn import preprocessing
from getROCFn import getROC
csv_file = '120103,010000000,UT,Austin,3378,Phasor.csv'
df = pd.read_csv(csv_file)

#angle1 = np.array(df.Austin_V1LPM_Angle)
angle1 = np.array(df['UT 3 phase_VALPM_Angle'])
#angle2 = np.array(df.HARRIS_V1LPM_Angle)
angle2 = np.array(df['McDonald 1P_V1LPM_Angle'])
dttme = list(df.Timestamp)


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




relangle = abs(np.unwrap(angle1) - np.unwrap(angle2))
plt.plot(tmeSec, relangle)
plt.title('Angle variation over time')
plt.xlabel('Time (s)')
plt.ylabel('Angle(degrees)')
plt.grid()
plt.show()

"""
#############
# visualizing the angle data
relangle = abs(np.unwrap(angle1) - np.unwrap(angle2))




# plot the difference between the two angles
plt.plot(tmeSec, relangle)
plt.grid()
plt.show()
"""

"""
# plot the angles at two different substations
plt.plot(tmeSec, np.unwrap(angle1),label = 'Austin')
plt.plot(tmeSec, np.unwrap(angle2),label = 'Harris')
plt.legend()
plt.grid()
plt.show()
"""
##############


"""
###########
# visualizing the fft transform
from scipy.fftpack import fft
# testing fft on relative angle data
# Number of sample points (maybe per second)
N = 30
# sample spacing
T = 1.0 / 120 # 120 time steps per second
xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
#plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
relangle = abs(np.unwrap(angle1) - np.unwrap(angle2))
drelangle = np.gradient(relangle)
drelangleArr = np.array(drelangle).reshape(-1,N)
for i in range(35,40):
    yf = fft(drelangleArr[i])
    #plt.plot(np.abs(yf),label = str(i))
    #plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]),label = str(i))
    plt.plot(2.0/N * np.abs(yf),label = str(i))
    #plt.plot(drelangleArr[i])
plt.grid()
plt.legend()
plt.show()
##############
"""


######################
"""
# clustering algorithm used on the relative angle data
relangle = abs(np.unwrap(angle1) - np.unwrap(angle2))
#relangleArray = np.array(relangle).reshape(-1,30) # 30 features
relangleArray = np.array(relangle).reshape(-1,300) # 300 features
#thresh = 100
#thresh = 140
thresh = 14
clusters = hcluster.fclusterdata(relangleArray, thresh, criterion="distance")
"""



"""
# histograms to get an idea of the threshold choice
distance = []
relangle = abs(np.unwrap(angle1) - np.unwrap(angle2))
relangleC = relangle # select a steady state part 
relangleArrayC = np.array(relangleC).reshape(-1,300)
for i in range(relangleArrayC.shape[0]-1):
    distance.append(LA.norm(relangleArrayC[i]-relangleArrayC[i+1]))

plt.hist(distance,bins = 'auto')
plt.show()
#dArray = np.array(distance)
#print dArray.mean()
#print dArray.std()
####
"""

"""
x = []
for cl in range(clusters.shape[0]):
    for i in range(relangleArray.shape[1]):
        x.append(clusters[cl])

f, (ax1, ax2) = plt.subplots(2, 1)
ax1.set_title('Illustration of clustering algorithm')
ax1.set_ylabel('V (pu)')
ax2.set_ylabel('Cluster No.')
ax2.set_xlabel('Sample No.')
#ax2.set_ylim(0,5)
ax1.plot(relangle)
ax2.plot(x)
ax1.grid(True)
ax2.grid(True)
plt.show()
"""

######################

"""
#############
# trying k-means
from sklearn.cluster import KMeans
wcss = []


relangle = abs(np.unwrap(angle1) - np.unwrap(angle2))
#relangleArray = np.array(relangle).reshape(-1,30) # 30 features
relangleArray = np.array(relangle).reshape(-1,300) # 300 features
X = relangleArray


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
kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0) 
y_kmeans = kmeans.fit_predict(X)


out = []
for cl in range(y_kmeans.shape[0]):
    for i in range(relangleArray.shape[1]):
        out.append(y_kmeans[cl])

f, (ax1, ax2) = plt.subplots(2, 1)
ax1.set_title('Illustration of clustering algorithm')
ax1.set_ylabel('V (pu)')
ax2.set_ylabel('Cluster No.')
ax2.set_xlabel('Sample No.')
#ax2.set_ylim(0,5)
ax1.plot(relangle)
ax2.plot(out)
ax1.grid(True)
ax2.grid(True)
plt.show()
#################
"""

"""
####
# visualize the rate of change of the angle and do clustering
relangle = abs(np.unwrap(angle1) - np.unwrap(angle2))
#drelangle = getROC(relangle,range(relangle.shape[0]))
drelangle = abs(np.gradient(relangle))
drelangleArr = np.array(drelangle).reshape(-1,300)
#plt.plot(abs(np.gradient(relangle)))
#plt.show()

plt.plot(drelangle)
plt.grid()
plt.show()
####
"""

"""
# histograms to get an idea of the threshold choice
distance = []
#relangle = abs(np.unwrap(angle1) - np.unwrap(angle2))
#relangleC = relangle # select a steady state part 
drelangleArr = np.array(drelangle).reshape(-1,30)
for i in range(drelangleArr.shape[0]-1):
#for i in range(30):
    distance.append(LA.norm(drelangleArr[i]-drelangleArr[i+1]))

plt.hist(distance,bins = 'auto')
plt.show()
#dArray = np.array(distance)
#print dArray.mean()
#print dArray.std()
"""



# get the distribution of the distance among the windows where spikes are seen

# make a list of all the time windows where spikes are seen
spikeList = []
nonSpikeList = []
relangle = abs(np.unwrap(angle1) - np.unwrap(angle2))

"""
plt.plot(tmeSec,relangle)
plt.title('Relative angle between two PMUs')
plt.xlabel('time (s)')
plt.ylabel('Relative angle (degrees)')
plt.grid()
plt.show()
"""


drelangle = abs(np.gradient(relangle))
"""
plt.plot(tmeSec,drelangle)
plt.title('Relative angle derivative between two PMUs')
plt.xlabel('time (s)')
plt.ylabel('Relative angle derivative (degrees/sec)')
plt.grid()
plt.show()
"""



drelangleArr = np.array(drelangle).reshape(-1,30)

"""
# Using the dendrogram to find the optimal number of clusters on the drelangle data
import scipy.cluster.hierarchy as sch
X = drelangleArr
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Relative angle data')
plt.ylabel('Euclidean distances')
plt.show()
"""

for i in range(drelangleArr.shape[0]):
    sig = drelangleArr[i]
    if sig.max() >= 0.4:
        spikeList.append(sig)
    else:
        nonSpikeList.append(sig)


"""
# plot 10 spiky samples
for sig in spikeList[:10]:
    plt.plot(sig)

plt.title('Random spiky samples')
plt.xlabel('Sample no.')
plt.ylabel('abs(drelAngle)')
plt.grid()
plt.show()
#


# plot 10 spiky samples
for sig in nonSpikeList[:10]:
    plt.plot(sig)

plt.title('Random non-spiky samples')
plt.xlabel('Sample no.')
plt.ylabel('abs(drelAngle)')
plt.grid()
plt.show()
#
"""





#### visualizing the fft transforms
from scipy.fftpack import fft


"""
# testing fft on relative angle data
n = drelangleArr[0].size # number of samples in a row
timestep = 1.0/n 
freq = np.fft.fftfreq(n, d=timestep)
#print freq

# spiky signals
spikySpectrum = []
for sig in spikeList:
    yf = fft(np.array(sig))
    spikySpectrum.append(abs(yf[4:10]))
    #plt.scatter(freq[:n/2],np.abs(yf[:n/2]),c='blue') # only plot the positive part of the spectrum
"""

"""
plt.title('Spiky spectrum')
plt.xlabel('Frequncy (Hz)')
plt.ylabel('Mag')
plt.grid()
plt.show()
"""



"""
# non spiky signals
nonSpikySpectrum = []
for sig in nonSpikeList:
    yf = fft(np.array(sig))
    nonSpikySpectrum.append(abs(yf[4:10]))
    #plt.scatter(freq[:n/2],np.abs(yf[:n/2]),c= 'red') # only plot the positive part of the spectrum
    #plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
    #plt.plot(2.0/N * np.abs(yf[0:N//2]))
    #plt.plot(np.abs(yf[:yf.shape[0]/2]))
"""


"""
plt.title('Non-spiky vs Spiky spectrum')
plt.xlabel('Frequncy (Hz)')
plt.ylabel('Mag')
plt.legend()
plt.grid()
plt.show()
"""



# get an array containing the cropped frequency spectrum
spectrumArray = []
for i in range(drelangleArr.shape[0]):
    sig = drelangleArr[i]
    yf = fft(np.array(sig))
    spectrumArray.append(abs(yf[4:10]))

spectrumArray = np.array(spectrumArray)

"""
# Using the dendrogram to find the optimal number of clusters on the spectrum data
import scipy.cluster.hierarchy as sch
X = spectrumArray
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Relative angle data')
plt.ylabel('Euclidean distances')
plt.show()
"""



# Fitting Hierarchical Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
X = spectrumArray
hc = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)


# getting the predictions in a way so that we can plot
out = []
for cl in range(y_hc.shape[0]):
    for i in range(drelangleArr.shape[1]):
        out.append(y_hc[cl])

# Visualizing the performance
f, (ax1, ax2) = plt.subplots(2, 1)
ax1.set_title('Illustration of clustering algorithm')
ax1.set_ylabel('abs(relative angle derivative)')
ax2.set_ylabel('Cluster No.')
ax2.set_xlabel('Sample No.')
#ax2.set_ylim(0,5)
ax1.plot(drelangle)
ax2.plot(out)
ax1.grid(True)
ax2.grid(True)
ax2.set_ylim(-0.5,1.5)
plt.show()


#####



"""
### Applying k means with 2 clusters on the spectrum data
from sklearn.cluster import KMeans
X = spectrumArray
kmeans = KMeans(n_clusters = 2, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0) 
y_kmeans = kmeans.fit_predict(X)


out = []
for cl in range(y_kmeans.shape[0]):
    for i in range(drelangleArr.shape[1]):
        out.append(y_kmeans[cl])
"""

"""
f, (ax1, ax2) = plt.subplots(2, 1)
ax1.set_title('Illustration of clustering algorithm')
ax1.set_ylabel('abs(relative angle derivative)')
ax2.set_ylabel('Cluster No.')
ax2.set_xlabel('Sample No.')
#ax2.set_ylim(0,5)
ax1.plot(drelangle)
ax2.plot(out)
ax1.grid(True)
ax2.grid(True)
plt.show()
"""
###



"""
#### histograms to get an idea of the threshold choice
distanceSpiky = []
for i in range(len(spikySpectrum)):
    for j in range(len(spikySpectrum)):
        if i == j:
            continue
        distanceSpiky.append(LA.norm(spikySpectrum[i]-spikySpectrum[j]))



plt.hist(distanceSpiky,bins = 'auto')
plt.title('Spiky distance distribution')
plt.grid()
plt.show()
plt.close()

distanceNonSpiky = []
for i in range(len(nonSpikySpectrum)):
    for j in range(len(nonSpikySpectrum)):
        if i == j:
            continue
        distanceNonSpiky.append(LA.norm(nonSpikySpectrum[i]-nonSpikySpectrum[j]))



plt.hist(distanceNonSpiky,bins = 'auto')
plt.title('NonSpiky distance distribution')
plt.grid()
plt.show()
plt.close()
#dArray = np.array(distance)
#print dArray.mean()
#print dArray.std()

######
"""







"""
# histograms to get an idea of the threshold choice
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
distance = []
dtwdistList = []
#relangle = abs(np.unwrap(angle1) - np.unwrap(angle2))
#relangleC = relangle # select a steady state part 
drelangleArr = np.array(drelangle).reshape(-1,30)
for i in range(len(spikeList)):
    for j in range(len(spikeList)):
        if i == j:
            continue
        distance.append(LA.norm(spikeList[i]-spikeList[j]))
        dtwdist, path = fastdtw(spikeList[i],spikeList[j], dist=euclidean)
        dtwdistList.append(dtwdist)


plt.hist(distance,bins = 'auto')
plt.show()
plt.hist(dtwdistList,bins = 'auto')
plt.show()
#dArray = np.array(distance)
#print dArray.mean()
#print dArray.std()
"""





"""
# evaluate hierarchical clustering
threshold = 0.3
clusters = hcluster.fclusterdata(drelangleArr, threshold, criterion="distance")

x = []
for cl in range(clusters.shape[0]):
    for i in range(drelangleArr.shape[1]):
        x.append(clusters[cl])

f, (ax1, ax2) = plt.subplots(2, 1)
ax1.set_title('Illustration of clustering algorithm')
ax1.set_ylabel('Rate of change of relative angle')
ax2.set_ylabel('Cluster No.')
ax2.set_xlabel('Sample No.')
#ax2.set_ylim(0,5)
ax1.plot(drelangle)
ax2.plot(x)
ax1.grid(True)
ax2.grid(True)
plt.show()
##
"""





"""
# evaluate k means clustering
# Applying k means with 2 clusters
from sklearn.cluster import KMeans
X = drelangleArr[1:,:]
kmeans = KMeans(n_clusters = 2, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0) 
y_kmeans = kmeans.fit_predict(X)


out = []
for cl in range(y_kmeans.shape[0]):
    for i in range(X.shape[1]):
        out.append(y_kmeans[cl])


f, (ax1, ax2) = plt.subplots(2, 1)
ax1.set_title('Illustration of clustering algorithm')
ax1.set_ylabel('V (pu)')
ax2.set_ylabel('Cluster No.')
ax2.set_xlabel('Sample No.')
#ax2.set_ylim(0,5)
ax1.plot(drelangle)
ax2.plot(out)
ax1.grid(True)
ax2.grid(True)
plt.show()
"""


####

########################


"""
##########
# experimenting with dynamic time warping
import numpy as np
from scipy.spatial.distance import euclidean

from fastdtw import fastdtw

relangle = abs(np.unwrap(angle1) - np.unwrap(angle2))
# mean normalization 
relangleScaled = preprocessing.scale(relangle)

# part where data is scaled
scaler = preprocessing.MinMaxScaler(feature_range=(0,1)) # feature to scale the values
#x = np.array(x).reshape((len(x),)) # convert to numpy array
#x=np.log(x) # preferred by the author, dont know the exact reason behind this
#x=x.reshape(-1,1) # makes x a 2D array (required by scaler)
x = np.array(relangle).reshape(-1,1)
x = scaler.fit_transform(x) # scaled
relangleScaledU = x.reshape(-1)



totalSamples = 1500
window = 30
num = totalSamples/window

for i in range(num):
    print('x: {}:{}'.format(window*i,window*(i+1)))
    print('y: {}:{}'.format(window*(i+1),window*(i+2)))
    x = relangle[window*i:window*(i+1)]
    y = relangle[window*(i+1):window*(i+2)]
    distance, path = fastdtw(x, y, dist=euclidean)
    euc_dist = LA.norm(x-y)
    print('DTW distance:{} '.format(distance))
    print('Euclidean distance:{}'.format(euc_dist))


#plt.plot(relangle[:1000])

plt.plot(relangleScaledU)
plt.grid()
plt.show()
############
"""





################
"""
# clustering algorihtm tested on the voltage data
sampleV = np.array(df.Austin_V1LPM_Magnitude) #you can also use df['Austin_V1LPM_Magnitude']
sampleVArray = np.array(sampleV).reshape(-1,30) # 30 features
#thresh = 100
thresh = 140
clusters = hcluster.fclusterdata(sampleVArray, thresh, criterion="distance")

with open('c.txt','w') as f:
    for i in range(clusters.shape[0]):
        f.write('{}'.format(clusters[i]))
        f.write('\n')


eventDict = {}
for ind in range(clusters.shape[0]):
    cl = clusters[ind]
    if cl != 1:
        if cl not in eventDict:
            eventDict[cl] = [ind]
        else:
            eventDict[cl].append(ind)

for cl in eventDict:
    tw = np.array(sampleVArray[eventDict[cl]]).reshape(-1)
    plt.plot(tw)
plt.grid()
plt.show()
"""

"""
for i in range(10):
    tw = np.array(sampleVArray[i]).reshape(-1)
    plt.plot(tw)
plt.grid()
plt.show()
"""
#plt.hist(clusters,bins = 'auto')
#plt.show()

"""
# histograms to get an idea of the threshold choice
distance = []
for i in range(sampleVArray.shape[0]-1):
    distance.append(LA.norm(sampleVArray[i]-sampleVArray[i+1]))

#plt.hist(distance,bins = 'auto')
#plt.show()
dArray = np.array(distance)
print dArray.mean()
print dArray.std()
####
"""
###############
