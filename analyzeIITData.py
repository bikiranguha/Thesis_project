import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from EventDetectPCAFn import eventdetectPCA
import scipy.cluster.hierarchy as sch
import os

# use regex to match keys in the dictionary
import re
s = re.compile(r'SB[L|P]_V[ABC]{2}_m') # this will only return true if the signal is a voltage quantity


class anomalies():
    def __init__(self, V, t):
        self.V = V
        self.t = t
"""
####
# scan through all the voltage data in the IIT files and detect anomalies
filedict = {}
fileList = os.listdir('.')
for filename in fileList:
    if not filename.endswith('.csv'):
        continue

    anomalydict = {}
    column_names = ['time', 'SBL_VAB_m', 'SBL_VAB_a', 'SBL_VBC_m', 'SBL_VBC_a', 'SBL_VCA_m', 'SBL_VCA_a', 'SBL_IA_m', 'SBL_IA_a', 'SBL_IB_m', 'SBL_IB_a', 'SBL_IC_m', 'SBL_IC_a',
                        'SBP_VAB_m', 'SBP_VAB_a', 'SBP_VBC_m', 'SBP_VBC_a', 'SBP_VCA_m', 'SBP_VCA_a', 'SBP_IA_m', 'SBP_IA_a', 'SBP_IB_m', 'SBP_IB_a', 'SBP_IC_m', 'SBP_IC_a']
    iitdata = pd.read_csv(filename, names = column_names)

    t = iitdata.time.values[:-1]/1000

    # ######
    # # some visualization of a couple of angles
    # SBL_VAB_a = np.unwrap(iitdata.SBL_VAB_a.values[:-1])
    # SBL_VBC_a = np.unwrap(iitdata.SBL_VBC_a.values[:-1])
    # relangle = SBL_VAB_a - SBL_VBC_a
    # drelangle = np.gradient(relangle)
    # 
    # plt.plot(t,drelangle)
    # plt.grid()
    # plt.show()
    # ######



    # gather time windows where the min/max exceed a certain threshold
    minThreshold = 0.96
    maxThreshold = 1.04

    minThreshold = 0.95
    maxThreshold = 1.05
    for key in iitdata:
        m = s.match(key)
        if m:

            # get the voltages and normalize according to the mean
            v = iitdata[key].values[:-1]
            meanV = v.mean()
            vpu = v/meanV

            # window the voltage data
            windowSize = 300
            cap = len(vpu)%300
            vpu_windowed = vpu[:-cap].reshape(-1,300)
            t_windowed = t[:-cap].reshape(-1,300)



            suspectedAnomalies = []
            anomalyTimes = []
            for i in range(len(vpu_windowed)):
                currentWindow = vpu_windowed[i]
                currentWindowMax = currentWindow.max()
                currentWindowMin = currentWindow.min()
                if currentWindowMin < minThreshold or currentWindowMax > maxThreshold:
                    suspectedAnomalies.append(currentWindow)
                    anomalyTimes.append(t_windowed[i])

            suspectedAnomalies = np.array(suspectedAnomalies)
            anomalyTimes = np.array(anomalyTimes)
            anomalydict[key] = anomalies(suspectedAnomalies,anomalyTimes)
    filedict[filename] = anomalydict


for file in filedict:
    print('In file {}'.format(file))
    print('\n')
    anomalydict = filedict[file]
    for key in anomalydict:
        v = anomalydict[key].V
        print('In {}, number of anomalies detected: {}'.format(key,len(v)))
    print('\n')
###########
"""




column_names = ['time', 'SBL_VAB_m', 'SBL_VAB_a', 'SBL_VBC_m', 'SBL_VBC_a', 'SBL_VCA_m', 'SBL_VCA_a', 'SBL_IA_m', 'SBL_IA_a', 'SBL_IB_m', 'SBL_IB_a', 'SBL_IC_m', 'SBL_IC_a',
                    'SBP_VAB_m', 'SBP_VAB_a', 'SBP_VBC_m', 'SBP_VBC_a', 'SBP_VCA_m', 'SBP_VCA_a', 'SBP_IA_m', 'SBP_IA_a', 'SBP_IB_m', 'SBP_IB_a', 'SBP_IC_m', 'SBP_IC_a']
iitdata = pd.read_csv('01_26_SB.csv', names = column_names)
#iitdata = pd.read_csv('01_29_SB.csv', names = column_names)
t = iitdata.time.values[:-1]/(1000*3600) # time is now in hours




# v = iitdata.SBL_VAB_m.values[:-1]
# meanV = v.mean()
# vpu = v/meanV

# plt.plot(t,vpu)
# plt.title('Voltage on 01_29')
# plt.xlabel('Time(s)')
# plt.ylabel('V (normalized)')
# plt.grid()
# plt.show()



# suspectedAnomalies = []
# anomalyTimes = []
# for i in range(len(SBL_VAB_pu_windowed)):
#     currentWindow = SBL_VAB_pu_windowed[i]
#     currentWindowMax = currentWindow.max()
#     currentWindowMin = currentWindow.min()
#     if currentWindowMin < minThreshold or currentWindowMax > maxThreshold:
#         suspectedAnomalies.append(currentWindow)
#         anomalyTimes.append(t_windowed[i])

# suspectedAnomalies = np.array(suspectedAnomalies).reshape(-1)
# anomalyTimes = np.array(anomalyTimes).reshape(-1)
# plt.plot(suspectedAnomalies)
# plt.grid()
# plt.show()





#analyze the voltage for any abnormalities
SBL_VAB_m = iitdata.SBL_VAB_m.values[:-1]
meanV = SBL_VAB_m.mean()
SBL_VAB_pu = SBL_VAB_m/meanV

# window the voltage data
windowSize = 10000
cap = len(SBL_VAB_pu)%windowSize
SBL_VAB_pu_windowed = SBL_VAB_pu[:-cap].reshape(-1,windowSize)
t_windowed = t[:-cap].reshape(-1,windowSize)


# ## apply pca event detection and see the effects
# yPredSteady, abnormalTimeInd  = eventdetectPCA(t,SBL_VAB_pu,0,10000,0.01)
# plt.plot(yPredSteady)
# plt.grid()
# plt.show()
##

#Using the dendrogram to find the optimal number of clusters or the optimal threshold
# import scipy.cluster.hierarchy as sch
# X = SBL_VAB_pu_windowed

# distList = []
# for i in range(len(X)):
#     for j in range(len(X)):
#         dist = np.linalg.norm(X[i]-X[j])
#         distList.append(dist)

# plt.hist(distList,bins = 'auto')
# plt.show()

# dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
# plt.title('Dendrogram')
# plt.xlabel('Voltage windows')
# plt.ylabel('Euclidean distances')
# plt.show()


#plt.plot(X.reshape(-1))
#plt.ylim(0,1.2)
#plt.grid()
#plt.show()

# # Fitting Hierarchical Clustering to the dataset using agglomerative clustering
# from sklearn.cluster import AgglomerativeClustering
# X = SBL_VAB_pu_windowed
# hc = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage = 'ward')
# y_hc = hc.fit_predict(X)


# # evaluate hierarchical clustering using thresholds
import scipy.cluster.hierarchy as sch
X = SBL_VAB_pu_windowed
#threshold = 0.04 # use for a window of 300
#threshold = 0.075 # use for window of 1000
threshold = 0.35
y_hc = sch.fclusterdata(X, threshold, criterion="distance")
#y_hc = sch.fclusterdata(X, threshold)
x = []
for cl in range(y_hc.shape[0]):
    for i in range(X.shape[1]):
        x.append(y_hc[cl])

# mark all the time windows
wEndPt = range(0,len(t[:-cap]),windowSize)
spclTimePts = []
spclVoltPts = []
Xflat = X.reshape(-1)
for pt in wEndPt:
    spclTimePts.append(t[pt])
    spclVoltPts.append(Xflat[pt])




# getting the predictions in a way so that we can plot
out = []
for cl in range(y_hc.shape[0]):
    for i in range(X.shape[1]):
        out.append(y_hc[cl])

# Visualizing the performance
f, (ax1, ax2) = plt.subplots(2, 1)
ax1.set_title('Illustration of clustering algorithm using threshold')
ax1.set_ylabel('V (normalized)')
ax2.set_ylabel('Cluster No.')
ax2.set_xlabel('Time (hours).')
#ax2.set_ylim(0,5)
#ax1.set_xlim(17,17.30)
#ax2.set_xlim(17,17.30)
ax1.plot(t_windowed.reshape(-1),Xflat)
ax1.plot(spclTimePts,spclVoltPts,ls="", marker="o")
ax2.plot(t_windowed.reshape(-1),out)
ax1.grid(True)
ax2.grid(True)
#ax2.set_ylim(-0.5,1.5)
plt.show()









