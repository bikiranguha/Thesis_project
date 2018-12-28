# visualize the anomalous part in the current data using red highlights

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


class anomalies():
    def __init__(self, V, t):
        self.V = V
        self.t = t


# find anomalous windows using thresholds
minThreshold = 0.98
maxThreshold = 1.02


windowsize = 300
filename = '01_26_SB.csv'
column_names = ['time', 'SBL_VAB_m', 'SBL_VAB_a', 'SBL_VBC_m', 'SBL_VBC_a', 'SBL_VCA_m', 'SBL_VCA_a', 'SBL_IA_m', 'SBL_IA_a', 'SBL_IB_m', 'SBL_IB_a', 'SBL_IC_m', 'SBL_IC_a',
                    'SBP_VAB_m', 'SBP_VAB_a', 'SBP_VBC_m', 'SBP_VBC_a', 'SBP_VCA_m', 'SBP_VCA_a', 'SBP_IA_m', 'SBP_IA_a', 'SBP_IB_m', 'SBP_IB_a', 'SBP_IC_m', 'SBP_IC_a']
iitdata = pd.read_csv(filename,names = column_names)

#comedProcessed= iitdata[x]
t = t = iitdata.time.values[:-1]/(1000*3600) # time is now in hours.Seconds.values/3600





### test the iit voltage data

# gather time windows where the min/max exceed a certain threshold
anomalydict = {}

windowsize = 300
key = 'SBL_VAB_m'
# get the current and normalize according to the mean
V = iitdata[key].values[:-1]
meanV = V.mean()
Vpu = V/meanV

# window the voltage data

cap = len(Vpu)%windowsize
Vpu_windowed = Vpu[:-cap].reshape(-1,windowsize)
t_windowed = t[:-cap].reshape(-1,windowsize)



suspectedAnomalies = []
anomalyTimes = []
for i in range(len(Vpu_windowed)):
    currentWindow = Vpu_windowed[i]
    currentWindowMax = currentWindow.max()
    currentWindowMin = currentWindow.min()
    if currentWindowMin < minThreshold or currentWindowMax > maxThreshold:
        suspectedAnomalies.append(currentWindow)
        anomalyTimes.append(t_windowed[i])

suspectedAnomalies = np.array(suspectedAnomalies)
anomalyTimes = np.array(anomalyTimes)
anomalydict[key] = anomalies(suspectedAnomalies,anomalyTimes)



# now plot the whole data, but highlight the anomalous data using some other color, like red
plt.plot(t,Vpu)

# mark all the time windows



anomalousDataArray = anomalydict[key].V
anomalousTimeArray = anomalydict[key].t

# anomalousDataList = list(anomalousDataArray.reshape(-1))
# anomalousTimeList = list(anomalousTimeArray.reshape(-1))



for i in range(len(anomalousDataArray)):
    data = anomalousDataArray[i]
    t = anomalousTimeArray[i]
    plt.plot(t,data,color = 'red')


#plt.plot(anomalousTimeList,anomalousDataList, color = 'red')
#plt.plot(anomalousTimeList,anomalousDataList, linestyle = '', marker = 'o')
plt.title('Voltage plot 01_26_SB iit')
plt.xlabel('Time (h)')
plt.ylabel('SBL_VAB_m (normalized)')
plt.grid()
plt.show()