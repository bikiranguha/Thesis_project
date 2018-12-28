# visualize the anomalous part in the current data using red highlights

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


class anomalies():
    def __init__(self, V, t):
        self.V = V
        self.t = t

# get the header values
headerfile = pd.read_csv('header.csv')
column_names =[]
for h in list(headerfile.columns):
    column_names.append(h.strip("'"))
    #print(h)

####

# find anomalous windows using thresholds
minThreshold = 0.98
maxThreshold = 1.02


windowsize = 300
filename = 'Data141128.csv'

testcomedata = pd.read_csv(filename,names = column_names)
 
x=testcomedata['IAWPM_Magnitude'] != 0.0 # indices where the value is zero

comedProcessed= testcomedata[x]
t = comedProcessed.Seconds.values/3600


### test the comed voltage data

# gather time windows where the min/max exceed a certain threshold
anomalydict = {}


key = 'IAWPM_Magnitude'
# get the current and normalize according to the mean
i = comedProcessed[key].values
meanI = i.mean()
ipu = i/meanI

# window the voltage data

cap = len(ipu)%windowsize
ipu_windowed = ipu[:-cap].reshape(-1,windowsize)
t_windowed = t[:-cap].reshape(-1,windowsize)



suspectedAnomalies = []
anomalyTimes = []
for i in range(len(ipu_windowed)):
    currentWindow = ipu_windowed[i]
    currentWindowMax = currentWindow.max()
    currentWindowMin = currentWindow.min()
    if currentWindowMin < minThreshold or currentWindowMax > maxThreshold:
        suspectedAnomalies.append(currentWindow)
        anomalyTimes.append(t_windowed[i])

suspectedAnomalies = np.array(suspectedAnomalies)
anomalyTimes = np.array(anomalyTimes)
anomalydict[key] = anomalies(suspectedAnomalies,anomalyTimes)



# now plot the whole data, but highlight the anomalous data using some other color, like red
plt.plot(t,ipu)

# mark all the time windows



anomalousDataArray = anomalydict[key].V
anomalousTimeArray = anomalydict[key].t

for i in range(len(anomalousDataArray)):
    data = anomalousDataArray[i]
    t = anomalousTimeArray[i]
    plt.plot(t,data,color = 'red')



plt.title('Current plot 141128_11 comed')
plt.xlabel('Time (h)')
plt.ylabel('IAWPM_Magnitude (normalized)')
plt.grid()
plt.show()