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
#filename = 'Data141123.csv'
filename = 'Data141128.csv' # 6 anomalies noted
#filename = 'Data141124.csv'
signal = 'IAWPM_Magnitude'
#signal = 'IBWPM_Magnitude'
# just test a sample
testcomedata = pd.read_csv(filename,names = column_names)
x=testcomedata[signal] != 0.0 # indices where the value is zero

comedProcessed= testcomedata[x]
currentA = comedProcessed[signal].values
currentApu = currentA/currentA.mean()
dcurrentApu = np.gradient(currentApu)
#angle = np.unwrap(angle)
#dangle = np.gradient(currentA)
t = comedProcessed.Seconds.values/3600

plt.plot(t,dcurrentApu)
plt.xlabel('time (h)')
plt.ylabel('dCurrent/dt(normalized)')
plt.grid()
plt.show()
###



"""
####

# find anomalous windows using thresholds
minThreshold = 0.98
maxThreshold = 1.02

minThreshold = 0.95
maxThreshold = 1.05


windowsize = 300

# scan all the current data using thresholds
filedict = {}
fileList = os.listdir('.')

import re
s = re.compile(r'I[ABC]WPM_Magnitude') # this will only return true if the signal is a voltage quantity
for filename in fileList:

    if not filename.endswith('.csv') or filename == 'header.csv':
        continue
    print('Reading file: {}'.format(filename))

    #filename = 'Data141123.csv'

    testcomedata = pd.read_csv(filename,names = column_names)
     
    x=testcomedata['IAWPM_Magnitude'] != 0.0 # indices where the value is zero

    comedProcessed= testcomedata[x]
    t = comedProcessed.Seconds.values/3600


    ### test the comed voltage data

    # gather time windows where the min/max exceed a certain threshold
    anomalydict = {}

    for key in comedProcessed:
        m = s.match(key)
        if m:

            # get the voltages and normalize according to the mean
            v = comedProcessed[key].values
            meanV = v.mean()
            vpu = v/meanV

            # window the voltage data
            
            cap = len(vpu)%windowsize
            vpu_windowed = vpu[:-cap].reshape(-1,windowsize)
            t_windowed = t[:-cap].reshape(-1,windowsize)



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

####
"""