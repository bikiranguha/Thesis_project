import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
class anomalies():
    def __init__(self, V, t):
        self.V = V
        self.t = t

import re
s = re.compile(r'V[ABC]VPM_Magnitude') # this will only return true if the signal is a voltage quantity


# get the header values
headerfile = pd.read_csv('header.csv')
column_names =[]
for h in list(headerfile.columns):
    column_names.append(h.strip("'"))
    #print(h)


#define the thresholds and the window size
minThreshold = 0.98
maxThreshold = 1.02
windowsize = 300



"""
####
filedict = {}
fileList = os.listdir('.')


for filename in fileList:

    if not filename.endswith('.csv') or filename == 'header.csv':
        continue
    print('Reading file: {}'.format(filename))

    #filename = 'Data141123.csv'

    testcomedata = pd.read_csv(filename,names = column_names)
     
    x=testcomedata['VAVPM_Magnitude'] != 0.0 # indices where the value is zero

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


# filename = 'Data141123.csv'
# testcomedata = pd.read_csv(filename,names = column_names)
# sampleV = testcomedata['VAVPM_Magnitude'].values
# sampleVpu = sampleV/sampleV.mean()
# t = testcomedata.Seconds.values/3600
# plt.plot(t,sampleVpu)
# plt.grid()
# plt.show()