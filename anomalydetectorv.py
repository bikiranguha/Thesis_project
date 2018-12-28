import os
import pandas as pd
import numpy as np
# use regex to match keys in the dictionary
import re
s = re.compile(r'SB[L|P]_V[ABC]{2}_m') # this will only return true if the signal is a voltage quantity


# gather time windows where the min/max exceed a certain threshold

minThreshold = 0.96
maxThreshold = 1.04

timewindowSteps = 300

class anomalies():
    def __init__(self, V, t):
        self.V = V
        self.t = t



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

    t = iitdata.time.values[:-1]/(1000*3600)





    for key in iitdata:
        m = s.match(key)
        if m:

            # get the voltages and normalize according to the mean
            v = iitdata[key].values[:-1]
            meanV = v.mean()
            vpu = v/meanV

            # group the voltage data into windows
            cap = len(vpu)%timewindowSteps
            vpu_windowed = vpu[:-cap].reshape(-1,timewindowSteps)
            t_windowed = t[:-cap].reshape(-1,timewindowSteps)



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


# output the results
for file in filedict:
    print('In file {}'.format(file))
    print('\n')
    anomalydict = filedict[file]
    for key in anomalydict:
        v = anomalydict[key].V
        print('In {}, number of anomalies detected: {}'.format(key,len(v)))
    print('\n')
###########
