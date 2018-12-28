import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
from avgFilterFn import avgFilter
# # calculate the starting sample and the number of samples needed
# distStart = startTime*30
# distEnd = endTime*30
# distSamples = int((distEnd - distStart))

csv_file = '120103,010000000,UT,Austin,3378,Phasor.csv'

# get the voltage data
df = pd.read_csv(csv_file)
# normalize the voltage data
v = np.array(df.Austin_V1LPM_Magnitude)
v = v/v.mean()



def spike(startTime, endTime, originalSignal, spikeHeight):

    #startTime = 1000
    #endTime = 1001

    # calculate the starting sample and the number of samples needed
    distStart = startTime*30
    distEnd = endTime*30
    distSamples = int((distEnd - distStart))

    #numSineWaves = 100
    numSineWaves = 20 # number to sine waves to add to get the final timeseries during the dip
    #dropV = 0.01 # drop in the voltage
    #dropVariance = 0.125 # ratio of variance wrt drop
    dropVariance = 0.5 # ratio of max variance of the sine waves with respect to drop
    frequencyRange = np.linspace(1,5,10)
    #frequencyRange = np.linspace(1,20,10)
    #frequencyRange = np.linspace(1,100,10)

    #dist = np.zeros(distSamples)
    #dist = dist - dropV 
    spike = np.zeros(distSamples)
    for i in range(numSineWaves):

        freq = random.choice(frequencyRange)
        Fs = distSamples # sampling frquency
        #sample = 50
        s = np.arange(distSamples)
        y = np.sin(2 * np.pi * freq * s / Fs)
        spike += y



    #waveRange = spike.max() - spike.min()
    spike = spike/spike.max()*spikeHeight
    z = np.zeros(originalSignal.shape[0])
    z[distStart:distStart+spike.shape[0]] = spike
    # apply smoothing on the disturbance and noise
    z  =np.array(avgFilter(z,6))
    z += np.random.normal(0,0.001,z.shape[0])
    newSignal = originalSignal + z
    return newSignal


spike(3000, 3001, v, 0.1)