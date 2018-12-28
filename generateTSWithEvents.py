# script to various events (faults, line outage, gen outage, tf outage, bad transformer) into the synchrophasor voltage data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from avgFilterFn import avgFilter
import csv


def getEventDict(dataFileName, eventKeyFile):
    # function to return an organized  outage event dict from the given filedata

    dataFile = open(dataFileName,'rb')
    reader=csv.reader(dataFile,quoting=csv.QUOTE_NONNUMERIC) # so that the entries are converted to floats



    # make an organized dictionary
    # read the event file
    eventList = []
    with open(eventKeyFile,'r') as f:
        fileLines = f.read().split('\n')
        for line in fileLines[1:]:
            if line == '':
                continue
            eventList.append(line.strip())


    SimpleEventDict = {}
    for idx, row in enumerate(reader):
        eventKey = eventList[idx]
        SimpleEventDict[eventKey] = row


    return SimpleEventDict
###########




csv_file = '120103,010000000,UT,Austin,3378,Phasor.csv'

# get the voltage data
df = pd.read_csv(csv_file)
# normalize the voltage data
v = np.array(df.Austin_V1LPM_Magnitude)
v = v/v.mean()

# generate 5 hours of data from 1 hour of data
# concatenate the same waveform to make longer timeseries
v = np.concatenate((v,v,v,v,v))
t = np.array(range(v.shape[0]))/30.0



# visualize the data
# plt.plot(v)
# plt.grid()
# plt.ylim(0.7,1.1)
# #plt.ylim(70000,80000)
# plt.show()


# ### get the time data
# dttme = list(df.Timestamp)
# def getSec(dtTmeString):
#     splt = dtTmeString.split()
#     tme = splt[1]
#     tmesplt = tme.strip().split(':')
#     hour = float(tmesplt[0])
#     mnt = float(tmesplt[1])
#     sec = float(tmesplt[2])
#     totsec = hour*3600+ mnt*60 + sec
#     #return hour, mnt, sec  
#     return totsec 

# # convert the datetime object to seconds
# tmeSec = []

# startSec =  getSec(dttme[0])

# for t in dttme:
#     currentSec = getSec(t)
#     relSec = currentSec - startSec
#     tmeSec.append(relSec)

###


# put a spike somewhere in the timeseries
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




# generate a transformer failure TS


#def tfFailure(distStart,distSamples,dropV, originalSignal):
def tfFailure(startTime,endTime,dropV, originalSignal):  

    # calculate the starting sample and the number of samples needed
    distStart = startTime*30
    distEnd = endTime*30
    distSamples = int((distEnd - distStart))


    #numSineWaves = 100
    numSineWaves = 20 # number to sine waves to add to get the final timeseries during the dip
    #dropV = 0.01 # drop in the voltage
    #dropVariance = 0.125 # ratio of variance wrt drop
    dropVariance = 0.5 # ratio of max variance of the sine waves with respect to drop
    #frequencyRange = np.linspace(1,5,10)
    frequencyRange = np.linspace(1,20,10)
    #frequencyRange = np.linspace(1,100,10)

    #distSamples = 10000
    dist = np.zeros(distSamples)
    dist = dist - dropV 
    wave = np.zeros(distSamples)
    for i in range(numSineWaves):

        freq = random.choice(frequencyRange)
        Fs = distSamples # sampling frquency
        #sample = 50
        s = np.arange(distSamples)
        y = np.sin(2 * np.pi * freq * s / Fs)
        wave += y

    waveRange = wave.max() - wave.min()
    wave = wave/waveRange*dropV*dropVariance
    dist += wave

    z = np.zeros(originalSignal.shape[0])
    z[distStart:distStart+dist.shape[0]] = dist
    # apply smoothing on the disturbance and noise
    z  =np.array(avgFilter(z,100))
    z += np.random.normal(0,0.001,z.shape[0])
    newSignal = originalSignal + z
    return newSignal



newV = tfFailure(1000,1150,0.01, v)
# plt.plot(t,newV)
# plt.grid()
# plt.ylim(0.2,1.1)
# plt.show()



###
## add a spike
newV = spike(15000, 15001, newV, 0.1)







# add random faults, generator outage and line outages from the simulations i have


### get the randomly sampled fault data
faultDir = 'G:/My Drive/My PhD research/Running TS3ph/fault3ph/Long'
vFileName = '{}/vData3phLISamples.csv'.format(faultDir) # csv file containing voltage data (different types of fault)
eventKeyFile = '{}/sampleVEventID.txt'.format(faultDir)

# read the event file
eventList = []
with open(eventKeyFile,'r') as f:
    fileLines = f.read().split('\n')
    for line in fileLines[1:]:
        if line == '':
            continue
        eventList.append(line.strip())


FaultDict = {}
vFile = open(vFileName,'rb')
readerV=csv.reader(vFile,quoting=csv.QUOTE_NONNUMERIC) # so that the entries are converted to floats
minList = []
for idx, row in enumerate(readerV):
    minval = np.array(row).min()
    minList.append(minval)
    eventKey = eventList[idx]
    FaultDict[eventKey] = row





sampleKey = random.choice(FaultDict.keys())
faultSample = np.array(FaultDict[sampleKey])


# shift the fault voltage so that prefault voltage is 1.0 pu
precontvolt = faultSample[0]
faultSample += (1.0-precontvolt)
sampleLength = faultSample.shape[0]
# add noise 
faultSample += np.random.normal(0,0.001,faultSample.shape[0])
# add smoothing
#faultSample = avgFilter(faultSample,6)

startTime = 6000
startSample = startTime*30

newV[startSample:startSample+sampleLength] = faultSample
# plt.plot(t,newV)
# plt.grid()
# plt.ylim(0.2,1.1)
# plt.show()
####

### add generator outage data
GenOutDict = {}
genoutDir = 'G:/My Drive/My PhD research/Running TS3ph/GenOut'
vFileName = '{}/vGenOut.csv'.format(genoutDir) # csv file containing voltage data (different types of fault)
eventKeyFile = '{}/eventGenOut.txt'.format(genoutDir)

GenOutDict =  getEventDict(vFileName, eventKeyFile)

sampleKey = random.choice(GenOutDict.keys())
genoutSample = np.array(GenOutDict[sampleKey])


# shift the fault voltage so that prefault voltage is 1.0 pu
precontvolt = genoutSample[0]
genoutSample += (1.0-precontvolt)
sampleLength = genoutSample.shape[0]
# add noise 
genoutSample += np.random.normal(0,0.001,genoutSample.shape[0])
# add smoothing
#genoutSample = avgFilter(genoutSample,6)

startTime = 4000
startSample = startTime*30


newV[startSample:startSample+sampleLength] = genoutSample
endSample = startSample+sampleLength
# try to smooth the transition
transitionWindow = newV[startSample-1000:endSample+1000]
transitionWindow = np.array(avgFilter(transitionWindow,100))
transitionWindow += np.random.normal(0,0.001,transitionWindow.shape[0])
#newV[startSample-1000:endSample+1000] = transitionWindow
newV[endSample:endSample+1000] = transitionWindow[-1000:] # only incorporate the transition after the event ends




# plt.plot(t,newV)
# plt.grid()
# plt.ylim(0.2,1.1)
# plt.show()
######


### add line outage data
LineOutDict = {}
lineOutDir = 'G:/My Drive/My PhD research/Running TS3ph/LineOut'
vFileName = '{}/vLineOut.csv'.format(lineOutDir) # csv file containing voltage data (different types of fault)
eventKeyFile = '{}/eventLineOut.txt'.format(lineOutDir)

LineOutDict =  getEventDict(vFileName, eventKeyFile)

sampleKey = random.choice(LineOutDict.keys())
lineoutsample = np.array(LineOutDict[sampleKey])


# shift the fault voltage so that prefault voltage is 1.0 pu
precontvolt = lineoutsample[0]
lineoutsample += (1.0-precontvolt)
sampleLength = lineoutsample.shape[0]

# add noise 
lineoutsample += np.random.normal(0,0.001,lineoutsample.shape[0])
# add smoothing
#lineoutsample = avgFilter(lineoutsample,6)

startTime = 12000
startSample = startTime*30


newV[startSample:startSample+sampleLength] = lineoutsample
endSample = startSample+sampleLength
# try to smooth the transition
transitionWindow = newV[startSample-1000:endSample+1000]
transitionWindow = np.array(avgFilter(transitionWindow,100))
transitionWindow += np.random.normal(0,0.001,transitionWindow.shape[0])
#newV[startSample-1000:endSample+1000] = transitionWindow
newV[endSample:endSample+1000] = transitionWindow[-1000:] # only incorporate the transition after the event ends


# plt.plot(t,newV)
# plt.grid()
# plt.ylim(0.2,1.1)
# plt.show()
######




### add tf outage data
TFOutDict = {}
TFOutDir = 'G:/My Drive/My PhD research/Running TS3ph/TFOut'
vFileName = '{}/vTFOut.csv'.format(TFOutDir) # csv file containing voltage data (different types of fault)
eventKeyFile = '{}/eventTFOut.txt'.format(TFOutDir)

TFOutDict =  getEventDict(vFileName, eventKeyFile)

sampleKey = random.choice(TFOutDict.keys())
TFOutsample = np.array(TFOutDict[sampleKey])


# shift the fault voltage so that prefault voltage is 1.0 pu
precontvolt = TFOutsample[0]
TFOutsample += (1.0-precontvolt)
sampleLength = TFOutsample.shape[0]

# add noise 
TFOutsample += np.random.normal(0,0.001,TFOutsample.shape[0])
# add smoothing
#TFOutsample = avgFilter(TFOutsample,6)

startTime = 8000
startSample = startTime*30


newV[startSample:startSample+sampleLength] = TFOutsample
endSample = startSample+sampleLength
# try to smooth the transition
transitionWindow = newV[startSample-1000:endSample+1000]
transitionWindow = np.array(avgFilter(transitionWindow,100))
transitionWindow += np.random.normal(0,0.001,transitionWindow.shape[0])
#newV[startSample-1000:endSample+1000] = transitionWindow
newV[endSample:endSample+1000] = transitionWindow[-1000:] # only incorporate the transition after the event ends


plt.plot(t,newV)
plt.grid()
plt.ylim(0.2,1.1)
plt.show()
######

# write the generated voltage sample into a csv file for analysis
outFile = 'sampleVAnomalyDetection.csv'



outFile = open(outFile, 'wb')
writerObjVTFOut = csv.writer(outFile)


writerObjVTFOut.writerow(newV)

outFile.close()