# script to plot the dv_dt values for cases in given file (with class separation)
# marks at placed at the intervals used for input and target for the machine learning algorithm

import matplotlib.pyplot as plt
import pickle
import numpy as np

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)




# Inputs needed
fault_clearance_time = 0.31
inputFeatureNo = 60
steadyStateSamples = 100
#inputFile = 'CaseV.txt' # classification of V
#inputFile = 'Casedvdt.txt' # classification of dv dt
inputFile = 'CasedvdtTmp.txt'

# make directories
import os
currentdir = os.getcwd()


# create directories for the false positives and false negatives
errorPlotDir = currentdir + '/Visualizations'
# for V
#class0Dir = errorPlotDir + '/class0PlotsV'
#class1Dir = errorPlotDir + '/class1PlotsV'
# for dv_dt visualization
#class0Dir = errorPlotDir + '/class0Plotsdvdt'
#class1Dir = errorPlotDir + '/class1Plotsdvdt'

# just a temp dv_dt threshold test
class0Dir = errorPlotDir + '/class0PlotsdvdtTmpdvdt'
class1Dir = errorPlotDir + '/class1PlotsdvdtTmpdvdt'
##############

# create directories if they dont exist
if not os.path.isdir(errorPlotDir):
    os.mkdir(errorPlotDir)
if not os.path.isdir(class0Dir):
    os.mkdir(class0Dir)

if not os.path.isdir(class1Dir):
    os.mkdir(class1Dir)



# read the file
with open(inputFile,'r') as f:
    fileLines = f.read().split('\n')

fpStartIndex = fileLines.index('Class 0:') + 1
fpEndIndex = fileLines.index('Class 1:')
fnStartIndex = fpEndIndex + 1

# get the false positive cases
class0Lines = []
class1Lines = []
for i in range(fpStartIndex,fpEndIndex):
    line = fileLines[i]
    class0Lines.append(line)

# get the false positive cases
for i in range(fnStartIndex,len(fileLines)):
    line = fileLines[i]
    if line == '':
        continue
    class1Lines.append(line)





VoltageDataDict = load_obj('VoltageData') # this voltage data dictionary has been generated from the script 'TS3phSimN_2FaultDataSave.py', see it for key format

# save all the class 1 data
k=1
for key in class1Lines:
    #key = '154,3008,1;151,201,1;F151/151'
    words = key.split('/')
    event = words[0].strip()
    Bus = words[1].strip()
    voltage = VoltageDataDict[key] 
    tme = VoltageDataDict['time']
    timestep = tme[1] - tme[0]
    VmagSize = voltage.shape[0]
    dv_dt = np.zeros(VmagSize) # initialize dv_dt array with all zeros
    for i in range(VmagSize):
        try:
            dv_dt[i] = abs((voltage[i] - voltage[i-1]))/timestep
        except: # will happen if i = 0, since there is no i-1
            continue




    ind_input_start = int(fault_clearance_time/timestep) + 1 # time index when fault is cleared
    ind_input_end = ind_input_start + inputFeatureNo # end time index for ML input

    ind_target_start = len(tme) - steadyStateSamples

    # get the special co-ordinates to mark
    spclTimePts = []
    spcldvdtplots = []

    spclTimePts.append(tme[ind_input_start])
    spcldvdtplots.append(dv_dt[ind_input_start])

    spclTimePts.append(tme[ind_input_end])
    spcldvdtplots.append(dv_dt[ind_input_end])

    spclTimePts.append(tme[ind_target_start])
    spcldvdtplots.append(dv_dt[ind_target_start])

    #timeStep = range(len(voltageValues))
    plt.plot(tme, dv_dt)
    plt.plot(spclTimePts,spcldvdtplots, ls="", marker="o", label="special points")
    titleStr = 'Event: ' + event +  'Bus ' + Bus
    plt.title(titleStr)
    plt.ylabel('dv_dt (pu)')
    plt.xlabel('Time (s)')
    plt.ticklabel_format(useOffset=False)
    #plt.xlabel('Time step after line clearance')
    plt.ylim(0.0,0.1)
    plt.grid()
    plt.legend()
    #plt.show()
    figName = class1Dir + '/' + 'Plot' + str(k) + '.png'
    plt.savefig(figName)
    plt.close()
    k+=1





"""

# generate random 100  samples from the class 0 data (since there are too many to plot)
k=1
import random
randomInd = [] # random sample index for the class 0 data
for i in range(100):
    out = random.choice(range(len(class0Lines)))
    randomInd.append(out)
# save all the class 0 data
for j in range(len(randomInd)):
    key = class0Lines[randomInd[j]]
    #key = '154,3008,1;151,201,1;F151/151'
    words = key.split('/')
    event = words[0].strip()
    Bus = words[1].strip()
    voltage = VoltageDataDict[key] 
    tme = VoltageDataDict['time']
    timestep = tme[1] - tme[0]
    ind_input_start = int(fault_clearance_time/timestep) + 1 # time index when fault is cleared
    ind_input_end = ind_input_start + inputFeatureNo # end time index for ML input

    ind_target_start = len(tme) - steadyStateSamples

    # get the special co-ordinates to mark
    spclTimePts = []
    spclVoltPts = []

    spclTimePts.append(tme[ind_input_start])
    spclVoltPts.append(voltage[ind_input_start])

    spclTimePts.append(tme[ind_input_end])
    spclVoltPts.append(voltage[ind_input_end])

    spclTimePts.append(tme[ind_target_start])
    spclVoltPts.append(voltage[ind_target_start])

    #timeStep = range(len(voltageValues))
    plt.plot(tme, voltage)
    plt.plot(spclTimePts,spclVoltPts, ls="", marker="o", label="special points")
    titleStr = 'Event: ' + event +  'Bus ' + Bus
    plt.title(titleStr)
    plt.ylabel('Voltage (pu)')
    plt.xlabel('Time (s)')
    plt.ticklabel_format(useOffset=False)
    #plt.xlabel('Time step after line clearance')
    #plt.ylim(0.75,1.5)
    plt.grid()
    plt.legend()
    #plt.show()
    figName = class0Dir + '/' + 'Plot' + str(k) + '.png'
    plt.savefig(figName)
    plt.close()
    k+=1
"""
