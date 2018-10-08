# read the false positive and false negative errrors from the specified text file
# generate plots for the errors and save them in an organized fashion
import matplotlib.pyplot as plt
errorsFile = 'MLErrors.txt'
fpLines = []
fNLines = []
with open(errorsFile,'r') as f:
    fileLines = f.read().split('\n')

fpStartIndex = fileLines.index('False positives:') + 1
fpEndIndex = fileLines.index('False negatives:')
fnStartIndex = fpEndIndex + 1

# get the false positive cases
for i in range(fpStartIndex,fpEndIndex):
    line = fileLines[i]
    fpLines.append(line)

# get the false positive cases
for i in range(fnStartIndex,len(fileLines)):
    line = fileLines[i]
    if line == '':
        continue
    fNLines.append(line)


import os
currentdir = os.getcwd()

# create directories for the false positives and false negatives
errorPlotDir = currentdir + '/ErrorPlots'
fpDir = errorPlotDir + '/fpPlots'
fnDir = errorPlotDir + '/fnPlots'

if not os.path.isdir(errorPlotDir):
    os.mkdir(errorPlotDir)
if not os.path.isdir(fpDir):
    os.mkdir(fpDir)

if not os.path.isdir(fnDir):
    os.mkdir(fnDir)

# get the voltage dictionary
import pickle
def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

VoltageDataDict = load_obj('VoltageData') # this voltage data dictionary has been generated from the script 'TS3phSimN_2FaultDataSave.py', see it for key format

# carry out the false positive plots

k = 1
for line in fpLines:

    words = line.split('/')
    event = words[0].strip()
    Bus = words[1].strip()
    voltageStr = VoltageDataDict[line]
    voltageStrList = voltageStr.split(',')
    voltageValues = [float(i) for i in voltageStrList]
    """
    if line == '151,201,1;151,152,1;F152/154':
        tme = VoltageDataDict['time']
        words = tme.split(',')
        tme = [float(i) for i in words]
        timestep = tme[1] - tme[0]
        time_1s = int(1.0/timestep)
        steadyV =   voltageValues[time_1s:time_1s+100]
        abnormalVList = [v for v in steadyV if v< 0.9 or v> 1.1]
        print abnormalVList
    """


    """
    timeStep = range(len(voltageValues))
    plt.plot(timeStep, voltageValues)
    titleStr = 'Event: ' + event +  'Bus ' + Bus
    plt.title(titleStr)
    plt.ylabel('Voltage (pu)')
    plt.ticklabel_format(useOffset=False)
    plt.xlabel('Time step after line clearance')
    plt.ylim(0.75,1.5)
    plt.grid()
    figName = fpDir + '/' + 'Plot' + str(k) + '.png'
    plt.savefig(figName)
    plt.close()
    k+=1
    """

"""
# carry out the false positive plots
k = 1
for line in fNLines:
    
    words = line.split('/')
    event = words[0].strip()
    Bus = words[1].strip()
    voltageStr = VoltageDataDict[line]
    voltageStrList = voltageStr.split(',')
    voltageValues = [float(i) for i in voltageStrList]
    timeStep = range(len(voltageValues))
    plt.plot(timeStep, voltageValues)
    titleStr = 'Event: ' + event +  'Bus ' + Bus
    plt.title(titleStr)
    plt.ylabel('Voltage (pu)')
    plt.ticklabel_format(useOffset=False)
    plt.xlabel('Time step after line clearance')
    plt.ylim(0.75,1.5)
    plt.grid()
    figName = fnDir + '/' + 'Plot' + str(k) + '.png'
    plt.savefig(figName)
    plt.close()
    k+=1
"""