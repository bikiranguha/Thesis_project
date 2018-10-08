# plot all the unstable cases (not predicted by the ML classifier but demarcated by us as unstable)
UnstableCaseFile = 'UnstableCaseList.txt'

with open(UnstableCaseFile,'r') as f:
    fileLines = f.read().split('\n')


unstableCases = []
for line in fileLines:
    if line == '':
        continue
    unstableCases.append(line)

import os
currentdir = os.getcwd()
PlotDir = currentdir + '/UnstablePlots'

if not os.path.isdir(PlotDir):
    os.mkdir(PlotDir)

# get the voltage dictionary
import pickle
def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

VoltageDataDict = load_obj('VoltageData') # this voltage data dictionary has been generated from the script 'TS3phSimN_2FaultDataSave.py', see it for key format

# get the time array
tme = VoltageDataDict['time']
words = tme.split(',')
tme = [float(i) for i in words]

timestep = tme[1] - tme[0]
time_start = int(0.31/timestep) # in the simulations, the lines were cleared at 0.31 sec
tme = tme[time_start:]



import matplotlib.pyplot as plt

k = 0
for line in unstableCases:

    words = line.split('/')
    event = words[0].strip()
    Bus = words[1].strip()
    voltageStr = VoltageDataDict[line]
    voltageStrList = voltageStr.split(',')
    voltageValues = [float(i) for i in voltageStrList]




    #timeStep = range(len(voltageValues))
    plt.plot(tme[:len(voltageValues)], voltageValues)
    titleStr = 'Event: ' + event +  'Bus ' + Bus
    plt.title(titleStr)
    plt.ylabel('Voltage (pu)')
    plt.ticklabel_format(useOffset=False)
    plt.xlabel('Time step after line clearance')
    plt.ylim(0.75,1.5)
    plt.grid()
    figName = PlotDir + '/' + 'Plot' + str(k) + '.png'
    plt.savefig(figName)
    plt.close()
    k+=1