# plot all the steady state gen angles together for a N-2 F simulation (which gets queried here)
print('Importing modules...')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from getBusDataFn import getBusData

eventsFile = 'obj/eventN_2FNew.txt'
buslistfile = 'obj/buslistN_2F.csv'
aFileSteady = 'obj/aN_2FNewSteady.csv' # generated in integrateN_2Va.py

refRaw = 'savnw.raw'
busdatadict = getBusData(refRaw)
# get the event list
eventwiseList = []
with open(eventsFile,'r') as f:
    fileLines = f.read().split('\n')
    for line in fileLines:
        if line == '':
            continue
        eventwiseList.append(line.strip())

# get the bus list
with open(buslistfile,'r') as f:
    filecontent = f.read().strip()
    buslist = filecontent.split(',')


print('Getting the steady state angles...')
adfS  = pd.read_csv(aFileSteady,header = None)

inputArrayEventWise = adfS.values[:-1] # the last row is incomplete, so take it out
#anglesBusWise = inputArrayEventWise.reshape(-1,120)

# Query the event you want to plot
event = '106/3005,3006,1;152,3004,1/F152' # average voltage oscillation in steady state: 0.273462384939
event = '100/153,154,1;152,202,1/F152'
event  = '106/3001,3003,1;154,205,1/F205'
event = '105/201,202,1;152,3004,1/F3004' # damped oscillation: 5.0
event = '104/152,3004,1;201,204,1/F204' # 10.0
event = '104/151,201,1;153,3006,1/F3006'
event = '105/201,202,1;151,152,2/F152'
event = '105/3005,3006,1;152,3004,1/F3004'
event = '106/3005,3006,1;201,204,1/F204'
event = '105/201,202,1;151,152,1/F152'
eventInd = eventwiseList.index(event)
angles = inputArrayEventWise[eventInd]
anglesBusWise = angles.reshape(-1,120)

for i in range(anglesBusWise.shape[0]):
    # plot if its a gen bus
    bus = buslist[i]
    bustype = busdatadict[bus].type
    if bustype == '2' or bustype == '3':
        currAngle = anglesBusWise[i]
        plt.plot(currAngle)

plt.grid()
plt.show()


