# provide an event
# plot all bus voltages and angles in TS3ph

import platform
currentOS = platform.system()
from runSimFn import runSim
from getBusDataFn import getBusData
import numpy as np
import matplotlib.pyplot as plt
HVBusSet = set()

# Functions and classes
def convertFileLinux(file,currentOS):
	# function to convert file from crlf to lf (if needed)
	if currentOS == 'Linux':
		text = open(file, 'rb').read().replace('\r\n', '\n')
		open(file, 'wb').write(text)


rawPath = 'test_cases/savnw/savnw_sol.raw'



# convert raw file crlf to lf (needed for linux)
convertFileLinux(rawPath,currentOS)
rawBusDataDict = getBusData(rawPath)
# generate the HV bus set
for Bus in rawBusDataDict:
	BusVolt = float(rawBusDataDict[Bus].NominalVolt)
	BusType = rawBusDataDict[Bus].type
	if BusVolt >= 34.5: # no need to consider type 4 buses since they are already filtered out in the get Bus Data function
		HVBusSet.add(Bus)
"""
# specify the event
event1Flag = '-event01'
event1Param = '0.1,OUT,TRANSFORMER,151,101,,1,7,,,,,'

#event2Flag = '-event02'
#event2Param = '0.1,OUT,LINE,151,152,,2,7,,,,,'

exitFlag = '-event03'
exitParam = '0.2,EXIT,,,,,,,,,,,'

#EventList = [event1Flag,event1Param,event2Flag,event2Param,exitFlag,exitParam]
EventList = [event1Flag,event1Param,exitFlag,exitParam]
"""
"""
event1Flag = '-event01'
event1Param = '0.1,OUT,LINE,151,201,,1,7,,,,,'

event2Flag = '-event02'
event2Param = '0.2,FAULTON,ABCG,151,,,,1.0e-6,1.0e-6,1.0e-6,0.0,0.0,0.0'

event3Flag = '-event03'
event3Param = '0.3,FAULTOFF,ABCG,151,,,,,,,,,'

event4Flag = '-event04'
event4Param = '0.31,OUT,LINE,151,152,,1,7,,,,,'

exitFlag = '-event05'
exitParam = '10,EXIT,,,,,,,,,,,'
EventList = [event1Flag, event1Param, event2Flag, event2Param,event3Flag, event3Param, event4Flag, event4Param, exitFlag, exitParam]
"""


# fault with no lines out
event1Flag = '-event01'
event1Param = '0.1,FAULTON,ABCG,151,,,,1.0e-6,1.0e-6,1.0e-6,0.0,0.0,0.0'

event2Flag = '-event02'
event2Param = '0.2,FAULTOFF,ABCG,151,,,,,,,,,'

exitFlag = '-event03'
exitParam = '10,EXIT,,,,,,,,,,,'
EventList = [event1Flag, event1Param, event2Flag, event2Param, exitFlag, exitParam]

Results = runSim(rawPath,EventList,'TS3phTFOut.log')


# get plots for all the buses in the HV set
# angles
for Bus in rawBusDataDict:

	time = Results['time']
	Angle = Results[int(Bus)].ang
	plt.plot(time, Angle)
	titleStr = 'Bus ' + Bus
	plt.title(titleStr)
	plt.ylabel('Angle (degrees)')
	plt.ticklabel_format(useOffset=False)
	plt.xlabel('Time (sec)')
	plt.ylim(-180, 180)
	figName = 'Bus'+ Bus+'Angle.png'
	plt.savefig(figName)
	plt.close()

# volt
for Bus in rawBusDataDict:

	time = Results['time']
	volt = Results[int(Bus)].ang
	plt.plot(time, volt)
	titleStr = 'Bus ' + Bus
	plt.title(titleStr)
	plt.ylabel('Volt (pu)')
	plt.ticklabel_format(useOffset=False)
	plt.xlabel('Time (sec)')
	#plt.ylim(-180, 180)
	figName = 'Bus'+ Bus+'Volt.png'
	plt.savefig(figName)
	plt.close()