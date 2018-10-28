# provide an event
# plot all HV bus voltages in TS3ph

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

# specify the event

"""
# N-2 line outage contingency
event1Flag = '-event01'
event1Param = '0.1,OUT,LINE,151,152,,1,7,,,,,'

event2Flag = '-event02'
event2Param = '0.1,OUT,LINE,151,152,,2,7,,,,,'
"""

"""
# load shed
event1Flag = '-event01'
event1Param = '0.1,OUT,LOAD,205,,,1,7,,,,,'

exitFlag = '-event02'
exitParam = '1,EXIT,,,,,,,,,,,'
#EventList = [event1Flag,event1Param,exitFlag,exitParam]
"""


# one line out then a fault
event1Flag = '-event01'
event1Param = '0.1,OUT,LINE,201,202,,1,7,,,,,'

event2Flag = '-event02'
event2Param = '0.2,FAULTON,ABCG,151,,,,1.0e-6,1.0e-6,1.0e-6,0.0,0.0,0.0'

event3Flag = '-event03'
event3Param = '0.3,FAULTOFF,ABCG,151,,,,,,,,,'

event4Flag = '-event04'
event4Param = '0.31,OUT,LINE,151,201,,1,7,,,,,'

exitFlag = '-event05'
exitParam = '10,EXIT,,,,,,,,,,,'
EventList = [event1Flag,event1Param,event2Flag,event2Param,event3Flag,event3Param,event4Flag,event4Param,exitFlag,exitParam]
"""
# just N-2 line outages
# one line out then a fault
event1Flag = '-event01'
event1Param = '0.1,OUT,LINE,201,204,,1,7,,,,,'


event2Flag = '-event02'
event2Param = '0.11,OUT,LINE,151,152,,1,7,,,,,'

exitFlag = '-event03'
exitParam = '3,EXIT,,,,,,,,,,,'
EventList = [event1Flag,event1Param,event2Flag,event2Param,exitFlag,exitParam]
"""

Results = runSim(rawPath,EventList,'TS3phLoadOut.log')

# print the final steady state voltage of some bus
#print Results[205].mag[-1]
# get plots for all the buses in the HV set
# plot to see if everything is ok
for Bus in list(HVBusSet):

	time = Results['time']
	vMag = Results[int(Bus)].mag
	plt.plot(time, vMag)
	titleStr = 'Bus ' + Bus
	plt.title(titleStr)
	plt.ylabel('Voltage magnitude (pu)')
	plt.ticklabel_format(useOffset=False)
	plt.xlabel('Time (sec)')
	plt.ylim(-0.1,1.5)
	plt.grid()
	figName = 'Bus'+ Bus+'VMag.png'
	plt.savefig(figName)
	plt.close()