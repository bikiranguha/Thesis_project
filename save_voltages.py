# provide an event
# plot all bus voltages in TS3ph for a certain event

import platform
currentOS = platform.system()
from runSimFn import runSim
from getBusDataFn import getBusData
import numpy as np
import matplotlib.pyplot as plt
import pickle
HVBusSet = set()

# Functions and classes
def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def convertFileLinux(file,currentOS):
	# function to convert file from crlf to lf (if needed)
	if currentOS == 'Linux':
		text = open(file, 'rb').read().replace('\r\n', '\n')
		open(file, 'wb').write(text)


rawPath = 'test_cases/savnw/savnw_sol.raw'



# convert raw file crlf to lf (needed for linux)
convertFileLinux(rawPath,currentOS)
rawBusDataDict = getBusData(rawPath)


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

# Fault at b151

event1Flag = '-event01'
event1Param = '0.1,FAULTON,ABCG,151,,,,1.0e-6,1.0e-6,1.0e-6,0.0,0.0,0.0'

event2Flag = '-event02'
event2Param = '0.2,FAULTOFF,ABCG,151,,,,,,,,,'

exitFlag = '-event03'
exitParam = '5,EXIT,,,,,,,,,,,'
EventList = [event1Flag,event1Param,event2Flag,event2Param,exitFlag,exitParam]
Results151F = runSim(rawPath,EventList,'TS3phF.log')


# get the voltage mag
v151 = {}
for bus in Results151F:
	if bus == 'time':
		v151[bus] = Results151F[bus]
		continue
	v = Results151F[bus].mag
	v151[bus] = v







# Fault at b152
event1Flag = '-event01'
event1Param = '0.1,FAULTON,ABCG,152,,,,1.0e-6,1.0e-6,1.0e-6,0.0,0.0,0.0'

event2Flag = '-event02'
event2Param = '0.2,FAULTOFF,ABCG,152,,,,,,,,,'

exitFlag = '-event03'
exitParam = '5,EXIT,,,,,,,,,,,'
EventList = [event1Flag,event1Param,event2Flag,event2Param,exitFlag,exitParam]
Results152F = runSim(rawPath,EventList,'TS3phF.log')



# get the voltage mag
v152 = {}
for bus in Results152F:
	if bus == 'time':
		v151[bus] = Results152F[bus]
		continue
	v = Results152F[bus].mag
	v152[bus] = v

# save the results dictionaries to pickle objects
save_obj(v151,'F151Vmag')
save_obj(v152,'F152Vmag')

