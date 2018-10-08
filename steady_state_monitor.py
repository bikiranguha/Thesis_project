# get info on when the voltages reach steady state after a fault

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

"""
# one line out then a fault
event1Flag = '-event01'
event1Param = '0.1,OUT,LINE,151,201,,1,7,,,,,'

event2Flag = '-event02'
event2Param = '0.2,FAULTON,ABCG,151,,,,1.0e-6,1.0e-6,1.0e-6,0.0,0.0,0.0'

event3Flag = '-event03'
event3Param = '0.3,FAULTOFF,ABCG,151,,,,,,,,,'

event4Flag = '-event04'
event4Param = '0.31,OUT,LINE,151,152,,1,7,,,,,'

exitFlag = '-event05'
exitParam = '5,EXIT,,,,,,,,,,,'
EventList = [event1Flag,event1Param,event2Flag,event2Param,event3Flag,event3Param,event4Flag,event4Param,exitFlag,exitParam]
"""

# one line out then a fault
event1Flag = '-event01'
event1Param = '0.1,OUT,LINE,154,205,,1,7,,,,,'

event2Flag = '-event02'
event2Param = '0.2,FAULTON,ABCG,154,,,,1.0e-6,1.0e-6,1.0e-6,0.0,0.0,0.0'

event3Flag = '-event03'
event3Param = '0.3,FAULTOFF,ABCG,154,,,,,,,,,'

event4Flag = '-event04'
event4Param = '0.31,OUT,LINE,154,203,,1,7,,,,,'

exitFlag = '-event05'
exitParam = '5,EXIT,,,,,,,,,,,'
EventList = [event1Flag,event1Param,event2Flag,event2Param,event3Flag,event3Param,event4Flag,event4Param,exitFlag,exitParam]




"""
event1Flag = '-event01'
event1Param = '0.1,FAULTON,ABCG,151,,,,1.0e-6,1.0e-6,1.0e-6,0.0,0.0,0.0'

event2Flag = '-event02'
event2Param = '0.2,FAULTOFF,ABCG,151,,,,,,,,,'

event3Flag = '-event03'
event3Param = '0.21,OUT,LINE,151,152,,1,7,,,,,'

exitFlag = '-event04'
exitParam = '5,EXIT,,,,,,,,,,,'
EventList = [event1Flag,event1Param,event2Flag,event2Param,event3Flag,event3Param,exitFlag,exitParam]
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

Results = runSim(rawPath,EventList,'TS3phLoadOut.log')
time = list(Results['time'])
for Bus in rawBusDataDict:
	vMag = Results[int(Bus)].mag
	# calculations for dv_dt
	VmagSize = vMag.shape[0]
	timestep = time[1] - time[0]
	dv_dt = np.zeros(VmagSize) # initialize dv_dt array with all zeros
	for i in range(VmagSize):
		try:
			dv_dt[i] = (vMag[i] - vMag[i-1])/timestep
		except: # will happen if i = 0, since there is no i-1
			continue


	# get the steady state mean dv_dt
	ind_steady = [ind for ind, t in enumerate(time) if t>= 2.0][0] # get the time index when its greater than 1 sec (after the 2nd line out)
	cropped_dv_dt = np.absolute(dv_dt[ind_steady:ind_steady+10]) # ten cycles after 2 seconds
	# Now simulate the fault at 151,152,1 to see what the values are
	VmagSteady = vMag[ind_steady:]
	min_Vmag = VmagSteady.min()
	dv_dt_mean = np.mean(cropped_dv_dt)

	print 'For Bus ' + Bus + ':'
	print 'Mean 10 cycles after 2 s: ' + str(dv_dt_mean)
	print 'Min voltage value: ' + str(min_Vmag)




	"""
	# get the min and max abs dv_dt for each bus after fault clearance
	ind_fault_clearance = [ind for ind, t in enumerate(time) if t>= 0.21][0] # get the time index when its greater than 1 sec (after the 2nd line out)
	cropped_dv_dt = np.absolute(dv_dt[ind_fault_clearance:])

	print 'For Bus ' + Bus + ':'
	print 'Max dv_dt: ' + str(cropped_dv_dt.max())
	print 'Min dv_dt: ' + str(cropped_dv_dt.min())
	"""


"""
# get plots for all the buses in the HV set
# plot to see if everything is ok
for Bus in list(HVBusSet):

	time = Results['time']
	# plot voltage
	vMag = Results[int(Bus)].mag
	
	plt.plot(time, vMag)
	titleStr = 'Bus ' + Bus
	plt.title(titleStr)
	plt.ylabel('dv/dt (pu)')
	plt.ticklabel_format(useOffset=False)
	plt.xlabel('Time (sec)')
	plt.ylim(-0.1,1.5)
	plt.grid()
	figName = 'Bus'+ Bus+'VMag.png'
	plt.savefig(figName)
	plt.close()


	# plot dv_dt
	plt.plot(time, dv_dt)
	titleStr = 'Bus ' + Bus
	plt.title(titleStr)
	plt.ylabel('dv/dt (pu)')
	plt.ticklabel_format(useOffset=False)
	plt.xlabel('Time (sec)')
	plt.ylim(-0.6,0.6)
	plt.grid()
	figName = 'Bus'+ Bus+'dVDt.png'
	plt.savefig(figName)
	plt.close()
"""