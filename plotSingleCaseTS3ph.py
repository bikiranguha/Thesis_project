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


rawFile = 'test_cases/PSSE/pf_ornl0823conz.raw'
dyrFile = 'test_cases/PSSE/pf_ornl_all.dyr'

rawPath = rawFile
TS3phOutFile = 'TS3phoutput.out'



rawFlag = '-ts_raw_dir'
rawPath = rawFile

dyrFlag = '-ts_dyr_dir'
dyrPath = dyrFile

state_varFlag = '-state_var_out_file'
state_varFile = TS3phOutFile




# convert raw file crlf to lf (needed for linux)
convertFileLinux(rawPath,currentOS)
rawBusDataDict = getBusData(rawPath)

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
"""

# #####
# # fault with (or without) some impedance (pf_ornl)
# event1Flag = '-event01'
# event1Param = '0.1,FAULTON,ABCG,101,,,,1.0e-6,1.0e-6,1.0e-6,0.0,0.0,0.0'

# event2Flag = '-event02'
# event2Param = '0.2,FAULTOFF,ABCG,101,,,,,,,,,'

# exitFlag = '-event03'
# exitParam = '10,EXIT,,,,,,,,,,,'
# #EventList = [event1Flag, event1Param, event2Flag, event2Param, exitFlag, exitParam]
# EventList = [state_varFlag, state_varFile ,rawFlag, rawPath, dyrFlag, dyrPath,event1Flag, event1Param, event2Flag, event2Param, exitFlag, exitParam]

# Results = runSim(rawPath,EventList,'TS3phTFOut.log')
# ####


# N-2 with a fault in between (pf_ornl)
#eventID = '1920,2000,1;900,901,1/F901'
#eventID = '1920,2000,1;101,900,2/F101'

event = '1920,2000,1;101,900,2/F101'

# extract all the relevant info from the event id
eventKeyWords = event.split('/')
lines = eventKeyWords[0].strip()
L1Elements = lines.split(';')[0].split(',')
L2Elements = lines.split(';')[1].split(',')

L1Bus1 = L1Elements[0].strip()
L1Bus2 = L1Elements[1].strip()
L1cktID = L1Elements[2].strip()


L2Bus1 = L2Elements[0].strip()
L2Bus2 = L2Elements[1].strip()
L2cktID = L2Elements[2].strip()

FaultBus  = eventKeyWords[1].strip()
FaultBus = FaultBus[1:]



# generate event flags and params
event1Flag = '-event01'
event1Param = '0.1,OUT,LINE,{},{},,{},7,,,,,'.format(L1Bus1,L1Bus2,L1cktID)

event2Flag = '-event02'
event2Param = '0.2,FAULTON,ABCG,{},,,,1.0e-6,1.0e-6,1.0e-6,0.0,0.0,0.0'.format(FaultBus)

event3Flag = '-event03'
event3Param = '0.3,FAULTOFF,ABCG,{},,,,,,,,,'.format(FaultBus)

event4Flag = '-event04'
event4Param = '0.31,OUT,LINE,{},{},,{},7,,,,,'.format(L2Bus1,L2Bus2,L2cktID)

exitFlag = '-event05'
exitParam = '30,EXIT,,,,,,,,,,,'
#EventList = [event1Flag, event1Param, event2Flag, event2Param,event3Flag, event3Param, event4Flag, event4Param, exitFlag, exitParam]
EventList = [state_varFlag, state_varFile ,rawFlag, rawPath, dyrFlag, dyrPath,event1Flag, event1Param, event2Flag, event2Param,event3Flag, event3Param, event4Flag, event4Param, exitFlag, exitParam]
Results = runSim(rawFile, EventList, 'TS3phN_2F.log', TS3phOutFile)
###

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
	figName = 'tmpPlots/Bus'+ Bus+'Angle.png'
	plt.savefig(figName)
	plt.close()

# volt
for Bus in rawBusDataDict:

	time = Results['time']
	volt = Results[int(Bus)].mag
	plt.plot(time, volt)
	titleStr = 'Bus ' + Bus
	plt.title(titleStr)
	plt.ylabel('Volt (pu)')
	plt.ticklabel_format(useOffset=False)
	plt.xlabel('Time (sec)')
	#plt.ylim(-180, 180)
	figName = 'tmpPlots/Bus'+ Bus+'Volt.png'
	plt.savefig(figName)
	plt.close()