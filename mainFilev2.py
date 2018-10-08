import matplotlib.pyplot as plt
import numpy as np
from input_data import RealWorldEventList, TS3phEventList, rawPath, EventDescriptionDict
from runSimFn import runSim # function to run simulation and plot the results
Results_Actual = runSim(rawPath,RealWorldEventList,'TS3phFault.log')
#Results_Sim = runSim(rawPath,TS3phEventList,'TS3phNoDist.log')

"""
# plot to see if everything is ok
time = Results_Actual['time']
vMag = Results_Actual[153].mag
plt.plot(time, vMag)
plt.title('Bus 153')
plt.ylabel('Voltage magnitude (pu)')
plt.ticklabel_format(useOffset=False)
plt.xlabel('Time (sec)')
plt.ylim(-0.1,1.5)
plt.savefig('Bus153VMag.png')
"""




time = Results_Actual['time'] 
totSimTime = int(round(float(time[-1]))) # get the total time to simulate
# dynamic dictionary of events for the simulation
EventDict = {}

#EvenDict['EXIT'] = ['-event01', exitParamString] # initialize the dictionary, the flag will need to be updated everytime an event is detected

mostRecentEventIndex = 0 # keeps track of how many events have been recorded
#exitFlag = '-event' + str(mostRecentEventIndex+1).zfill(2)
exitParamString = str(totSimTime) + ',EXIT,,,,,,,,,,,'




while True:
	EventList = [] # reset event list
	# get the most updated event list from the event dictionary
	for i in range(len(EventDict)):
		currentEventList = EventDict[i] 
		for ele in currentEventList:
			EventList.append(ele)

	# add the exit event
	exitFlag = '-event' + str(len(EventDict)+1).zfill(2)
	EventList.append(exitFlag)
	EventList.append(exitParamString)

	# run the simulation
	Results_Sim = runSim(rawPath,EventList,'TS3phNoDist.log')
	DiffInstant ={}
	DifFound = 0 # flag which is set when new event is detected
	# get the exact time when the results are different
	for key in Results_Sim:
		if key == 'time':
			continue
		Vmag1 = Results_Sim[key].mag
		Vmag2 = Results_Actual[key].mag
	#	print type(Vmag1)
	#	print Vmag1.size
	#	print Vmag2.size
		i = 0
		while i < Vmag1.size:
			currentVmag1 = Vmag1[i]
			currentVmag2 = Vmag2[i]
			error = abs((currentVmag2 - currentVmag1)/currentVmag2)*100

			if error > 5: # sudden difference detected
				diffStartTime = i
				diffTime = time[i]
				totDiff = 0.0
				for j in range(diffStartTime,diffStartTime+11): # scan the next 10 steps for errors
					currentVmag1 = Vmag1[j]
					currentVmag2 = Vmag2[j]
					error = abs((currentVmag2 - currentVmag1)/currentVmag2)*100
					totDiff += error
				avgError = totDiff/10
				#print avgError
				if avgError > 5: # average error more than 5% in the next ten time steps
					DiffInstant[key] = diffTime
					DifFound = 1
					break
				else: # differene was not sustained, skip ahead next 10 time steps and continue analysis
					i+=10
			else:  # error < 5%
				i+=1

	# no difference found, so exact match, break out of loop
	if DifFound == 0:
		break
	# get the majority vote, which is the final verdict on the exact time when the simulations start to differ
	AllDiffTimes = []
	for Bus in DiffInstant:
		AllDiffTimes.append(DiffInstant[Bus])

	TimeWhenUseless  = max(AllDiffTimes, key=AllDiffTimes.count) # the final verdict on the time instant when the TS3ph sim becomes useless
	# update the event dictionary
	Event = EventDescriptionDict[mostRecentEventIndex]
	eventIndex = mostRecentEventIndex
	mostRecentEventIndex +=1
	eventFlag = '-event' + str(mostRecentEventIndex).zfill(2)
	TimeStr = '%.2f' %TimeWhenUseless
	eventParam = TimeStr + ',' + Event
	print 'New Event Found:'
	print eventFlag + ' ' + eventParam
	EventDict[eventIndex] = [eventFlag,eventParam]


# plot
#vMag = Results_Sim[153][:time.shape[0]] 
vMag = Results_Sim[153].mag
time = Results_Sim['time']
plt.plot(time, vMag)
plt.title('Bus 153')
plt.ylabel('Voltage magnitude (pu)')
plt.ticklabel_format(useOffset=False)
plt.xlabel('Time (sec)')
plt.ylim(-0.1,1.5)
plt.savefig('Bus153VMag.png')

#print TimeWhenUseless
"""
totSimTime = 5.0

remainingTime = totSimTime - TimeWhenUseless


# new simulation event list
event1_flag =  '-event01'
event1_param =  '0.01,FAULTON,ABCG,153,,,,1.0e-6,1.0e-6,1.0e-6,0.0,0.0,0.0'

event2_flag = '-event02'
event2_param = str(remainingTime) +',EXIT,,,,,,,,,,,'
NewSimcurrentEventList = Eve [event1_flag,event1_param,event2_flag,event2_param]

# carry out simulation and get results
Results_Rem = runSim(rawPath,NewSimEventList,'TS3phRem.log')


# append the results of the two simulations

for i in range(len(time)):
	if time[i] > TimeWhenUseless:
		diffTimeInd = i
		break

AppendedSimResults = {}

for key in Results_Sim:
	if key == 'time':
		continue
	Vmag1 = Results_Sim[key].mag
	VmagPart1 = Vmag1[:diffTimeInd+1].ravel()
	#print type(VmagPart1)
	#print type(Vmag2)
	Vmag2 = Results_Rem[key].mag.ravel()
	VmagAppended = np.concatenate([VmagPart1,Vmag2]) 
	AppendedSimResults[key] = VmagAppended

#print AppendedSimResults[153].shape
#print time.shape
# plot to see if everything is ok
#time = Results_Actual['time']
# make sure the length of this array and the array time is
vMag = AppendedSimResults[153][:time.shape[0]] 
plt.plot(time, vMag)
plt.title('Bus 153')
plt.ylabel('Voltage magnitude (pu)')
plt.ticklabel_format(useOffset=False)
plt.xlabel('Time (sec)')
plt.ylim(-0.1,1.5)
plt.savefig('Bus153VMag.png')

"""
# run at the end, deletes all useless pyc files
import delete_all_pyc