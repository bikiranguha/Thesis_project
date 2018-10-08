# script to get a simulation from the real world (predetermined in our case) and try to capture the event timings and reconstruct the whole event in another copy of TS3ph
# since we cannot do dynamic initialization at the moment, everytime a new event is detected, it is added to the event list and the whole simulation is rerun.
# This process is repeated till we get a good match with the real world simulation
import matplotlib.pyplot as plt
import numpy as np
from input_data import RealWorldEventList, TS3phEventList, rawPath, EventDescriptionDict
from runSimFn import runSim # function to run simulation and plot the results
Results_Actual = runSim(rawPath,RealWorldEventList,'TS3phFault.log') # case representing real world data
#Results_Sim = runSim(rawPath,TS3phEventList,'TS3phNoDist.log')






time = Results_Actual['time'] 
totSimTime = int(round(float(time[-1]))) # get the total time to simulate

EventDict = {} # dynamic dictionary of events for the simulation

mostRecentEventIndex = 0 # keeps track of how many events have been recorded
exitParamString = str(totSimTime) + ',EXIT,,,,,,,,,,,'



# loop to redo the whole simulation when new events are detected
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


# plot to see how it all looks
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

# run at the end, deletes all useless pyc files
import delete_all_pyc