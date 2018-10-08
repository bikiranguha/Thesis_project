# run all the N-2 contingencies (with faults in between) (except for ones which cause topology issues) and see which ones lead to unstable voltages
# Unstable voltage is when the dv_dt value for 10 cycles ( 1 sec after the final line outage) is greater than 0.1 pu, or if the  voltage at any bus goes below
# 0.9 pu
# For all the events where some issues are detected, plots are generated for all the buses where issues are detected
# Also a summary of the LV or high dvdt buses for each suspicious event is detected

# external stuff
from getBusDataFn import getBusData
from runSimFn import runSim  # function to run simulation and plot the results
import matplotlib.pyplot as plt
import os
import numpy as np
# create plot directory
currentdir = os.getcwd()
plot_dir = currentdir +  '/VIssuePlots'
if not os.path.isdir(plot_dir):
    os.mkdir(plot_dir)
# LV report class structure


class LVReport(object):
    def __init__(self):
        self.LVBuses = []
        self.highDvDTBuses = []
        self.minV = []
        self.averageDV_dt = []


# files
raw = 'test_cases/savnw/savnw_sol.raw'
# lists all the N-2 contingencies which cause topology inconsistencies
topology_inconsistency_file = 'topology_inconsistency_cases_savnw.txt'

# variables
HVBusSet = set()
HVLineSet = set()
rawBusDataDict = getBusData(raw)
topology_inconsistent_set = set()
LVReportDict = {}
outputLines = []
SimulationDoneSet = set()

# constants
LVThreshold = 0.90
dv_dt_threshold = 0.1


# generate the HV bus set
for Bus in rawBusDataDict:
    BusVolt = float(rawBusDataDict[Bus].NominalVolt)
    BusType = rawBusDataDict[Bus].type
    if BusVolt >= 34.5:  # no need to consider type 4 buses since they are already filtered out in the get Bus Data function
        HVBusSet.add(Bus)


# get the N-2 events which cause topology inconsistencies
with open(topology_inconsistency_file, 'r') as f:
    fileLines = f.read().split('\n')
    for line in fileLines:
        if line == '':
            continue
        topology_inconsistent_set.add(line.strip())


# read the raw file and get the HV line set
with open(raw, 'r') as f:
    fileLines = f.read().split('\n')
branchStartIndex = fileLines.index(
    '0 / END OF GENERATOR DATA, BEGIN BRANCH DATA') + 1
branchEndIndex = fileLines.index(
    '0 / END OF BRANCH DATA, BEGIN TRANSFORMER DATA')
# extract all the HV lines
for i in range(branchStartIndex, branchEndIndex):
    line = fileLines[i]
    words = line.split(',')
    Bus1 = words[0].strip()
    Bus2 = words[1].strip()
    cktID = words[2].strip("'").strip()
    status = words[13].strip()
    if Bus1 in HVBusSet and Bus2 in HVBusSet and status != '0':
        key = Bus1 + ',' + Bus2 + ',' + cktID
        HVLineSet.add(key)

# total N-2 contingencies to be carried out
# totalSims = len(HVLineSet)**2 - len(topology_inconsistent_set)
# totalSims = 274 # found from status update

totalSims = 0
# nested loops to count how many events will be run
for line1 in list(HVLineSet):
    for line2 in list(HVLineSet):
        # the lines cannot be the same
        if line1 == line2:
            continue
        # part to ensure there is no duplication of events
        currentSet = line1+';'+line2
        currentSetReverse = line2 + ';' + line1
        # if case causes topology inconsistencies, continue
        if currentSet in topology_inconsistent_set or currentSetReverse in topology_inconsistent_set:
            continue

        # simulation already done
        # if currentSet in SimulationDoneSet:
        #    continue
        # else:
        #    currentSetReverse = line2+';'+line1
		#	SimulationDoneSet.add(currentSet)
        #    SimulationDoneSet.add(currentSetReverse)
        # 2 simulations, one for each bus (as fault bus) in line 2
        totalSims += 2


"""
# temporary part for debugging
HVLineSet = set()
HVLineSet.add('151,152,2')
HVLineSet.add('153,154,1')
"""

# run nested loops to see if there are any abnormal low voltages
simCount = 0  # to keep track of how many simulations are already done
SimulationDoneSet = set()  # need to reset
for line1 in list(HVLineSet):
    for line2 in list(HVLineSet):
        # stability_indicator = 1
        # Bus_issues = [] # list of buses where issues (low voltage or high dv_dt) are reported
        # the lines cannot be the same
        if line1 == line2:
            continue
        # part to ensure there is no duplication of events
        currentSet = line1+';'+line2
        currentSetReverse = line2 + ';' + line1
        # if case causes topology inconsistencies, continue
        if currentSet in topology_inconsistent_set or currentSetReverse in topology_inconsistent_set:
            continue

        # simulation already done
        # if currentSet in SimulationDoneSet:
        #    continue
        # else:
        #    currentSetReverse = line2+';'+line1
        #    SimulationDoneSet.add(currentSet)
        #    SimulationDoneSet.add(currentSetReverse)

        # event key
        # key = line1 + ';' + line2

        line1Elements = line1.split(',')
        line2Elements = line2.split(',')

        # Line 1 params
        L1Bus1 = line1Elements[0]
        L1Bus2 = line1Elements[1]
        L1cktID = line1Elements[2]

        # Line 2 params
        L2Bus1 = line2Elements[0]
        L2Bus2 = line2Elements[1]
        L2cktID = line2Elements[2]

        # generate the event
        # one line out then a fault
        # list of buses where faults will be applied
        FaultBusList = [L2Bus1, L2Bus2]
        for FaultBus in FaultBusList:  # simulate faults on each side of the 2nd line
            event1Flag = '-event01'
            event1Param = '0.1,OUT,LINE,' + L1Bus1 + ',' + L1Bus2 + ',,' + L1cktID + ',7,,,,,'

            event2Flag = '-event02'
            event2Param = '0.2,FAULTON,ABCG,' + FaultBus + ',,,,1.0e-6,1.0e-6,1.0e-6,0.0,0.0,0.0'

            event3Flag = '-event03'
            event3Param = '0.3,FAULTOFF,ABCG,' + FaultBus + ',,,,,,,,,'

            event4Flag = '-event04'
            event4Param = '0.31,OUT,LINE,' + L2Bus1 + ',' + L2Bus2 + ',,' + L2cktID + ',7,,,,,'

            exitFlag = '-event05'
            exitParam = '3,EXIT,,,,,,,,,,,'
            EventList = [event1Flag, event1Param, event2Flag, event2Param,event3Flag, event3Param, event4Flag, event4Param, exitFlag, exitParam]
            Results = runSim(raw, EventList, 'TS3phEvent.log')
            currentEvent = currentSet + ';' + 'F' + FaultBus
            # print 'Current event: ' + currentEvent

            time = list(Results['time'])
            # get the time index when its greater than 1 sec (after the 2nd line out)
            ind_1sec = [ind for ind, t in enumerate(time) if t >= 1.31][0]
            # print time[ind_1sec]

            # extract LV results if any
            for key in Results:
                if key == 'time':
                    continue

                vMag = Results[key].mag
                # get the dv_dt values
                VmagSize = vMag.shape[0]
                timestep = time[1] - time[0]
                # initialize dv_dt array with all zeros
                dv_dt = np.zeros(VmagSize)
                for i in range(VmagSize):
                    try:
                        dv_dt[i] = (vMag[i] - vMag[i-1])/timestep
                    except:  # will happen if i = 0, since there is no i-1
                        continue
                # ten cycles after 2 seconds
                cropped_dv_dt = np.absolute(dv_dt[ind_1sec:ind_1sec+10])
                # get the relevant parameters on which to base the decision
                VmagSteady = vMag[ind_1sec:]
                min_Vmag = VmagSteady.min()
                dv_dt_mean = np.mean(cropped_dv_dt)

                if dv_dt_mean > dv_dt_threshold:
                    if currentEvent not in LVReportDict:
                        LVReportDict[currentEvent] = LVReport()
                    LVReportDict[currentEvent].highDvDTBuses.append(
                        key)  # get the bus
                    LVReportDict[currentEvent].averageDV_dt.append(dv_dt_mean)

                if min_Vmag < LVThreshold:
                    if currentEvent not in LVReportDict:
                        LVReportDict[currentEvent] = LVReport()
                    LVReportDict[currentEvent].LVBuses.append(key)  # get the bus
                    # get the minimum voltage recorded for that bus
                    LVReportDict[currentEvent].minV.append(min_Vmag)

            # event has issues, raise the instability flag, plot the results and save
            if currentEvent in LVReportDict:
                # stability_indicator = 0


                eventdir = plot_dir + '/' +  currentEvent
                # make directory to put the plots for this event
                if not os.path.isdir(eventdir):
                    os.mkdir(eventdir)
                LVBuses = LVReportDict[currentEvent].LVBuses
                highDvDTBuses = LVReportDict[currentEvent].highDvDTBuses
                for Bus in LVBuses:
                    vMag = Results[Bus].mag
                    
                    # plot voltages
                    plt.plot(time, vMag)
                    titleStr = 'Bus ' + str(Bus)
                    plt.title(titleStr)
                    plt.ylabel('Voltage (pu)')
                    plt.ticklabel_format(useOffset=False)
                    plt.xlabel('Time (sec)')
                    plt.ylim(-0.1,1.5)
                    plt.grid()
                    figName = eventdir + '/' +  'Bus'+ str(Bus)+'VMag.png'
                    plt.savefig(figName)
                    plt.close()

                for Bus in highDvDTBuses:
                    vMag = Results[Bus].mag
                    VmagSize = vMag.shape[0]
                    # get the dv_dt values
                    dv_dt = np.zeros(VmagSize) # initialize dv_dt array with all zeros
                    for i in range(VmagSize):
                        try:
                            dv_dt[i] = (vMag[i] - vMag[i-1])/timestep
                        except: # will happen if i = 0, since there is no i-1
                            continue

                    # plot voltages
                    plt.plot(time, vMag)
                    titleStr = 'Bus ' + str(Bus)
                    plt.title(titleStr)
                    plt.ylabel('Voltage (pu)')
                    plt.ticklabel_format(useOffset=False)
                    plt.xlabel('Time (sec)')
                    plt.ylim(-0.1,1.5)
                    plt.grid()
                    figName = eventdir + '/' +  'Bus'+ str(Bus)+'VMag.png'
                    plt.savefig(figName)
                    plt.close()

                    # plot dv_dt
                    plt.plot(time, dv_dt)
                    titleStr = 'Bus ' + str(Bus)
                    plt.title(titleStr)
                    plt.ylabel('dv/dt (pu)')
                    plt.ticklabel_format(useOffset=False)
                    plt.xlabel('Time (sec)')
                    plt.ylim(-0.5,0.5)
                    plt.grid()
                    figName = eventdir + '/' + 'Bus'+ str(Bus)+'dVDt.png'
                    plt.savefig(figName)
                    plt.close()



            # status update       
            simCount+=1
            print 'Simulations done:' + str(simCount) + ' out of ' + str(totalSims)


# generate output lines
for event in LVReportDict:
    LVBusList = LVReportDict[event].LVBuses
    highDVDTBusList = LVReportDict[event].highDvDTBuses
    LVString = event + ';' + 'LV Buses: ' + str(LVBusList)
    outputLines.append(LVString)
    highDvDTString = event + ';' + 'High DVDT Buses: ' + str(highDVDTBusList)
    outputLines.append(highDvDTString)
    outputLines.append('\n')


# write to output file
with open('VoltageStabilityReport.txt', 'w') as f:
    for line in outputLines:
        f.write(line)
        f.write('\n')

