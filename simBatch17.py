from getBusDataFn import getBusData
from runSimFn import runSim 
import csv
import numpy as np
## define the raw, dyr path and the events list file
rawFile = 'test_cases/PSSE/pf_ornl0823conz.raw'
dyrFile = 'test_cases/PSSE/pf_ornl_all.dyr'
eventListFile = 'N_2FEvents.txt'


BusDataDict = getBusData(rawFile)
outdir = 'PFORNLSimSeq'
#### get the events list
eventsList = []
with open(eventListFile,'r') as f:
    fileLines = f.read().split('\n')
    for line in fileLines:
        if line == '':
            continue
        eventsList.append(line.strip())
#####

# dump the arrays into csv files
simList = eventsList[1600:1700]
i = 17
k=0
TS3phOutFile = 'TS3phoutput{}.out'.format(k)

# calculation of the start and end indices of the steady state and transient part
timestep = 1/120.0
transientEnd = int(1.5/timestep) # get roughly 1.5 seconds of initial data
steadyStateStart = int(1.0/timestep)  # get roughly 1 second of final (steady state data)


# files to dump the transient part (0 to 1.5 sec)
vNameT = outdir+'/vTransient{}.csv'.format(i)
aNameT = outdir+'/aTransient{}.csv'.format(i)

eventsFileName = outdir+'/eventKeysPFORNL_{}.csv'.format(i)
vFileT = open(vNameT, 'wb') # 'wb' needed to avoid blank space in between lines
aFileT = open(aNameT, 'wb')


vWriterT = csv.writer(vFileT)
aWriterT = csv.writer(aFileT)




# files to dump the steady state part (last second)
vNameS = outdir+'/vSteady{}.csv'.format(i)
aNameS = outdir+'/aSteady{}.csv'.format(i)

#eventsFileName = outdir+'/eventKeys_{}.csv'.format(i)
vFileS = open(vNameS, 'wb') # 'wb' needed to avoid blank space in between lines
aFileS = open(aNameS, 'wb')


vWriterS = csv.writer(vFileS)
aWriterS = csv.writer(aFileS)


####################



#print('Please make sure the file and event options are commented in petscopt')
## define the raw and dyr files
rawFlag = '-ts_raw_dir'
rawPath = rawFile

dyrFlag = '-ts_dyr_dir'
dyrPath = dyrFile

state_varFlag = '-state_var_out_file'
state_varFile = TS3phOutFile


suspectEvents = []
eventListBusWise = []
# for each event in sim list, run a 30 second simulation
for event in simList:
    print('Process {} event: {}'.format(k, event))
    ### get all the relevant info from the event id string
    eventWords = event.split('/')
    linesOutage = eventWords[0].strip()
    FaultBus = eventWords[1].strip()[1:] # exclude the 'F' at the beginning

    # get the line params
    line1Elements = linesOutage.split(';')[0].strip()
    line2Elements = linesOutage.split(';')[1].strip()


    # Line 1 params
    line1 = line1Elements.split(',')
    L1Bus1 = line1[0].strip()
    L1Bus2 = line1[1].strip()
    L1cktID = line1[2].strip("'").strip()


    # Line 2 params
    line2 = line2Elements.split(',')
    L2Bus1 = line2[0].strip()
    L2Bus2 = line2[1].strip()
    L2cktID = line2[2].strip("'").strip()


    # specify the cape protection origin
# -cape_sim_area_fbus 300
# -cape_sim_area_tbus 301
# -cape_sim_area_cid 1

    protection_fbus_flag = '-cape_sim_area_fbus'
    protection_fbus_param = L2Bus1

    protection_tbus_flag = '-cape_sim_area_tbus'
    protection_tbus_param = L2Bus2

    protection_cid_flag = '-cape_sim_area_cid'
    protection_cid_param = L1cktID

    ###

    ### create flags and parameters for the event
    event1Flag = '-event01'
    event1Param = '0.1,OUT,LINE,' + L1Bus1 + ',' + L1Bus2 + ',,' + L1cktID + ',7,,,,,'

    event2Flag = '-event02'
    event2Param = '0.2,FAULTON,ABCG,' + FaultBus + ',,,,1.0e-6,1.0e-6,1.0e-6,0.0,0.0,0.0'

    event3Flag = '-event03'
    event3Param = '0.3,FAULTOFF,ABCG,' + FaultBus + ',,,,,,,,,'

    event4Flag = '-event04'
    event4Param = '0.31,OUT,LINE,' + L2Bus1 + ',' + L2Bus2 + ',,' + L2cktID + ',7,,,,,'

    exitFlag = '-event05'
    exitParam = '30,EXIT,,,,,,,,,,,'
    EventList = [state_varFlag, state_varFile ,rawFlag, rawPath, dyrFlag, dyrPath,event1Flag, event1Param, event2Flag, event2Param,event3Flag, event3Param, event4Flag, event4Param, exitFlag, exitParam,
                protection_fbus_flag,protection_fbus_param,protection_tbus_flag,protection_tbus_param,protection_cid_flag, protection_cid_param]

    eventLogfile = 'TS3phEvent{}.log'.format(k)

    #Results = runSim(rawFile, EventList, 'TS3phEvent{}.log'.format(k), TS3phOutFile)


    # get a suspect event if ts3ph exits due to some reason
    try:
        Results = runSim(rawFile, EventList, eventLogfile, TS3phOutFile)
    except: # some error happened, log the event which caused error and continue
        print('Suspect event: {}'.format(event))
        suspectEvents.append(event)
        continue

    # get any cases which do not converge
    with open(eventLogfile,'r') as f:
        eventLog = f.read()

    matchString = 'Exiting TS3ph at t = 30.0000'
    if matchString not in eventLog:
        suspectEvents.append(event)
        continue
    ##

    # get the voltage and angles and save them in separate files
    #tme = Results['time']
    for bus in BusDataDict:

        signalID = '{}/{}'.format(event,bus)
        eventListBusWise.append(signalID)

        Angle = Results[int(bus)].ang
        volt = Results[int(bus)].mag

        aWriterT.writerow(Angle[:transientEnd])
        aWriterS.writerow(Angle[-steadyStateStart:])

        vWriterT.writerow(volt[:transientEnd])
        vWriterS.writerow(volt[-steadyStateStart:])





# timeArrayT = np.array(tme[:transientEnd])
# timeArrayS = np.array(tme[-steadyStateStart:])

vFileT.close()
aFileT.close()


vFileS.close()
aFileS.close()






# write the corresponding event keys into a text file
with open(eventsFileName,'w') as f:
    for event in eventListBusWise:
        f.write(event)
        f.write('\n')

with open(outdir+'/suspectFile{}.txt'.format(i),'w') as f:
    for event in suspectEvents:
        f.write(event)
        f.write('\n')
