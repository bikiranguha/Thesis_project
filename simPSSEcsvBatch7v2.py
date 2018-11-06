from __future__ import with_statement
  
#def runPSSESimBatches(simList,dyrFile,objName):
import sys,os
import pickle
import h5py
from getBusDataFn import getBusData
import numpy as np
import csv
# add psspy to the system path
sys.path.append(r"C:\Program Files (x86)\PTI\PSSE33\PSSBIN")
os.environ['PATH'] = (r"C:\Program Files (x86)\PTI\PSSE33\PSSBIN;"
                       + os.environ['PATH'])
refRaw = 'savnw_conp.raw'
BusDataDict = getBusData(refRaw)
eventListFile = 'events.txt'

eventsList = []
with open(eventListFile,'r') as f:
    fileLines = f.read().split('\n')
    for line in fileLines:
        if line == '':
            continue
        eventsList.append(line.strip())

# Names to change
# dump the arrays into csv files
simList = eventsList[4800:5600]
vName = 'obj/v{}.csv'.format(7)
aName = 'obj/a{}.csv'.format(7)
fName = 'obj/f{}.csv'.format(7)
eventsFileName = 'obj/eventKeys_{}.csv'.format(7)
vFile = open(vName, 'wb') # 'wb' needed to avoid blank space in between lines
aFile = open(aName, 'wb')
fFile = open(fName, 'wb')

vWriter = csv.writer(vFile)
aWriter = csv.writer(aFile)
fWriter = csv.writer(fFile)

####################









dyrFile = 'savnw_dy_sol_0905.dyr'

from contextlib import contextmanager
#from io import StringIO # for python3 
import StringIO
from getBusDataFn import getBusData


def save_obj(obj, name ):
    # save as pickle object
    currentdir = os.getcwd()
    objDir = currentdir + '/obj'
    if not os.path.isdir(objDir):
        os.mkdir(objDir)
    with open(objDir+ '/' +  name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


@contextmanager
def silence(file_object=None):

    #Discard stdout (i.e. write to null device) or
    #optionally write to given file-like object.
    
    if file_object is None:
        file_object = open(os.devnull, 'w')

    old_stdout = sys.stdout
    try:
        sys.stdout = file_object
        yield
    finally:
        sys.stdout = old_stdout
        if file_object is None:
            file_object.close()




# Local imports
import redirect
import psspy
import dyntools


# getting the raw file



##### Get everything set up on the PSSE side
redirect.psse2py()


#output = StringIO.StringIO()
with silence():
    psspy.psseinit(buses=80000)
    _i=psspy.getdefaultint()
    _f=psspy.getdefaultreal()
    _s=psspy.getdefaultchar()


# some important parameters
FaultRpu = 1e-06
Sbase = 100.0





out_file = 'output2.out'
#### iterate over all the events in the given list
keyList = []
voltList = []
angleList = []
freqList = []
for event in simList:
    eventWords = event.split('/')
    RawFileIndicator = eventWords[0].strip()
    linesOutage = eventWords[1].strip()
    FaultBus = eventWords[2].strip()[1:] # exclude the 'F' at the beginning


    # get the line params
    line1Elements = linesOutage.split(';')[0].strip()
    line2Elements = linesOutage.split(';')[1].strip()


    # Line 1 params
    line1 = line1Elements.split(',')
    L1Bus1 = int(line1[0].strip())
    L1Bus2 = int(line1[1].strip())
    L1cktID = line1[2].strip("'").strip()


    # Line 2 params
    line2 = line2Elements.split(',')
    L2Bus1 = int(line2[0].strip())
    L2Bus2 = int(line2[1].strip())
    L2cktID = line2[2].strip("'").strip()

     # get the raw file name, to be used for getting the corresponding names of the sav and snp files
    if RawFileIndicator == '100':
        rawFileName = 'savnw_conp'
    else:
        rawFileName = 'savnw_conp{}'.format(RawFileIndicator)       


    savFile = rawFileName + '.sav'
    snpFile = rawFileName + '.snp'


    print('Event: {}'.format(event))
    # get the nominal voltages as well as the fault impedance in ohms
    FaultBusNomVolt = float(BusDataDict[str(FaultBus)].NominalVolt)
    Zbase = FaultBusNomVolt**2/Sbase  # float since Sbase is a float
    Rohm = FaultRpu*Zbase # fault impedance in ohms 
             
    # run simulation till just before the fault
    output = StringIO.StringIO()
    with silence(output):
        # load the sav and snp file
        psspy.case(savFile)
        psspy.rstr(snpFile)
    output = StringIO.StringIO()
    with silence(output):
        ierr = psspy.strt(0,out_file)
        ierr = psspy.run(0,0.1,1,1,1)
        ierr = psspy.dist_branch_trip(L1Bus1, L1Bus2, L1cktID)

    output = StringIO.StringIO()
    with silence(output):
        ierr = psspy.run(0,0.2,1,1,1) #fault on time

    outputStr = output.getvalue()
    if "Network not converged" in outputStr:
        print('For ' + event + ':')
        print('Network did not converge between branch 1 trip and fault application, skipping...')
        continue
    #######

    # check for convergence during fault 
    output = StringIO.StringIO()
    with silence(output):
        ierr = psspy.dist_bus_fault(int(FaultBus), 3, 0.0, [Rohm, 0.0])
        ierr = psspy.run(0,0.3,1,1,1) #fault off time
        ierr = psspy.dist_clear_fault(1)
        
    outputStr = output.getvalue()
    if "Network not converged" in outputStr:
        print('For ' + event + ':')
        print('Network did not converge during fault, skipping...')
        continue

    # check for convergence between fault clearance and second branch trip
    #output = StringIO.StringIO()
    with silence(output):
        ierr = psspy.run(0,0.31,1,1,1) #fault off time
        ierr = psspy.dist_branch_trip(L2Bus1, L2Bus2,L2cktID)
        ierr = psspy.run(0,0.35,1,1,1) #fault off time

    # check for non-convergence
    output = StringIO.StringIO()
    outputStr = output.getvalue()
    if "Network not converged" in outputStr:
        print('For ' + event + ':')
        print('Network did not converge between fault clearance and branch 2 trip, skipping...')
        continue


    # select run time ##############################################################
    output = StringIO.StringIO()
    with silence(output):
        ierr = psspy.run(0,10.0,1,1,1) #exit time (second argument is the end time)
    ################################################################################
    # check for non-convergence
    
    outputStr = output.getvalue()
    if "Network not converged" in outputStr:
        print('For ' + event + ':')
        print('Network did not converge sometime after 2nd branch trip, skipping...')
        continue

    # write to output file
    #with open('outputTmp.txt','w') as f:
    #   f.write(outputStr)


    outputData = dyntools.CHNF(out_file)

    data = outputData.get_data()

    channelDict = data[1] # dictionary where the value is the channel description
    valueDict = data[2] # dictionary where the values are the signal values, keys match that of channelDict


    tme = valueDict['time'] # get time
    keySet = set()
    for key in channelDict:
        if key == 'time':
            continue

        signalDescr = channelDict[key]
        words = signalDescr.split()
        signalType = words[0].strip()
        bus = words[1].strip()
        fullKey = event + '/' +  bus
        if fullKey not in keySet:
            keyList.append(fullKey)
            keySet.add(fullKey)

        if signalType == 'VOLT':
            vWriter.writerow(valueDict[key])
            #voltList.append(valueDict[key])

        elif signalType == 'ANGL':
            aWriter.writerow(valueDict[key])
            #angleList.append(valueDict[key])
        elif signalType == 'FREQ':
            fWriter.writerow(valueDict[key])
            #freqList.append(valueDict[key])
###############

"""
# convert the signal lists to arrays
voltArray = np.array(voltList)
angleArray = np.array(angleList)
freqArray = np.array(freqList)

"""
timeArray = np.array(tme)

"""
# save all the signal arrays to an h5py file
h5fFileName = 'obj/{}.h5'.format(rawFileName)
h5f = h5py.File(h5fFileName, 'w')
h5f.create_dataset('volt', data=voltArray)
h5f.create_dataset('angle', data=angleArray)
h5f.create_dataset('freq', data=freqArray)
h5f.create_dataset('time',data=timeArray)
h5f.close()
"""




#np.savetxt(vName, voltArray, delimiter=",")
#np.savetxt(aName, angleArray, delimiter=",")
#np.savetxt(fName, freqArray, delimiter=",")
vFile.close()
aFile.close()
fFile.close()
# write the corresponding event keys into a text file

with open(eventsFileName,'w') as f:
    for key in keyList:
        f.write(key)
        f.write('\n')


# need to save time array only once
np.savetxt("obj/timeArray.csv", timeArray, delimiter=",")