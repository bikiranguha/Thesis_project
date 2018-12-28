# write a script which outages all the generators and saves voltage, angle, frequency

from __future__ import with_statement
  
#def runPSSESimBatches(simList,dyrFile,objName):
import sys,os
import pickle
import h5py
from getBusDataFn import getBusData
import numpy as np

from contextlib import contextmanager
#from io import StringIO # for python3 
import StringIO
from getBusDataFn import getBusData
import matplotlib.pyplot as plt
# add psspy to the system path
sys.path.append(r"C:\Program Files (x86)\PTI\PSSE33\PSSBIN")
os.environ['PATH'] = (r"C:\Program Files (x86)\PTI\PSSE33\PSSBIN;"
                       + os.environ['PATH'])


rawFile = 'savnw_conp.raw'
dyrFile = 'savnw_dy_sol_0905.dyr'
BusDataDict = getBusData(rawFile)

# get the set of generator buses alongwith the machine id
GenDict = {}
with open(rawFile, 'r') as f:
    fileLines = f.read().split('\n')
genStartIndex = fileLines.index('0 / END OF FIXED SHUNT DATA, BEGIN GENERATOR DATA') + 1
genEndIndex = fileLines.index('0 / END OF GENERATOR DATA, BEGIN BRANCH DATA')
for i in range(genStartIndex, genEndIndex):
    line = fileLines[i]
    words = line.split(',')
    bus = words[0].strip()
    cktID = words[1].strip("'").strip()
    GenDict[bus] = [bus, cktID]

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

#Parameters. CONFIGURE THIS
settings = {
# use the same raw data in PSS/E and TS3ph #####################################
    'filename':rawFile, #use the same raw data in PSS/E and TS3ph
################################################################################
    'dyr_file':dyrFile,
    'out_file':'output2.out',
    'pf_options':[
        0,  #disable taps
        0,  #disable area exchange
        0,  #disable phase-shift
        0,  #disable dc-tap
        0,  #disable switched shunts
        0,  #do not flat start
        0,  #apply var limits immediately
        0,  #disable non-div solution
    ]
}


genbus = '101'
genID = '1'
voltList = []
angleList = []
freqList = []
rawFileName = 'savnw_conp'
with silence():
    ierr = psspy.read(0, settings['filename'])
    #This is for the power flow. I'll use the solved case instead
    ierr = psspy.fnsl(settings['pf_options'])


    ##### Prepare case for dynamic simulation
    # Load conversion (multiple-step)
    psspy.conl(_i,_i,1,[0,_i],[_f,_f,_f,_f])
    # all constant power load to constant current, constant reactive power load to constant admittance
    # standard practice for dynamic simulations, constant MVA load is not acceptable
    psspy.conl(1,1,2,[_i,_i],[100.0, 0.0,0.0, 100.0]) 
    psspy.conl(_i,_i,3,[_i,_i],[_f,_f,_f,_f])
    

    ierr = psspy.cong(0) #converting generators
    ierr = psspy.ordr(0) #order the network nodes to maintain sparsity
    ierr = psspy.fact()  #factorise the network admittance matrix
    ierr = psspy.tysl(0) #solving the converted case
    ierr = psspy.dynamicsmode(0) #enter dynamics mode

    print("\n Reading dyr file:",settings['dyr_file'])

    ierr = psspy.dyre_new([1,1,1,1], settings['dyr_file'])
    ierr=psspy.docu(0,1,[0,3,1]) #print the starting point of state variables

    # select time step ##############################################################
    ierr = psspy.dynamics_solution_params([_i,_i,_i,_i,_i,_i,_i,_i], [_f,_f,0.00833333333333333,_f,_f,_f,_f,_f], 'out_file') # the number here is the time step
    ################################################################################

    ##### select channels
    ierr = psspy.delete_all_plot_channels() # clear channels

    BusDataDict = getBusData(rawFile)
    # get all the bus voltages, angles and frequencies
    for bus  in BusDataDict:
        bus = int(bus)
        ierr = psspy.voltage_and_angle_channel([-1, -1, -1, bus])
        ierr = psspy.bus_frequency_channel([-1, bus])


    savFile = rawFileName + '.sav'
    snpFile = rawFileName + '.snp'
    ierr = psspy.save(savFile)
    ierr = psspy.snap([_i,_i,_i,_i,_i],snpFile)







output = StringIO.StringIO()
with silence(output):
    # load the sav and snp file
    psspy.case(savFile)
    psspy.rstr(snpFile)

# trip gen
output = StringIO.StringIO()
with silence(output):
    ierr = psspy.strt(0,settings['out_file'])
    ierr = psspy.run(0,0.1,1,1,1)
    ierr = psspy.dist_machine_trip(int(genbus),genID)

output = StringIO.StringIO()
with silence(output):
    ierr = psspy.run(0,0.2,1,1,1) #fault on time

outputStr = output.getvalue()
if "Network not converged" in outputStr:
    print('For ' + eventStr + ':')
    print('Network did not converge between branch 1 trip and fault application, skipping...')
        #continue
#######


outputData = dyntools.CHNF(settings['out_file'])

data = outputData.get_data()

channelDict = data[1] # dictionary where the value is the channel description
valueDict = data[2] # dictionary where the values are the signal values, keys match that of channelDict


tme = valueDict['time'] # get time

for key in channelDict:
    if key == 'time':
        continue

    signalDescr = channelDict[key]
    words = signalDescr.split()
    signalType = words[0].strip()
    bus = words[1].strip()
    #fullKey = eventStr + '/' +  bus
    #keyList.append(fullKey)

    if signalType == 'VOLT':
        voltList.append(valueDict[key])

    elif signalType == 'ANGL':
        angleList.append(valueDict[key])
    elif signalType == 'FREQ':
        freqList.append(valueDict[key])

# convert the signal lists to arrays
voltArray = np.array(voltList)
angleArray = np.array(angleList)
freqArray = np.array(freqList)
timeArray = np.array(tme)


# plot
for freq in freqList:
    plt.plot(freq)
plt.grid()
plt.show()

"""
vName = 'obj/' + rawFileName + 'v.csv'
aName = 'obj/' + rawFileName + 'a.csv'
fName = 'obj/' + rawFileName + 'f.csv'

np.savetxt(vName, voltArray, delimiter=",")
np.savetxt(aName, angleArray, delimiter=",")
np.savetxt(fName, freqArray, delimiter=",")
    



# need to save time array only once
np.savetxt("obj/timeArray.csv", timeArray, delimiter=",")
"""