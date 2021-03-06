# run a single N-2 contingency in PSSE
# checks for non-convergence during all the important steps and prints a warning if there is any non-convergence
from __future__ import with_statement
from contextlib import contextmanager
import os,sys
from getBusDataFn import getBusData
import dill as pickle # used to load more complicated data structures (in this case, a dictionary containing a class)
import StringIO
import csv
import numpy as np

# add psspy to the system path
sys.path.append(r"C:\Program Files (x86)\PTI\PSSE33\PSSBIN")
os.environ['PATH'] = (r"C:\Program Files (x86)\PTI\PSSE33\PSSBIN;"
                       + os.environ['PATH'])
"""
# Select working path (current working directory) ##########################################################
os.chdir(r"C:\Users\Guest\Desktop\CAPE-TS Simulations\compare_scripts_CELIN")
################################################################################
"""
#################
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
    """
    Discard stdout (i.e. write to null device) or
    optionally write to given file-like object.
    """
    if file_object is None:
        file_object = open(os.devnull, 'w')

    old_stdout = sys.stdout # save the current std output
    try:
        sys.stdout = file_object
        yield
    finally:
        sys.stdout = old_stdout # after the function is done, get back the original standard output


# output files
vName = 'obj/vTest.csv'
aName = 'obj/aTest.csv'
fName = 'obj/fTest.csv'
eventsFileName = 'obj/eventKeysTest.csv'
vFile = open(vName, 'wb') # 'wb' needed to avoid blank space in between lines
aFile = open(aName, 'wb')
fFile = open(fName, 'wb')

vWriter = csv.writer(vFile)
aWriter = csv.writer(aFile)
fWriter = csv.writer(fFile)



############################
# Local imports
import redirect
import psspy
import dyntools
import matplotlib.pyplot as plt
#rawFile = 'savnw_dy_sol_0905.raw'
#rawFile = 'savnw.raw'
rawFile = 'savnw_conp.raw'
dyrFile = 'savnw_dy_sol_0905.dyr'
psse_log = 'output.txt'

BusDataDict = getBusData(rawFile)
#ResultsDict = {} # key: bus, value: Results structure
EventsDict = {}
''' Drives a PSS/E Dynamic simulation and returns values '''

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















##### Get everything set up on the PSSE side
redirect.psse2py()

#output = StringIO.StringIO()
with silence():
    psspy.psseinit(buses=80000)
    _i=psspy.getdefaultint()
    _f=psspy.getdefaultreal()
    _s=psspy.getdefaultchar()

    """
    # Redirect any psse outputs to psse_log
    psspy.report_output(2,psse_log,[0,0])
    psspy.progress_output(2,psse_log,[0,0]) #ignored
    psspy.alert_output(2,psse_log,[0,0]) #ignored
    psspy.prompt_output(2,psse_log,[0,0]) #ignored
    """

    ##### Load Raw Datafile and do power flow

    print "\n Reading raw file:",settings['filename']
    #    " Reading raw file: {0:s}".format('text')

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

    print "\n Reading dyr file:",settings['dyr_file']

    ierr = psspy.dyre_new([1,1,1,1], settings['dyr_file'])

    ierr=psspy.docu(0,1,[0,3,1]) #print the starting point of state variables

    # select time step ##############################################################
    ierr = psspy.dynamics_solution_params([_i,_i,_i,_i,_i,_i,_i,_i], [_f,_f,0.00833333333333333,_f,_f,_f,_f,_f], 'out_file') # the number here is the time step
    ################################################################################

    ##### select channels
    ierr = psspy.delete_all_plot_channels() # clear channels


    # get all the bus voltages, angles and frequencies
    for bus  in BusDataDict:
        bus = int(bus)
        ierr = psspy.voltage_and_angle_channel([-1, -1, -1, bus])
        ierr = psspy.bus_frequency_channel([-1, bus])



    FaultRpu = 1e-06
    Sbase = 100.0


    # simulate the event
    #event = '152,3004,1;3007,3008,1/F3008'
    event = '151,201,1;201,202,1/F201'

    eventWords = event.split('/')
    lines = eventWords[0].strip()
    FaultBus = int(eventWords[1][1:]) # skip 'F' 

    lineElements = lines.split(';')

    line1Elements = lineElements[0].split(',')
    line2Elements = lineElements[1].split(',')

    # Line 1 params
    L1Bus1 = int(line1Elements[0])
    L1Bus2 = int(line1Elements[1])
    L1cktID = line1Elements[2].strip("'").strip()

    # Line 2 params
    L2Bus1 = int(line2Elements[0])
    L2Bus2 = int(line2Elements[1])
    L2cktID = line2Elements[2].strip("'").strip()




    # get the nominal voltages as well as the fault impedance in ohms
    FaultBusNomVolt = float(BusDataDict[str(FaultBus)].NominalVolt)
    Zbase = FaultBusNomVolt**2/Sbase  # float since Sbase is a float
    Rohm = FaultRpu*Zbase # fault impedance in ohms 
    #print Rohm           
    # run simulation
    ierr = psspy.strt(0,settings['out_file'])
    ierr = psspy.run(0,1.0,1,1,1)
    ierr = psspy.dist_branch_trip(L1Bus1, L1Bus2, L1cktID) # trip branch

output = StringIO.StringIO() # reset output
with silence(output):
    ierr = psspy.run(0,2.0,1,1,1) #fault on time

outputStr = output.getvalue()
if "Network not converged" in outputStr:
    print 'Case does not converge after branch outage before fault'
#print outputStr



#######
output = StringIO.StringIO()
with silence(output):
    # run fault, clear fault
    ierr = psspy.dist_bus_fault(FaultBus, 3, 0.0, [Rohm, 0.0])
    ierr = psspy.run(0,2.1,1,1,1) #fault off time
    ierr = psspy.dist_clear_fault(1)
outputStr = output.getvalue()
if "Network not converged" in outputStr:
    print 'Case does not converge during fault'


output = StringIO.StringIO()
with silence(output):
    # branch trip
    ierr = psspy.run(0,2.11,1,1,1) #fault off time
    ierr = psspy.dist_branch_trip(L2Bus1, L2Bus2,L2cktID) # trip branch
outputStr = output.getvalue()
if "Network not converged" in outputStr:
    print 'Case does not converge after fault cleared but before 2nd branch outage'


output = StringIO.StringIO()
with silence(output):
    # run till end time
    ierr = psspy.run(0,10.0,1,1,1) #exit time (second argument is the end time)
outputStr = output.getvalue()
if "Network not converged" in outputStr:
    print 'Case does not converge after 2nd branch outage'


outputData = dyntools.CHNF(settings['out_file'])

data = outputData.get_data()

channelDict = data[1] # dictionary where the value is the channel description
valueDict = data[2] # dictionary where the values are the signal values, keys match that of channelDict

keyList = []
tme = valueDict['time'] # get time
keySet = set()
for key in channelDict:
    if key == 'time':
        continue

    signalDescr = channelDict[key]
    words = signalDescr.split()
    signalType = words[0].strip()
    bus = words[1].strip()
    #print Bus + ' ' + signalType
    fullKey = event + '/' +  bus
    if fullKey not in keySet:
        keyList.append(fullKey)
        keySet.add(fullKey)
    if signalType == 'VOLT':
        vWriter.writerow(valueDict[key])

    elif signalType == 'ANGL':
        aWriter.writerow(valueDict[key])

    elif signalType == 'FREQ':
        fWriter.writerow(valueDict[key])





vFile.close()
aFile.close()
fFile.close()
# write the corresponding event keys into a text file

with open(eventsFileName,'w') as f:
    for key in keyList:
        f.write(key)
        f.write('\n')


timeArray = np.array(tme)
np.savetxt("obj/timeTest.csv", timeArray, delimiter=",")

