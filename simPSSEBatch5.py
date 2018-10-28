from __future__ import with_statement
class Results():
    def __init__(self):
        self.volt = []
        self.angle = []
        self.freq = []
        
#def runPSSESimBatches(simList,dyrFile,objName):
import sys,os
import dill as pickle
# add psspy to the system path
sys.path.append(r"C:\Program Files (x86)\PTI\PSSE33\PSSBIN")
os.environ['PATH'] = (r"C:\Program Files (x86)\PTI\PSSE33\PSSBIN;"
                       + os.environ['PATH'])





eventListFile = 'events.txt'

eventsList = []
with open(eventListFile,'r') as f:
    fileLines = f.read().split('\n')
    for line in fileLines:
        if line == '':
            continue
        eventsList.append(line.strip())

dyrFile = 'savnw_dy_sol_0905.dyr'

from contextlib import contextmanager
import StringIO
from getBusDataFn import getBusData


def save_obj(obj, name ):
    # save as pickle object
    currentdir = os.getcwd()
    objDir = currentdir + '/obj'
    if not os.path.isdir(objDir):
        os.mkdir(objDir)
    with open(objDir+ '/' +  name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL,recurse = 'True')


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
EventsDict = {}



simList = eventsList[3200:4000]


for event in simList:
    eventWords = event.split('/')
    RawFileIndicator = eventWords[0].strip()
    linesOutage = eventWords[1].strip()
    FaultBus = eventWords[2].strip()[1:] # exclude the 'F' at the beginning

    # get the raw file
    if RawFileIndicator == '100':
        rawFile = 'savnw_conp.raw'
    else:
        rawFile = 'savnw_conp{}.raw'.format(RawFileIndicator)



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


    
    output = StringIO.StringIO()
    with silence(output):
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

        BusDataDict = getBusData(rawFile)
        # get all the bus voltages, angles and frequencies
        for bus  in BusDataDict:
            bus = int(bus)
            ierr = psspy.voltage_and_angle_channel([-1, -1, -1, bus])
            ierr = psspy.bus_frequency_channel([-1, bus])



    print 'Event: {}'.format(event)
    
    # get the nominal voltages as well as the fault impedance in ohms
    FaultBusNomVolt = float(BusDataDict[str(FaultBus)].NominalVolt)
    Zbase = FaultBusNomVolt**2/Sbase  # float since Sbase is a float
    Rohm = FaultRpu*Zbase # fault impedance in ohms 
 
    # run simulation till just before the fault
    ResultsDict = {}

    

    # get the line params
    line1Elements = linesOutage.split(';')[0].strip()
    line2Elements = linesOutage.split(';')[1].strip()


    # Line 1 params
    line1 = line1Elements.split(',')
    L1Bus1 = int(line1[0].strip())
    L1Bus2 = int(line1[1].strip())
    L1cktID = line1[2].strip("'").strip()
    #print L1Bus1
    #print L1Bus2
    #print L1cktID

    # Line 2 params
    line2 = line2Elements.split(',')
    L2Bus1 = int(line2[0].strip())
    L2Bus2 = int(line2[1].strip())
    L2cktID = line2[2].strip("'").strip()
    #print L2Bus1
    #print L2Bus2
    #print L2cktID


    #output = StringIO.StringIO()
    with silence(output):
        ierr = psspy.strt(0,settings['out_file'])
        ierr = psspy.run(0,0.1,1,1,1)
        ierr = psspy.dist_branch_trip(L1Bus1, L1Bus2, L1cktID)

    #output = StringIO.StringIO()
    with silence(output):
        ierr = psspy.run(0,0.2,1,1,1) #fault on time

    outputStr = output.getvalue()
    if "Network not converged" in outputStr:
        print 'For ' + event + ':'
        print 'Network did not converge between branch 1 trip and fault application, skipping...'
        continue
    #######

    # check for convergence during fault 
    #output = StringIO.StringIO()
    with silence(output):
        ierr = psspy.dist_bus_fault(int(FaultBus), 3, 0.0, [Rohm, 0.0])
        ierr = psspy.run(0,0.3,1,1,1) #fault off time
        ierr = psspy.dist_clear_fault(1)
        
    outputStr = output.getvalue()
    if "Network not converged" in outputStr:
        print 'For ' + event + ':'
        print 'Network did not converge during fault, skipping...'
        continue

    # check for convergence between fault clearance and second branch trip
    #output = StringIO.StringIO()
    with silence(output):
        ierr = psspy.run(0,0.31,1,1,1) #fault off time
        ierr = psspy.dist_branch_trip(L2Bus1, L2Bus2,L2cktID)
        ierr = psspy.run(0,0.35,1,1,1) #fault off time

    # check for non-convergence
    #output = StringIO.StringIO()
    outputStr = output.getvalue()
    if "Network not converged" in outputStr:
        print 'For ' + event + ':'
        print 'Network did not converge between fault clearance and branch 2 trip, skipping...'
        continue


    # select run time ##############################################################
    #output = StringIO.StringIO()
    with silence(output):
        ierr = psspy.run(0,10.0,1,1,1) #exit time (second argument is the end time)
    ################################################################################
    # check for non-convergence
    
    outputStr = output.getvalue()
    if "Network not converged" in outputStr:
        print 'For ' + event + ':'
        print 'Network did not converge sometime after 2nd branch trip, skipping...'
        continue

    # write to output file
    #with open('outputTmp.txt','w') as f:
    #   f.write(outputStr)


    outputData = dyntools.CHNF(settings['out_file'])

    data = outputData.get_data()

    channelDict = data[1] # dictionary where the value is the channel description
    valueDict = data[2] # dictionary where the values are the signal values, keys match that of channelDict


    tme = valueDict['time'] # get time
    ResultsDict['time'] = tme
    for key in channelDict:
        if key == 'time':
            continue

        signalDescr = channelDict[key]
        words = signalDescr.split()
        signalType = words[0].strip()
        bus = words[1].strip()
        #print Bus + ' ' + signalType
        if bus not in ResultsDict:
            ResultsDict[bus] = Results()

        if signalType == 'VOLT':
            ResultsDict[bus].volt = valueDict[key]

        elif signalType == 'ANGL':
            ResultsDict[bus].angle = valueDict[key]
        elif signalType == 'FREQ':
            ResultsDict[bus].freq = valueDict[key]

    
    EventsDict[event] = ResultsDict


save_obj(EventsDict,'Event4')