# Function to run N-2 + fault contingencies in PSSE with different load levels
# currently saves the voltage, angle and frequency of the bus
# skips simulation if there are convergence issues and prints out a message
# System imports
from __future__ import with_statement
from contextlib import contextmanager
import os,sys
from getBusDataFn import getBusData
import dill as pickle # used to load more complicated data structures (in this case, a dictionary containing a class)
import StringIO

# add psspy to the system path
sys.path.append(r"C:\Program Files (x86)\PTI\PSSE33\PSSBIN")
os.environ['PATH'] = (r"C:\Program Files (x86)\PTI\PSSE33\PSSBIN;"
                       + os.environ['PATH'])
"""
# Select working path (current working directory) ##########################################################
os.chdir(r"C:\Users\Guest\Desktop\CAPE-TS Simulations\compare_scripts_CELIN")
################################################################################
"""

def save_obj(obj, name ):
    # save as pickle object
    currentdir = os.getcwd()
    objDir = currentdir + '/obj'
    if not os.path.isdir(objDir):
        os.mkdir(objDir)
    with open(objDir+ '/' +  name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

"""
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
"""

class Results():
    def __init__(self):
        self.volt = []
        self.angle = []
        self.freq = []



# Local imports
import redirect
import psspy
import dyntools
import matplotlib.pyplot as plt





# get the list of raw files to be considered for the simulation
fileList = os.listdir('.')
RawFileList = []
for file in fileList:
    if file.endswith('.raw') and 'savnw_conp' in file:
        #print file
        RawFileList.append(file)


refRaw = 'savnw_conp.raw'
dyrFile = 'savnw_dy_sol_0905.dyr'
psse_log = 'output.txt'
topology_inconsistency_file = 'topology_inconsistency_cases_savnw.txt'
BusDataDict = getBusData(refRaw)
#ResultsDict = {} # key: bus, value: Results structure
EventsDict = {}



# generate the HV bus set
HVBusSet = set()
for Bus in BusDataDict:
    BusVolt = float(BusDataDict[Bus].NominalVolt)
    BusType = BusDataDict[Bus].type
    if BusVolt >= 34.5:  # no need to consider type 4 buses since they are already filtered out in the get Bus Data function
        HVBusSet.add(Bus)


# get the N-2 events which cause topology inconsistencies
topology_inconsistent_set = set()
with open(topology_inconsistency_file, 'r') as f:
    fileLines = f.read().split('\n')
    for line in fileLines:
        if line == '':
            continue
        topology_inconsistent_set.add(line.strip())

# read the raw file and get the HV line set
HVLineSet = set()
with open(refRaw, 'r') as f:
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




# get the total simulations which are going to be carried out, to generate percentage updates while actually 
# running the simulations
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
        totalSims += 2

totalSims = totalSims*len(RawFileList)




''' Drives a PSS/E Dynamic simulation and returns values '''
simulatedEvents = set()


##### Get everything set up on the PSSE side
totalSimCount = 0

while totalSimCount < totalSims:
    redirect.psse2py()


    #output = StringIO.StringIO()
    #with silence():
    psspy.psseinit(buses=80000)
    _i=psspy.getdefaultint()
    _f=psspy.getdefaultreal()
    _s=psspy.getdefaultchar()


    # Redirect any psse outputs to psse_log
    psspy.report_output(2,psse_log,[0,0])
    psspy.progress_output(2,psse_log,[0,0]) #ignored
    psspy.alert_output(2,psse_log,[0,0]) #ignored
    psspy.prompt_output(2,psse_log,[0,0]) #ignored





    #k=1
    simCount = 0  # to keep track of how many simulations are already done
    for rawFile in RawFileList:

        # get the percentage loading from the raw file name
        if rawFile == 'savnw_conp.raw':
            PL = '100'
        else:
            rawFileName = rawFile.replace('.raw','')
            PL = rawFileName[-3:]


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


        ##### Load Raw Datafile and do power flow

        print "\n Reading raw file:",settings['filename']
        #    " Reading raw file: {0:s}".format('text')



        FaultRpu = 1e-06
        Sbase = 100.0

        #FaultBusNomVolt = float(BusDataDict[FaultBus].NominalVolt)


        #Zbase = FaultBusNomVolt**2/Sbase  # float since Sbase is a float
        #Rohm = FaultRpu*Zbase # fault impedance in ohms
        ##########################




        # run nested loops to see if there are any abnormal low voltages

        croppedHVLineSet = list(HVLineSet)

        for line1 in croppedHVLineSet:
            for line2 in croppedHVLineSet:
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


                line1Elements = line1.split(',')
                line2Elements = line2.split(',')

                # Line 1 params
                L1Bus1 = int(line1Elements[0])
                L1Bus2 = int(line1Elements[1])
                L1cktID = line1Elements[2].strip("'").strip()

                # Line 2 params
                L2Bus1 = int(line2Elements[0])
                L2Bus2 = int(line2Elements[1])
                L2cktID = line2Elements[2].strip("'").strip()


                FaultBusList = [L2Bus1, L2Bus2] # apply faults at both buses




                for FaultBus in FaultBusList:
                    eventStr = PL + '/' +  line1 + ';' + line2 + '/F' + str(FaultBus)
                    if eventStr in simulatedEvents:
                        continue
                    #output = StringIO.StringIO()
                    #with silence():
                    psse_log_old = psse_log
                    psse_log = 'output' + str(totalSimCount) + '.txt'

                    # Redirect any psse outputs to psse_log
                    psspy.report_output(2,psse_log,[0,0])
                    psspy.progress_output(2,psse_log,[0,0]) #ignored
                    psspy.alert_output(2,psse_log,[0,0]) #ignored
                    psspy.prompt_output(2,psse_log,[0,0]) #ignored




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

                    #print "\n Reading dyr file:",settings['dyr_file']

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




                    # add event to set
                    simulatedEvents.add(eventStr)

                    print 'Event: {}'.format(eventStr)
                    # get the nominal voltages as well as the fault impedance in ohms
                    FaultBusNomVolt = float(BusDataDict[str(FaultBus)].NominalVolt)
                    Zbase = FaultBusNomVolt**2/Sbase  # float since Sbase is a float
                    Rohm = FaultRpu*Zbase # fault impedance in ohms 
                 
                    # run simulation till just before the fault
                    ResultsDict = {}
                    

                    #output = StringIO.StringIO()
                    #with silence(output):
                    ierr = psspy.strt(0,settings['out_file'])
                    ierr = psspy.run(0,0.1,1,1,1)
                    ierr = psspy.dist_branch_trip(L1Bus1, L1Bus2, L1cktID)

                    #output = StringIO.StringIO()
                    #with silence(output):
                    ierr = psspy.run(0,0.2,1,1,1) #fault on time

                    """
                    outputStr = output.getvalue()
                    if "Network not converged" in outputStr:
                        print 'For ' + eventStr + ':'
                        print 'Network did not converge between branch 1 trip and fault application, skipping...'
                        continue
                    """
                    #######

                    # check for convergence during fault 
                    #output = StringIO.StringIO()
                    #with silence(output):
                    ierr = psspy.dist_bus_fault(FaultBus, 3, 0.0, [Rohm, 0.0])
                    ierr = psspy.run(0,0.3,1,1,1) #fault off time
                    ierr = psspy.dist_clear_fault(1)
                        
                    """
                    outputStr = output.getvalue()
                    if "Network not converged" in outputStr:
                        print 'For ' + eventStr + ':'
                        print 'Network did not converge during fault, skipping...'
                        continue
                    """

                    # check for convergence between fault clearance and second branch trip
                    #output = StringIO.StringIO()
                    #with silence(output):
                    ierr = psspy.run(0,0.31,1,1,1) #fault off time
                    ierr = psspy.dist_branch_trip(L2Bus1, L2Bus2,L2cktID)
                    ierr = psspy.run(0,0.35,1,1,1) #fault off time

                    """
                    # check for non-convergence
                    #output = StringIO.StringIO()
                    outputStr = output.getvalue()
                    if "Network not converged" in outputStr:
                        print 'For ' + eventStr + ':'
                        print 'Network did not converge between fault clearance and branch 2 trip, skipping...'
                        continue
                    """


                    # select run time ##############################################################
                    #output = StringIO.StringIO()
                    #with silence(output):
                    ierr = psspy.run(0,10.0,1,1,1) #exit time (second argument is the end time)
                    ################################################################################
                    """
                    # check for non-convergence
                    outputStr = output.getvalue()
                    if "Network not converged" in outputStr:
                        print 'For ' + eventStr + ':'
                        print 'Network did not converge sometime after 2nd branch trip, skipping...'
                        continue
                    """

                    with open(psse_log,'r') as f:
                        filecontent = f.read()

                    # clear the old log file
                    os.remove(psse_log_old)


                    if 'Network not converged' in filecontent:
                        print '{} has convergence issues. Skipping...'.format(eventStr)
                        simCount +=1
                        totalSimCount +=1
                        continue




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

                    
                    EventsDict[eventStr] = ResultsDict
                    
                    print 'Simulation ' + str(totalSimCount) + ' out of ' + str(totalSims)
                    simCount +=1
                    totalSimCount +=1

                    # Uncomment next two lines if you want to see the output
                    #with open('output'+str(k) + '.txt','w') as f:
                    #    f.write(outputStr)
                    #k+=1


                    if simCount >= 10:
                        break
                if simCount >= 10:
                    break
            if simCount >= 10:
                break               
        if simCount >= 10:
            break

    if simCount >= 10:
        #ierr = psspy.stop_2()
        break

save_obj(EventsDict,'EventData')
