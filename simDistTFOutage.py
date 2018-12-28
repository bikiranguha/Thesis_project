# simulate transformer outages using PSS/E

from __future__ import with_statement
  
#def runPSSESimBatches(simList,dyrFile,objName):
import sys,os
import pickle
import h5py
from getBusDataFn import getBusData
import numpy as np
import csv
from contextlib import contextmanager
import StringIO
# add psspy to the system path
sys.path.append(r"C:\Program Files (x86)\PTI\PSSE33\PSSBIN")
os.environ['PATH'] = (r"C:\Program Files (x86)\PTI\PSSE33\PSSBIN;"
                       + os.environ['PATH'])


# read the baseline raw file (with standard load)
refRaw = 'savnw_conp.raw'
#refRaw = 'savnw.raw'
# get the generator buses and corresponding circuit id
with open(refRaw, 'r') as f:
    filecontent = f.read()
    fileLines = filecontent.split('\n')


# get the list of raw files to be considered for the simulation
fileList = os.listdir('.')
RawFileList = []
for file in fileList:
    if file.endswith('.raw') and 'savnw_conp' in file:
        RawFileList.append(file)


###
# get the transformer ids in the raw file
tfStartIndex = fileLines.index('0 / END OF BRANCH DATA, BEGIN TRANSFORMER DATA') + 1
tfEndIndex = fileLines.index('0 / END OF TRANSFORMER DATA, BEGIN AREA DATA')

tfSet = set() # set of transformers in the raw file
tfKeySample =   "  3008,  3018,     0,'1 '"
tfKeyLen = len(tfKeySample)
i = tfStartIndex
while i < tfEndIndex:
    line = fileLines[i]
    words = line.split(',')
    Bus1 = words[0].strip()
    Bus2 = words[1].strip()
    Bus3 = words[2].strip()
    cktID = words[3].strip("'").strip()
    tfKey = line[:tfKeyLen]
    tfSet.add(tfKey)
    #print(tfKey)
    if Bus3 == '0':
        i+=4
        continue
    else:
        i+=5 
        continue
####

# file objects for line outages
voltTFOut = 'vTFOut.csv'
anglTFOut = 'aTFOut.csv'
freqTFOut = 'fTFOut.csv'
timeDataFileName = 'tTFOut.csv'
eventIDFileTFOut = 'eventTFOut.txt'
vFileTFOut = open(voltTFOut, 'wb') # 'wb' needed to avoid blank space in between lines
aFileTFOut = open(anglTFOut, 'wb')
fFileTFOut = open(freqTFOut, 'wb')
tFile = open(timeDataFileName, 'wb')


eventFileObjTFOut = open(eventIDFileTFOut, 'w')
eventHeader = 'TFOut/Bus'
eventFileObjTFOut.write(eventHeader)
eventFileObjTFOut.write('\n')


writerObjVTFOut = csv.writer(vFileTFOut)
writerObjATFOut = csv.writer(aFileTFOut)
writerObjFTFOut = csv.writer(fFileTFOut)
timeObj = csv.writer(tFile)
####


# define the silence function
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



redirect.psse2py()


#output = StringIO.StringIO()
with silence():
    psspy.psseinit(buses=80000)
    _i=psspy.getdefaultint()
    _f=psspy.getdefaultreal()
    _s=psspy.getdefaultchar()



######
# define the raw file and get the stuff needed from raw file
#rawFile  = 'savnw_conp.raw'
dyrFile = 'savnw_dy_sol_0905.dyr'

totSimTime = 5.0
eventSet = set()
for rawFile in RawFileList:

    print 'Reading raw file: {}'.format(rawFile)
    rawFileName = rawFile.replace('.raw','')
    # get the percentage loading from the raw file name
    if rawFile == 'savnw_conp.raw':
        PL = '100'
        rawFileName = rawFile.replace('.raw','')
    else:
        rawFileName = rawFile.replace('.raw','')
        PL = rawFileName[-3:]




    ###############
    # run simulations





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


    # set up the sav and snp files
    
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


    #######
    # simulate line outages
    print 'Simulating tf outages....'
    for tf in list(tfSet):

        # get the line elements
        tfElements = tf.split(',')
        bus1 = tfElements[0].strip()
        bus2 = tfElements[1].strip()
        bus3 = tfElements[2].strip()
        cktID  = tfElements[3].strip("'").strip()


        eventStr = 'R{}/T{}'.format(PL,tf)
        #cktID = GenDict[genbus]
        output = StringIO.StringIO()
        with silence(output):
            # load the sav and snp file
            psspy.case(savFile)
            psspy.rstr(snpFile)
        #output = StringIO.StringIO()
        with silence(output):
            ierr = psspy.strt(0,settings['out_file'])
            ierr = psspy.run(0,0.1,1,1,1)
            ierr = psspy.dist_branch_trip(int(bus1), int(bus2),cktID) # this line can be used for 2 winding tf as well

        #output = StringIO.StringIO()
        with silence(output):
            ierr = psspy.run(0,totSimTime,1,1,1) #simulation end time

        outputStr = output.getvalue()
        if "Network not converged" in outputStr:
            print('For ' + eventStr + ':')
            print('Network did not converge after tf trip, skipping...')
            continue


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

            # append the event description
            eventID = '{}/B{}'.format(eventStr, bus)
            if eventID not in eventSet:
                eventFileObjTFOut.write(eventID)
                eventFileObjTFOut.write('\n')
                eventSet.add(eventID)

            # add the voltage, freq and angle data
            if signalType == 'VOLT':
                writerObjVTFOut.writerow(valueDict[key])

            elif signalType == 'ANGL':
                writerObjATFOut.writerow(valueDict[key])

            elif signalType == 'FREQ':
                writerObjFTFOut.writerow(valueDict[key])
    ########

with open('tmp.txt','w') as f:
    f.write(outputStr)
vFileTFOut.close()
aFileTFOut.close()
fFileTFOut.close()
eventFileObjTFOut.close()

# save the time data (needed only once)
timeObj.writerow(tme)
tFile.close()