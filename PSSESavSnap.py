# script to simulate various kinds of scenarios in PSSE and plot them
import sys,os
from getBusDataFn import getBusData
psse_log = 'log.log'
# add psspy to the system path
sys.path.append(r"C:\Program Files (x86)\PTI\PSSE33\PSSBIN")
os.environ['PATH'] = (r"C:\Program Files (x86)\PTI\PSSE33\PSSBIN;"
                       + os.environ['PATH'])

# save the .sav and .snp file
from N_2Inputs import raw as rawFile, dyrFile
#rawFile = 'pf_ornl0823.raw'
#dyrFile = 'pf_ornl_all.dyr'
rawFileName = rawFile.replace('.raw','')

# Local imports
import redirect
import psspy
import dyntools


##### Get everything set up on the PSSE side
redirect.psse2py()

psspy.psseinit(buses=80000)
_i=psspy.getdefaultint()
_f=psspy.getdefaultreal()
_s=psspy.getdefaultchar()

# Redirect any psse outputs to psse_log
psspy.report_output(2,psse_log,[0,0])
psspy.progress_output(2,psse_log,[0,0]) #ignored
psspy.alert_output(2,psse_log,[0,0]) #ignored
psspy.prompt_output(2,psse_log,[0,0]) #ignored

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