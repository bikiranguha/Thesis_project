# script to simulate various kinds of scenarios in PSSE and plot them
import sys,os
import matplotlib.pyplot as plt
from getBusDataFn import getBusData
import numpy as np
# add psspy to the system path
sys.path.append(r"C:\Program Files (x86)\PTI\PSSE33\PSSBIN")
os.environ['PATH'] = (r"C:\Program Files (x86)\PTI\PSSE33\PSSBIN;"
                       + os.environ['PATH'])


"""
FaultBus = '3006'
L1Bus1 = '152'
L1Bus2 = '3004'
L1cktID = '1'
L2Bus1 = '153'
L2Bus2 = '3006'
L2cktID = '1'
"""

"""
FaultBus = '153'
L1Bus1 = '151'
L1Bus2 = '201'
L1cktID = '1'
L2Bus1 = '153'
L2Bus2 = '154'
L2cktID = '2'
"""

# FaultBus = '151'
# L1Bus1 = '201'
# L1Bus2 = '202'
# L1cktID = '1'
# L2Bus1 = '151'
# L2Bus2 = '201'
# L2cktID = '1'



# L1Bus1 = '151'
# L1Bus2 = '201'
# L1cktID = '1'
# L2Bus1 = '151'
# L2Bus2 = '152'
# L2cktID = '1'
# FaultBus = '151'

# #event = '152,202,1;151,201,1/F201'
# event = '103/152,202,1;151,201,1/F201'


# eventKeyWords = event.split('/')
# LP = eventKeyWords[0].strip()

# lines = eventKeyWords[1].strip()
# L1Elements = lines.split(';')[0].split(',')
# L2Elements = lines.split(';')[1].split(',')

# L1Bus1 = L1Elements[0].strip()
# L1Bus2 = L1Elements[1].strip()
# L1cktID = L1Elements[2].strip()


# L2Bus1 = L2Elements[0].strip()
# L2Bus2 = L2Elements[1].strip()
# L2cktID = L2Elements[2].strip()

# FaultBus  = eventKeyWords[2].strip()
# FaultBus = FaultBus[1:]


# if LP == '100':
#     rawFile = 'savnw_conp.raw'
    
# else:
#     rawFile = 'savnw_conp{}.raw'.format(LP)

# rawFileName = rawFile.replace('.raw','')
# dyrFile = 'savnw_dy_sol_0905.dyr'
# psse_log = 'log.txt'


rawFile = 'pf_ornl0823.raw'
dyrFile = 'pf_ornl_all.dyr'
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


"""
#### gen outages

#specs
totSimTime = 10.0
genbus = '101'
cktID = '1'
# load the sav and snp file
psspy.case(savFile)
psspy.rstr(snpFile)

ierr = psspy.strt(0,settings['out_file'])
ierr = psspy.run(0,0.1,1,1,1)
ierr = psspy.dist_machine_trip(int(genbus),cktID)


ierr = psspy.run(0,totSimTime,1,1,1) #simulation end time
#####
"""




"""
#### line outages

#specs
totSimTime = 1.0
bus1 = '151'
bus2 = '152'
cktID = '1'
# load the sav and snp file
psspy.case(savFile)
psspy.rstr(snpFile)

ierr = psspy.strt(0,settings['out_file'])
ierr = psspy.run(0,0.1,1,1,1)
ierr = psspy.dist_branch_trip(int(bus1), int(bus2),cktID)


ierr = psspy.run(0,totSimTime,1,1,1) #simulation end time
####
"""


####### N-2 plus fault


Sbase = 100.0
FaultRpu = 1.0e-6


# get the nominal voltages as well as the fault impedance in ohms
FaultBusNomVolt = float(BusDataDict[str(FaultBus)].NominalVolt)
Zbase = FaultBusNomVolt**2/Sbase  # float since Sbase is a float
Rohm = FaultRpu*Zbase # fault impedance in ohms 
             
# load the sav and snp file
psspy.case(savFile)
psspy.rstr(snpFile)

ierr = psspy.strt(0,settings['out_file'])
#ierr = psspy.run(0,0.1,1,1,1)
#ierr = psspy.dist_branch_trip(int(L1Bus1), int(L1Bus2), L1cktID)


ierr = psspy.run(0,0.2,1,1,1) #fault on time




ierr = psspy.dist_bus_fault(int(FaultBus), 3, 0.0, [Rohm, 0.0])
ierr = psspy.run(0,0.3,1,1,1) #fault off time
ierr = psspy.dist_clear_fault(1)
        



#ierr = psspy.run(0,0.31,1,1,1) #fault off time
#ierr = psspy.dist_branch_trip(int(L2Bus1), int(L2Bus2),L2cktID)

# simulating the following gen trip
# In 103/152,202,1;151,201,1/F201
# 102:7.25830841064
# 101:7.25830841064


#ierr = psspy.run(0,7.25,1,1,1) # gen trip

# trip the separating generators to avoid loss of synchronism
#ierr = psspy.dist_machine_trip(101,'1')
#ierr = psspy.dist_machine_trip(102,'1')

ierr = psspy.run(0,10.0,1,1,1) #exit time (second argument is the end time)

########



##### get the voltage, angle and frequencies
vDict = {}
aDict = {}
freqDict = {}

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


    # add the voltage, freq and angle data
    if signalType == 'VOLT':
        vDict[bus] = valueDict[key]

    elif signalType == 'ANGL':
        aDict[bus] = valueDict[key]

    elif signalType == 'FREQ':
        freqdevpu = np.array(valueDict[key])
        freq = 60*(1+ freqdevpu)
        freqDict[bus] = freq
######



### Plots


plotdir = 'testPlotsPFORNL'
currentdir = os.getcwd()
obj_dir = currentdir +  '/' + plotdir
if not os.path.isdir(obj_dir):
    os.mkdir(obj_dir)
#### plot all the bus voltages
for bus in vDict:
    v = vDict[bus]
    plt.plot(tme,v)
    plt.title('Bus {} voltage'.format(bus))
    plt.ylim(-0.1,1.5)
    plt.xlabel('Time (s)')
    plt.ylabel('V (pu)')
    plt.grid()
    figName = plotdir + '/V{}.png'.format(bus)
    plt.savefig(figName)
    plt.close()





#### plot all the bus angles
for bus in aDict:
    v = aDict[bus]
    plt.plot(tme,v)
    plt.title('Bus {} angle'.format(bus))
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (deg)')
    plt.grid()
    figName = plotdir + '/a{}.png'.format(bus)
    plt.savefig(figName)
    plt.close()





#### plot all the bus frequencies
for bus in freqDict:
    v = freqDict[bus]
    plt.plot(tme,v)
    plt.title('Bus {} frequency'.format(bus))
    plt.xlabel('Time (s)')
    plt.ylabel('Freq (Hz)')
    plt.grid()
    plt.ylim(-55,65)
    figName = plotdir + '/f{}.png'.format(bus)
    plt.savefig(figName)
    plt.close()















