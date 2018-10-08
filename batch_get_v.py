#! /usr/bin/env python

'''
Runs a TS3ph simulation and saves all the bus voltages and magnitudes



'''

# Import modules

import os, csv
import subprocess
import time
import numpy as np
from   numpy import linalg as la
from os import listdir
from os.path import isfile, join
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Raw file, dyr file locations#####################################################   
#rawDir = 'test_cases/PSSE/'
rawDir = 'test_cases/savnw/'
#rawfile = 'pf_ornl150_1008.raw'
rawfile = 'savnw.raw'
rawPath = rawDir + '/' + rawfile
dyr_directory = 'test_cases/BASE_CASES'
#model = 'pf_ornl'
#model = 'savnw'

#save TS3ph log to this file
f='TS3phlogNodist.txt'
out_file = open(f, 'w')

################################################################################

# class to save the complex voltage results for each bus
class VoltData(object):
	def __init__(self,mag,angle):
		self.mag = mag
		self.ang = angle


SimResults = {} # Dictionary which will contain all the sim data

""" get a list of all the buses in the raw file """

BusList = [] # BusList contains a list of all the bus numbers in string format
with open(rawPath,'r') as f:
    filecontent = f.read()
    fileLines = filecontent.split("\n")
    for line in fileLines:

        if 'PSS(R)E' in line or line == '': # skip the first line and any blank lines
            continue
        if 'END OF BUS DATA' in line:   # stop if we have reached end of bus data
            break
        words = line.split(',')
        if len(words)<2: # continue to next iteration of loop if its a blank line
            continue
        BusList.append(words[0].strip())

BusListInt = [] # list of all the bus numbers as integers
for Bus in BusList:
    BusListInt.append(int(Bus))



def readBinary(filename):

    '''
    name: readBinary
    input: file address

    description: this function reads a dense petsc
    matrix with the specific datatypes used in TS3ph
    '''

    scalartype = np.dtype('>f8') # 64 bit float point
    inttype = np.dtype('>i4')    # 32 bit signed integer format


    fid = open(filename, 'rb')  # 'rb' is for reading binary
    header = np.fromfile(fid, dtype=inttype, count=1)[0]
    M,N,nz = np.fromfile(fid, dtype=inttype, count=3)
    nz = M*N

    mat = np.fromfile(fid, dtype=scalartype, count=nz)
    mat = mat.reshape((M,N))  # reshape the data to a M*N array

    return mat


def get_volt_angle(output,dynsize,bus_index,act_length,sbus):
	output_lenght = output.shape[1]  # shape returns a tuple of (row, column). So this gets the column numbers
	volta = output[dynsize + 1 + 6*bus_index][2:act_length] + \
	    1j * output[dynsize + 4 + 6*bus_index][2:act_length]
	mag = np.absolute(volta)
	angle = (180.0/math.pi)*np.unwrap(np.angle(volta))
	SimResults[sbus] = VoltData(mag,angle)







#Some seaborn plotiing options
sns.despine()
sns.set_context("talk")


#user defined parameters

dyr_flag = '-ts_dyr_dir'
gen_size = 6*7  # 6 generators with 7 states each
mod_size = 0
gen_num = 6


#parse files









# FaultBus = raw_input('Please enter the fault bus number: ').strip() # PLease type the fault bus here
# FaultBusInt = int(FaultBus)



# if you want to ignore TS3ph output, uncomment the next line
#        FNULL = open(os.devnull,'w')  # use this stdout in subprocess.call

    




#        dyr_dire = dyr_directory + '/' + current_dyr
# event1_flag = '-event01'
# event1_param = '0.2,FAULTON,ABCG,' + FaultBus +',,,,1.0e-6,1.0e-6,1.0e-6,0.0,0.0,0.0'

# event2_flag = '-event02'
# event2_param = '1,EXIT,,,,,,,,,,,'

# log_flag = '-event_log_file'

# log_filename = 'Event_testing' + FaultBus +  '.txt'
# log_param = log_directory +'/'+ log_filename

# sim_id_flag = '-event_sim_id' 
# sim_id_param = 'TestBus' + FaultBus

##        event3_flag = '-event03'
##        event3_param = '2,EXIT,,,,,,,,,,,'

# Simulate a nodist
event1_flag = '-event01'
event1_param = '5,EXIT,,,,,,,,,,,'

#run ts3ph in a stream
print '\n'

#       TS3ph_string = './TS3ph' + fault_start + fault_end + sim_exit
#       call_string = ['./TS3ph', dyr_flag, dyr_dire, event1_flag,event1_param, event2_flag, event2_param, log_flag, log_param, sim_id_flag, sim_id_param] # first the flag, then the parameter
#        call_string = ['./TS3ph', dyr_flag, dyr_dire] # first the flag, then the parameter
call_string = ['./TS3ph', even1_flag, event1_param]
try:
#            result = subprocess.call(call_string, stdout=FNULL)
    result = subprocess.call(call_string, stdout = out_file)
except:
    print 'Unable to run TS3ph'
    quit()

print "TS3ph executed succesfully"

#load metadata
filename = "metaout.xml"
tree = ET.parse(filename)
root = tree.getroot()
nnodes = int(root.find('nbus').text)
dynsize = int(root.find('dynsize').text)


#load output file
output = readBinary('TS3phoutput.out')
output_lenght = output.shape[1]
# Find actual length of time, ignoring extra 0's at the end
act_length = output_lenght
while act_length>0:
    if output[0][act_length-3] == 0:
        act_length-=1
    else:
        break
act_length = act_length-2
time=output[0][2:act_length]
SimResults['time'] = time
##############################

#        time = output[0][2:output_lenght - 3]   # had to delete some columns at the end because it was garbage

print "Sizes"
print dynsize
print output.shape

#    model = current_dyr.replace('.dyr','')

# for bus in BusList:
#     if FaultBus == bus:
#         faultBusIndex = BusList.index(bus)
#         break



""" plot the comparison results of all  bus voltage and magnitudes"""
for i in range(len(BusListInt)):
	get_volt_angle(output,dynsize,i,act_length,BusListInt[i])

""" compare all the speed deviations"""
# dSpeedIndexList = range(5,dynsize,7)  # list of indices for the speed deviation state, the 1st row is time and has index of 0
# stateCount = 0 # initialize stateCount, used to match with pssedat index for the states
# for j in dSpeedIndexList:

#     ts3ph_state =  output[j][2:output_lenght - 3]
#     compare_state(time, ts3ph_state, pssdat, directory, model, stateCount)
#     stateCount+=1


#clean output files

os.system('rm TS3phoutput.out')
os.system('rm TS3phoutput.out.info')



