#! /usr/bin/env python

'''
Runs a TS3ph simulation and saves all the bus voltages and magnitudes
For now, this script is intended to serve as a real world case, where a disturbance happens



'''

# Import modules

import os
import subprocess
import numpy as np
import xml.etree.ElementTree as ET
import math
#from input_data import EventList, rawPath


def runSim(rawPath,EventList,TS3phOutFile):

    #save TS3ph log to this file
    #f='TS3phlogFault.txt'
    out_file = open(TS3phOutFile, 'w')

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


    # generate the call string
    call_string = ['./TS3ph']
    for event in EventList:
        call_string.append(event)

        
    try:
    #            result = subprocess.call(call_string, stdout=FNULL)
        result = subprocess.call(call_string, stdout = out_file)
    except:
        print 'Unable to run TS3ph'
        quit()

    #print "TS3ph executed succesfully"

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

    #print "Sizes"
    #print dynsize
    #print output.shape



    """ get the voltage and angles of all the buses in the system"""
    for i in range(len(BusListInt)):
    	get_volt_angle(output,dynsize,i,act_length,BusListInt[i])



    #clean output files

    os.system('rm TS3phoutput.out')
    os.system('rm TS3phoutput.out.info')
    return SimResults



