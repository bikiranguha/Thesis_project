#! /usr/bin/env python

'''
Runs a TS3ph simulation and writes all the 3 phase voltage and angle data into text and csv files



'''

# Import modules

import os
import subprocess
import numpy as np
import xml.etree.ElementTree as ET
import math
import csv
#from input_data import EventList, rawPath


def runSim(rawPath,EventList,TS3phLogFile,faultBus, faultType, eventFileObj, writerObjV, writerObjA,rawFileName):
    # eventFileObj: The text file object in which we write the event key
    # writerObjV: write the voltages into rows in the voltage csv file
    # writerObjA: write the angles into rows in the angle csv file
    #save TS3ph log to this file
    #f='TS3phlogFault.txt'
    out_file = open(TS3phLogFile, 'w')

    ################################################################################







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
        # function to return 3 ph voltage magnitude and angle
    	output_lenght = output.shape[1]  # shape returns a tuple of (row, column). So this gets the column numbers

        # get the complex voltage 
    	volta = output[dynsize + 1 + 6*bus_index][2:act_length] + \
    	    1j * output[dynsize + 4 + 6*bus_index][2:act_length]

        voltb = output[dynsize + 2 + 6*bus_index][2:act_length] + \
            1j * output[dynsize + 5 + 6*bus_index][2:act_length]

        voltc = output[dynsize + 3 + 6*bus_index][2:act_length] + \
            1j * output[dynsize + 6 + 6*bus_index][2:act_length]

        # get the voltage magnitude and angles
    	magA = np.absolute(volta)
    	angleA = (180.0/math.pi)*np.unwrap(np.angle(volta))


        magB = np.absolute(voltb)
        angleB = (180.0/math.pi)*np.unwrap(np.angle(voltb))

        magC = np.absolute(voltc)
        angleC = (180.0/math.pi)*np.unwrap(np.angle(voltc))


        return magA, magB, magC, angleA, angleB, angleC


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
    #print type(time)
    ##############################

    # get the raw file key
    if rawFileName == 'savnw_conz':
        PL = '100'
    else:
        PL = rawFileName[-3:] # last 3 character contain the percentage loading


    """ get the voltage and angles of all the buses in the system"""
    for i in range(len(BusListInt)):
    	magA, magB, magC, angleA, angleB, angleC = get_volt_angle(output,dynsize,i,act_length,BusListInt[i])
        # phase A
        eventID = 'R{}/F{}/B{}/{}/{}'.format(PL ,faultBus, BusList[i], faultType,'A')
        eventFileObj.write(eventID)
        eventFileObj.write('\n')
        writerObjV.writerow(magA)
        writerObjA.writerow(angleA)

        # phase B
        eventID = 'R{}/F{}/B{}/{}/{}'.format(PL, faultBus, BusList[i], faultType,'B')
        eventFileObj.write(eventID)
        eventFileObj.write('\n')
        writerObjV.writerow(magB)
        writerObjA.writerow(angleB)

        # phase C
        eventID = 'R{}/F{}/B{}/{}/{}'.format(PL, faultBus, BusList[i], faultType,'C')
        eventFileObj.write(eventID)
        eventFileObj.write('\n')
        writerObjV.writerow(magC)
        writerObjA.writerow(angleC)

    return time
    #clean output files
    os.system('rm TS3phoutput.out')
    os.system('rm TS3phoutput.out.info')
    #return eventFileObj, writerObjV, writerObjA



