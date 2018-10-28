import os
import subprocess
import numpy as np
import xml.etree.ElementTree as ET
import math
import matplotlib.pyplot as plt
from runSimFnv2 import runSim




out_file = 'TS3ph.log'

# one line out then a fault
event1Flag = '-event01'
event1Param = '0.1,OUT,LINE,154,205,,1,7,,,,,'

event2Flag = '-event02'
event2Param = '0.2,FAULTON,ABCG,154,,,,1.0e-6,1.0e-6,1.0e-6,0.0,0.0,0.0'

event3Flag = '-event03'
event3Param = '0.3,FAULTOFF,ABCG,154,,,,,,,,,'

event4Flag = '-event04'
event4Param = '0.31,OUT,LINE,154,203,,1,7,,,,,'

exitFlag = '-event05'
exitParam = '5,EXIT,,,,,,,,,,,'
EventList = [event1Flag,event1Param,event2Flag,event2Param,event3Flag,event3Param,event4Flag,event4Param,exitFlag,exitParam]
MacInfoFile =  'MAC_INFO.txt'
raw = 'test_cases/savnw/savnw_sol.raw'

VoltResults, StateResults = runSim(raw,EventList,out_file, MacInfoFile)

time =VoltResults['time']
for key in StateResults:
    keyWords = key.split(',')
    Bus = keyWords[0].strip()
    dSpeed = StateResults[key].dSpeed
    Pm = StateResults[key].Pm
    # plot dSpeed
    plt.plot(time, dSpeed)
    plt.title(key)
    plt.ylabel('pu')
    plt.xlabel('Time (sec)')
    #plt.ylim(-0.1,1.5)
    plt.grid()
    figName = 'dSpeed' + Bus + '.png'
    plt.savefig(figName)
    plt.close()

    if Pm != None:
    # plot dSpeed
        plt.plot(time, Pm)
        plt.title(key)
        plt.ylabel('pu')
        plt.xlabel('Time (sec)')
        #plt.ylim(-0.1,1.5)
        plt.grid()
        figName = 'Pm' + Bus + '.png'
        plt.savefig(figName)
        plt.close()


"""
# generate the call string
call_string = ['./TS3ph']
for event in EventList:
    call_string.append(event)


try:
#   result = subprocess.call(call_string, stdout=FNULL)
    result = subprocess.call(call_string, stdout = out_file)
except:
    print 'Unable to run TS3ph'
    quit()

#print "TS3ph executed succesfully"



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



# get the states of the 1st generator
Eqp = output[1][2:act_length] # 2nd index, as first one is time
dSpeed = output[5][2:act_length] 
Pmech = output[14][2:act_length] # just added 1 to the index given in MAC INFO



# plots

# Eqp
plt.plot(time, Eqp)
plt.title('Gen 101 Eqp')
plt.ylabel('V (pu)')
plt.xlabel('Time (sec)')
#plt.ylim(-0.1,1.5)
plt.grid()
figName = 'Gen101Eqp.png'
plt.savefig(figName)
plt.close()


# dSpeed
plt.plot(time, dSpeed)
plt.title('Gen 101 dSpeed')
plt.ylabel('pu')
plt.xlabel('Time (sec)')
#plt.ylim(-0.1,1.5)
plt.grid()
figName = 'Gen101dSpeed.png'
plt.savefig(figName)
plt.close()

# Pmech
plt.plot(time, Pmech)
plt.title('Gen 101 Pmech')
plt.ylabel('MW')
plt.xlabel('Time (sec)')
#plt.ylim(-0.1,1.5)
plt.grid()
figName = 'Gen101Pmech.png'
plt.savefig(figName)
plt.close()
"""

