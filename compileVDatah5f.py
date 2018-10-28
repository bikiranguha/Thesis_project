# gather all the voltage data
# generate input and target arrays and the corresponding key list
# save the arrays to a new h5py file and the key list to a pickle object
import h5py
import pickle
from getROCFn import getROC
import numpy as np
import os
# load pickle object
def load_obj(name ):
    # load pickle object
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
def save_obj(obj, name ):
    # save as pickle object
    currentdir = os.getcwd()
    objDir = currentdir + '/obj'
    if not os.path.isdir(objDir):
        os.mkdir(objDir)
    with open(objDir+ '/' +  name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

# load voltage data
vFile = 'obj/v.h5'
h5f = h5py.File(vFile,'r')

# h5f is basically a dictionary where there are 9 chunks with keys: v0 to v8. The corresponding id are in key0.pkl to key8.pkl
v0 = h5f['v0'][:] # get the v0 array
# get the time object
tme = load_obj('time')
timestep = tme[1] - tme[0]
ind_fault_clearance = int(0.31/timestep)  + 1 #  the fault is cleared at this time 
ind_fc_1s = int(1.31/timestep)  + 1 # one sec after the fault is cleared

# important lists
inputVList = []
totalKeyList = []
targetOscList = []
targetAbnVList = []


# populate the 3 relevant lists
for i in range(9):
    keyFile = 'key{}'.format(i)
    currentKeyList =  load_obj(keyFile)
    vKey = 'v{}'.format(i)
    vArray = h5f[vKey][:]
    keyLength = len(currentKeyList)
    print('Loop {} out of 8'.format(i))
    for j in range(keyLength):
        event = currentKeyList[j]
        voltage = vArray[j]
        dv_dt =  getROC(voltage,tme)
        steadyV = voltage[-100:]
        dv_dtSteady = dv_dt[-100:]
        inputV = voltage[ind_fault_clearance:ind_fc_1s]
        inputVList.append(inputV)
        totalKeyList.append(event)

        highdvdtList = [steadyV[j] for j in range(steadyV.shape[0]) if dv_dtSteady[j] > 0.05] # based only on dv_dt thresholds
        if len(highdvdtList) > 10:
            targetOscList.append(1.0)
        else:
            targetOscList.append(0.0)

        abnormalVList = [steadyV[j] for j in range(steadyV.shape[0]) if (steadyV[j] < 0.95 or steadyV[j] > 1.1) and dv_dt[j] < 0.01]
        if len(abnormalVList) > 10:
            targetAbnVList.append(1.0)
        else:
            targetAbnVList.append(0.0)



    del vArray


h5f.close()

inputVArray = np.array(inputVList)
targetOscArray = np.array(targetOscList)
targetAbnVArray = np.array(targetAbnVList)



# create a new file and save the input and target arrays
h5f2 = h5py.File('obj/vInpClass.h5', 'w')
h5f2.create_dataset('inp', data=inputVArray)
h5f2.create_dataset('targetOsc', data=targetOscArray)
h5f2.create_dataset('targetAbnV', data=targetAbnVArray)
h5f2.close()


# save the event key list
save_obj(totalKeyList,'allKeys')









