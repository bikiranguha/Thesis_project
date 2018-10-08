import pickle
import numpy as np
# Functions
def load_obj(name ):
    # load pickle object
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

VoltageDataDict = load_obj('VoltageData')
tme = VoltageDataDict['time']
timestep = tme[1] - tme[0]
ind_fault_clearance = int(0.31/timestep)  + 1 #  the fault is cleared at this time 
key = '151,152,1;151,201,1;F201/3018'
croppedV = VoltageDataDict[key][ind_fault_clearance:]
steadyV = croppedV[-100:] # the final 100 samples of the voltage

# get the derivative of the steady state voltage
dv_dt = np.zeros(steadyV.shape[0])
for i in range(steadyV.shape[0]):
    try:
        diff = abs((steadyV[i]-steadyV[i-1])/timestep)
        dv_dt[i] = diff
    except: # when  i=0, since there is no i-1
        continue

medvdtList = [steadyV[j] for j in range(steadyV.shape[0]) if dv_dt[j] > 0.01 and dv_dt[j] < 0.05] # based only on dv_dt thresholds