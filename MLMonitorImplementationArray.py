# implement a saved machine learning model on voltage data and see how good it does
# For each event, all the voltage data is compiled inside an array, which is passed to the ML model which outputs the prediction
# The predictions are organized into a list, which is passed to the predictDict dict. It has keys for the event and a corresponding bus list
# which can be used to locate which buses had an instability predictor.
# All the false positives and false negatives are output to a text file

import pickle
import numpy as np
#from multiprocessing import Process
import os

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
"""
def runInParallel(*fns):
    proc = []
    for fn in fns:
      p = Process(target=fn)
      p.start()
      proc.append(p)
    for p in proc:
      p.join()
"""
class EventData(object):
    def __init__(self):
        self.VDict = {}

class predictData(object):
    def __init__(self):
        self.prediction = []
        self.BusList = []
"""
class ErrorData(object):
    def __init__(self):
        self.fpBuses = []
        self.fnBuses = []
"""

#if __name__ == '__main__': # needed for the multiprocessing module in windows
# get the ML model
currentdir = os.getcwd()
model_dir = currentdir +  '/MLModels'
LRFile = model_dir + '/' +  'LR_model.sav'
LRModel = pickle.load(open(LRFile, 'rb'))

# get the voltage data
VoltageDataDict = load_obj('VoltageData') # this voltage data dictionary has been generated from the script 'TS3phSimN_2FaultDataSave.py', see it for key format
predictDict = {}

EventDict = {} # key: event, value: EventData structure
WrongPredDict = {}
fpList = []
fnList = []
 
# Get timestep, the steady state time and 10 cycles time data for the ML input
tme = VoltageDataDict['time']
#words = tme.split(',')
#tme = [float(i) for i in words]

timestep = tme[1] - tme[0]
time_fault_clearance = int(1.31/timestep)
input_cyc = int((30.0/60)/timestep)  # 60 time step input


# organize the data, filter by events and buses
for key in VoltageDataDict:

    if key == 'time':
        continue

    keyWords = key.split('/')
    event = keyWords[0].strip()
    Bus = keyWords[1].strip()

    if event not in EventDict:
        EventDict[event]  = EventData()

    voltage = VoltageDataDict[key]
    #words = voltage.split(',')
    #voltageValues = np.zeros(len(words))
    #voltageValues = [float(i) for i in words]
    #voltageValues = np.asarray(voltageValues)
    EventDict[event].VDict[Bus] = voltage # this is the whole voltage data, from the start

# run the ML predictor event by event and for each event, all the voltage data is analyzed in parallel
for i in range(10): # run a loop 10 times to see if there is ever any error
    false_pos_count = 0
    false_neg_count = 0
    for event in EventDict:
        #print event
        VDict = EventDict[event].VDict
        predictDict[event] = predictData()
        BusList = VDict.keys()
        k = 0 
        VArray = np.zeros((len(BusList),60))
        # organize all the bus voltages into a numpy arrays
        act = [] # list of all the actual classification
        for Bus in BusList:

            # get the input voltages
            croppedV = VDict[Bus][time_fault_clearance:] # voltage data after the fault is cleared
            inputV = croppedV[:input_cyc] # the input to the ML model
            VArray[k] = inputV
            k += 1
            # get the actual prediction
            steadyV =   croppedV[-100:]

            # get the derivative of the steady state voltage
            dv_dt = np.zeros(steadyV.shape[0])
            for i in range(steadyV.shape[0]):
                try:
                    diff = abs((steadyV[i]-steadyV[i-1])/timestep)
                    dv_dt[i] = diff
                except: # when  i=0, since there is no i-1
                    continue

            #abnormalVList = [v for v in steadyV if v< 0.9 or v> 1.1]
            abnormalVList = [steadyV[j] for j in range(steadyV.shape[0]) if dv_dt[j] > 0.1] # based only on dv_dt thresholds
            if len(abnormalVList) > 10:
                act.append(1.0)
            else:
                act.append(0.0)

        # run the predictor and save it in the prediction structure
        pred = LRModel.predict(VArray)
        pred = np.array(pred).reshape(-1)
        pred = list(pred)
        predictDict[event].prediction = pred
        predictDict[event].BusList = BusList

        #wrong_ind = [ind for ind, x in enumerate(pred) if pred[ind] != act[ind]]
        false_pos_ind = [ind for ind, x in enumerate(pred) if pred[ind] == 1.0 and act[ind] == 0.0]
        false_neg_ind = [ind for ind, x in enumerate(pred) if pred[ind] == 0.0 and act[ind] == 1.0]
        error_buses = []
        if len(false_pos_ind) > 0 or len(false_neg_ind) > 0:
            false_pos_count += len(false_pos_ind)
            false_neg_count += len(false_neg_ind)
            #WrongPredDict[event] = ErrorData()
            #fpBuses = []
            #fnBuses = []
            for w in false_pos_ind:
                string = event + '/' + BusList[w]
                fpList.append(string)

            for w in false_neg_ind:
                string = event + '/' + BusList[w]
                fnList.append(string) 
            #WrongPredDict[event].fpBuses = fpBuses
            #WrongPredDict[event].fnBuses = fnBuses 
    print 'False positive count: ', false_pos_count
    print 'False negative count: ', false_neg_count
"""
# List all the false positives and false negatives
with open('MLErrors.txt','w') as f:
    f.write('False positives:')
    f.write('\n')
    for line in fpList:
        f.write(line)
        f.write('\n')
    f.write('False negatives:')
    f.write('\n')
    for line in fnList:
        f.write(line)
        f.write('\n')
"""

        


