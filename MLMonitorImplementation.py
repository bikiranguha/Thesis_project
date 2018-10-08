# implement a saved machine learning model on voltage data and see how good it does
# the idea is to run the model in parallel on the voltage data

import pickle
import numpy as np
from multiprocessing import Process
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
        self.prediction = {}

def runML(V,cyc,predictDict,event,Bus,model):
    croppedV = V[:cyc]
    croppedV = np.array(croppedV).reshape(1,-1)
    stab_pred = model.predict(croppedV)
    predictDict[event].prediction[Bus] = stab_pred
    #return stab_pred




if __name__ == '__main__': # needed for the multiprocessing module in windows
    # get the ML model
    currentdir = os.getcwd()
    model_dir = currentdir +  '/MLModels'
    LRFile = model_dir + '/' +  'LR_model.sav'
    LRModel = pickle.load(open(LRFile, 'rb'))

    # get the voltage data
    VoltageDataDict = load_obj('VoltageData') # this voltage data dictionary has been generated from the script 'TS3phSimN_2FaultDataSave.py', see it for key format
    predictDict = {}

    EventDict = {} # key: event, value: EventData structure
     
    # Get timestep, the steady state time and 10 cycles time data for the ML input
    tme = VoltageDataDict['time']
    words = tme.split(',')
    tme = [float(i) for i in words]

    timestep = tme[1] - tme[0]
    time_1s = int(1.0/timestep)
    time_10cyc = int((10.0/60)/timestep)


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
        words = voltage.split(',')
        #voltageValues = np.zeros(len(words))
        voltageValues = [float(i) for i in words]
        voltageValues = np.asarray(voltageValues)
        EventDict[event].VDict[Bus] = voltageValues

    # run the ML predictor event by event and for each event, all the voltage data is analyzed in parallel
    for event in EventDict:
        print event
        VDict = EventDict[event].VDict
        predictDict[event] = predictData()
        BusList = VDict.keys()
        proc = []
        for Bus in BusList:
            p = Process(target=runML,args=(VDict[Bus],time_10cyc,predictDict,event,Bus,LRModel))
            p.start()
            proc.append(p)
            for p in proc:
                p.join()

        


