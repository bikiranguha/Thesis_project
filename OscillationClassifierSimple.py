# get the voltage oscillation classifications
# plot distributions of the following parameters wrt class 0 (normal voltage) and class 1 (oscillating voltage):
#   ratio of first peak magnitude (after fault clearance) over the precontingency voltage
#   distance to fault bus
#   ratio of load at bus relative to the total load (MVA)
#   rise time
#   precontingency voltage
# also build a LR model based on any combo of the features mentioned above:
#   automated the process of trying out all possible combinations and getting the accuracy and the binary confusion matrix


import pickle
import numpy as np
import matplotlib.pyplot as plt
from getLoadDataFn import getLoadData
import math
from generateNeighboursFn import getNeighbours
from findPathFn import getPath
from getBusDataFn import getBusData
# importing evaluation metrics
from sklearn.metrics import accuracy_score, confusion_matrix
# Logistic Regression
from sklearn import linear_model as lm
# for splitting the data
from sklearn.model_selection import train_test_split
import random
from mpl_toolkits.mplot3d import Axes3D
import os

# Classes
class PathStruct(object):
    def __init__(self):
        self.restOfBuses = []
        self.depth = []

class Performance():
    def __init__(self):
        self.accuracy = 0.0
        self.conf = np.empty([2, 2]) 

# Functions
def load_obj(name ):
    # load pickle object
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def nearbyLoadDepthOne(Bus):
    # function to return the total MVA load (constant impedance) in a depth of one as a fraction of the total load 
    # in the raw file
    depthOneBuses = list(NeighbourDict[Bus])
    depthOneBuses.append(Bus)
    depthOneLoad = 0.0
    for b in depthOneBuses:
        if b in LoadDataDict:
            constZP = LoadDataDict[b].constZP
            constZQ = LoadDataDict[b].constZQ
            constZS = math.sqrt(constZP**2 + constZQ**2) 
            depthOneLoad += constZS      
    loadRatio = depthOneLoad/totalConstZMVA
    return loadRatio

def BusLoadRatio(Bus):
    # function to return the total MVA load (constant impedance) at the bus as a fraction of the total load 
    # in the raw file
    BusLoad = 0.0

    if Bus in LoadDataDict:
        constZP = LoadDataDict[Bus].constZP
        constZQ = LoadDataDict[Bus].constZQ
        constZS = math.sqrt(constZP**2 + constZQ**2) 
        BusLoad += constZS      
    loadRatio = BusLoad/totalConstZMVA
    return loadRatio


def dist2FaultBus(currentBus,FaultBus):
    # return the depth (distance between) fault bus and current bus
    #Path = getPath(Raw,currentBus,FaultBus)
    #depth = len(Path.split('->')) - 1 # since the starting bus is the bus itself
    nInd = depthDict[currentBus].restOfBuses.index(FaultBus)
    depth = depthDict[currentBus].depth[nInd]
    return depth

#################################
savnw_raw = 'savnw_dy_sol_0905.raw'
# get the total load (MVA) in the raw file
LoadDataDict = getLoadData(savnw_raw)
NeighbourDict = getNeighbours(savnw_raw)

totalConstZMVA = 0.0 # total constant impedance load in the raw file
for Bus in LoadDataDict:
    constZP = LoadDataDict[Bus].constZP
    constZQ = LoadDataDict[Bus].constZQ
    constZS = math.sqrt(constZP**2 + constZQ**2)
    totalConstZMVA += constZS


# organize the distance between any two buses in the system
BusDataDict = getBusData(savnw_raw)
depthDict = {} # stores the distance of all the other buses to the bus provided as key
for Bus in BusDataDict:
    depthDict[Bus] = PathStruct()
    for n in BusDataDict:
        if n != Bus:
            depthDict[Bus].restOfBuses.append(n)
            Path = getPath(savnw_raw,Bus,n)
            d = len(Path.split('->')) - 1 # since the starting bus is the bus itself
            depthDict[Bus].depth.append(d)
            



# load voltage data
VoltageDataDict = load_obj('VoltageData') # this voltage data dictionary has been generated from the script 'TS3phSimN_2FaultDataSave.py', see it for key format

tme = VoltageDataDict['time']
timestep = tme[1] - tme[0]

ind_fault_clearance = int(0.31/timestep)  + 1 # fault cleared
ind_line1_outage = int(0.1/timestep)  + 5 # time when line 1 is outaged (added 5 time steps to make sure the voltage settles to the new value)
ind_fc_1s = int(1.31/timestep)  + 1 # one sec after the fault is cleared

samplevCropped = VoltageDataDict[VoltageDataDict.keys()[0]][ind_fault_clearance:] # starting from the fault clearance till the end



#croppedVArray = np.zeros((len(VoltageDataDict)-1,samplevCropped.shape[0])) # make an array of zeros where each row is a sample (cropped) voltage



# dictionaries to gather the distribution data
class0MaxVDict = {} # key: max voltage after fault clearance, as a ratio of the precont voltage, value: event id
class1MaxVDict = {}

class0TDict = {} # key: the time it takes to reach the max value after fault clearance, value: event id
class1TDict = {}

tmaxClass0List = []
tmaxClass1List = []

loadRatioClass0 = []
loadRatioClass1 = []

precontvolt0 = []
precontvolt1 = []

depthToFaultBus0 = []
depthToFaultBus1 = []
#######################

# initialize the input and target array
inputArray = np.zeros((len(VoltageDataDict)-1,4)) # 4 features
#inputArray = np.zeros((len(VoltageDataDict)-1,3)) # 3 features
#inputArray = np.zeros((len(VoltageDataDict)-1,2)) # 2 features

targetVec = np.zeros(len(VoltageDataDict)-1) # target (class) vector

# loop to get all the voltage data and the features
k=0
for key in VoltageDataDict:
    if key == 'time':
        continue

    # get the current bus and the fault bus
    keyWords = key.split('/')
    Bus = keyWords[1].strip()
    keyLHS = keyWords[0].strip()
    keyLHSWords = keyLHS.split('F')
    FaultBus = keyLHSWords[1].strip()


    croppedV = VoltageDataDict[key][ind_fault_clearance:]
    inputV = VoltageDataDict[key][ind_fault_clearance:ind_fc_1s] # the voltage signal for 1 s after the fault is cleared
    max_inputV = np.amax(inputV)

    # get the time it takes to reach max from the instant of fault clearance
    croppedVList = list(croppedV)
    ind_max = croppedVList.index(max_inputV)
    t_max = ind_max*timestep # the time (s) the voltage takes to reach its first peak after fault clearance

    # get the ratio of the max voltage to precont volt and also get the steady state voltage
    pre_cont_volt = VoltageDataDict[key][ind_line1_outage] # the voltage value after the line 1 is outaged
    max_ratio = max_inputV/pre_cont_volt*100
    array_len = croppedV.shape[0]
    #croppedVArray[k] = croppedV
    steadyV = croppedV[-100:] # the final 100 samples of the voltage

    # get the derivative of the steady state voltage
    dv_dt = np.zeros(steadyV.shape[0])
    for i in range(steadyV.shape[0]):
        try:
            diff = abs((steadyV[i]-steadyV[i-1])/timestep)
            dv_dt[i] = diff
        except: # when  i=0, since there is no i-1
            continue

    # classify whether voltage is within bounds or not (but dont classify voltage oscillations)
    abnormalOscList = [steadyV[j] for j in range(steadyV.shape[0]) if dv_dt[j] > 0.05]
    #abnormalOscList = [steadyV[j] for j in range(steadyV.shape[0]) if (steadyV[j] < 0.95 or steadyV[j] > 1.1) and dv_dt[j] < 0.01]

    #lRatio = nearbyLoadDepthOne(Bus) # ratio of total load in depth 1 over the total load in the system
    lRatio = BusLoadRatio(Bus) # ratio of bus load over the total load in the system

    # get the depth to fault bus
    if Bus == FaultBus:
        depthF = 0
    else:
        depthF = dist2FaultBus(Bus,FaultBus)


    # input features to the classifier
    # diff combos
    # All 4 features
    input_features = [max_ratio, pre_cont_volt, lRatio, depthF] 
    # diff combos of three features
    #input_features = [max_ratio, pre_cont_volt, depthF]
    #input_features = [max_ratio, pre_cont_volt, lRatio]
    #input_features = [max_ratio, lRatio, depthF] 
    #input_features = [pre_cont_volt, lRatio, depthF] 
    # 2 features
    #input_features = [max_ratio, pre_cont_volt]
    #############




    input_features = np.asarray(input_features) # numpy array
    inputArray[k] = input_features


    if len(abnormalOscList) > 10: # class 1
        
        #class1MaxV.append(max_ratio)
        class1MaxVDict[max_ratio] = key
        tmaxClass1List.append(t_max)
        loadRatioClass1.append(lRatio)
        precontvolt1.append(pre_cont_volt)
        depthToFaultBus1.append(depthF)
        
        

        if t_max not in class1TDict:
            class1TDict[t_max] = [key]
        else:
            class1TDict[t_max].append(key)

        #abnormalVTarget[k] = 1.0
        #VClass1.append(key)
        
        targetVec[k] = 1
    

    else: # class 0
        
        highdvdtList = [steadyV[j] for j in range(steadyV.shape[0]) if dv_dt[j] > 0.05] # based only on dv_dt thresholds
        if len(highdvdtList) > 10: # exclude all the cases where there is considerable oscillation
            continue
        #class0MaxV.append(max_ratio)
        class0MaxVDict[max_ratio] = key
        tmaxClass0List.append(t_max)
        loadRatioClass0.append(lRatio)
        precontvolt0.append(pre_cont_volt)
        depthToFaultBus0.append(depthF)


        if t_max not in class0TDict:
            class0TDict[t_max] = [key]
        else:
            class0TDict[t_max].append(key)
        #VClass0.append(key)
        
        targetVec[k] = 0

    k+=1




"""
# testing with a certain combo of features
#x = inputArray
subInputArray = inputArray[:,[0,1,3]]
x = subInputArray
y = targetVec
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.10)
model_LR = lm.LogisticRegression(C=1e5,class_weight={0:0.01,1:1})
model_LR.fit(x_train, y_train)
y_pred_LR = model_LR.predict(x_test)
print 'Accuracy score:', accuracy_score(y_test,y_pred_LR)*100
print 'Confusion matrix:'
print confusion_matrix(y_test, y_pred_LR)
"""


"""
# 3d plot of the accuracy wrt the test size and the test size to train size ratio
subInputArray = inputArray[:,[0,1,3]]
x = subInputArray
y = targetVec


x_ax = np.linspace(0.1,0.8,8) # the test size
y_ax = np.linspace(0.01,1,10) # zero class weight 

#z_ax = np.zeros((x_ax.shape[0],y_ax.shape[0]))
z_ax = np.zeros((y_ax.shape[0],x_ax.shape[0]))
for i in range(x_ax.shape[0]):
    for j in range(y_ax.shape[0]):
        ts = x_ax[i]
        zw = y_ax[j]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = ts)
        cwDict = {0:zw,1:1}
        model_LR = lm.LogisticRegression(C=1e5,class_weight=cwDict)
        model_LR.fit(x_train, y_train)
        y_pred_LR = model_LR.predict(x_test)
        z_ax[j,i] = accuracy_score(y_test,y_pred_LR)*100
"""

"""
# Plotting the 3d plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(x_ax, y_ax, z_ax)
plt.show()
"""


# loops to select a subset of features and get accuracy and confusion matrix

explored = set()
PerformanceDict = {}
PerformanceDictDetailed = {}
innertrialNo = 50
for outertrial in range(innertrialNo):

    featureNo = random.randint(1,4) # number of features, generated randomly
    featureList = [0,1,2,3] # original indices
    randomColumns = random.sample(featureList,featureNo)
    randomColumnSet = frozenset(randomColumns)

    percent_complete = outertrial/float(innertrialNo)*100
    if percent_complete%10 == 0:
        print 'Percent done: ', percent_complete

    # check if this combo already tested
    if randomColumnSet in explored: # already explored
        continue
    else: # not explored, add
        explored.add(randomColumnSet)


    subInputArray = inputArray[:,randomColumns]

    randomColumnStr = [str(x) for x in randomColumns]
    featureKey = ','.join(randomColumnStr)

    #print randomColumns
    # build a low voltage classifier based on the set of features
    #x = inputArray
    accuracyList = [] # keeps track of all the acc
    tpList = []
    fnList = []
    fpList = []
    tnList = []
    # using the selected combination of features, split randomly and get the accuracy a 100 times
    for innertrial in range(10):
        x = subInputArray
        y = targetVec
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)
        model_LR = lm.LogisticRegression(C=1e5,class_weight={0:0.03,1:1})
        model_LR.fit(x_train, y_train)
        y_pred_LR = model_LR.predict(x_test)
        accuracyList.append(accuracy_score(y_test,y_pred_LR)*100)
        c = confusion_matrix(y_test, y_pred_LR)
        # getting the elements of the confusion matrix
        tp = c[0][0] # true positive
        fn = c[0][1] # false negative
        fp = c[1][0] # false positive
        tn = c[1][1] # true negative

        tpList.append(tp)
        fnList.append(fn)
        fpList.append(fp)
        tnList.append(tn)

    # get the average of the accuracy as well as the elements of the confusion matrix
    mean_accuracy = np.mean(accuracyList)
    mean_tp = np.mean(tp)
    mean_fn = np.mean(fn)
    mean_fp = np.mean(fp)
    mean_tn = np.mean(tn)
    mean_cf = np.array([[mean_tp,mean_fn],[mean_fp,mean_tn]])



    PerformanceDict[featureKey] = mean_accuracy # dictionary to help sort the data according to the accuracy value
    PerformanceDictDetailed[featureKey] = Performance()
    PerformanceDictDetailed[featureKey].accuracy = mean_accuracy
    PerformanceDictDetailed[featureKey].conf = mean_cf
    #print 'Accuracy score:', accuracy_score(y_test,y_pred_LR)*100
    #print 'Confusion matrix: a11: True negatives, a12: False positives, a21: False negatives, a22: True negatives'
    #print 'Confusion matrix:'
    #print confusion_matrix(y_test, y_pred_LR)

outputLines = []
for key, value in sorted(PerformanceDict.iteritems(), key=lambda (k,v): v, reverse = True):
    str1 = 'Indices: ' + key
    outputLines.append(str1)
    str2 = 'Accuracy:' + str(value)
    outputLines.append(str2)
    str3 = 'Confusion matrix:'
    outputLines.append(str3)
    outputLines.append(str(PerformanceDictDetailed[key].conf))
    outputLines.append('\n')


with open('PerformanceOsc.txt','w') as f:
    f.write('original input_feature indices = [max_ratio, pre_cont_volt, lRatio, depthF]')
    f.write('\n')
    f.write('Confusion matrix description: a11: True negatives, a12: False positives, a21: False negatives, a22: True negatives')
    f.write('\n')
    f.write('negative: minimal oscillation, positive: considerable oscillation')
    f.write('\n')
    for line in outputLines:
        f.write(line)
        f.write('\n')





# draw some histograms (to see distribution according to the classes)
plotdir = 'VoltOscDistr'
if not os.path.isdir(plotdir):
    os.mkdir(plotdir)
# for the max voltage
plt.hist(class0MaxVDict.keys(), bins='auto',label='Class 0')  
plt.hist(class1MaxVDict.keys(), bins='auto',label='Class 1') 
plt.legend()
titleStr = 'Max voltage (percent of prefault voltage) distribution'
plt.title(titleStr)
plt.xlabel('Voltage')
plt.ylabel('Samples')
#plt.show()
figName = plotdir + '/' +  'MaxV.png'
plt.savefig(figName)
plt.close()


# for the rise time
plt.hist(tmaxClass0List, bins='auto',label='Class 0')  
plt.hist(tmaxClass1List, bins='auto',label='Class 1') 
plt.legend()
titleStr = 'Rise time'
plt.title(titleStr)
plt.xlabel('s')
plt.ylabel('Samples')
#plt.show()
figName = plotdir + '/' +  'tMax.png'
plt.savefig(figName)
plt.close()

#  for the load ratio
plt.hist(loadRatioClass0, bins='auto',label='Class 0')  
plt.hist(loadRatioClass1, bins='auto',label='Class 1') 
plt.legend()
titleStr = 'Bus load ratio'
plt.title(titleStr)
plt.xlabel('Load ratio')
plt.ylabel('Samples')
#plt.show()
figName = plotdir + '/' +  'lRatio.png'
plt.savefig(figName)
plt.close()


#  for the precont volt
plt.hist(precontvolt0, bins='auto',label='Class 0')  
plt.hist(precontvolt1, bins='auto',label='Class 1') 
plt.legend()
titleStr = 'Pre-contingency voltage'
plt.title(titleStr)
plt.xlabel('Volt')
plt.ylabel('Samples')
#plt.show()
figName = plotdir + '/' +  'precontvolt.png'
plt.savefig(figName)
plt.close()


#  for the depth to fault bus
plt.hist(depthToFaultBus0, bins='auto',label='Class 0')  
plt.hist(depthToFaultBus1, bins='auto',label='Class 1') 
plt.legend()
titleStr = 'Depth to fault bus'
plt.title(titleStr)
plt.xlabel('depth')
plt.ylabel('Samples')
#plt.show()
figName = plotdir + '/' +  'depthF.png'
plt.savefig(figName)
plt.close()


"""
# test any key
#key = '154,205,1;3001,3003,1;F3003/154'
#key = '154,205,1;201,204,1;F204/204'
#key = '3005,3007,1;3007,3008,1;F3008/3007'
key = '154,205,1;153,154,1;F154/202'
croppedV = VoltageDataDict[key][ind_fault_clearance:]
inputV = VoltageDataDict[key][ind_fault_clearance:ind_fc_1s] # the voltage signal for 1 s after the fault is cleared
max_inputV = np.amax(inputV)

pre_cont_volt = VoltageDataDict[key][ind_line1_outage] # the voltage value after the line 1 is outaged
max_ratio = max_inputV/pre_cont_volt*100
array_len = croppedV.shape[0]
steadyV = croppedV[-100:] # the final 100 samples of the voltage


voltage = VoltageDataDict[key]
plt.plot(tme, voltage)
titleStr = key
plt.title(titleStr)
plt.grid()
plt.show()
"""





