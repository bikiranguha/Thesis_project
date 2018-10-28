# Script to rigorously test the LR classifier on the voltage oscillation data

print 'Importing modules...'
import pickle
import matplotlib.pyplot as plt
import numpy as np
# importing evaluation metrics
from sklearn.metrics import accuracy_score, confusion_matrix
# Logistic Regression
from sklearn import linear_model as lm
# for splitting the data
from sklearn.model_selection import train_test_split
import os
from getROCFn import getROC
from GenDataFn import getGenData
from getBusDataFn import getBusData
from generateNeighboursFn import getNeighbours
import skfuzzy as fuzz
import random
from mpl_toolkits.mplot3d import Axes3D
from pylab import meshgrid
import pickle as pl



#### classes

class Features(object):
    def __init__(self,inputV):
        self.max_ratio = 0.0
        self.genRatioF = 0.0
        self.similarityCL0 = 0.0
        self.similarityCL1 = 0.0
        self.t_max = 0.0
        self.osc = 0.0 # one if sample belongs to oscillatory class, otherwise 0
        self.inputV = inputV

raw = 'savnw.raw'
BusDataDict = getBusData(raw)
GenDataDict = getGenData(raw)
NeighbourDict = getNeighbours(raw)
FeatureDict = {}



############## Functions
def load_obj(name ):
    # load pickle object
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def GenRatioDepth1(Bus, totalGen):
    # function which returns the total generation in a depth of one of the bus 
    # as a ratio of the total gen in the system
    depthOneBuses = list(NeighbourDict[Bus])
    depthOneBuses.append(Bus)
    depthOneGen = 0.0

    for b in depthOneBuses:
        if b in GenDataDict:
            depthOneGen += float(GenDataDict[b].Pgen)
    genRatio = depthOneGen/totalGen

    return genRatio

def testMLModel(x,y, testSize, classWeightDict,noOfTrials):
    # train the ML a number of times with randomly selected training and test sets
    # return the average number of false positives and false negatives

    fpList = []
    fnList = []
    accuracyList = []
    y = np.array(y).reshape(-1)
    for i in range(noOfTrials):
       

        # partition the data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = testSize)


        # train LR
        model_LR = lm.LogisticRegression(C=1e5,class_weight=classWeightDict)
        model_LR.fit(x_train, y_train)
        y_pred_LR = model_LR.predict(x_test)
        cm = confusion_matrix(y_test, y_pred_LR)
        fnList.append(cm[0][1]) # false alarm
        fpList.append(cm[1][0]) # event undetected
        accuracyList.append(accuracy_score(y_test,y_pred_LR)*100)

    avg_fp = np.mean(fpList)
    avg_fn = np.mean(fnList)
    avg_accuracy = np.mean(accuracyList)
    return avg_fp, avg_fn, avg_accuracy

#########################


# get total generator data
totalGen = 0.0
for gen in GenDataDict:
    totalGen += float(GenDataDict[gen].Pgen)



# get the voltage templates for class 0 and class 1

# Load the voltage data
print 'Loading the voltage data from the object file...'
VoltageDataDict = load_obj('VoltageData') # this voltage data dictionary has been generated from the script 'TS3phSimN_2FaultDataSave.py', see it for key format


# crop the data till after fault clearance
print 'Formatting the data to be used by the LR model....'
tme = VoltageDataDict['time']
timestep = tme[1] - tme[0]
#ind_fault_clearance = int(1.31/timestep) #  the fault is cleared at this time 
ind_fault_clearance = int(0.31/timestep)  + 1 #  the fault is cleared at this time 
ind_fc_1s = int(1.31/timestep)  + 1 # one sec after the fault is cleared
ind_line1_outage = int(0.1/timestep)  + 5 # time when line 1 is outaged (added 5 time steps to make sure the voltage settles to the new value)
samplevCropped = VoltageDataDict[VoltageDataDict.keys()[0]][ind_fault_clearance:]



# get the input features and the classifications
croppedVArray = np.zeros((len(VoltageDataDict)-1,samplevCropped.shape[0])) # make an array of zeros where each row is a sample (cropped) voltage
dvdtTarget = np.zeros(len(VoltageDataDict)-1) # the target vector for dvdt classification




k= 0 # row number of croppedVArray
class0List = []
class1List = []
for key in VoltageDataDict:
    if key == 'time':
        continue
    voltage = VoltageDataDict[key]
    dv_dt =  getROC(voltage,tme)
    croppedV = voltage[ind_fault_clearance:]
    croppedVArray[k] = croppedV
    steadyV = voltage[-100:] # the final 100 samples of the voltage
    dv_dtSteady = dv_dt[-100:]

    ### get the input features
    # maximum overshoot
    inputV = voltage[ind_fault_clearance:ind_fc_1s] # the voltage signal for 1 s after the fault is cleared
    pre_cont_volt = voltage[ind_line1_outage] # the voltage value after the line 1 is outaged
    max_inputV = np.amax(inputV)
    max_ratio = max_inputV/pre_cont_volt*100
    # the time it takes to reach max from the instant of fault clearance
    croppedVList = list(croppedV)
    ind_max = croppedVList.index(max_inputV)
    t_max = ind_max*timestep # the time (s) the voltage takes to reach its first peak after fault clearance
    # get the gen ratio at a depth of one of the fault bus
    event = key.split('/')[0]
    faultBus = event.split('F')[1]
    genRatioF = GenRatioDepth1(faultBus, totalGen)

    FeatureDict[key] = Features(inputV)
    FeatureDict[key].max_ratio = max_ratio
    FeatureDict[key].genRatioF = genRatioF
    FeatureDict[key].t_max = t_max






    #################
    # classify instability according to the rate of change of voltage
    highdvdtList = [steadyV[j] for j in range(steadyV.shape[0]) if dv_dtSteady[j] > 0.05] # based only on dv_dt thresholds
    if len(highdvdtList) > 10:
        dvdtTarget[k] = 1.0
        FeatureDict[key].osc = 1.0
        class1List.append(inputV)
    else:
        class0List.append(inputV)

    k+=1


# convert the class lists to class arrays for generating the templates
class0Array = np.asarray(class0List)
class1Array = np.asarray(class1List)


# use fuzzy c-means to generate templates
# class 0
class0ArrayT = np.transpose(class0Array) # transpose required for fuzzy c-means
ncenters = 1 
tmpltCls0, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    class0ArrayT, ncenters, 2, error=0.005, maxiter=1000, init=None)

tmpltCls0 = np.array(tmpltCls0).reshape(-1)
meantmpltCls0 = np.mean(tmpltCls0)


# class 1
class1ArrayT = np.transpose(class1Array)
ncenters = 1 
tmpltCls1, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    class1ArrayT, ncenters, 10, error=0.005, maxiter=1000, init=None)

tmpltCls1 = np.array(tmpltCls1).reshape(-1)
meantmpltCls1 = np.mean(tmpltCls1)



# generate the similarity thresholds
k=0
inputArray = np.zeros((len(FeatureDict),5)) # max number of features possible
for key in FeatureDict:
    inputV = FeatureDict[key].inputV

    meanCroppedV = np.mean(inputV)
    # generate the offsets
    meanOffset0 = (meanCroppedV-meantmpltCls0)*np.ones(inputV.shape[0])
    meanOffset1 = (meanCroppedV-meantmpltCls1)*np.ones(inputV.shape[0])
    # using offset
    similarity0 = 1/np.linalg.norm(inputV-tmpltCls0-meanOffset0)
    similarity1 = 1/np.linalg.norm(inputV-tmpltCls1-meanOffset1)
    FeatureDict[key].similarityCL0 = similarity0
    FeatureDict[key].similarityCL1 = similarity1


    max_ratio = FeatureDict[key].max_ratio
    genRatioF = FeatureDict[key].genRatioF
    t_max = FeatureDict[key].t_max


    input_features = [max_ratio, similarity0, similarity1, t_max, genRatioF] 
    inputArray[k] = input_features

    k+=1





outputLines = []

# define costs of misclassification
fpCost = 10
fnCost = 100



# get the performance from just using the raw input voltage data
print 'Training the LR model for voltage oscillation using raw time series data...'
x = croppedVArray[:,:60] # the first 60 timesteps of the voltage array after line clearance
y = dvdtTarget
testSize = 0.25
classWeightDict = {0:0.03,1:1}
avg_fp, avg_fn, avg_accuracy = testMLModel(x,y, testSize, classWeightDict,100)
misclassification_cost = fpCost*avg_fp + fnCost*avg_fn


outputLines.append('Performance of the LR classifier based only on raw voltage data after fault clearance:')
outputLines.append('Average FP: {}'.format(avg_fp))
outputLines.append('Average FN: {}'.format(avg_fn))
outputLines.append('Average accuracy: {}'.format(avg_accuracy))
outputLines.append('Average misclassification cost: {}'.format(misclassification_cost))
outputLines.append('\n\n')


# get the performance analysis from various combinations of the selected features

print 'Training the LR model using various combinations of the preselected features...'
outputLines.append('Performance results of the selected feature combo:')
outputLines.append('Legend of features: 0: Max voltage overshoot percent, 1: Similarity to class 0, 2: Similarity to class 1, 3: Time to reach 1st peak after fault clearance, 4: Generator Depth 1 Ratio ')
innertrialNo = 100 # total number of trials with different feature combo
explored = set()
for outerTrial in range(innertrialNo):

    featureNo = random.randint(1,5) # number of features, generated randomly
    featureList = [0,1,2,3,4] # original indices
    randomColumns = random.sample(featureList,featureNo)
    randomColumnSet = frozenset(randomColumns)
    randomColumnStr = [str(x) for x in randomColumns]
    featureKey = ','.join(randomColumnStr)

    percent_complete = outerTrial/float(innertrialNo)*100
    if percent_complete%10 == 0:
        print 'Percent done: ', percent_complete

    # check if this combo already tested
    if randomColumnSet in explored: # already explored
        continue
    else: # not explored, add
        explored.add(randomColumnSet)


    subInputArray = inputArray[:,randomColumns]
    x = subInputArray
    y = dvdtTarget
    testSize = 0.25
    classWeightDict = {0:0.03,1:1}
    avg_fp, avg_fn, avg_accuracy = testMLModel(x,y, testSize, classWeightDict,100)
    misclassification_cost = fpCost*avg_fp + fnCost*avg_fn
    outputLines.append('Feature set: {}'.format(featureKey))
    outputLines.append('Average FP: {}'.format(avg_fp))
    outputLines.append('Average FN: {}'.format(avg_fn))
    outputLines.append('Average accuracy: {}'.format(avg_accuracy))
    outputLines.append('Average misclassification cost: {}'.format(misclassification_cost))
    outputLines.append('\n')
    

# Write the performance results
with open('LRPerformanceReport.txt','w') as f:
    for line in outputLines:
        f.write(line)
        f.write('\n')




"""
print 'Running simulations to save the plots...'
# run 3d plots to see the topology of accuracy wrt the test size and the  the number of time steps
x_ax = np.linspace(0.1,0.8,10) # the test size
y_ax = np.asarray(range(1,101)) # number of time steps to include
z_ax = np.zeros((y_ax.shape[0],x_ax.shape[0]))

totalSim = x_ax.shape[0]*y_ax.shape[0]
print 'Total simulations to be done: {}'.format(totalSim)
y = dvdtTarget

k= 0
for i in range(x_ax.shape[0]):
    for j in range(y_ax.shape[0]):
        ts = x_ax[i]
        #zw = y_ax[j]
        x = croppedVArray[:,:y_ax[j]]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = ts)
        cwDict = {0:03,1:1}
        model_LR = lm.LogisticRegression(C=1e5,class_weight=cwDict)
        model_LR.fit(x_train, y_train)
        y_pred_LR = model_LR.predict(x_test)
        z_ax[j,i] = accuracy_score(y_test,y_pred_LR)*100
        k+=1
        print 'Simulations done: {}'.format(k)

# Plotting the 3d plot
X_ax,Y_ax = np.meshgrid(x_ax, y_ax)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X_ax, Y_ax, z_ax)
plt.grid()
plt.xlabel('Test size')
plt.ylabel('No. of time steps')
plt.title('Accuracy')
plt.show()
fig.savefig('Plot3DAccuracy.png',bbox_inches='tight')
# save figure to pickle file

def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

save_obj(fig,'Plot3DAccuracy')
"""
