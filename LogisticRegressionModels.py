# get the voltage data from the pickle file (10 sec N-2 with fault in between)
# train the LR classifier on the data
# test with different initial time-steps (after fault clearance)

# implement (separate) logistic regression models to classify voltage oscillations and low (high) voltages

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


# Functions
def load_obj(name ):
    # load pickle object
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def trainMLModel(x,y, testSize, classWeightDict):
    #Train the LR model on the x (input array) and y (the target vector)

    #x = croppedVArray[:,:60] # take the first x time steps
    #y = dvdtTarget
    y = np.array(y).reshape(-1)

    # partition the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = testSize)


    # train LR
    model_LR = lm.LogisticRegression(C=1e5,class_weight=classWeightDict)
    model_LR.fit(x_train, y_train)
    return model_LR
    #y_pred_LR = model_LR.predict(x_test)

def tuneMLModel(lengthList,target,testSize,classWeightDict):
    # change the feature length of the model within a given range and get a plot of the accuracy
    accuracy = []
    for lenght in lengthList:
        #print 'Current number of features: ', lenght
        x = croppedVArray[:,:lenght] # take the first 100 time steps
        y = target
        y = np.array(y).reshape(-1)
        # status update of this loop
        percent_complete = float(lenght)/len(lengthList)*100
        if percent_complete%10 == 0:
            print 'Percentage complete: ', percent_complete

        # partition the data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = testSize)


        # train LR
        model_LR = lm.LogisticRegression(C=1e5,class_weight=classWeightDict)
        model_LR.fit(x_train, y_train)
        y_pred_LR = model_LR.predict(x_test)
        # predict
        accuracy.append(accuracy_score(y_test,y_pred_LR)*100)
        #print('Accuracy : {}'.format(accuracy_score(y_test, 
        #      y_pred_LR)*100))

    # Plot accuracy wrt no. of features
    features = lengthList
    plt.plot(features,accuracy)
    titleStr = 'Plot of LR model performance vs no. of features'
    plt.title(titleStr)
    plt.ylabel('Percentage accuracy')
    plt.ticklabel_format(useOffset=False)
    plt.xlabel('Number of features')
    plt.grid()
    plt.show() 

def confMatrix(croppedVArray,y,featureNo,testSize,classWeightDict):
    # get the confusion matrix of the classifier with given feature length, test size and class weights

    x = croppedVArray[:,:featureNo] # take the first 100 time steps
    #y = target
    y = np.array(y).reshape(-1)

    # partition the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = testSize)


    # train LR
    model_LR = lm.LogisticRegression(C=1e5,class_weight=classWeightDict)
    model_LR.fit(x_train, y_train)
    y_pred_LR = model_LR.predict(x_test)

    return confusion_matrix(y_test, y_pred_LR)

def tweakTestSize(testSizeList,croppedVArray,y,classWeightDict,noOfFeatures):
    # given a no. of features and class weights, plot accuracies wrt the testSize

    accuracy = []
    for testSize in testSizeList:
        #print 'Current number of features: ', lenght
        x = croppedVArray[:,:noOfFeatures] # take the first 100 time steps

        y = np.array(y).reshape(-1)

        # partition the data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = testSize)


        # train LR
        model_LR = lm.LogisticRegression(C=1e5,class_weight=classWeightDict)
        model_LR.fit(x_train, y_train)
        y_pred_LR = model_LR.predict(x_test)
        # predict
        accuracy.append(accuracy_score(y_test,y_pred_LR)*100)
        #print('Accuracy : {}'.format(accuracy_score(y_test, 
        #      y_pred_LR)*100))

    # Plot accuracy wrt test size ratio
    plt.plot(testSizeList,accuracy)
    titleStr = 'Plot of LR model performance vs test to train ratio'
    plt.title(titleStr)
    plt.ylabel('Percentage accuracy')
    plt.ticklabel_format(useOffset=False)
    plt.xlabel('Test to train ratio')
    plt.grid()
    plt.show() 
###########





print 'Loading the voltage data from the object file...'
VoltageDataDict = load_obj('VoltageData') # this voltage data dictionary has been generated from the script 'TS3phSimN_2FaultDataSave.py', see it for key format
"""
testV = VoltageDataDict[VoltageDataDict.keys()[0]]
plt.plot(tme, testV)
plt.show()
"""


# crop the data till after fault clearance
print 'Formatting the data to be used by the LR model....'
tme = VoltageDataDict['time']
timestep = tme[1] - tme[0]
#ind_fault_clearance = int(1.31/timestep) #  the fault is cleared at this time 
ind_fault_clearance = int(0.31/timestep)  + 1 #  the fault is cleared at this time 
samplevCropped = VoltageDataDict[VoltageDataDict.keys()[0]][ind_fault_clearance:]



croppedVArray = np.zeros((len(VoltageDataDict)-1,samplevCropped.shape[0])) # make an array of zeros where each row is a sample (cropped) voltage
dvdtTarget = np.zeros(len(VoltageDataDict)-1) # the target vector for dvdt classification
abnormalVTarget = np.zeros(len(VoltageDataDict)-1) # the target vector for abnormal voltage classification
dvdtClass1 = [] # event ids where high dv_dt is observed
dvdtClass0 = []
# temporary lists
dvdtTmpClass1 = []
dvdtTmpClass0 = []


VClass1 = [] # event ids where voltage is either consistently low or consistently high
VClass0 = []
k= 0 # row number of croppedVArray
for key in VoltageDataDict:
    if key == 'time':
        continue
    croppedV = VoltageDataDict[key][ind_fault_clearance:]
    array_len = croppedV.shape[0]
    croppedVArray[k] = croppedV
    steadyV = croppedV[-100:] # the final 100 samples of the voltage

    # get the derivative of the steady state voltage
    dv_dt = np.zeros(steadyV.shape[0])
    for i in range(steadyV.shape[0]):
        try:
            diff = abs((steadyV[i]-steadyV[i-1])/timestep)
            dv_dt[i] = diff
        except: # when  i=0, since there is no i-1
            continue
    #abnormalVList = [steadyV[j] for j in range(steadyV.shape[0]) if steadyV[j] < 0.9 or steadyV[j] > 1.1 or dv_dt[j] > 0.1] # based on voltage and dv_dt values
    # classify instability according to the rate of change of voltage
    highdvdtList = [steadyV[j] for j in range(steadyV.shape[0]) if dv_dt[j] > 0.05] # based only on dv_dt thresholds
    if len(highdvdtList) > 10:
        dvdtTarget[k] = 1.0
        dvdtClass1.append(key)
    else:
        dvdtClass0.append(key)


    """    
    # just a temporary test to see what kind of oscillations belong to this class
    medvdtList = [steadyV[j] for j in range(steadyV.shape[0]) if dv_dt[j] > 0.01 and dv_dt[j] < 0.1] # based only on dv_dt thresholds
    if len(medvdtList) > 50:
        #dvdtTarget[k] = 1.0
        dvdtTmpClass1.append(key)
    else:
        dvdtTmpClass0.append(key)
    """
    


    # classify whether voltage is within bounds or not (but dont classify voltage oscillations)
    abnormalVList = [steadyV[j] for j in range(steadyV.shape[0]) if (steadyV[j] < 0.9 or steadyV[j] > 1.1) and dv_dt[j] < 0.01]

    if len(abnormalVList) > 10:
        abnormalVTarget[k] = 1.0
        VClass1.append(key)
    else:
        VClass0.append(key)

    k+=1



# list all the events according to classification
print 'Outputting the classification to files...'
with open('Casedvdt.txt','w') as f:
    # put down all the class 0 dv_dt
    f.write('Class 0:')
    f.write('\n')
    for case in dvdtClass0:
        f.write(case)
        f.write('\n')

    # put down all the class 1 dv_dt
    f.write('Class 1:')
    f.write('\n')
    for case in dvdtClass1:
        f.write(case)
        f.write('\n')
"""
# tmp output
with open('CasedvdtTmp.txt','w') as f:
    # put down all the class 0 dv_dt
    f.write('Class 0:')
    f.write('\n')
    for case in dvdtTmpClass0:
        f.write(case)
        f.write('\n')

    # put down all the class 1 dv_dt
    f.write('Class 1:')
    f.write('\n')
    for case in dvdtTmpClass1:
        f.write(case)
        f.write('\n')
"""
# output for voltage classification
with open('CaseV.txt','w') as f:
    # put down all the class 0 dv_dt
    f.write('Class 0:')
    f.write('\n')
    for case in VClass0:
        f.write(case)
        f.write('\n')

    # put down all the class 1 dv_dt
    f.write('Class 1:')
    f.write('\n')
    for case in VClass1:
        f.write(case)
        f.write('\n')

"""
# train the LR model for voltage oscillation
print 'Training the LR model for voltage oscillation'
x = croppedVArray[:,:60] # the first 60 timesteps of the voltage array after line clearance
y = dvdtTarget
testSize = 0.25
classWeightDict = {0:0.03,1:1}
LR_modeldvdt = trainMLModel(x,y, testSize, classWeightDict)

# train the LR model for low and high voltages
print 'Training the LR model for abnormal voltages with minimal oscillation in steady state'
#x = croppedVArray[:,:60] # the first 60 timesteps of the voltage array after line clearance
x = croppedVArray[:,:3] # the first 60 timesteps of the voltage array after line clearance
y = abnormalVTarget
testSize = 0.25
classWeightDict = {0:0.03,1:1}
LR_modelV = trainMLModel(x,y, testSize, classWeightDict)

# save the models
# separate directory for storing the models
currentdir = os.getcwd()
model_dir = currentdir +  '/MLModels'
if not os.path.isdir(model_dir):
    os.mkdir(model_dir)


# dvdt
LRFiledvdt = model_dir + '/' +  'LR_modeldvdt.sav'
pickle.dump(LR_modeldvdt, open(LRFiledvdt, 'wb'))

# voltage
LRFileV = model_dir + '/' +  'LR_modelV.sav'
pickle.dump(LR_modelV, open(LRFileV, 'wb'))
"""


"""
# tune the model for voltage anomaly
print 'Tuning the model for voltage anomaly wrt feature size...'
lengthList = range(1,101)
testSize = 0.25
classWeightDict = {0:0.03,1:1}
tuneMLModel(lengthList,abnormalVTarget,testSize,classWeightDict)
"""
"""
# tune the model for voltage anomaly
print 'Tuning the model for voltage oscillations wrt feature size...'
lengthList = range(1,101)
testSize = 0.25
classWeightDict = {0:0.03,1:1}
tuneMLModel(lengthList,dvdtTarget,testSize,classWeightDict)
"""

"""
# get the confusion matrix for a certain model
print 'Getting the confusion matrix'
y = abnormalVTarget
featureNo = 1
testSize = 0.25
classWeightDict = {0:0.03,1:1}
print confMatrix(croppedVArray,y,featureNo,testSize,classWeightDict)
"""

"""
# plot variations of the accuracy wrt the test size
print 'Tweaking test size and plotting the accuracy...'
testSizeList = list(np.linspace(0.05,0.95,15))
y = abnormalVTarget
classWeightDict = {0:0.03,1:1}
noOfFeatures = 1
tweakTestSize(testSizeList,croppedVArray,y,classWeightDict,noOfFeatures)
"""






"""
# plot accuracy vs the number of features
accuracy = []
for lenght in range(1,101):
    print 'Current number of features: ', lenght
    x = croppedVArray[:,:lenght] # take the first 100 time steps
    y = abnormalVTarget
    y = np.array(y).reshape(-1)

    # partition the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.05)


    # train LR
    model_LR = lm.LogisticRegression(C=1e5,class_weight={0:0.03,1:1})
    model_LR.fit(x_train, y_train)
    y_pred_LR = model_LR.predict(x_test)
    # predict
    accuracy.append(accuracy_score(y_test,y_pred_LR)*100)
    #print('Accuracy : {}'.format(accuracy_score(y_test, 
    #      y_pred_LR)*100))

# Plot accuracy wrt no. of features
features = range(1,101)
plt.plot(features,accuracy)
plt.show()
"""

"""
# ML to predict voltage oscillations
x = croppedVArray[:,:60] # take the first x time steps
y = dvdtTarget
y = np.array(y).reshape(-1)

# partition the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)


# train LR
model_LR = lm.LogisticRegression(C=1e5,class_weight={0:0.03,1:1})
model_LR.fit(x_train, y_train)
y_pred_LR = model_LR.predict(x_test)


# Save the LR model





# expected cost 
# Here: 0: Negative, 1: Positive
benefit_tP = 50
cost_fN = -100
benefit_tN = 20
cost_fP = -10
cost = ExpCost(y_test,y_pred_LR,benefit_tP,cost_fN, benefit_tN, cost_fP)
print 'Expected Benefit: ', cost
# confusion matrix
#print confusion_matrix(y_test, y_pred_LR)
#accuracy.append(accuracy_score(y_test,y_pred_LR)*100)

# Plot tests
#testV = VoltageDataDict['3005,3007,1;3007,3008,1;F3008/3007']
#testV = VoltageDataDict['151,201,1;154,205,1;F154/154']
testV = VoltageDataDict['151,201,1;151,152,2;F151/152']
plt.plot(tme,testV)
plt.grid()
plt.ylim(0.8,1.1)
plt.show()
"""





