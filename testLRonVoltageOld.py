# get the voltage data from the pickle file (10 sec N-2 with fault in between)
# train the LR classifier on the data
# test with different initial time-steps (after fault clearance)

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



def load_obj(name ):
    # load pickle object
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

VoltageDataDict = load_obj('VoltageData') # this voltage data dictionary has been generated from the script 'TS3phSimN_2FaultDataSave.py', see it for key format
"""
testV = VoltageDataDict[VoltageDataDict.keys()[0]]
plt.plot(tme, testV)
plt.show()
"""


# crop the data till after fault clearance
tme = VoltageDataDict['time']
timestep = tme[1] - tme[0]
ind_fault_clearance = int(1.31/timestep) #  the fault is cleared at this time 
samplevCropped = VoltageDataDict[VoltageDataDict.keys()[0]][ind_fault_clearance:]



croppedVArray = np.zeros((len(VoltageDataDict)-1,samplevCropped.shape[0])) # make an array of zeros where each row is a sample (cropped) voltage
dvdtTarget = np.zeros(len(VoltageDataDict)-1)
abnormalVTarget = np.zeros(len(VoltageDataDict)-1)
unstableVCases = [] # for high dv_dt
abnormalVCases = [] # low or high voltage but no oscillations
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
    highdvdtList = [steadyV[j] for j in range(steadyV.shape[0]) if dv_dt[j] > 0.1] # based only on dv_dt thresholds
    if len(highdvdtList) > 10:
        dvdtTarget[k] = 1.0
        unstableVCases.append(key)

    # classify whether voltage is within bounds or not (but dont classify voltage oscillations)
    abnormalVList = [steadyV[j] for j in range(steadyV.shape[0]) if (steadyV[j] < 0.9 or steadyV[j] > 1.1) and dv_dt[j] < 0.01]

    if len(abnormalVList) > 10:
        abnormalVTarget[k] = 1.0
        abnormalVCases.append(key)

    k+=1



with open('AbnormalVList.txt','w') as f:
    for case in abnormalVCases:
        f.write(case)
        f.write('\n')

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

# separate directory for storing the models
currentdir = os.getcwd()
model_dir = currentdir +  '/MLModels'
if not os.path.isdir(model_dir):
    os.mkdir(model_dir)



LRFile = model_dir + '/' +  'LR_modeldvdt.sav'
pickle.dump(model_LR, open(LRFile, 'wb'))



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





