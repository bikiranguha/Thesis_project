# create a multi-class voltage oscillation classifier

# Modules to import
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model as lm
from sklearn.metrics import accuracy_score, confusion_matrix

# Functions
def load_obj(name ):
    # load pickle object
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def classifyOscillation(dvdt):
    # get the mean of the steady state oscillation and categorize according to it
    # the higher the category no., the more severe the oscillation
    cat = 0.0
    mean_dvdt = np.mean(dvdt)
    """
    if mean_dvdt > 0.1 and mean_dvdt <= 0.2:
        cat = 1
    elif mean_dvdt > 0.2 and mean_dvdt <= 0.3:
        cat = 2   
    elif mean_dvdt > 0.3 and mean_dvdt <= 0.4:
        cat = 3 
    elif mean_dvdt > 0.4 and mean_dvdt <= 0.5:
        cat = 4
    elif mean_dvdt > 0.5 and mean_dvdt <= 0.6:
        cat = 5
    elif mean_dvdt > 0.6 and mean_dvdt <= 0.7:
        cat = 6
    elif mean_dvdt > 0.7 and mean_dvdt <= 0.8:
        cat = 7
    elif mean_dvdt > 0.8 and mean_dvdt <= 0.9:
        cat = 8
    elif mean_dvdt > 0.9 and mean_dvdt <= 1.0:
        cat = 9
    """

    """
    if mean_dvdt > 0.1 and mean_dvdt <= 0.4:
        cat = 1.0
    elif mean_dvdt > 0.4 and mean_dvdt <= 0.7:
        cat = 2.0   
    elif mean_dvdt > 0.7 and mean_dvdt <= 1.0:
        cat = 3.0
    """
    # category definitions
    if mean_dvdt > 0.1 and mean_dvdt <= 0.7:
        cat = 1.0
  
    elif mean_dvdt > 0.7 and mean_dvdt <= 1.0:
        cat = 2.0
    elif mean_dvdt > 1.0:
        cat = 3.0



    return cat




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
    #mul_lr = lm.LogisticRegression(multi_class='multinomial', solver='newton-cg').fit(x_train, y_train)
    #return mul_lr,x_test,y_test
    return model_LR,x_test,y_test
    #y_pred_LR = model_LR.predict(x_test)
###########


# get the voltage data
print 'Loading the voltage data from the object file...'
# this voltage data dictionary has been generated from the script 'TS3phSimN_2FaultDataSave.py', see it for key format
VoltageDataDict = load_obj('VoltageData') 

classCountDict = {}
# crop the data till after fault clearance
print 'Formatting the data to be used by the LR model....'
tme = VoltageDataDict['time']
timestep = tme[1] - tme[0]
ind_fault_clearance = int(1.31/timestep) #  the fault is cleared at this time 
#ind_fault_clearance = int(0.31/timestep)  + 1 #  the fault is cleared at this time 
samplevCropped = VoltageDataDict[VoltageDataDict.keys()[0]][ind_fault_clearance:]

croppedVArray = np.zeros((len(VoltageDataDict)-1,samplevCropped.shape[0])) # make an array of zeros where each row is a sample (cropped) voltage
dvdtTarget = np.zeros(len(VoltageDataDict)-1) # the target vector for dvdt classification
dvdtClass1 = [] # event ids where high dv_dt is observed
dvdtClass0 = []
k=0
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

    cat = classifyOscillation(dv_dt)
    dvdtTarget[k] = cat
    """
    if cat not in classCountDict:
        classCountDict[cat] = 0
    else:
        classCountDict[cat] +=1
    """
    k+=1

# print how many cases belong to each class

#for key in classCountDict:
#    print str(key) + ':' + str(classCountDict[key])


# multi-class classifier
print 'Training the LR model for voltage oscillation....'
x = croppedVArray[:,:100] # the first 60 timesteps of the voltage array after line clearance
y = dvdtTarget
testSize = 0.25
classWeightDict = {0:0.03,1:1,2:1}
LR_modeldvdt,x_test,y_test = trainMLModel(x,y, testSize, classWeightDict)
# get the performance
y_pred_LR = LR_modeldvdt.predict(x_test) 
#print accuracy_score(y_test,y_pred_LR)*100 # print the accuracy score
print confusion_matrix(y_test, y_pred_LR)


