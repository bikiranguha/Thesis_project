# tests the LR model on the large voltage (using different load levels) dataset

import pickle
import matplotlib.pyplot as plt
import h5py
# importing evaluation metrics
from sklearn.metrics import accuracy_score, confusion_matrix
# Logistic Regression
from sklearn import linear_model as lm
# for splitting the data
from sklearn.model_selection import train_test_split
import numpy as np
import skfuzzy as fuzz # for fuzzy c means

def load_obj(name):
    # load pickle object
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def testMLModel(x,y, testSize, classWeightDict,noOfTrials):
    # train the ML a number of times with randomly selected training and test sets
    # return the average number of false positives and false negatives

    fpList = []
    fnList = []
    accuracyList = []
    y = np.array(y).reshape(-1)
    for i in range(noOfTrials):
       
        print('Loop {} out of {}'.format(i+1,noOfTrials))
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
#####################


vDataFile = h5py.File('obj/vInpClass.h5', 'r')
inputV = vDataFile['inp'][:] 
targetOsc= vDataFile['targetOsc'][:]

vDataFile.close()

"""
##########
# get average performance using first 100  samples
x = inputV[:,:100]
y = targetOsc
testSize = 0.25
#classWeightDict = {0:0.03,1:1}
classWeightDict = {0:0.1,1:1}
avg_fp, avg_fn, avg_accuracy = testMLModel(x,y, testSize, classWeightDict,10)
print('Average accuracy: {}'.format(avg_accuracy))
print('Average false positives: {}'.format(avg_fp))
print('Average false negatives: {}'.format(avg_fn))
##############
"""

"""
##############
# Performance wrt changing timesteps
#ts = np.linspace(0.1,0.8,8) # the test size
noOfSamples = range(10,101,10)
y = targetOsc
y = np.array(y).reshape(-1)
fnList = []
fpList = []
accuracyList = []
for samples in noOfSamples:
    print('Current sample size: {}'.format(samples))
    x = inputV[:,:samples]
    testSize = 0.25
    classWeightDict = {0:0.1,1:1}

    #avg_fp, avg_fn, avg_accuracy = testMLModel(x,y, testSize, classWeightDict,10)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = testSize)
    model_LR = lm.LogisticRegression(C=1e5,class_weight=classWeightDict)
    model_LR.fit(x_train, y_train)
    y_pred_LR = model_LR.predict(x_test)
    cm = confusion_matrix(y_test, y_pred_LR)
    fnList.append(cm[0][1]) # false alarm
    fpList.append(cm[1][0]) # event undetected
    accuracyList.append(accuracy_score(y_test,y_pred_LR)*100)


# plot the changes in accuracy, false positives and false negatives
plt.plot(noOfSamples,accuracyList)
plt.ylabel('Accuracy')
plt.xlabel('Number of timesteps')
plt.title('Accuracy wrt sample size')
plt.grid()
plt.show()
plt.close()

plt.plot(noOfSamples,fpList)
plt.ylabel('FP')
plt.xlabel('Number of timesteps')
plt.title('FP wrt sample size')
plt.grid()
plt.show()
plt.close()

plt.plot(noOfSamples,fnList)
plt.ylabel('FN')
plt.xlabel('Number of timesteps')
plt.title('FN wrt sample size')
plt.grid()
plt.show()
plt.close()
################
"""


"""
##############
# Performance wrt changing class weights
classWeightArray = np.linspace(0.1,1,10) # the class weights
#noOfSamples = range(10,101,10)
y = targetOsc
y = np.array(y).reshape(-1)
fnList = []
fpList = []
accuracyList = []
for i in range(len(classWeightArray)):
    weight0 = classWeightArray[i]
    print('Class weights: 0: {}, 1: 1'.format(weight0))
    x = inputV[:,:100]
    testSize = 0.25
    classWeightDict = {0:weight0,1:1}

    #avg_fp, avg_fn, avg_accuracy = testMLModel(x,y, testSize, classWeightDict,10)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = testSize)
    model_LR = lm.LogisticRegression(C=1e5,class_weight=classWeightDict)
    model_LR.fit(x_train, y_train)
    y_pred_LR = model_LR.predict(x_test)
    cm = confusion_matrix(y_test, y_pred_LR)
    fnList.append(cm[0][1]) # false alarm
    fpList.append(cm[1][0]) # event undetected
    accuracyList.append(accuracy_score(y_test,y_pred_LR)*100)


# plot the changes in accuracy, false positives and false negatives
plt.plot(classWeightArray,accuracyList)
plt.ylabel('Accuracy')
plt.xlabel('Class Zero to One Ratio')
plt.title('Accuracy wrt class ratio')
plt.grid()
plt.show()
plt.close()

plt.plot(classWeightArray,fpList)
plt.ylabel('FP')
plt.xlabel('Class Zero to One Ratio')
plt.title('FP wrt class ratio')
plt.grid()
plt.show()
plt.close()

plt.plot(classWeightArray,fnList)
plt.ylabel('FN')
plt.xlabel('Class Zero to One Ratio')
plt.title('FN wrt class ratio')
plt.grid()
plt.show()
plt.close()
################
"""


#######
# get the templates of the class 0 and class 1 data
class1Indices = [ind for ind, val in enumerate(list(targetOsc)) if val == 1.0]
class0List = []
class1List = []
for i in range(inputV.shape[0]):
    vData  = inputV[i]
    if i in class1Indices:
        class1List.append(vData)
    else:
        class0List.append(vData)

class0Array = np.array(class0List)
class1Array = np.array(class1List)

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


num = range(tmpltCls0.shape[0])
plt.plot(num,tmpltCls0, label = 'Class 0')
plt.plot(num,tmpltCls1, label = 'Class 1')
plt.ylabel('Template plots')
plt.xlabel('No.')
plt.title('Plots of the mean templates')
plt.grid()
plt.legend()
plt.show()
plt.close()
#########################