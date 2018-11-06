# tests the LR model on the large voltage (using different load levels) dataset
print 'Importing modules'
import pickle
import matplotlib.pyplot as plt
import h5py
# importing evaluation metrics
from sklearn.metrics import accuracy_score, confusion_matrix
# Logistic Regression
from sklearn import linear_model as lm, preprocessing
# for splitting the data
from sklearn.model_selection import train_test_split, cross_val_predict, cross_validate, cross_val_score
import numpy as np
import skfuzzy as fuzz # for fuzzy c means

def load_obj(name):
    # load pickle object
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
# Functions
def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

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


def testMLModel2(x,y, testSize, classWeightDict,noOfFolds):
    # use cross-validation with stratified k fold (preserving the percentage of classes) for the data splits

    fpList = []
    fnList = []
    tpList = []
    tnList = []
    accuracyList = []
    y = np.array(y).reshape(-1)
    #skf = StratifiedKFold(n_splits=noOfFolds)
    #train, test = skf.split(x,y)

    model_LR = lm.LogisticRegression(C=1e5,class_weight=classWeightDict)
    y_pred  =  cross_val_predict(model_LR, x, y, cv=noOfFolds)
    #scores  =  cross_val_score(model_LR, x, y, cv=noOfFolds)
    conf_mat = confusion_matrix(y, y_pred)
    accuracy = accuracy_score(y,y_pred)*100
    tp = conf_mat[0][0]
    fn = conf_mat[0][1] # false alarm
    fp = conf_mat[1][0] # event undetected
    tn = conf_mat[1][1]


    #return scores
    return y_pred, accuracy, fp, fn, tp, tn
    #avg_fp = np.mean(fpList)
    #avg_fn = np.mean(fnList)
    #avg_accuracy = np.mean(accuracyList)
    #return avg_fp, avg_fn, avg_accuracy
#####################

print 'Importing data'
vDataFile = h5py.File('obj/vInpClass.h5', 'r')
inputV = vDataFile['inp'][:] 
targetOsc= vDataFile['targetOsc'][:]

vDataFile.close()


##########
# get average performance using first 100  samples
x = inputV[:,:100]
x = preprocessing.scale(x) # the scaled inputs have zero mean and unit variance for each feature
x = preprocessing.scale(x,axis=1) # the scaled inputs have zero mean and unit variance for every sample
y = targetOsc
testSize = 0.25
#classWeightDict = {0:0.03,1:1}
#classWeightDict = {0:0.1,1:1}
classWeightDict = {0:1,1:1}
"""
# if not using cross-validate
avg_fp, avg_fn, avg_accuracy = testMLModel(x,y, testSize, classWeightDict,10)
print('Average accuracy: {}'.format(avg_accuracy))
print('Average false positives: {}'.format(avg_fp))
print('Average false negatives: {}'.format(avg_fn))
"""


# if using cross-validate
y_pred, accuracy, fp, fn, tp, tn = testMLModel2(x,y, testSize, classWeightDict,10)
print('Accuracy: {}'.format(accuracy))
print('False positives: {}'.format(fp))
print('False negatives: {}'.format(fn))
print('True positives: {}'.format(tp))
print('True negatives: {}'.format(tn))

##############


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

"""
#######
# get the templates of the class 0 and class 1 data
print 'Getting the templates'
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





# generate the similarity thresholds
print 'Getting similarities...'
similarity0List = []
similarity1List = []

for i in range(inputV.shape[0]):
    v =  inputV[i][:60]
    vMean = np.mean(v)
    croppedtmplt0 = tmpltCls0[:60]
    croppedtmplt1 = tmpltCls1[:60]
    meantmpltCls0 = np.mean(croppedtmplt0)
    meantmpltCls1 = np.mean(croppedtmplt1)
    # generate the offsets
    meanOffset0 = (vMean-meantmpltCls0)*np.ones(v.shape[0])
    meanOffset1 = (vMean-meantmpltCls1)*np.ones(v.shape[0])
    
    # using offset
    #similarity0 = 1/np.linalg.norm(v-croppedtmplt0-meanOffset0)
    #similarity1 = 1/np.linalg.norm(v-croppedtmplt1-meanOffset1)
    
    # not using offset
    similarity0 = 1/np.linalg.norm(v-croppedtmplt0)
    similarity1 = 1/np.linalg.norm(v-croppedtmplt1)
    
    similarity0List.append(similarity0)
    similarity1List.append(similarity1)

similarity0Array = np.array(similarity0List)
similarity1Array = np.array(similarity1List)

similarityArray = np.zeros((2,len(similarity0List)))
similarityArray[0,:] = similarity0Array
similarityArray[1,:] = similarity1Array
similarityArray = np.transpose(similarityArray)

# saving the similarity array
save_obj(similarityArray,'tmpSimilarity') # use this to save time generating the same similarity array

##################
"""
# Use this to save time after the array has been saved for the first time
#similarityArray = load_obj('tmpSimilarity')

"""
# plot the templates
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
"""

###########



"""
# train the LR model on just the similarity features
print 'Training using both similarity 0 and 1'
x = similarityArray
y = targetOsc
y = np.array(y).reshape(-1)
testSize = 0.25
classWeightDict = {0:0.1,1:1}
avg_fp, avg_fn, avg_accuracy = testMLModel(x,y, testSize, classWeightDict,10)
print('Average accuracy: {}'.format(avg_accuracy))
print('Average false positives: {}'.format(avg_fp))
print('Average false negatives: {}'.format(avg_fn))


print 'Training using only similarity 0'
x = similarityArray[:,0]
x = np.array(x).reshape(-1,1)
y = targetOsc
y = np.array(y).reshape(-1)
testSize = 0.25
classWeightDict = {0:0.1,1:1}
avg_fp, avg_fn, avg_accuracy = testMLModel(x,y, testSize, classWeightDict,10)
print('Average accuracy: {}'.format(avg_accuracy))
print('Average false positives: {}'.format(avg_fp))
print('Average false negatives: {}'.format(avg_fn))


print 'Training using only similarity 1'
x = similarityArray[:,1]
x = np.array(x).reshape(-1,1)
y = targetOsc
y = np.array(y).reshape(-1)
testSize = 0.25
classWeightDict = {0:0.1,1:1}
avg_fp, avg_fn, avg_accuracy = testMLModel(x,y, testSize, classWeightDict,10)
print('Average accuracy: {}'.format(avg_accuracy))
print('Average false positives: {}'.format(avg_fp))
print('Average false negatives: {}'.format(avg_fn))



# train the LR model on just the similarity features using cross-validation
print 'Training using both similarity 0 and 1 using feature scaling'
x = similarityArray
x = preprocessing.scale(x) # the scaled inputs have zero mean and unit variance
y = targetOsc
y = np.array(y).reshape(-1)
testSize = 0.25
classWeightDict = {0:0.1,1:1}
y_pred, accuracy, fp, fn = testMLModel2(x,y, testSize, classWeightDict,10)
#print conf_mat
print('Cross-validation accuracy: {}'.format(accuracy))
print('Cross-validation false positives: {}'.format(fp))
print('Cross-validation false negatives: {}'.format(fn))

###############
"""

"""
###########
# get a list of all the max voltages after fault clearance
print 'Getting the max voltages after fault clearance'
vMaxList = []
vMaxClass0 = []
vMaxClass1 = []
class1Indices = [ind for ind, val in enumerate(list(targetOsc)) if val == 1.0]
for i in range(inputV.shape[0]):
    v =  inputV[i][:60]
    maxV = np.max(v)
    vMaxList.append(maxV)
    if i in class1Indices:
        vMaxClass1.append(maxV)
    else:
        vMaxClass0.append(maxV)

vMaxArray = np.array(vMaxList)
# saving the similarity array
save_obj(vMaxList,'tmpvMax') # use this to save time generating the same similarity array
####################
"""

######
"""
# plot the frequency density
plt.hist(vMaxClass0, bins='auto',label='Class 0')  
plt.hist(vMaxClass1, bins='auto',label='Class 1')
plt.legend()
titleStr = 'Max voltage distribution'
plt.title(titleStr)
plt.xlabel('Voltage')
plt.ylabel('Samples')
plt.show()
"""
#########



"""
################
# uncomment this when the array is saved
vMaxArray =  load_obj('tmpvMax')
print 'Evaluating performance using similarity as well as the max voltages'
similarityArray = np.transpose(similarityArray)
featureArray = np.zeros((3,similarityArray.shape[1]))
featureArray[0:2,:] = similarityArray
featureArray[2,:] = vMaxArray

featureArray = np.transpose(featureArray)
x= featureArray
x = preprocessing.scale(x) # the scaled inputs have zero mean and unit variance
y = targetOsc
y = np.array(y).reshape(-1)
testSize = 0.25
classWeightDict = {0:0.1,1:1}
y_pred, accuracy, fp, fn = testMLModel2(x,y, testSize, classWeightDict,10)
#print conf_mat
print('Cross-validation accuracy: {}'.format(accuracy))
print('Cross-validation false positives: {}'.format(fp))
print('Cross-validation false negatives: {}'.format(fn))
########
"""







