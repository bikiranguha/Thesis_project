# analyze the steady state angle for the N-2 plus fault cases in the savnw case
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from getBusDataFn import getBusData
# these files are generated from integrateN_2VA.py
aFileSteady = 'obj/aN_2FNewSteady.csv'
aFileInput = 'obj/aN_2FNewTransient.csv'
eventsFile = 'obj/eventN_2FNew.txt'
buslistfile = 'obj/buslistN_2F.csv'

# get the event list
eventwiseList = []
with open(eventsFile,'r') as f:
    fileLines = f.read().split('\n')
    for line in fileLines:
        if line == '':
            continue
        eventwiseList.append(line.strip())

# get the bus list
with open(buslistfile,'r') as f:
    filecontent = f.read().strip()
    buslist = filecontent.split(',')


# get a buswise list
buswiselist = []
for event in eventwiseList:
    for bus in buslist:
        currentstr = '{}/{}'.format(event,bus)
        buswiselist.append(currentstr)

refRaw = 'savnw.raw'
busdatadict = getBusData(refRaw)


# get the inputs
print('Getting the inputs...')
originalColumnSize = 23*120
adfT  = pd.read_csv(aFileInput,header = None)

inputArrayEventWise = adfT.values[:-1] # the last row is incomplete, so take it out
inputArrayA = inputArrayEventWise.reshape(-1,120)
inputArrayA = inputArrayA[:,:60]
inputArrayEventWiseNew = inputArrayA.reshape(-1,originalColumnSize/2)

# #### get the target voltage array

print('Getting the steady state angles and generate targets...')
adfS  = pd.read_csv(aFileSteady,header = None)

targetArrayEventWise = adfS.values[:-1] # the last row is incomplete, so take it out
#aDataBusWise = targetArrayEventWise.reshape(-1,120)

angleDevDict = {}
threshold = 6.5
oscTarget = []
for i in range(targetArrayEventWise.shape[0]):
    currentEvent = targetArrayEventWise[i]
    currentEventBusWise = currentEvent.reshape(-1,120)
    event = eventwiseList[i]
    # isolate the gen data
    genAngleDataList = []
    for j in range(currentEventBusWise.shape[0]):
        bus = buslist[j]
        bustype = busdatadict[bus].type
        if bustype == '2' or bustype == '3':
            genAngleDataList.append(currentEventBusWise[j])
    genAngleDataArray = np.array(genAngleDataList)
    meanAngles = np.mean(genAngleDataArray,axis=0)
    rangelist = []
    for k in range(genAngleDataArray.shape[0]):
    
        relAngle = abs(meanAngles-genAngleDataArray[k])
        rng = relAngle.max() - relAngle.min()
        rangelist.append(rng)
        # if rng > 10.0:
        #     abnormalEvents.append(event)
        #     break
    maxRange = np.max(rangelist)
    if maxRange > 6.5:
        oscTarget.append(1)
    else:
        oscTarget.append(0)
    angleDevDict[event] = maxRange

oscTarget = np.array(oscTarget)
# with open('AbnormalEventDataOsc.txt','w') as f:
#     f.write('Event: Max deviation in angle noted')
#     f.write('\n')
#     for key, value in sorted(angleDevDict.iteritems(), key=lambda (k,v): v, reverse = True): # descending order
#         value = '%.3f' % value
#         line = '{}:{}'.format(key,value)
#         f.write(line)
#         f.write('\n')

# # see how many positive and negative samples there are
# pos = len([i for i in oscTarget if i == 1])
# neg = len(oscTarget) - pos
# print('Positive samples: {}'.format(pos))
# print('Negative samples: {}'.format(neg))


### analyze the performance of various classifier models

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, precision_score, f1_score, recall_score

X = inputArrayEventWiseNew
Y = oscTarget

# # Feature Scaling (mean normalization and unit variance)
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X = sc.fit_transform(X)


X_pca = X
####
# implement PCA to find out the optimal number of features

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Applying PCA
from sklearn.decomposition import PCA
# use n_components = None and then use pca.explained_variance_ratio_ to see how many 
# features are necessary
#pca = PCA(n_components = None) 
pca = PCA(n_components = 100) 
X_train = pca.fit_transform(X_train)
#X_test = pca.transform(X_test)
# see the ranked variances per feature
#explained_variance = pca.explained_variance_ratio_
#print(explained_variance)

# transform the high dimensional X to a 2 dimensional X_pca to be used for prediction
X_pca = pca.transform(X)
print(X_pca.shape)
####


# Test various classifiers
print('Testing various classifiers...')
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
f1results = []
precision_results= []
recall_results = []
names = []
results = []
scoring = 'accuracy'
for name, model in models:

# ### using accuracies
#     kfold = model_selection.KFold(n_splits=3)
#     cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
#     results.append(cv_results)
#     names.append(name)
#     msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
#     print(msg)
# # boxplot algorithm comparison
# fig = plt.figure()
# fig.suptitle('Algorithm Comparison using accuracy')
# ax = fig.add_subplot(111)
# plt.boxplot(results)
# ax.set_xticklabels(names)
# plt.show()
####

#### getting f-1 scores, precision, recall and f-1 score 
    names.append(name)
    kfold = StratifiedKFold(n_splits=3, shuffle=True)
    f1_scoreList = []
    precision_list = []
    recall_list = []
    for train, test in kfold.split(X_pca, Y):
        
        model.fit(X_pca[train],Y[train])

        y_pred = model.predict(X_pca[test])
        #get the f-1 score for class 1 (the abnormalities)

        precision = precision_score(Y[test], y_pred)
        recall = recall_score(Y[test], y_pred)
        f1_s = f1_score(Y[test], y_pred)

        precision_list.append(precision)
        recall_list.append(recall)
        f1_scoreList.append(f1_s)


    precision_results.append(precision_list)
    pre_results_array = np.array(precision_list)

    recall_results.append(recall_list)
    recall_array = np.array(recall_list)

    f1results.append(f1_scoreList)
    f1scoreArray = np.array(f1_scoreList)

    msg = "Precision: %s: %f (%f)" % (name, pre_results_array.mean(), pre_results_array.std())
    print(msg)

    msg = "Recall: %s: %f (%f)" % (name, recall_array.mean(), recall_array.std())
    print(msg)

    msg = "F-1score: %s: %f (%f)" % (name, f1scoreArray.mean(), f1scoreArray.std())
    print(msg)


# f-1 score
fig = plt.figure()
fig.suptitle('Algorithm Comparison using f-1 score')
ax = fig.add_subplot(111)
plt.boxplot(f1results)
ax.set_xticklabels(names)
plt.show()


# precision
fig = plt.figure()
fig.suptitle('Algorithm Comparison using precision')
ax = fig.add_subplot(111)
plt.boxplot(precision_results)
ax.set_xticklabels(names)
plt.show()

# recall
fig = plt.figure()
fig.suptitle('Algorithm Comparison using recall')
ax = fig.add_subplot(111)
plt.boxplot(recall_results)
ax.set_xticklabels(names)
plt.show()
# #####
