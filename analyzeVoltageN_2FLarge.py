# analyze the steady state individual bus voltages for the N-2 plus fault cases
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# these files are generated from integrateN_2VA.py
vFileSteady = 'obj/vN_2FNewSteady.csv'
vFileInput = 'obj/vN_2FNewTransient.csv'
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


#### get the input voltage array
print('Getting the inputs...')
vdfT  = pd.read_csv(vFileInput,header = None)

inputArrayEventWise = vdfT.values[:-1] # the last row is incomplete, so take it out
inputArrayV = inputArrayEventWise.reshape(-1,120)
inputArrayV = inputArrayV[:,:60]
# for i in range(10):
#     plt.plot(inputArrayV[i])
# plt.grid()
# plt.show()

######






# #### get the target voltage array
print('Getting the targets...')
vdfS  = pd.read_csv(vFileSteady,header = None)

targetArrayEventWise = vdfS.values[:-1] # the last row is incomplete, so take it out
vDataBusWise = targetArrayEventWise.reshape(-1,120)



threshold = 0.35
threshold = 0.2
targetArray = []
class0List = []
class1List = []
for i in range(len(vDataBusWise)):
    value = vDataBusWise[i]
    dvalue = abs(np.gradient(value)*120.0) # get the absolute rate of change of voltage
    if dvalue.mean() > 0.35:
        targetArray.append(1)
        class1List.append(buswiselist[i])
    else:
        targetArray.append(0)
        class0List.append(buswiselist[i])


targetArray = np.array(targetArray)
# pos = len([i for i in targetArray if i == 1])
# neg = len(targetArray) - pos
# print('Positive samples: {}'.format(pos))
# print('Negative samples: {}'.format(neg))
######

## save the classification in a text file
# with open('tmpCls1List.txt','w') as f:
#     for e in class1List:
#         f.write(e)
#         f.write('\n')

# with open('tmpCls0List.txt','w') as f:
#     for e in class0List:
#         f.write(e)
#         f.write('\n')
##


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

X = inputArrayV
Y = targetArray





print('Testing various classifiers...')
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
#models.append(('SVM', SVC()))
# evaluate each model in turn
f1results = []
precision_results= []
recall_results = []
names = []
scoring = 'accuracy'
for name, model in models:

#### using accuracies
#     kfold = model_selection.KFold(n_splits=3)
#     cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
#     results.append(cv_results)
#     names.append(name)
#     msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
#     print(msg)
# # boxplot algorithm comparison
# fig = plt.figure()
# fig.suptitle('Algorithm Comparison')
# ax = fig.add_subplot(111)
# plt.boxplot(results)
# ax.set_xticklabels(names)
# plt.show()


#### using f-1 score
    names.append(name)
    kfold = StratifiedKFold(n_splits=3, shuffle=True)
    f1_scoreList = []
    precision_list = []
    recall_list = []
    for train, test in kfold.split(X, Y):
        
        model.fit(X[train],Y[train])

        y_pred = model.predict(X[test])
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



































# Some test code:



"""
#### experiments to determine the threshold for classification



### get an idea of the average oscillations 
# avgOsc = [] # list of average abs rate of change of voltage
# oscDict = {}
# for i in range(len(vDataBusWise)):
#     value = vDataBusWise[i]
#     dvalue = abs(np.gradient(value)*120.0) # get the absolute rate of change of voltage
#     oscDict[buswiselist[i]] = dvalue.mean()
#     #avgOsc.append(dvalue.mean())
#     #avgOsc.append(dvalue.mean())

# with open('t.txt','w') as f:
#     for key, value in sorted(oscDict.iteritems(), key=lambda (k,v): v, reverse = True):
#         f.write('{}:{}'.format(key,value))
#         f.write('\n')

# see the distribution of mean rate of change of voltage
# avgOsc = np.array(avgOsc)
# plt.hist(avgOsc,bins = 'auto')
# plt.grid()
# plt.show()


###

# visualize the steady state voltage
event = '106/153,154,1;201,204,1/F201/3004' # not so much osc
event = '101/201,204,1;151,152,2/F152/202'
event = '103/151,152,1;201,202,1/F202/211' # not clear yet
event = '106/152,3004,1;151,201,1/F201/151' # definitely too large
event = '104/151,152,1;201,202,1/F201/101' # maybe just right

eventIndex = buswiselist.index(event)
plt.plot(vDataBusWise[eventIndex])
#plt.plot(np.gradient(vDataBusWise[eventIndex])*120.0)
plt.ylim(0,1.2)
plt.grid()
plt.show()


# ### investigate if there are any nan values
# nanindices = np.isnan(vDataBusWise)
# nanwhere = []
# naneventset = set()
# for i in range(len(nanindices)):
#     for j in range(nanindices.shape[1]):
#         val = nanindices[i,j]
#         if val == True:
#             naneventset.add(buswiselist[i])
#             nanwhere.append([i,j])
#####


# event = '105/3005,3006,1;203,205,2/F203/3011'
# event = '106/151,201,1;152,3004,1/F152/152'
# event = '100/151,201,1;151,152,1/F151/151'
# eventIndex = buswiselist.index(event)
# print(eventIndex)


 
# np.savetxt('tmp.txt',vDataBusWise[-1],delimiter = ',')
# plt.plot(vDataBusWise[eventIndex])
# #plt.plot(np.gradient(vDataBusWise[eventIndex])*120.0)
# plt.ylim(0,1.2)
# plt.grid()
# plt.show()




# for e in list(naneventset):
#     print e
# ###

# avgOsc = []
# for i in range(len(vDataBusWise)):
#     value = vDataBusWise[i]
#     dvalue = np.gradient(value)
#     avgOsc.append(np.nanmean(dvalue))
#     #avgOsc.append(dvalue.mean())

# avgOsc = np.array(avgOsc)
# plt.hist(avgOsc,bins = 'auto')
# plt.grid()
# plt.show()
########



##########
"""