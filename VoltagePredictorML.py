# apply different machine learning algorithms on the voltage data and see how well the classifiers work
# see how long it takes to get each model ready
# save the models, so that they can be used on actual cases

import pickle
import numpy as np
import os
from sklearn.model_selection import train_test_split
import time
# importing evaluation metrics
from sklearn.metrics import accuracy_score
# Logistic Regression
from sklearn import linear_model as lm
# SVM
from sklearn import svm
# Neural Network
from sklearn.neural_network import MLPClassifier

def load_obj(name ):
	with open('obj/' + name + '.pkl', 'rb') as f:
		return pickle.load(f)

VoltageDataDict = load_obj('VoltageData') # this voltage data dictionary has been generated from the script 'TS3phSimN_2FaultDataSave.py', see it for key format
tmpList = []
# Build an input feature vector of initial 10 cycles of data, and then a binary target vector (which classifies the case as stable or unstable)
tme = VoltageDataDict['time']
words = tme.split(',')
tme = [float(i) for i in words]

timestep = tme[1] - tme[0]
time_1s = int(1.0/timestep)
time_10cyc = int((10.0/60)/timestep)


NNInputVec = np.zeros((len(VoltageDataDict)-1,20)) # -1 to exclude time
classVec = np.zeros((len(VoltageDataDict)-1,1))
unstableVID = []
k= 0 # row number in array input vector
for key in VoltageDataDict:

	if key == 'time':
		continue
	# key contains the event as well as the bus number (look into TS3phN_2FaultDataSave.py for the key structure)
	voltage = VoltageDataDict[key]
	words = voltage.split(',')
	voltageValues = np.zeros(len(words))
	for i in range(voltageValues.shape[0]):
		voltageValues[i] = float(words[i])

	# now get the 1st 10 cycles of data as input, and 100 cycles after the time_1s index (steady state)
	# if the steady state goes beyond the given limit for more than 10 times, then system is characterized as unstable
	# build a binary vector for unstability, and a numpy array with  a row for each sample and columns representing each timestep (20 columns)
	# these two will be used for training
	croppedV = voltageValues[:time_10cyc]
	NNInputVec[k] = croppedV
	#for j in range(croppedV.shape[0]):
	#	NNInputVec[k][j] = croppedV[j]
	#print NNInputVec[i]

	steadyStateV = list(voltageValues[time_1s:time_1s+100]) # 100 cycles, starting 1 sec after the fault clearance
	abnormalVList = [v for v in steadyStateV if v< 0.9 or v> 1.1]

	if len(abnormalVList) > 10:
		classVec[k] = 1
		unstableVID.append(key)

	k+=1 # go to next row


# use the input and classification vector to build your neural network

# partition the data
x = NNInputVec
y = classVec
y = np.array(y).reshape(-1)

"""
# old code for test and train splitting
end = len(x) - 1
learn_end = int(end*0.954)
x_train = x[0:learn_end-1]
x_test = x[learn_end:end-1]
y_train = y[1:learn_end]
y_test = y[learn_end+1:end]

y_train = np.array(y_train).reshape(-1)
y_test = np.array(y_test).reshape(-1)
"""
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.05)
# =============================================================================
# Given below are sample models that were tested on the given dataset
# The models were not optimized to the fullest but below are just simple
# implementations
# =============================================================================



LRStart = time.time()
model_LR = lm.LogisticRegression(C=1e5)
model_LR.fit(x_train, y_train)
y_pred_LR = model_LR.predict(x_test)
LREnd = time.time()
print 'LR  took', LREnd-LRStart, 'seconds.'
#print('Accuracy using Logistic Regression: {}'.format(accuracy_score(y_test, 
#      y_pred_LR)*100))

# Support Vector Machine

SVMStart = time.time()
model_SVM = svm.SVC()
model_SVM.fit(x_train, y_train)
y_pred_SVM = model_SVM.predict(x_test)
SVMEnd = time.time()
print 'SVM  took', SVMEnd-SVMStart, 'seconds.'
#print('Accuracy using Support Vector Machine: {}'.format(accuracy_score(y_test,
#      y_pred_SVM)*100))

# Neural Network
NNStart = time.time()
model_NN = MLPClassifier(hidden_layer_sizes=(10,), max_iter=100, alpha=1e-4,
                    solver='sgd', verbose=False, tol=1e-4, random_state=1,
                    learning_rate_init=.1)
model_NN.fit(x_train, y_train)
NNEnd = time.time()
#y_pred_NN = model_NN.predict(x_test)
print 'NN  took', NNEnd-NNStart, 'seconds.'
#print("Test set score: %f" % (model_NN.score(x_test, y_test)*100))

# save the models to disk

# separate directory for storing the models
currentdir = os.getcwd()
model_dir = currentdir +  '/MLModels'
if not os.path.isdir(model_dir):
    os.mkdir(model_dir)

# Save the LR model
LRFile = model_dir + '/' +  'LR_model.sav'
pickle.dump(model_LR, open(LRFile, 'wb'))

# Save the SVM model
SVMFile = model_dir + '/' +  'SVM_model.sav'
pickle.dump(model_SVM, open(SVMFile, 'wb'))


# Save the NN model

NNFile = model_dir + '/' + 'NN_model.sav'
pickle.dump(model_NN, open(NNFile, 'wb'))


