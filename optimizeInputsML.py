# apply different machine learning algorithms on the voltage data and see how well the classifiers work

import pickle
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import linear_model as lm
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

def load_obj(name ):
	with open('obj/' + name + '.pkl', 'rb') as f:
		return pickle.load(f)

VoltageDataDict = load_obj('VoltageData')
tmpList = []
# Build an input feature vector of initial 10 cycles of data, and then a binary target vector (which classifies the case as stable or unstable)
time = VoltageDataDict['time']
words = time.split(',')
time = [float(i) for i in words]

timestep = time[1] - time[0]
time_1s = int(1.0/timestep)


# dictionaries to store the accuracy data
LRDict = {} # key: Number of features in input, value: accuracy score
SVMDict = {} # key: Number of features in input, value: accuracy score
NNDict = {} # key: Number of features in input, value: accuracy score
loopNo = 0 # just to print how many loops have been traversed

classVec = np.zeros((len(VoltageDataDict)-1,1))
VArray = np.zeros((len(VoltageDataDict)-1,565)) # static allocation, based on observation

k= 0 # row number in array input vector
for key in VoltageDataDict:

	if key == 'time':
		continue
	# key contains the event as well as the bus number (look into TS3phN_2FaultDataSave.py for the key structure)
	voltage = VoltageDataDict[key]
	words = voltage.split(',')
	voltageValues = np.zeros(len(words))
	voltageValues =  [float(i) for i in words]
	voltageValues = np.asarray(voltageValues)
	#for i in range(voltageValues.shape[0]):
	#	voltageValues[i] = float(words[i])

	steadyStateV = list(voltageValues[time_1s:time_1s+100]) # 100 cycles, starting 1 sec after the fault clearance
	abnormalVList = [v for v in steadyStateV if v< 0.9 or v> 1.1]
	VArray[k] = voltageValues

	if len(abnormalVList) > 10:
		classVec[k] = 1


	k+=1 # go to next row

y = classVec
y = np.array(y).reshape(-1)
for cycNo in range(1,5+1): # experiment with different input time cycles (between 5 to 20)



	input_cyc = int((float(cycNo)/60)/timestep)
	loopNo +=1
	print 'Loop ' + str(loopNo) +  ' out of 5'

	#NNInputVec = np.zeros((len(VoltageDataDict)-1,input_cyc)) # -1 to exclude time
	#croppedV = voltageValues[:input_cyc]
	#NNInputVec[k] = croppedV
	x = VArray[:,:input_cyc]





	# use the input and classification vector to build your neural network

	# partition the data
	#x = NNInputVec
	print 'Number of features in inputs: ', x.shape[1]
	
	"""
	end = len(x) - 1
	learn_end = int(end*0.954)
	x_train = x[0:learn_end-1]
	x_test = x[learn_end:end-1]
	y_train = y[1:learn_end]
	y_test = y[learn_end+1:end]

	y_train = np.array(y_train).reshape(-1)
	y_test = np.array(y_test).reshape(-1)
	"""
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)
	# =============================================================================
	# Given below are sample models that were tested on the given dataset
	# The models were not optimized to the fullest but below are just simple
	# implementations
	# =============================================================================

	# importing evaluation metrics


	# Logistic Regression
	model_LR = lm.LogisticRegression(C=1e5)
	model_LR.fit(x_train, y_train)
	y_pred_LR = model_LR.predict(x_test)
	LRDict[input_cyc] = accuracy_score(y_test,y_pred_LR)*100
	#print('Accuracy using Logistic Regression: {}'.format(accuracy_score(y_test, 
	#      y_pred_LR)*100))

	# Support Vector Machine
	model_SVM = svm.SVC()
	model_SVM.fit(x_train, y_train)
	y_pred_SVM = model_SVM.predict(x_test)
	SVMDict[input_cyc] = accuracy_score(y_test,y_pred_SVM)*100
	#print('Accuracy using Support Vector Machine: {}'.format(accuracy_score(y_test,
	#      y_pred_SVM)*100))

	# Neural Network
	model_NN = MLPClassifier(hidden_layer_sizes=(10,), max_iter=100, alpha=1e-4,
	                    solver='sgd', verbose=False, tol=1e-4, random_state=1,
	                    learning_rate_init=.1)
	model_NN.fit(x_train, y_train)
	NNDict[input_cyc] = model_NN.score(x_test,y_test)*100
	#print("Test set score: %f" % (model_NN.score(x_test, y_test)*100))


