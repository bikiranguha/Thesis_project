# build neural network model using voltage data after line clearance (of an N-2 contingency )

import pickle
import numpy as np

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
time_10cyc = int((10.0/60)/timestep)

#print time_1s
#print time_10cyc
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
"""
end = len(x) - 1
learn_end = int(end*0.954)
x_train = x[0:learn_end-1]
x_test = x[learn_end:end-1]
y_train = y[1:learn_end]
y_test = y[learn_end+1:end]
"""
	
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
y = np.array(y).reshape(-1)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.05) 

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# Fitting Logistic Regression to the training set

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0) # to get same results
classifier.fit(X_train, y_train)


# Predicting the test set results
y_pred = classifier.predict(X_test)
errors = [x for i,x in enumerate(y_test) if y_pred[i] != y_test[i]]
print len(errors)
#s = classifier.score(X_test,y_test)




