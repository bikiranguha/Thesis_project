# here the classifier only outputs the fault type using the voltage data of all the buses at the time of fault
# the input data is also shifted by the time window - 1
# 6 cycle filter and noise added to simulate pmu data
# classifiers tested here are: SVM, xgboost, LSTM
print('Importing modules')
import csv
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_predict, cross_validate, cross_val_score, KFold
from sklearn.metrics import accuracy_score, confusion_matrix
from avgFilterFn import avgFilter # to emulate filtered data
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from getBusDataFn import getBusData
import random

refRaw = 'savnw.raw'
BusDataDict = getBusData(refRaw)


class Signals(object):
    def __init__(self):
        self.SignalDict = {}



print('Reading the csv files')
vFileName = 'fault3ph/vData3phLI.csv' # csv file containing voltage data (different types of fault)
tFileName = 'fault3ph/tData3ph.csv' # csv file containing the time data
eventKeyFile = 'fault3ph/eventIDFileLI.txt'
# python 2
#vFile = open(vFileName,'rb')
#tFile = open(tFileName,'rb')
# python 3
vFile = open(vFileName,'r')
tFile = open(tFileName,'r')

readerV=csv.reader(vFile,quoting=csv.QUOTE_NONNUMERIC) # so that the entries are converted to floats
readerT = csv.reader(tFile,quoting=csv.QUOTE_NONNUMERIC)
tme = [row for idx, row in enumerate(readerT) if idx==0][0]

###
timeWindow = 10
start = 0.1
startind = min([idx for idx,val in enumerate(tme) if val >= start])
endind = startind + timeWindow



# make an organized dictionary
# read the event file
print('Organizing the csv data...')
eventList = []
with open(eventKeyFile,'r') as f:
    fileLines = f.read().split('\n')
    for line in fileLines[1:]:
        if line == '':
            continue
        eventList.append(line.strip())


EventDict = {}
SignalDict = {}
for idx, row in enumerate(readerV):
    eventKey = eventList[idx]
    eventKeyWords = eventKey.split('/')
    PL = eventKeyWords[0][1:].strip()
    faultbus = eventKeyWords[1][1:].strip()
    currentbus = eventKeyWords[2][1:].strip()
    faulttype = eventKeyWords[3].strip()
    phase = eventKeyWords[4].strip()
    faultZ = eventKeyWords[5].strip()
    eventID = 'R{}/F{}/{}/{}'.format(PL,faultbus,faulttype,faultZ)
    if eventID not in EventDict:
        EventDict[eventID] = Signals()

    # pass the input through a 6 cycle average filter
    out = avgFilter(row,6)
    # add some noise to the outputs
    out = np.array(out) + np.random.normal(0,0.01,len(out)) # normal noise with standard deviation of 0.01


    EventDict[eventID].SignalDict['B{}/{}'.format(currentbus,phase)] = out


"""
# get an event sample
eventID = 'R100/F151/ABCG/1.0e-6'
busData = 'B152/A'

v = EventDict[eventID].SignalDict[busData]
plt.plot(tme,v)
plt.title('After smoothing and noise addition')
plt.xlabel('Time')
plt.ylabel('V (pu)')
plt.grid()
plt.show()
"""



# arrange all the fault event data into rows sequentially

AllEventList = []
targetList = []
eventList = []
for event in EventDict:
    s = EventDict[event].SignalDict
    # get the event class
    eventKeyWords = event.split('/')
    faulttype = eventKeyWords[2].strip()
    if faulttype == 'ABCG':

        #for i in range(timeWindow/2):
        for i in range(timeWindow-1): 
            eventList.append(event)  
            targetList.append(0)
            allSignals = []
            for b in BusDataDict:
                valA = s['B{}/A'.format(b)][startind-i:endind-i]
                valB = s['B{}/B'.format(b)][startind-i:endind-i]
                valC = s['B{}/C'.format(b)][startind-i:endind-i]
                for v in valA:
                    allSignals.append(v)
                    
                for v in valB:
                    allSignals.append(v)

                for v in valC:
                    allSignals.append(v)

            AllEventList.append(allSignals)



    elif faulttype == 'AG':
        # get SLG A data

        #for i in range(timeWindow/2):
        for i in range(timeWindow-1):
            eventList.append(event) 
            targetList.append(1)
            allSignals = []
            for b in BusDataDict:
                valA = s['B{}/A'.format(b)][startind-i:endind-i]
                valB = s['B{}/B'.format(b)][startind-i:endind-i]
                valC = s['B{}/C'.format(b)][startind-i:endind-i]
                for v in valA:
                    allSignals.append(v)
                    
                for v in valB:
                    allSignals.append(v)

                for v in valC:
                    allSignals.append(v)

            AllEventList.append(allSignals)



        # get SLG B data
        #for i in range(timeWindow/2):
        for i in range(timeWindow-1):
            eventList.append(event.replace('AG','BG')) 
            targetList.append(2)
            allSignals = []
            for b in BusDataDict:
                valA = s['B{}/A'.format(b)][startind-i:endind-i]
                valB = s['B{}/B'.format(b)][startind-i:endind-i]
                valC = s['B{}/C'.format(b)][startind-i:endind-i]
                for v in valB:
                    allSignals.append(v)
                    
                for v in valA:
                    allSignals.append(v)

                for v in valC:
                    allSignals.append(v)

            AllEventList.append(allSignals)  


        # get SLG C data
        #for i in range(timeWindow/2):
        for i in range(timeWindow-1):
            eventList.append(event.replace('AG','CG')) 
            targetList.append(3)
            allSignals = []
            for b in BusDataDict:
                valA = s['B{}/A'.format(b)][startind-i:endind-i]
                valB = s['B{}/B'.format(b)][startind-i:endind-i]
                valC = s['B{}/C'.format(b)][startind-i:endind-i]
                for v in valC:
                    allSignals.append(v)
                    
                for v in valB:
                    allSignals.append(v)

                for v in valA:
                    allSignals.append(v)

            AllEventList.append(allSignals)  

AllEventArray = np.array(AllEventList)
targetArray = np.array(targetList)

"""
###### 
#Illustrating the time shift
for i in range(9):
    plt.plot(AllEventArray[i,:9])

    #print eventList[i]

plt.title('Voltage shifting illustration')
plt.xlabel('Timestep')
plt.ylabel('V (pu)')
plt.grid()
plt.show()
####
"""



"""
##### evaluate SVM classifier
print 'Evaluating the classifier'
from sklearn.svm import SVC
x = AllEventArray
#x = fullArrayFil
y = targetArray.astype(int)

svm_model_linear = SVC(kernel = 'linear', C = 1)
#svm_model_linear = SVC(kernel = 'rbf', C = 1) # is not as good as linear

cv = KFold(n_splits=10)
y_pred  =  cross_val_predict(svm_model_linear, x, y, cv=cv)
accuracy = accuracy_score(y,y_pred)*100

print 'Accuracy: {}'.format(accuracy)
conf_mat = confusion_matrix(y, y_pred)
print conf_mat

scores = cross_val_score(svm_model_linear, x, y, cv=cv)
print("Accuracy: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))
#####
"""

"""
#### Evaluate xgboost
print('Evaluating XGBoost')
import xgboost as xgb
x = AllEventArray
y = targetArray

train_X, test_X, train_Y, test_Y = train_test_split(x, y, test_size = 0.25)
# classifier part (old)
xg_train = xgb.DMatrix(train_X, label=train_Y)
xg_test = xgb.DMatrix(test_X, label=test_Y)
# setup parameters for xgboost
param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softmax'
# scale weight of positive examples
param['eta'] = 0.1
param['max_depth'] = 6
param['silent'] = 1
param['nthread'] = 4
param['num_class'] = 4 # number of classes

watchlist = [(xg_train, 'train'), (xg_test, 'test')]
num_round = 10
bst = xgb.train(param, xg_train, num_round, watchlist)
# get prediction
pred = bst.predict(xg_test)
#error_rate = np.sum(pred != test_Y) / test_Y.shape[0]
accuracy = np.sum(pred == test_Y) / test_Y.shape[0]*100
#print('Test error using softmax = {}'.format(error_rate))
print('Test accuracy = {}'.format(accuracy))

# do the same thing again, but output probabilities
param['objective'] = 'multi:softprob'
bst = xgb.train(param, xg_train, num_round, watchlist)
# Note: this convention has been changed since xgboost-unity
# get prediction, this is in 1D array, need reshape to (ndata, nclass)
pred_prob = bst.predict(xg_test).reshape(test_Y.shape[0], 6)
pred_label = np.argmax(pred_prob, axis=1)
error_rate = np.sum(pred_label != test_Y) / test_Y.shape[0]
print(Test error using softprob = {}'.format(error_rate))

######
"""



# reshape the input array to describe timeseries
AllEvent3D = []

for i in range(AllEventArray.shape[0]):
    eventData = AllEventArray[i]
    eventData = np.array(eventData).reshape(-1,10)
    eventData = np.transpose(eventData)
    AllEvent3D.append(eventData)

Input3DArray = np.array(AllEvent3D)

########
# Evaluate LSTM
# create the model
# import modules required for LSTM
from sklearn.preprocessing import LabelEncoder
#from keras.utils import np_utils
#from keras.models import Sequential
#from keras.layers import LSTM
#from keras.layers.core import Dense, Activation

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(targetArray)
encoded_Y = encoder.transform(targetArray)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)




x = Input3DArray
y = dummy_y

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)


#print(model.summary())
#model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=3, batch_size=1000)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)
#model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=3, batch_size=1000)


model = Sequential()
#model.add(LSTM(23)) # the number of units set to the number of buses
model.add(LSTM(69, input_shape=(x_train.shape[1], x_train.shape[2]))) # the number refers to features in a timestep
model.add(Dense(4, activation='sigmoid')) # sigmoid used for classification problems
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=50, batch_size=1000, verbose = 2)
# Final evaluation of the model
scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
###############

### build an algorithm which ranks the proximity of the PMU voltage to the fault
### according to the voltage drop of the sample


#### pick a random sample
x = AllEventArray
ind = random.choice(range(x.shape[0]))
random_sample = x[ind]
event = eventList[ind]
print event
# reshape into an array where each row represents a three phase voltage sample
# over the time window
sampleArray = random_sample.reshape(-1,30)
######


###### test a certain event
#event = 'R100/F151/ABCG/4.6e-02'
#event = 'R100/F151/ABCG/1.0e-6'
event = 'R100/F151/AG/1.0e-6'
print event
x = AllEventArray
eventInd = eventList.index(event)
sample = x[eventInd]
# reshape into an array where each row represents a three phase voltage sample
# over the time window
sampleArray = sample.reshape(-1,30)



# plot the event
plt.plot(sampleArray[0])
plt.grid()
plt.show()
####
"""



"""
# build a dictionary of the minimum voltage of each row
minVDict = {}
for i in range(sampleArray.shape[0]):
    minVDict[i] = sampleArray[i].min()

print 'Bus ID: Min voltage (sorted)'
for ind, value in sorted(minVDict.iteritems(), key=lambda (k,v): v): 
    print '{}:{}'.format(BusDataDict.keys()[ind],value)

######################
###########
"""


