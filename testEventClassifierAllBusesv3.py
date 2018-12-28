# test an event classifier where the events to be classified are: (balanced and unbalanced) faults, generator and line outages (grouped together)
# voltage data is used
# smoothing and noise addition is carried out
# tests SVM 
# tests LSTM with train, cv and test (with cross-validation in between)
print 'Importing modules'
import csv
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_predict, cross_validate, cross_val_score, KFold
from sklearn.metrics import accuracy_score, confusion_matrix
from avgFilterFn import avgFilter # to emulate filtered data
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from getBusDataFn import getBusData

refRaw = 'savnw.raw'
BusDataDict = getBusData(refRaw)


class Signals(object):
    def __init__(self):
        self.SignalDict = {}


#####################
# organize the fault data
eventTracker = []
print 'Reading the csv files for fault data...'
vFileName = 'fault3ph/vData3phLI.csv' # csv file containing voltage data (different types of fault)
tFileName = 'fault3ph/tData3ph.csv' # csv file containing the time data
eventKeyFile = 'fault3ph/eventIDFileLI.txt'
vFile = open(vFileName,'rb')
tFile = open(tFileName,'rb')
readerV=csv.reader(vFile,quoting=csv.QUOTE_NONNUMERIC) # so that the entries are converted to floats
readerT = csv.reader(tFile,quoting=csv.QUOTE_NONNUMERIC)
tme = [row for idx, row in enumerate(readerT) if idx==0][0]

###
timesteps = 20
start = 0.1
startind = min([idx for idx,val in enumerate(tme) if val >= start])
endind = startind + timesteps
#endind = startind + 20



# make an organized dictionary
# read the event file
print 'Organizing the csv data...'
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

    # pass the input through a 6 cycle average filter
    out = avgFilter(row,6)
    # add some noise to the outputs
    out = np.array(out) + np.random.normal(0,0.01,len(out)) # normal noise with standard deviation of 0.01


    # ignore all faults with relatively high impedance
    #if float(faultZ) > 0.5e-2:
    #    continue
    eventID = 'R{}/F{}/{}/{}'.format(PL,faultbus,faulttype,faultZ)
    if eventID not in EventDict:
        EventDict[eventID] = Signals()
    EventDict[eventID].SignalDict['B{}/{}'.format(currentbus,phase)] = out

# arrange all the fault event data into rows sequentially

AllEventList = []
targetList = []
for event in EventDict:
    s = EventDict[event].SignalDict
    # get the event class
    eventKeyWords = event.split('/')
    faulttype = eventKeyWords[2].strip()
    if faulttype == 'ABCG':

        targetList.append(0)
        eventTracker.append(event)
        allSignals = []
        for b in BusDataDict:
            valA = s['B{}/A'.format(b)][startind:endind]
            valB = s['B{}/B'.format(b)][startind:endind]
            valC = s['B{}/C'.format(b)][startind:endind]
            for v in valA:
                allSignals.append(v)
                
            for v in valB:
                allSignals.append(v)

            for v in valC:
                allSignals.append(v)

        AllEventList.append(allSignals)



    elif faulttype == 'AG':
        # get SLG A data
        targetList.append(1)
        eventTracker.append(event)
        allSignals = []
        for b in BusDataDict:
            valA = s['B{}/A'.format(b)][startind:endind]
            valB = s['B{}/B'.format(b)][startind:endind]
            valC = s['B{}/C'.format(b)][startind:endind]
            for v in valA:
                allSignals.append(v)
                
            for v in valB:
                allSignals.append(v)

            for v in valC:
                allSignals.append(v)

        AllEventList.append(allSignals)



        # get SLG B data
        targetList.append(2)
        eventTracker.append(event.replace('AG','BG'))
        allSignals = []
        for b in BusDataDict:
            valA = s['B{}/A'.format(b)][startind:endind]
            valB = s['B{}/B'.format(b)][startind:endind]
            valC = s['B{}/C'.format(b)][startind:endind]
            for v in valB:
                allSignals.append(v)
                
            for v in valA:
                allSignals.append(v)

            for v in valC:
                allSignals.append(v)

        AllEventList.append(allSignals)  


        # get SLG C data
        targetList.append(3)
        eventTracker.append(event.replace('AG','CG'))
        allSignals = []
        for b in BusDataDict:
            valA = s['B{}/A'.format(b)][startind:endind]
            valB = s['B{}/B'.format(b)][startind:endind]
            valC = s['B{}/C'.format(b)][startind:endind]
            for v in valC:
                allSignals.append(v)
                
            for v in valB:
                allSignals.append(v)

            for v in valA:
                allSignals.append(v)

        AllEventList.append(allSignals)  

FaultEventArray = np.array(AllEventList)
faulttargetArray = np.array(targetList)
#############






#####
eventSet = set() # to keep track of the event ids
N_1EventDict = {}

##############
# generator outages
genoutvdata = []
# get the generator outage data
genOutDir = 'GenOut'

vFilePath = '{}/vGenOut.csv'.format(genOutDir)
aFilePath = '{}/aGenOut.csv'.format(genOutDir)
fFilePath = '{}/fGenOut.csv'.format(genOutDir)
eventFilePath = '{}/eventGenOut.txt'.format(genOutDir)
timeDataFilePath = '{}/t.csv'.format(genOutDir)



# file objects
eventList = []
with open(eventFilePath,'r') as f:
    fileLines = f.read().split('\n')
    for line in fileLines[1:]:
        if line == '':
            continue
        eventList.append(line.strip())


vFilePath = open(vFilePath, 'rb') # 'wb' needed to avoid blank space in between lines
aFilePath = open(aFilePath, 'rb')
fFilePath = open(fFilePath, 'rb')
timeDataFilePath = open(timeDataFilePath, 'rb')


vReader = csv.reader(vFilePath,quoting=csv.QUOTE_NONNUMERIC) # 'wb' needed to avoid blank space in between lines
aReader = csv.reader(aFilePath,quoting=csv.QUOTE_NONNUMERIC)
fReader = csv.reader(fFilePath,quoting=csv.QUOTE_NONNUMERIC)
tReader = csv.reader(timeDataFilePath,quoting=csv.QUOTE_NONNUMERIC)


tme = [row for idx, row in enumerate(tReader) if idx==0][0]

# get the voltage data (and simulate balanced three phase)
for idx, row in enumerate(vReader):
    eventKey = eventList[idx]
    eventKeyWords = eventKey.split('/')
    PL = eventKeyWords[0][1:].strip()
    genbus = eventKeyWords[1][1:].strip()
    currentbus = eventKeyWords[2][1:].strip()
    eventID = 'R{}/G{}'.format(PL,genbus)
    if eventID not in N_1EventDict:
        N_1EventDict[eventID] = []
        #eventTracker.append(eventID)
        #eventSet.add(eventID)


    # pass the input through a 6 cycle average filter
    row = avgFilter(row,6)
    # add some noise to the outputs
    row = np.array(row) + np.random.normal(0,0.01,len(row)) # normal noise with standard deviation of 0.01



    for bs in range(3): # simulate three phase
        for v in row[startind:endind]:
            #genoutvdata.append(v)
            N_1EventDict[eventID].append(v)

#genoutArray = np.array(genoutvdata)

# close all files
vFilePath.close()
aFilePath.close()
fFilePath.close()
timeDataFilePath.close()
###########





####### 
# line outage data
lineoutvdata = []
lineOutDir = 'LineOut'

vFilePath = '{}/vLineOut.csv'.format(lineOutDir)
aFilePath = '{}/aLineOut.csv'.format(lineOutDir)
fFilePath = '{}/fLineOut.csv'.format(lineOutDir)
eventFilePath = '{}/eventLineOut.txt'.format(lineOutDir)
timeDataFilePath = '{}/t.csv'.format(lineOutDir)

# file objects
eventList = []
with open(eventFilePath,'r') as f:
    fileLines = f.read().split('\n')
    for line in fileLines[1:]:
        if line == '':
            continue
        eventList.append(line.strip())


vFilePath = open(vFilePath, 'rb') # 'wb' needed to avoid blank space in between lines
aFilePath = open(aFilePath, 'rb')
fFilePath = open(fFilePath, 'rb')
timeDataFilePath = open(timeDataFilePath, 'rb')


vReader = csv.reader(vFilePath,quoting=csv.QUOTE_NONNUMERIC) # 'wb' needed to avoid blank space in between lines
aReader = csv.reader(aFilePath,quoting=csv.QUOTE_NONNUMERIC)
fReader = csv.reader(fFilePath,quoting=csv.QUOTE_NONNUMERIC)
tReader = csv.reader(timeDataFilePath,quoting=csv.QUOTE_NONNUMERIC)


tme = [row for idx, row in enumerate(tReader) if idx==0][0]

# examine the voltage data
vDict = {}

for idx, row in enumerate(vReader):
    eventKey = eventList[idx]
    eventKeyWords = eventKey.split('/')
    PL = eventKeyWords[0][1:].strip()
    line = eventKeyWords[1][1:].strip()
    currentbus = eventKeyWords[2][1:].strip()
    eventID = 'R{}/L{}'.format(PL,line)
    if eventID not in N_1EventDict:
        #eventTracker.append(eventID)
        #eventSet.add(eventID)
        N_1EventDict[eventID] = []


    # pass the input through a 6 cycle average filter
    row = avgFilter(row,6)
    # add some noise to the outputs
    row = np.array(row) + np.random.normal(0,0.01,len(row)) # normal noise with standard deviation of 0.01


    for bs in range(3): # simulate three phase
        for v in row[startind:endind]:
            #lineoutvdata.append(v)
            N_1EventDict[eventID].append(v)

#lineoutArray = np.array(lineoutvdata).reshape(-1,)





# close all files
vFilePath.close()
aFilePath.close()
fFilePath.close()
timeDataFilePath.close()
######################

N_1Events =[]
for event in N_1EventDict:
    N_1Events.append(N_1EventDict[event])
    eventTracker.append(event)

N_1EventArray = np.array(N_1Events)


N_1targetArray = np.array([4]*(N_1EventArray.shape[0]))

#N_1targetArray = np.array([4]*(genoutArray.shape[0] + lineoutArray.shape[0]))

# stack the arrays
fullArray = np.concatenate((FaultEventArray, N_1EventArray),axis=0) # vertical stacks: faults, gen out, line out
fullTargetArray = np.concatenate((faulttargetArray,N_1targetArray))


with open('EventClassifierLog.txt','w') as f:
    for event in eventTracker:
        f.write(event)
        f.write('\n')



"""
##### evaluate SVM classifier
print('Evaluating SVM')
from sklearn.svm import SVC
x = fullArray
#x = fullArrayFil
y = fullTargetArray.astype(int)

svm_model_linear = SVC(kernel = 'linear', C = 1)
#svm_model_linear = SVC(kernel = 'rbf', C = 1) # is not as good as linear

cv = KFold( n_splits=10)
y_pred  =  cross_val_predict(svm_model_linear, x, y, cv=cv)
accuracy = accuracy_score(y,y_pred)*100

print 'Accuracy: {}'.format(accuracy)
conf_mat = confusion_matrix(y, y_pred)
print accuracy
print conf_mat

scores = cross_val_score(svm_model_linear, x, y, cv=cv)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
######
"""




########
print('Evaluating LSTM using cross-validation')
# reshape the input array to describe timeseries
AllEvent3D = []

for i in range(fullArray.shape[0]):
    eventData = fullArray[i]
    eventData = np.array(eventData).reshape(-1,10)
    eventData = np.transpose(eventData)
    AllEvent3D.append(eventData)

Input3DArray = np.array(AllEvent3D)



# create the model
# import modules required for LSTM
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
from sklearn.model_selection import StratifiedKFold
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers.core import Dense, Activation

"""
def targetEncoder(targetArray):
    # encode class values as integers (needed by LSTM)


    encoder = LabelEncoder()
    encoder.fit(targetArray)
    encoded_Y = encoder.transform(targetArray)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = np_utils.to_categorical(encoded_Y)  
    return dummy_y
"""

x = Input3DArray
y = fullTargetArray

# binarize the label values
lb = LabelBinarizer()
lb.fit(y)
#y_ohe = lb.transform(y) # transform to binarize labels


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
# use cross-validation
#kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=212)
kfold = StratifiedKFold(n_splits=2, shuffle=True) # preserve the distribution of classes in folds
cvscores = []
for train, test in kfold.split(x_train, y_train):
    # this part measures performance using the folds (the test set here is essentially a cross-validation set)


    model = Sequential()
    #model.add(LSTM(23)) # the number of units set to the number of buses
    model.add(LSTM(69, input_shape=(x_train.shape[1], x_train.shape[2]))) # the number refers to features in a timestep
    model.add(Dense(5, activation='sigmoid')) # sigmoid used for classification problems
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])


    dummy_y_train = lb.transform(y_train[train]) # binarize the labels before feeding into LSTM model
    model.fit(x_train[train], dummy_y_train, epochs=75, batch_size=1000, verbose = 2)


    #model.fit(x_train[train], y_train[train])    
    
    # evaluate the model
    #dummy_pred = model.predict(x_train[test])
    #y_pred_prob = model.predict_proba(x_train[test])
    
    # evaluate predictions
    #accuracy = accuracy_score(y_train[test], y_pred)
    dummy_y_test = lb.transform(y_train[test])
    accuracy = model.evaluate(x_train[test], dummy_y_test, verbose=0)
    print("Accuracy cv: %.2f%%" % (accuracy[1] * 100.0))
    #print(classification_report(y_train[test], y_pred))
    cvscores.append(accuracy[1] * 100)

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

# Evaluate on test data
#y_test_pred = model.predict(x_test)
accuracy = model.evaluate(x_test, lb.transform(y_test), verbose=0)
print("Accuracy test: %.2f%%" % (accuracy[1] * 100.0))
#accuracy = accuracy_score(targetEncoder(y_test), y_test_pred)
#print("Accuracy test: %.2f%%" % (accuracy[1] * 100.0))

# Experiments to carry out:
# get a confusion matrix function
# see what is the optimal number of time steps
# optimal number of epochs to train over
# optimal number of LSTMs

####






"""
# split the data into training, cross-validate and test sets
model = Sequential()
#model.add(LSTM(23)) # the number of units set to the number of buses
model.add(LSTM(69, input_shape=(x_train.shape[1], x_train.shape[2]))) # the number refers to features in a timestep
model.add(Dense(5, activation='sigmoid')) # sigmoid used for classification problems
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=75, batch_size=1000, verbose = 2)
# Final evaluation of the model
scores = model.evaluate(x_test, y_test, verbose=0)
y_pred = model.predict(x_test)
print("Accuracy: %.2f%%" % (scores[1]*100))
#conf_mat = confusion_matrix(y_test, y_pred)
#print(conf_mat)
"""
###############




"""
# plot any event
eventID  = 'R100/F3003/ABCG/8.0e-03'
eventInd = eventTracker.index(eventID)
sampleData = fullArray[eventInd]
plt.plot(sampleData)
plt.title(eventID)
plt.grid()
plt.show()
"""