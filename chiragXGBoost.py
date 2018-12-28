import csv
import numpy as np
import matplotlib.pyplot as plt
from getBusDataFn import getBusData
from sklearn.model_selection import train_test_split, cross_val_predict, cross_validate, cross_val_score, KFold
from sklearn.metrics import accuracy_score, confusion_matrix

refRaw = 'savnw.raw'
BusDataDict = getBusData(refRaw)


class Signals(object):
    def __init__(self):
        self.SignalDict = {}



print ('Reading the input csv files')
vFileName = 'fault3ph/vData3phLI.csv' # csv file containing voltage data (different types of fault)
tFileName = 'fault3ph/tData3ph.csv' # csv file containing the time data
eventKeyFile = 'fault3ph/eventIDFileLI.txt' # file containing the event keys
vFile = open(vFileName,'rt')
tFile = open(tFileName,'rt')
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
print ('Organizing the csv data...')
eventList = []
with open(eventKeyFile,'r') as f:
    fileLines = f.read().split('\n')
    for line in fileLines[1:]:
        if line == '':
            continue
        eventList.append(line.strip())

# make a dictionary to organize all the data eventwise
EventDict = {} # key: event id, value: Signaldict for the event
SignalDict = {}# key: bus id, value: three phase voltage data
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
    EventDict[eventID].SignalDict['B{}/{}'.format(currentbus,phase)] = row

# arrange all the fault event data into rows sequentially
AllEventList = [] # each row contains an event data (all the three phase voltage for the event)
targetList = [] 
eventList = [] # list to keep track of the events in the array
for event in EventDict:
    s = EventDict[event].SignalDict
    # get the event class
    eventKeyWords = event.split('/')
    faulttype = eventKeyWords[2].strip()
    if faulttype == 'ABCG':
        # shift data to simulate faults happening at different times within the sample
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

        # shift data to simulate faults happening at different times within the sample
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
        # shift data to simulate faults happening at different times within the sample
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
        # shift data to simulate faults happening at different times within the sample
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

AllEventArray = np.array(AllEventList) # input (samples along rows)
targetArray = np.array(targetList) #  target

# Description of the data
# There are 23 buses (or nodes) in the system which simultaneously gather the voltage info
# Each bus gets three phase data (voltage A, B and C)
# For each phase, i collect 10 timesteps for the problem, so 30 timesteps per bus
# Since there are 23 buses, the total number of timesteps i collected: 23*30 = 690
# So, in each row, 1st 30 columns is the timeseries data from bus 1, 31-60: from bus 2 and so on...

# so a whole sample looks like this:
plt.plot(AllEventArray[0])
plt.title('Visualization of one row of the input')
plt.ylabel('Voltage (normalized)')
plt.xlabel('Column number')
plt.grid()
plt.show()


# now if i reshape like this, i get the signals from separate buses into separate rows
sampleArray = np.array(AllEventArray[0]).reshape(-1,30)
plt.plot(sampleArray[0])
plt.title('Visualization of three phase voltage of the first bus (fault at phase A)')
plt.ylabel('Voltage (normalized)')
plt.xlabel('Column number')
plt.grid()
plt.show()


# Please note: 
# The three phase info should stick together, because they characterize the classes (type of faults)
# For example:
# Class 0: Three phase fault (All phases show voltage dip)
# Class 1,2 and 3: Single phase fault at phase A (dip only at phase A), B and C

##### evaluate SVM classifier with every row as a separate input sample
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        AllEventArray, targetArray.astype(int), test_size=0.2, random_state=42)

print ('Evaluating SVM')
from sklearn.svm import SVC
svm_model_linear = SVC(kernel = 'linear', C = 1)
#svm_model_linear = SVC(kernel = 'rbf', C = 1) # is not as good as linear

# This is wrong as you are training and testing on the same data
#cv = KFold(n_splits=10)
#y_pred  =  cross_val_predict(svm_model_linear, X_train, y_train, cv=cv)
#accuracy = accuracy_score(y,y_pred)*100
#
#print ('Accuracy: {}'.format(accuracy))
#conf_mat = confusion_matrix(y, y_pred)
#print (conf_mat)
#
#scores = cross_val_score(svm_model_linear, x, y, cv=cv)
#print("Accuracy: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))

svm_model_linear.fit(X_train, y_train)
y_pred = svm_model_linear.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)*100
print ('Accuracy: {}'.format(accuracy))
conf_mat = confusion_matrix(y_test, y_pred)
print (conf_mat)
#####

# XGBoost
print('Evaluating XGBoost')
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=212)
cvscores = []
for train, test in kfold.split(X_train, y_train):

    # Train the model
    model = XGBClassifier(learning_rate=0.01, reg_alpha= 0.01, reg_lambda=0.01, max_depth=7, 
                          max_delta_step= 1.8, colsample_bytree= 0.4,
                          subsample= 0.8, gamma= 0.65, n_estimators= 700)
    model.fit(X_train[train], y_train[train])    
    
    # evaluate the model
    y_pred = model.predict(X_train[test])
    y_pred_prob = model.predict_proba(X_train[test])
    
    # evaluate predictions
    accuracy = accuracy_score(y_train[test], y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print(classification_report(y_train[test], y_pred))
    cvscores.append(accuracy * 100)

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

# Evaluate on test data
y_test_pred = model.predict(X_test)



