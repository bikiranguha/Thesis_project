# Event (fault and gen outage and motor start) classification using three phase voltage of one bus as a sample
# added noise and smoothing
# added time shift
# added motor start cases
print 'Importing modules'
import csv
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_predict, cross_validate, cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from avgFilterFn import avgFilter # to emulate filtered data
import matplotlib.pyplot as plt
import random


print 'Reading the csv files'
vFileName = 'fault3ph/vData3phLI.csv' # csv file containing voltage data (different types of fault)
tFileName = 'fault3ph/tData3ph.csv' # csv file containing the time data
eventKeyFile = 'fault3ph/eventIDFileLI.txt'
vFile = open(vFileName,'rb')
tFile = open(tFileName,'rb')
readerV=csv.reader(vFile,quoting=csv.QUOTE_NONNUMERIC) # so that the entries are converted to floats
readerT = csv.reader(tFile,quoting=csv.QUOTE_NONNUMERIC)
tme = [row for idx, row in enumerate(readerT) if idx==0][0]



# get the indices for fault on and fault off
faultontime = 0.1
#timesteps = 40
#timesteps = 10
#timesteps = 20
timesteps = 30
numcyc = 6
shiftRange = 5
std = 0.001 # standard deviation of noise
startind = min([idx for idx,val in enumerate(tme) if val >= faultontime])
endind = startind + timesteps


list3PHA = []
list3PHB = []
list3PHC = []


list1PHA = []
list1PHB = []
list1PHC = []

listSteadyA = []
listSteadyB = []
listSteadyC = []

condensedEventKey3ph = []
condensedEventKey1ph = []
condensedEventKeySteady = []
#condensedset = set()
eventTarget3ph = [] # targets for all the 3 ph fault data
eventTargetSLGA = [] # targets for all the single phase fault data at phase A
eventTargetSLGB = [] # SLG phase B
eventTargetSLGC = [] # SLG phase C

# read the event file
eventList = []
with open(eventKeyFile,'r') as f:
    fileLines = f.read().split('\n')
    for line in fileLines[1:]:
        if line == '':
            continue
        eventList.append(line.strip())

print 'Organizing all the data into arrays for the classifier'
for idx, row in enumerate(readerV):
    eventKey = eventList[idx]
    eventKeyWords = eventKey.split('/')
    PL = eventKeyWords[0][1:].strip()
    faultbus = eventKeyWords[1][1:].strip()
    currentbus = eventKeyWords[2][1:].strip()
    faulttype = eventKeyWords[3].strip()
    phase = eventKeyWords[4].strip()
    faultZ = eventKeyWords[5].strip()
    condensedKey = 'R{}/F{}/B{}/{}/{}'.format(PL,faultbus,currentbus,faulttype,faultZ)


    # pass the input through a 6 cycle average filter
    row = avgFilter(row,numcyc)
    # add some noise to the outputs
    row = np.array(row) + np.random.normal(0,std,len(row)) # normal noise with standard deviation of std



    if phase == 'A':
        if faulttype == 'ABCG':
            for i in range(shiftRange):
                list3PHA.append(row[startind-i:endind-i])
                condensedEventKey3ph.append(condensedKey)
                eventTarget3ph.append(0)





        elif faulttype == 'AG':
            for i in range(shiftRange):
                list1PHA.append(row[startind-i:endind-i])
                condensedEventKey1ph.append(condensedKey)
                eventTargetSLGA.append(1)

        

    elif phase == 'B':
        if faulttype == 'ABCG':
            for i in range(shiftRange):
                list3PHB.append(row[startind-i:endind-i])


        elif faulttype == 'AG':
            for i in range(shiftRange):
                list1PHB.append(row[startind-i:endind-i])


        

    elif phase == 'C':
        if faulttype == 'ABCG':
            for i in range(shiftRange):
                list3PHC.append(row[startind-i:endind-i])


        elif faulttype == 'AG': 
            for i in range(shiftRange):
                list1PHC.append(row[startind-i:endind-i])



# contains all the 3 phase data
array3PHA = np.array(list3PHA)
array3PHB = np.array(list3PHB)
array3PHC = np.array(list3PHC)
# contains all the single phase data
array1PHA = np.array(list1PHA) # the faulted phase
array1PHB = np.array(list1PHB)
array1PHC = np.array(list1PHC)

event3phArray = np.concatenate((array3PHA,array3PHB,array3PHC),axis=1) # stack left to right
eventSLGA = np.concatenate((array1PHA,array1PHB,array1PHC),axis=1) # SLG fault at A
eventSLGB = np.concatenate((array1PHB,array1PHA,array1PHC),axis=1) # SLG fault at B
eventSLGC = np.concatenate((array1PHC,array1PHB,array1PHA),axis=1) # SLG fault at C



# get 1000 samples from each type of fault
Sample3phInd = random.sample(range(event3phArray.shape[0]),1000)
SLGASampleInd = random.sample(range(eventSLGA.shape[0]),1000)
SLGBSampleInd = random.sample(range(eventSLGB.shape[0]),1000)
SLGCSampleInd = random.sample(range(eventSLGC.shape[0]),1000)


event3phArray = event3phArray[Sample3phInd,:]
eventSLGA = eventSLGA[SLGASampleInd,:]
eventSLGB = eventSLGB[SLGBSampleInd,:]
eventSLGC = eventSLGC[SLGCSampleInd,:]




event1phArray = np.concatenate((eventSLGA,eventSLGB, eventSLGC),axis=0) # stack (top to bottom) SLG A, then SLG B, then SLG C

faultarray = np.concatenate((event3phArray, event1phArray),axis=0) # vertical stacks: 3 ph data, 1 ph data 

for target in eventTargetSLGA:

    eventTargetSLGB.append(2)
    eventTargetSLGC.append(3)

#steadyTarget = np.zeros(steadyArray.shape[0])
event3phTargetArray = np.array(eventTarget3ph)
TargetSLGAFault = np.array(eventTargetSLGA)
TargetSLGBFault = np.array(eventTargetSLGB)
TargetSLGCFault = np.array(eventTargetSLGC)


# take the corresponding 500 targets
event3phTargetArray = event3phTargetArray[Sample3phInd]
TargetSLGAFault = TargetSLGAFault[SLGASampleInd]
TargetSLGBFault = TargetSLGBFault[SLGBSampleInd]
TargetSLGCFault = TargetSLGCFault[SLGCSampleInd]

faultTargetArray = np.concatenate((event3phTargetArray,TargetSLGAFault,TargetSLGBFault,TargetSLGCFault)) # stacking 3 times because (AG, BG, CG) all have same target

"""
# arrange the keys properly
allKeys = []
# 3 ph
for key in condensedEventKey3ph:
    allKeys.append(key)

# SLG A
for key in condensedEventKey1ph:
    allKeys.append(key)

# SLG B
for key in condensedEventKey1ph:
    #key = key[:-2] + 'BG'
    key = key.replace('/AG/','/BG/')
    allKeys.append(key)

# SLG C
for key in condensedEventKey1ph:
    #key = key[:-2] + 'CG'
    key = key.replace('/AG/','/CG/')
    allKeys.append(key)
"""

"""
#### plot some samples of each class

ind3ph = random.sample(range(event3phArray.shape[0]),5)
indSLGA = random.sample(range(eventSLGA.shape[0]),5)
indSLGB = random.sample(range(eventSLGB.shape[0]),5)
indSLGC = random.sample(range(eventSLGC.shape[0]),5)


# 3 ph fault
for s in ind3ph:
    plt.plot(event3phArray[s])
plt.grid()
plt.title('Class 0 (three phase fault) Samples')
plt.xlabel('Sample number')
plt.ylabel('V (pu)')
plt.ylim(-0.2,1.2)
plt.show()
plt.close()



# SLG A fault
for s in indSLGA:
    plt.plot(eventSLGA[s])
plt.grid()
plt.title('Class 1 (Single phase fault phase A) Samples')
plt.xlabel('Sample number')
plt.ylabel('V (pu)')
plt.ylim(-0.2,1.2)
plt.show()


# SLG B fault
for s in indSLGB:
    plt.plot(eventSLGB[s])
plt.grid()
plt.title('Class 2 (Single phase fault phase B) Samples')
plt.xlabel('Sample number')
plt.ylabel('V (pu)')
plt.ylim(-0.2,1.2)
plt.show()

# SLG C fault
for s in indSLGC:
    plt.plot(eventSLGC[s])
plt.grid()
plt.title('Class 3 (Single phase fault phase C) Samples')
plt.xlabel('Sample number')
plt.ylabel('V (pu)')
plt.ylim(-0.2,1.2)
plt.show()

#####
"""


# close files
vFile.close()
tFile.close()



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

listGenOut = []
# get the voltage data (and simulate balanced three phase)
for idx, row in enumerate(vReader):
    eventKey = eventList[idx]
    eventKeyWords = eventKey.split('/')
    PL = eventKeyWords[0][1:].strip()
    genbus = eventKeyWords[1][1:].strip()
    currentbus = eventKeyWords[2][1:].strip()
    #eventID = 'R{}/G{}'.format(PL,genbus)

    # pass the input through a 6 cycle average filter
    row = avgFilter(row,numcyc)
    # add some noise to the outputs
    row = np.array(row) + np.random.normal(0,std,len(row)) # normal noise with standard deviation of std

    #allKeys.append(eventKey)
    
    for i in range(shiftRange):
        currentList = []
        for bs in range(3): # simulate three phase
            for v in row[startind-i:endind-i]:
                currentList.append(v)

        listGenOut.append(currentList)


genoutArray = np.array(listGenOut)
genoutTarget = np.array([4]*genoutArray.shape[0])
# close all files
vFilePath.close()
aFilePath.close()
fFilePath.close()
timeDataFilePath.close()
###########


"""
## plot generator outage samples
indGenOut = random.sample(range(genoutArray.shape[0]),5)


# 3 ph fault
for s in indGenOut:
    plt.plot(genoutArray[s])
plt.grid()
plt.title('Class 4 (Generator outage) Samples')
plt.xlabel('Sample number')
plt.ylabel('V (pu)')
plt.ylim(-0.2,1.2)
plt.show()
plt.close()
"""



### getting the motor start data
startTime = 1.0
numSamples = 30
shiftRange = 5
numcyc = 6
std = 0.001
vFileName = 'mStartVpu.csv' # csv file containing voltage data (different types of fault)
tFileName = 'mStartTime.csv' # csv file containing the time data
#eventKeyFile = 'fault3ph/eventIDFileLI.txt'
vFile = open(vFileName,'rb')
tFile = open(tFileName,'rb')
readerV=csv.reader(vFile,quoting=csv.QUOTE_NONNUMERIC) # so that the entries are converted to floats
readerT = csv.reader(tFile,quoting=csv.QUOTE_NONNUMERIC)
#tme = [row for idx, row in enumerate(readerT) if idx==0][0]

tDict = {}
vDict = {}

for idx, row in enumerate(readerT):
    tDict[idx] = row

for idx, row in enumerate(readerV):
    vDict[idx] = row

croppedV = []
for idx in tDict:
    t = tDict[idx]
    startInd =  min([ind for ind, val in enumerate(t) if val >=startTime])
    endInd = startInd + numSamples
    v = vDict[idx]

    # add noise and smoothing
    # pass the input through a 6 cycle average filter
    v = avgFilter(v,numcyc)
    # add some noise to the outputs
    v = np.array(v) + np.random.normal(0,std,len(v)) # normal noise with standard deviation of std


    for ts in range(shiftRange):
        lst = []
        for rep in range(3):
            for val in v[startInd-ts:endInd-ts]:
                lst.append(val)
        croppedV.append(lst)

mStartArray = np.array(croppedV)


mStartTarget = np.array([5]*mStartArray.shape[0])

vFile.close()
tFile.close()

############

"""
## plot motor start samples
indmStart = random.sample(range(mStartArray.shape[0]),5)


for s in indmStart:
    plt.plot(mStartArray[s])
plt.grid()
plt.title('Class 5 (Motor start) Samples')
plt.xlabel('Sample number')
plt.ylabel('V (pu)')
plt.ylim(-0.2,1.2)
plt.show()
plt.close()
###
"""




# concatenate the fault, genout and mStart array
fullArray = np.concatenate((faultarray, genoutArray,mStartArray),axis=0) # vertical stacks: 3 ph data, 1 ph data 
fullTargetArray = np.concatenate((faultTargetArray,genoutTarget,mStartTarget))





##### evaluate SVM classifier
print 'Evaluating SVM'
from sklearn.svm import SVC
x = fullArray
#x = fullArrayFil
y = fullTargetArray.astype(int)

svm_model_linear = SVC(kernel = 'linear', C = 1)
#svm_model_linear = SVC(kernel = 'rbf', C = 1) # is not as good as linear


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
"""
### quick test
svm_model_linear.fit(x_train, y_train)
y_pred = svm_model_linear.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)*100
print ('Accuracy: {}'.format(accuracy))
conf_mat = confusion_matrix(y_test, y_pred)
print (conf_mat)
####
"""


### extensive test
kfold = StratifiedKFold(n_splits=3, shuffle=True) # preserve the distribution of classes in folds
cvscores = []
for train, test in kfold.split(x_train, y_train):
    # this part measures performance using the folds (the test set here is essentially a cross-validation set)
    model = SVC(kernel = 'linear', C = 1)
    #model = SVC(kernel = 'rbf', C = 1)
    model.fit(x_train[train], y_train[train])

  
    
    # evaluate the model
    y_pred = model.predict(x_train[test])
    #y_pred_prob = model.predict_proba(x_train[test])
    
    # evaluate predictions
    accuracy = accuracy_score(y_train[test], y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    #print(classification_report(y_train[test], y_pred))
    cvscores.append(accuracy * 100)

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

# Evaluate on test data
y_test_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_test_pred)*100
print ('Accuracy: {}'.format(accuracy))
conf_mat = confusion_matrix(y_test, y_test_pred)
print (conf_mat)
###


###########
"""
# plot a certain event
plt.plot(x[0]) # a three phase fault
#plt.plot(x[-1]) # a gen outage
plt.show()
"""
