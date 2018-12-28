# Event (fault, gen outage, line outage, tf outage) classification using three phase voltage of one bus as a sample
# added noise and smoothing
# limited number (5) of PMUs available
# added angle data as well for inputs
print 'Importing modules'
import csv
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_predict, cross_validate, cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from avgFilterFn import avgFilter # to emulate filtered data
import matplotlib.pyplot as plt
import random
from getBusDataFn import getBusData



#### 
## Function to get any type of outage data
# generator outages


def  outageFn(dataFileName,tFileName,eventFileName, classInd):
    # use this function to get any sort of data from outage events

    # file objects
    eventList = []
    with open(eventFileName,'r') as f:
        fileLines = f.read().split('\n')
        for line in fileLines[1:]:
            if line == '':
                continue
            eventList.append(line.strip())


    dataFileName = open(dataFileName, 'rb') # 'wb' needed to avoid blank space in between lines
    tFileName = open(tFileName, 'rb')


    vReader = csv.reader(dataFileName,quoting=csv.QUOTE_NONNUMERIC) # 'wb' needed to avoid blank space in between lines

    tReader = csv.reader(tFileName,quoting=csv.QUOTE_NONNUMERIC)


    tme = [row for idx, row in enumerate(tReader) if idx==0][0]

    listOut = []
    # get the voltage data (and simulate balanced three phase)
    for idx, row in enumerate(vReader):
        eventKey = eventList[idx]
        eventKeyWords = eventKey.split('/')
        currentbus = eventKeyWords[2][1:].strip()

        if currentbus in PMUBuses:
            # add some noise to the outputs
            row = np.array(row) + np.random.normal(0,std,len(row)) # normal noise with standard deviation of std
            # pass the input through a 6 cycle average filter
            row = avgFilter(row,numcyc)


            allKeys.append(eventKey)
            
            for i in range(shiftRange):
                currentList = []
                for bs in range(3): # simulate three phase
                    for v in row[startind-i:endind-i]:
                        currentList.append(v)

                listOut.append(currentList)


    eventoutArray = np.array(listOut)
    eventoutTarget = np.array([classInd]*eventoutArray.shape[0])
    # close all files
    dataFileName.close()
    tFileName.close()

    return eventoutArray, eventoutTarget


####











##### get a random set of buses where to place the PMUs
refRaw = 'savnw.raw'
BusDataDict = getBusData(refRaw)
tapbuslist = ['3007','204']
buslist = [bus for bus in BusDataDict.keys() if bus not in tapbuslist]
numBuses = len(buslist)

PMUBuses = random.sample(buslist,5)
print('Selected PMU buses: {}'.format(PMUBuses))
####



print('Reading fault data')
vFileName = 'fault3ph/vData3phLI.csv' # csv file containing voltage data (different types of fault)
aFileName = 'fault3ph/aData3phLI.csv' # csv file containing angle data (different types of fault)
tFileName = 'fault3ph/tData3ph.csv' # csv file containing the time data
eventKeyFile = 'fault3ph/eventIDFileLI.txt'
vFile = open(vFileName,'rb')
aFile = open(aFileName,'rb')
tFile = open(tFileName,'rb')
readerV=csv.reader(vFile,quoting=csv.QUOTE_NONNUMERIC) # so that the entries are converted to floats
readerA = csv.reader(aFile,quoting=csv.QUOTE_NONNUMERIC) # so that the entries are converted to floats
readerT = csv.reader(tFile,quoting=csv.QUOTE_NONNUMERIC)
tme = [row for idx, row in enumerate(readerT) if idx==0][0]



# get the indices for fault on and fault off
faultontime = 0.1
#timesteps = 40
#timesteps = 10
#timesteps = 20
timesteps = 40
numcyc = 6
shiftRange = 5
std = 0.001 # standard deviation of noise
startind = min([idx for idx,val in enumerate(tme) if val >= faultontime])
endind = startind + timesteps

# lists to contain the volt data
list3PHAV = []
list3PHBV = []
list3PHCV = []


list1PHAV = []
list1PHBV = []
list1PHCV = []


# lists to contain the angle data
list3PHAA = []
list3PHBA = []
list3PHCA = []


list1PHAA = []
list1PHBA = []
list1PHCA = []



condensedEventKey3ph = []
condensedEventKey1ph = []
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

#print 'Organizing all the data into arrays for the classifier'


# get the voltage data
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

    if currentbus in PMUBuses:


        # add some noise to the outputs
        row = np.array(row) + np.random.normal(0,std,len(row)) # normal noise with standard deviation of std
        # pass the input through a 6 cycle average filter
        row = avgFilter(row,numcyc)


        if phase == 'A':
            if faulttype == 'ABCG':
                for i in range(shiftRange):
                    list3PHAV.append(row[startind-i:endind-i])
                    condensedEventKey3ph.append(condensedKey)
                    eventTarget3ph.append(0)





            elif faulttype == 'AG':
                for i in range(shiftRange):
                    list1PHAV.append(row[startind-i:endind-i])
                    condensedEventKey1ph.append(condensedKey)
                    eventTargetSLGA.append(1)

            

        elif phase == 'B':
            if faulttype == 'ABCG':
                for i in range(shiftRange):
                    list3PHBV.append(row[startind-i:endind-i])


            elif faulttype == 'AG':
                for i in range(shiftRange):
                    list1PHBV.append(row[startind-i:endind-i])


            

        elif phase == 'C':
            if faulttype == 'ABCG':
                for i in range(shiftRange):
                    list3PHCV.append(row[startind-i:endind-i])


            elif faulttype == 'AG': 
                for i in range(shiftRange):
                    list1PHCV.append(row[startind-i:endind-i])



# contains all the 3 phase data
array3PHAV = np.array(list3PHAV)
array3PHBV = np.array(list3PHBV)
array3PHCV = np.array(list3PHCV)
# contains all the single phase data
array1PHAV = np.array(list1PHAV) # the faulted phase
array1PHBV = np.array(list1PHBV)
array1PHCV = np.array(list1PHCV)



# get the angle data

for idx, row in enumerate(readerA):
    eventKey = eventList[idx]
    eventKeyWords = eventKey.split('/')
    PL = eventKeyWords[0][1:].strip()
    faultbus = eventKeyWords[1][1:].strip()
    currentbus = eventKeyWords[2][1:].strip()
    faulttype = eventKeyWords[3].strip()
    phase = eventKeyWords[4].strip()
    faultZ = eventKeyWords[5].strip()
    #condensedKey = 'R{}/F{}/B{}/{}/{}'.format(PL,faultbus,currentbus,faulttype,faultZ)

    if currentbus in PMUBuses:


        # add some noise to the outputs
        row = np.array(row) + np.random.normal(0,std,len(row)) # normal noise with standard deviation of std
        # pass the input through a 6 cycle average filter
        row = avgFilter(row,numcyc)


        if phase == 'A':
            if faulttype == 'ABCG':
                for i in range(shiftRange):
                    list3PHAA.append(row[startind-i:endind-i])
                    #condensedEventKey3ph.append(condensedKey)
                    #eventTarget3ph.append(0)





            elif faulttype == 'AG':
                for i in range(shiftRange):
                    list1PHAA.append(row[startind-i:endind-i])
                    #condensedEventKey1ph.append(condensedKey)
                    #eventTargetSLGA.append(1)

            

        elif phase == 'B':
            if faulttype == 'ABCG':
                for i in range(shiftRange):
                    list3PHBA.append(row[startind-i:endind-i])


            elif faulttype == 'AG':
                for i in range(shiftRange):
                    list1PHBA.append(row[startind-i:endind-i])


            

        elif phase == 'C':
            if faulttype == 'ABCG':
                for i in range(shiftRange):
                    list3PHCA.append(row[startind-i:endind-i])


            elif faulttype == 'AG': 
                for i in range(shiftRange):
                    list1PHCA.append(row[startind-i:endind-i])
#####


# contains all the 3 phase voltage  data
array3PHAV = np.array(list3PHAV)
array3PHBV = np.array(list3PHBV)
array3PHCV = np.array(list3PHCV)
# contains all the single phase data
array1PHAV = np.array(list1PHAV) # the faulted phase
array1PHBV = np.array(list1PHBV)
array1PHCV = np.array(list1PHCV)

# contains all the 3 phase angle  data
array3PHAA = np.array(list3PHAA)
array3PHBA = np.array(list3PHBA)
array3PHCA = np.array(list3PHCA)
# contains all the single phase data
array1PHAA = np.array(list1PHAA) # the faulted phase
array1PHBA = np.array(list1PHBA)
array1PHCA = np.array(list1PHCA)

# stack the voltage and angle data left to right
array3PHA = np.concatenate((array3PHAV,array3PHAA),axis=1) # stack left to right
array3PHB = np.concatenate((array3PHBV,array3PHBA),axis=1)
array3PHC = np.concatenate((array3PHCV,array3PHCA),axis=1)

array1PHA = np.concatenate((array1PHAV,array1PHAA),axis=1) # stack left to right
array1PHB = np.concatenate((array1PHBV,array1PHBA),axis=1)
array1PHC = np.concatenate((array1PHCV,array1PHCA),axis=1)




# now form the full input arrays and targets
event3phArray = np.concatenate((array3PHA,array3PHB,array3PHC),axis=1) # stack left to right
eventSLGA = np.concatenate((array1PHA,array1PHB,array1PHC),axis=1) # SLG fault at A
eventSLGB = np.concatenate((array1PHB,array1PHA,array1PHC),axis=1) # SLG fault at B
eventSLGC = np.concatenate((array1PHC,array1PHB,array1PHA),axis=1) # SLG fault at C

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

faultTargetArray = np.concatenate((event3phTargetArray,TargetSLGAFault,TargetSLGBFault,TargetSLGCFault)) # stacking 3 times because (AG, BG, CG) all have same target


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

# close files
vFile.close()
tFile.close()



##############
# generator outages
print('Reading gen out data...')
genOutDir = 'GenOut'
vFilePath = '{}/vGenOut.csv'.format(genOutDir)
aFilePath = '{}/aGenOut.csv'.format(genOutDir)
eventFilePath = '{}/eventGenOut.txt'.format(genOutDir)
timeDataFilePath = '{}/tGenOut.csv'.format(genOutDir)
genoutArrayV, genoutTarget =  outageFn(vFilePath,timeDataFilePath,eventFilePath, 4)
genoutArrayA, _ =  outageFn(aFilePath,timeDataFilePath,eventFilePath, 4)

genoutArray = np.concatenate((genoutArrayV,genoutArrayA),axis = 1)

###########


##############
# line  outages
print('Reading line out data...')
# get the generator outage data
lineOutDir = 'LineOut'

vFilePath = '{}/vLineOut.csv'.format(lineOutDir)
aFilePath = '{}/aLineOut.csv'.format(lineOutDir)
eventFilePath = '{}/eventLineOut.txt'.format(lineOutDir)
timeDataFilePath = '{}/tLineOut.csv'.format(lineOutDir)
lineOutArrayV, lineOutTarget =  outageFn(vFilePath,timeDataFilePath,eventFilePath, 5)
lineOutArrayA, _ =  outageFn(aFilePath,timeDataFilePath,eventFilePath, 5)
lineOutArray = np.concatenate((lineOutArrayV,lineOutArrayA),axis = 1)
###########



##############
# transformer  outages
print('Reading tf out data...')
# get the generator outage data
TFOutDir = 'TFOut'

vFilePath = '{}/vTFOut.csv'.format(TFOutDir)
aFilePath = '{}/aTFOut.csv'.format(TFOutDir)
eventFilePath = '{}/eventTFOut.txt'.format(TFOutDir)
timeDataFilePath = '{}/tTFOut.csv'.format(TFOutDir)
TFOutArrayV, TFOutTarget =  outageFn(vFilePath,timeDataFilePath,eventFilePath, 6)
TFOutArrayA, _ =  outageFn(aFilePath,timeDataFilePath,eventFilePath, 6)
TFOutArray = np.concatenate((TFOutArrayV,TFOutArrayA),axis = 1)

###########




# concatenate the fault and genout array
fullArray = np.concatenate((faultarray, genoutArray,lineOutArray,TFOutArray),axis=0) # vertical stacks: 3 ph data, 1 ph data 
fullTargetArray = np.concatenate((faultTargetArray,genoutTarget,lineOutTarget,TFOutTarget))



plt.plot(fullArray[-1][:120])
plt.grid()
plt.show()


# ##### evaluate SVM classifier
# print 'Evaluating SVM'
# from sklearn.svm import SVC
# x = fullArray
# print('Shape of input array: {}'.format(x.shape))
# #x = fullArrayFil
# y = fullTargetArray.astype(int)

# svm_model_linear = SVC(kernel = 'linear', C = 1)
# #svm_model_linear = SVC(kernel = 'rbf', C = 1) # is not as good as linear


# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

# # ### quick test
# # svm_model_linear.fit(x_train, y_train)
# # y_pred = svm_model_linear.predict(x_test)
# # accuracy = accuracy_score(y_test, y_pred)*100
# # print ('Accuracy: {}'.format(accuracy))
# # conf_mat = confusion_matrix(y_test, y_pred)
# # print (conf_mat)
# # ####



# ### extensive test
# kfold = StratifiedKFold(n_splits=3, shuffle=True) # preserve the distribution of classes in folds
# cvscores = []
# for train, test in kfold.split(x_train, y_train):
#     # this part measures performance using the folds (the test set here is essentially a cross-validation set)
#     model = SVC(kernel = 'linear', C = 1)
#     model.fit(x_train[train], y_train[train])

  
    
#     # evaluate the model
#     y_pred = model.predict(x_train[test])
#     #y_pred_prob = model.predict_proba(x_train[test])
    
#     # evaluate predictions
#     accuracy = accuracy_score(y_train[test], y_pred)
#     print("Accuracy: %.2f%%" % (accuracy * 100.0))
#     #print(classification_report(y_train[test], y_pred))
#     cvscores.append(accuracy * 100)

# print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

# # Evaluate on test data
# y_test_pred = model.predict(x_test)
# accuracy = accuracy_score(y_test, y_test_pred)*100
# print ('Accuracy: {}'.format(accuracy))
# conf_mat = confusion_matrix(y_test, y_test_pred)
# print (conf_mat)
# ###


###########

