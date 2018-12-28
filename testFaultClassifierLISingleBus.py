# classify faults only with variable impedances and also scaled loads
# invidual bus fault classifier
# simulations contain seq data
print 'Importing modules'
import csv
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_predict, cross_validate, cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from avgFilterFn import avgFilter # to emulate filtered data
import matplotlib.pyplot as plt
import matplotlib.patches as patches


print 'Reading the csv files'

##files containg seq data
vFileName = 'fault3ph/vData3phLISeq.csv' # csv file containing voltage data (different types of fault)
tFileName = 'fault3ph/tData3phLISeq.csv' # csv file containing the time data
eventKeyFile = 'fault3ph/eventIDFileLISeq.txt'
##

"""
## files not containing seq data
vFileName = 'fault3ph/vData3phLI.csv' # csv file containing voltage data (different types of fault)
tFileName = 'fault3ph/tData3ph.csv' # csv file containing the time data
eventKeyFile = 'fault3ph/eventIDFileLI.txt'
"""
##
vFile = open(vFileName,'rb')
tFile = open(tFileName,'rb')
readerV=csv.reader(vFile,quoting=csv.QUOTE_NONNUMERIC) # so that the entries are converted to floats
readerT = csv.reader(tFile,quoting=csv.QUOTE_NONNUMERIC)
tme = [row for idx, row in enumerate(readerT) if idx==0][0]




def outFilteredData(dataList,cyc):

    filDataList = []
    for data in dataList:
        filData = avgFilter(data,cyc)
        filDataList.append(filData)
    return filDataList




# read the event file
eventList = []
with open(eventKeyFile,'r') as f:
    fileLines = f.read().split('\n')
    for line in fileLines[1:]:
        if line == '':
            continue
        eventList.append(line.strip())

# get the indices for fault on and fault off
faultontime = 0.1
timsteps = 20
std = 0.001
startind = min([idx for idx,val in enumerate(tme) if val >= faultontime])
endind = startind + timsteps


list3PHA = []
list3PHB = []
list3PHC = []


list1PHA = []
list1PHB = []
list1PHC = []


condensedEventKey3ph = []
condensedEventKey1ph = []

#condensedset = set()
eventTarget3ph = [] # targets for all the 3 ph fault data
eventTargetSLGA = [] # targets for all the single phase fault data at phase A
eventTargetSLGB = [] # SLG phase B
eventTargetSLGC = [] # SLG phase C
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
    # implement 6 cycle filter and smoothing
    row = avgFilter(row,6)
    row = np.array(row) + np.random.normal(0,std,len(row)) # normal noise with standard deviation of std
    if phase == 'A':
        if faulttype == 'ABCG':
            list3PHA.append(row[startind:endind])

            condensedEventKey3ph.append(condensedKey)
            eventTarget3ph.append(0) # three phase fault






        elif faulttype == 'AG':
            list1PHA.append(row[startind:endind])
            condensedEventKey1ph.append(condensedKey)
            eventTargetSLGA.append(1)   # SLG A fault 






        

    elif phase == 'B':
        if faulttype == 'ABCG':
            list3PHB.append(row[startind:endind])

        elif faulttype == 'AG':
            list1PHB.append(row[startind:endind])
        

    elif phase == 'C':
        if faulttype == 'ABCG':
            list3PHC.append(row[startind:endind])

        elif faulttype == 'AG': 
            list1PHC.append(row[startind:endind])
        



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

event1phArray = np.concatenate((eventSLGA,eventSLGB, eventSLGC),axis=0) # stack (top to bottom) SLG A, then SLG B, then SLG C





fullArray = np.concatenate((event3phArray, event1phArray),axis=0) # vertical stacks: steady state data, 3 ph data, 1 ph data 

# arrange the targets properly


for target in eventTargetSLGA:
    if target == 1:
        eventTargetSLGB.append(2) # SLG B 
        eventTargetSLGC.append(3) # SLG C


#steadyTarget = np.zeros(steadyArray.shape[0])
event3phTargetArray = np.array(eventTarget3ph)
TargetSLGAFault = np.array(eventTargetSLGA)
TargetSLGBFault = np.array(eventTargetSLGB)
TargetSLGCFault = np.array(eventTargetSLGC)
#fullTargetArray = np.concatenate((steadyTarget,event3phTargetArray,TargetSLGAFault,TargetSLGBFault,TargetSLGCFault)) # stacking 3 times because (AG, BG, CG) all have same target
fullTargetArray = np.concatenate((event3phTargetArray,TargetSLGAFault,TargetSLGBFault,TargetSLGCFault)) # stacking 3 times because (AG, BG, CG) all have same target
# arrange the keys properly
allKeys = []

# append all steady state keys
#for key in condensedEventKeySteady:
#    allKeys.append(key)

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

"""
# output all the condensed event keys
with open('tmp.txt','w') as f:
    for key in allKeys:
        f.write(key)
        f.write('\n')
"""


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
#############



"""
#############
# test a data stream 
from sklearn.svm import SVC
#x = fullArray
x = fullArrayFil
y = fullTargetArray.astype(int)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)
svm_model_linear = SVC(kernel = 'linear', C = 1)
svm_model_linear.fit(x_train, y_train)

# get an event
vFile = open(vFileName,'rb')
tFile = open(tFileName,'rb')
readerV=csv.reader(vFile,quoting=csv.QUOTE_NONNUMERIC) # so that the entries are converted to floats
readerT = csv.reader(tFile,quoting=csv.QUOTE_NONNUMERIC)
tme = [row for idx, row in enumerate(readerT) if idx==0][0]
interestingInd = []

streamSize = endind - startind
# visualize 3 ph at the fault bus
#event = 'R100/F151/B151/ABCG' # three phase fault here
#event = 'R106/F151/B211/ABCG' # three phase fault not here
#event = 'R100/F205/B205/AG' # SLG A Fault here
#event = 'R100/F205/B205/BG' # SLG B Fault here
#event = 'R100/F101/B101/CG' # SLG C Fault here
event = 'R100/F102/B101/ABCG' # Three phase fault somewhere else
event = 'R100/F101/B101/ABCG' # Three phase fault here
#event = 'R105/F205/B3011/AG' # SLG fault A somewhere else
#event = 'R106/F3018/B101/AG' # SLG fault at A somewhere else, but no voltage dip causes misclassification
#event = 'R104/F152/B153/AG' # SLG fault at A somewhere else, but high voltage dip causes misclassification
eventWords = event.split('/')
faultType = eventWords[-1].strip()
eventOrg = event


if faultType == 'BG' or faultType == 'CG':
    faultType = 'AG'
    eventWords[-1] = faultType
    event = '/'.join(eventWords)

eventKeyA = '{}/A'.format(event)
eventIndA = eventList.index(eventKeyA)

eventKeyB = '{}/B'.format(event)
eventIndB = eventList.index(eventKeyB)


eventKeyC = '{}/C'.format(event)
eventIndC = eventList.index(eventKeyC)

interestingInd.append(eventIndA)
interestingInd.append(eventIndB)
interestingInd.append(eventIndC)
#print eventInd
interestingrows = [row for idx, row in enumerate(readerV) if idx in interestingInd]
#vA = interestingrows[0]
#vB = interestingrows[1]
#vC = interestingrows[2]

vA = avgFilter(interestingrows[0],6)
vB = avgFilter(interestingrows[1],6)
vC = avgFilter(interestingrows[2],6)


# close files
vFile.close()
tFile.close()



##### test the whole data stream
outList = []
for strt in range(0,len(vA),streamSize):

    currVA = vA[strt:strt+streamSize]
    currVB = vB[strt:strt+streamSize]
    currVC = vC[strt:strt+streamSize]

    if len(currVA) < streamSize: # need to pad data with last recorded value
        #print currVA
        currVA = vA[strt:]
        currVB = vB[strt:]
        currVC = vC[strt:] 
        lenDiff = streamSize - len(currVA)
        for a in range(lenDiff):
            currVA.append(currVA[-1]) 
            currVB.append(currVB[-1])
            currVC.append(currVC[-1]) 

    currVA = np.array(currVA)
    currVB = np.array(currVB)
    currVC = np.array(currVC)
    if faultType == 'BG':
       inpArray =   np.concatenate((currVB,currVA,currVC)) 
    elif faultType == 'CG':
        inpArray =   np.concatenate((currVC,currVB,currVA)) 
    else:
        inpArray =   np.concatenate((currVA,currVB,currVC))
    inpArray = np.array(inpArray).reshape(1,-1)

    yTPred = svm_model_linear.predict(inpArray)
    for i in range(streamSize):
        outList.append(yTPred[0])

#print outList





f, (ax1, ax2) = plt.subplots(2, 1)
if faultType == 'BG':
    ax1.plot(tme, vA ,label = 'B')
    ax1.plot(tme, vB ,label = 'A')
    ax1.plot(tme, vC ,label = 'C')
elif faultType == 'CG':
    ax1.plot(tme, vA ,label = 'C')
    ax1.plot(tme, vB ,label = 'B')
    ax1.plot(tme, vC ,label = 'A')
else:
    ax1.plot(tme, vA ,label = 'A')
    ax1.plot(tme, vB ,label = 'B')
    ax1.plot(tme, vC ,label = 'C')

#ax1.set_title('Sharing Y axis')
ax2.plot(tme, outList[:len(tme)])
ax1.grid(True)
ax2.grid(True)

ax1.set_xlabel('Time (s)')
ax2.set_xlabel('Time (s)')
ax1.set_ylabel('Voltage (pu)')
ax2.set_ylabel('Class')
ax1.set_ylim(-0.5,1.5)
#ax1.set_xlim(0,0.4)
ax2.set_ylim(-1,6)
#ax2.set_xlim(0,0.4)
evenTitle = '{} Voltage'.format(eventOrg)
#evenTitle = 'SLG Phase A Fault at Bus'
ax1.set_title(evenTitle)
ax2.set_title('Classifier')
ax1.legend()
ax2.text(0.35,3,'0: Nothing')
ax2.text(0.35,2.5,'1: 3ph fault here')
ax2.text(0.35,2.0,'2: SLG A')
ax2.text(0.35,1.5,'3: SLG B ')
ax2.text(0.35,1.0,'4: SLG C ')
ax2.text(0.35,0.5,'5: Something wrong ')

plt.show()
#####
"""



"""
########### tests with shifted inputs
# test shifted fault input
# include steady state samples in the beginning
print 'Results if steady state samples are included in the beginning'
for i in range(10):
    startInd = startind-i
    endInd = startInd + 13
    currVA = vA[startInd:endInd]
    currVB = vB[startInd:endInd]
    currVC = vC[startInd:endInd]

    inpArray =   np.concatenate((currVA,currVB,currVC))
    inpArray = np.array(inpArray).reshape(1,-1)
    yTPred = svm_model_linear.predict(inpArray)
    print 'Lag {}: Class {}'.format(i,yTPred)

# include fault cleared samples at the end
print 'Results if fault cleared samples are included at the end'
for i in range(10):
    startInd = startind+i
    endInd = startInd + 13
    currVA = vA[startInd:endInd]
    currVB = vB[startInd:endInd]
    currVC = vC[startInd:endInd]

    inpArray =   np.concatenate((currVA,currVB,currVC))
    inpArray = np.array(inpArray).reshape(1,-1)
    yTPred = svm_model_linear.predict(inpArray)
    print 'Lead {}: Class {}'.format(i,yTPred)
####################
"""