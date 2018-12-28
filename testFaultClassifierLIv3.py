# script to prepare the fault data and test using various classifiers
# here the main improvement from v2 is that each type of SLG fault (phase A, B and C) have their own separate class
print 'Importing modules'
import csv
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_predict, cross_validate, cross_val_score, KFold
from sklearn.metrics import accuracy_score, confusion_matrix
from avgFilterFn import avgFilter # to emulate filtered data
import matplotlib.pyplot as plt
import matplotlib.patches as patches


print 'Reading the csv files'
vFileName = 'fault3ph/vData3phLI.csv' # csv file containing voltage data (different types of fault)
tFileName = 'fault3ph/tData3ph.csv' # csv file containing the time data
eventKeyFile = 'fault3ph/eventIDFileLI.txt'
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
faultofftime = 0.2
faultonind = min([idx for idx,val in enumerate(tme) if val >= faultontime])
faultoffind = min([idx for idx,val in enumerate(tme) if val >= faultofftime])


list3PHA = []
list3PHB = []
list3PHC = []
list3PHAFil = []
list3PHBFil = []
list3PHCFil = []

list1PHA = []
list1PHB = []
list1PHC = []
list1PHAFil = []
list1PHBFil = []
list1PHCFil = []

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
    rowFil = avgFilter(row,6)
    if phase == 'A':
        if faulttype == 'ABCG':
            list3PHA.append(row[faultonind:faultoffind])
            list3PHAFil.append(rowFil[faultonind:faultoffind]) # 6 cycle filter
            #listSteadyA.append(row[:faultonind-1]) # isolate the steady state part
            condensedEventKey3ph.append(condensedKey)
            steadyKey = condensedKey + '/steady'
            #condensedEventKeySteady.append(steadyKey)
            if currentbus == faultbus: # 3 phase fault at current bus
                eventTarget3ph.append(0)
            else: # three phase fault at different bus
                eventTarget3ph.append(4)




        elif faulttype == 'AG':
            list1PHA.append(row[faultonind:faultoffind])
            list1PHAFil.append(rowFil[faultonind:faultoffind]) # 6 cycle filter
            #listSteadyA.append(row[:faultonind-1]) # isolate the steady state part
            condensedEventKey1ph.append(condensedKey)
            #condensedEventKeySteady.append(steadyKey)           

            if currentbus == faultbus: # SLG fault phase A at current bus
                eventTargetSLGA.append(1)
            else: # three phase fault at different bus
                eventTargetSLGA.append(4)

        

    elif phase == 'B':
        if faulttype == 'ABCG':
            list3PHB.append(row[faultonind:faultoffind])
            list3PHBFil.append(rowFil[faultonind:faultoffind]) # 6 cycle filter
            #listSteadyB.append(row[:faultonind-1]) # isolate the steady state part
        elif faulttype == 'AG':
            list1PHB.append(row[faultonind:faultoffind])
            list1PHBFil.append(rowFil[faultonind:faultoffind]) # 6 cycle filter
            #listSteadyB.append(row[:faultonind-1])  # isolate the steady state part
        

    elif phase == 'C':
        if faulttype == 'ABCG':
            list3PHC.append(row[faultonind:faultoffind])
            list3PHCFil.append(rowFil[faultonind:faultoffind]) # 6 cycle filter
            #listSteadyC.append(row[:faultonind-1]) # isolate the steady state part
        elif faulttype == 'AG': 
            list1PHC.append(row[faultonind:faultoffind])
            list1PHCFil.append(rowFil[faultonind:faultoffind]) # 6 cycle filter
            #listSteadyC.append(row[:faultonind-1]) # isolate the steady state part
        



# contains all the 3 phase data
array3PHA = np.array(list3PHA)
array3PHB = np.array(list3PHB)
array3PHC = np.array(list3PHC)
# contains all the single phase data
array1PHA = np.array(list1PHA) # the faulted phase
array1PHB = np.array(list1PHB)
array1PHC = np.array(list1PHC)

"""
# contains all the steady state data
arraySteadyA = np.array(listSteadyA)
arraySteadyB = np.array(listSteadyB)
arraySteadyC = np.array(listSteadyC)
"""
event3phArray = np.concatenate((array3PHA,array3PHB,array3PHC),axis=1) # stack left to right
eventSLGA = np.concatenate((array1PHA,array1PHB,array1PHC),axis=1) # SLG fault at A
eventSLGB = np.concatenate((array1PHB,array1PHA,array1PHC),axis=1) # SLG fault at B
eventSLGC = np.concatenate((array1PHC,array1PHB,array1PHA),axis=1) # SLG fault at C

event1phArray = np.concatenate((eventSLGA,eventSLGB, eventSLGC),axis=0) # stack (top to bottom) SLG A, then SLG B, then SLG C

#steadyArray = np.concatenate((arraySteadyA,arraySteadyB,arraySteadyC),axis=1) # stack left to right
#steadyArray = steadyArray[:,:event3phArray.shape[1]] # crop columns to match that of eventArray


#fullArray = np.concatenate((steadyArray,event3phArray, event1phArray),axis=0) # vertical stacks: steady state data, 3 ph data, 1 ph data 
fullArray = np.concatenate((event3phArray, event1phArray),axis=0) # vertical stacks: steady state data, 3 ph data, 1 ph data 

# arrange the targets properly


for target in eventTargetSLGA:
    if target == 1:
        eventTargetSLGB.append(2)
        eventTargetSLGC.append(3)
    else:
        eventTargetSLGB.append(target)
        eventTargetSLGC.append(target)

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









"""
###### Generate dataset which has 6 cycle filtering
# contains all the 3 phase data filtered
array3PHAFil = np.array(list3PHAFil)
array3PHBFil = np.array(list3PHBFil)
array3PHCFil = np.array(list3PHCFil)
# contains all the single phase data filtered
array1PHAFil = np.array(list1PHAFil) # the faulted phase
array1PHBFil = np.array(list1PHBFil)
array1PHCFil = np.array(list1PHCFil)






event3phArrayFil = np.concatenate((array3PHAFil,array3PHBFil,array3PHCFil),axis=1) # stack left to right
eventSLGA = np.concatenate((array1PHAFil,array1PHBFil,array1PHCFil),axis=1) # SLG fault at A
eventSLGB = np.concatenate((array1PHBFil,array1PHAFil,array1PHCFil),axis=1) # SLG fault at B
eventSLGC = np.concatenate((array1PHCFil,array1PHBFil,array1PHAFil),axis=1) # SLG fault at C

event1phArrayFil = np.concatenate((eventSLGA,eventSLGB, eventSLGC),axis=0) # stack (top to bottom) SLG A, then SLG B, then SLG C




fullArrayFil = np.concatenate((steadyArray,event3phArrayFil, event1phArrayFil),axis=0) # vertical stacks: steady state data, 3 ph data, 1 ph data 

###############
"""






print 'Evaluating the classifier'
##### evaluate SVM classifier
from sklearn.svm import SVC
x = fullArray
#x = fullArrayFil
y = fullTargetArray.astype(int)

#svm_model_linear = SVC(kernel = 'linear', C = 1)
svm_model_linear = SVC(kernel = 'rbf', C = 1) # is not as good as linear

cv = KFold( n_splits=10)
y_pred  =  cross_val_predict(svm_model_linear, x, y, cv=cv)
accuracy = accuracy_score(y,y_pred)*100

print 'Accuracy: {}'.format(accuracy)
conf_mat = confusion_matrix(y, y_pred)
print accuracy
print conf_mat

scores = cross_val_score(svm_model_linear, x, y, cv=cv)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#results = cross_validate(svm_model_linear, x, y, cv=)
#print results['test_score'] 

"""
# get the indices where y_pred != y
wrongInd = []
for i in range(y.shape[0]):
    if y_pred[i] != y[i]:
        wrongInd.append(i)


for i in wrongInd:
    print '{},{},{}'.format(allKeys[i],y[i],y_pred[i])
"""
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

streamSize = faultoffind - faultonind
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
    startInd = faultonind-i
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
    startInd = faultonind+i
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