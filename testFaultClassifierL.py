# script to prepare the fault data and test using various classifiers
import csv
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_predict, cross_validate, cross_val_score, KFold
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.patches as patches

vFileName = 'fault3ph/vData3phL.csv' # csv file containing voltage data (different types of fault)
tFileName = 'fault3ph/tData3ph.csv' # csv file containing the time data
eventKeyFile = 'fault3ph/eventIDFileL.txt'
vFile = open(vFileName,'rb')
tFile = open(tFileName,'rb')
readerV=csv.reader(vFile,quoting=csv.QUOTE_NONNUMERIC) # so that the entries are converted to floats
readerT = csv.reader(tFile,quoting=csv.QUOTE_NONNUMERIC)
tme = [row for idx, row in enumerate(readerT) if idx==0][0]


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


listPHA = []
listPHB = []
listPHC = []

listSteadyA = []
listSteadyB = []
listSteadyC = []

condensedEventKey = []
condensedset = set()
eventTarget = []
for idx, row in enumerate(readerV):
    eventKey = eventList[idx]
    eventKeyWords = eventKey.split('/')
    PL = eventKeyWords[0][1:].strip()
    faultbus = eventKeyWords[1][1:].strip()
    currentbus = eventKeyWords[2][1:].strip()
    faulttype = eventKeyWords[3].strip()
    phase = eventKeyWords[-1].strip()

    if phase == 'A':
        listPHA.append(row[faultonind:faultoffind])
        listSteadyA.append(row[:faultonind])


        

    elif phase == 'B':
        listPHB.append(row[faultonind:faultoffind])
        listSteadyB.append(row[:faultonind])

        

    elif phase == 'C':
        listPHC.append(row[faultonind:faultoffind])
        listSteadyC.append(row[:faultonind])

        



    condensedKey = 'R{}/F{}/B{}/{}'.format(PL,faultbus,currentbus,faulttype)
    if condensedKey not in condensedset:
        condensedEventKey.append(condensedKey)
        condensedset.add(condensedKey)


        

        if faulttype == 'ABCG' and currentbus == faultbus: # 3 phase fault at current bus
            eventTarget.append(1)
        elif faulttype == 'AG' and currentbus == faultbus: # single phase fault at phase A, B or C
            eventTarget.append(2)
        else: # fault at some other bus
            eventTarget.append(3)



arrayPHA = np.array(listPHA)
arrayPHB = np.array(listPHB)
arrayPHC = np.array(listPHC)


arraySteadyA = np.array(listSteadyA)
arraySteadyB = np.array(listSteadyB)
arraySteadyC = np.array(listSteadyC)

eventArray = np.concatenate((arrayPHA,arrayPHB,arrayPHC),axis=1)
steadyArray = np.concatenate((arraySteadyA,arraySteadyB,arraySteadyC),axis=1)
steadyArray = steadyArray[:,:eventArray.shape[1]] # crop columns to match that of eventArray
fullArray = np.concatenate((steadyArray,eventArray),axis=0) # append all the steady state cases at the top


steadyTarget = np.zeros(steadyArray.shape[0])
eventTargetArray = np.array(eventTarget)
fullTargetArray = np.concatenate((steadyTarget,eventTargetArray))

#np.savetxt(vName, fullArray, delimiter=",")

# get a list of all event keys


allKeys = []
# generate event keys for the steady part
for key in condensedEventKey:

    newKey = key + '/' + 'steady'
    allKeys.append(newKey)

#allKeys = ['steady']*steadyArray.shape[0]
for key in condensedEventKey:
    allKeys.append(key)

# close files
vFile.close()
tFile.close()


# evaluate SVM classifier
from sklearn.svm import SVC
x = fullArray
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

#results = cross_validate(svm_model_linear, x, y, cv=)
#print results['test_score'] 


# get the indices where y_pred != y
wrongInd = []
for i in range(y.shape[0]):
    if y_pred[i] != y[i]:
        wrongInd.append(i)


for i in wrongInd:
    print '{},{},{}'.format(allKeys[i],y[i],y_pred[i])


"""
# test a data stream 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)
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
#event = 'R100/F151/B151/ABCG' # class 1
event = 'R106/F151/B211/ABCG' # class 3
event = 'R105/F205/B205/AG' # class 2 (shows error in the beginning due to low voltage)
event = 'R100/F205/B205/AG'
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
vA = interestingrows[0]
vB = interestingrows[1]
vC = interestingrows[2]



# close files
vFile.close()
tFile.close()

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
    inpArray =   np.concatenate((currVA,currVB,currVC))
    inpArray = np.array(inpArray).reshape(1,-1)
    yTPred = svm_model_linear.predict(inpArray)
    for i in range(streamSize):
        outList.append(yTPred[0])

#print outList



# visualize
# build a rectangle in axes coords
left, width = 0.35, .05
bottom, height = 1.25, 1.5
right = left + width
top = bottom + height



f, (ax1, ax2) = plt.subplots(2, 1)
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
ax1.set_ylim(0,1.5)
ax2.set_ylim(-1,4)
evenTitle = '{} Voltage'.format(event)
ax1.set_title(evenTitle)
ax2.set_title('Classifier')
ax1.legend()
ax2.text(0.35,3,'0: Nothing')
ax2.text(0.35,2.5,'1: 3ph fault here')
ax2.text(0.35,2.0,'2: SLG fault here')
ax2.text(0.35,1.5,'3: fault not here ')
"""


"""
# axes coordinates are 0,0 is bottom left and 1,1 is upper right
p = patches.Rectangle(
    (left, bottom), width, height,
    fill=False, transform=ax2.transAxes, clip_on=False
    )

ax2.add_patch(p)
"""

"""
plt.show()
"""







    

