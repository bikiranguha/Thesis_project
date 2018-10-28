#Load each event pickle object separately and then make lists out of the voltage data, and corresponding event keys are saved in keyxxx.pkl files
# It also saves the time list into a separate pkl file
import dill as pickle
import h5py
import numpy as np 
import time
import os

def load_obj(name ):
    # load pickle object
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def save_obj(obj, name ):
    # save as pickle object
    currentdir = os.getcwd()
    objDir = currentdir + '/obj'
    if not os.path.isdir(objDir):
        os.mkdir(objDir)
    with open(objDir+ '/' +  name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL,recurse = 'True')
"""
########
# example: https://stackoverflow.com/questions/20928136/input-and-output-numpy-arrays-to-h5py/20938742#20938742
# save numpy array
a = np.random.random(size=(100,20))
h5f = h5py.File('data.h5', 'w')
h5f.create_dataset('dataset_1', data=a)

# load numpy array
h5f = h5py.File('data.h5','r')
b = h5f['dataset_1'][:]
################
"""


# get the time
EventDict = load_obj('Event0')
event0 = EventDict.keys()[0]
ResultsDict = EventDict[event0]
tme = ResultsDict['time']
save_obj(tme,'time')


"""
###################
##### get all the voltage data by getting each chunk separately
eventStart = time.time()
EventDict = load_obj('Event0')
print 'Done loading event dictionary'
eventEnd = time.time()
eventTime = eventEnd -eventStart
print 'Took {} sec to load event dictionary'.format(eventTime)



loopStart = time.time()
vList = []
tme = []
keyList = []
numEvents = len(EventDict)
eventNo = 1
for event in EventDict:
    ResultsDict = EventDict[event]
    print 'Scanning event {} out of {}'.format(eventNo, numEvents)
    for ele in ResultsDict:
        if ele == 'time':
            #tme = ResultsDict[ele]
            continue
        key = event + '/' + ele
        v = ResultsDict[ele].volt
        vList.append(v)
        keyList.append(key)
    eventNo +=1
loopEnd = time.time()
loopTime = loopEnd -loopStart
print 'Took {} sec to generate lists from dictionary'.format(loopTime)
save_obj(vList, 'v0')
save_obj(vList, 'v1')
save_obj(keyList,'key1')
del EventDict
del vList
del keyList


#  dataset 1
print 'Scanning dataset 1'
EventDict = load_obj('Event1')
print 'Done loading event object'
vList = []
tme = []
keyList = []
numEvents = len(EventDict)
eventNo = 1
for event in EventDict:
    ResultsDict = EventDict[event]
    for ele in ResultsDict:
        if ele == 'time':
            #tme = ResultsDict[ele]
            continue
        key = event + '/' + ele
        v = ResultsDict[ele].volt
        vList.append(v)
        keyList.append(key)
    eventNo +=1

print 'Saving list'
save_obj(vList, 'v1')
save_obj(keyList,'key1')
del EventDict
del vList
del keyList


##  dataset 2
print 'Scanning dataset 2'
EventDict = load_obj('Event2')
vList = []
tme = []
keyList = []
numEvents = len(EventDict)
eventNo = 1
for event in EventDict:
    ResultsDict = EventDict[event]
    for ele in ResultsDict:
        if ele == 'time':
            #tme = ResultsDict[ele]
            continue
        key = event + '/' + ele
        v = ResultsDict[ele].volt
        vList.append(v)
        keyList.append(key)
    eventNo +=1

print 'Saving list'
save_obj(vList, 'v2')
save_obj(keyList,'key2')
del EventDict
del vList
del keyList



##  dataset 3
print 'Scanning dataset 3'
EventDict = load_obj('Event3')
vList = []
tme = []
keyList = []
numEvents = len(EventDict)
eventNo = 1
for event in EventDict:
    ResultsDict = EventDict[event]
    for ele in ResultsDict:
        if ele == 'time':
            #tme = ResultsDict[ele]
            continue
        key = event + '/' + ele
        v = ResultsDict[ele].volt
        vList.append(v)
        keyList.append(key)
    eventNo +=1

print 'Saving list'
save_obj(vList, 'v3')
save_obj(keyList,'key3')
del EventDict
del vList
del keyList



##  dataset 4
print 'Scanning dataset 4'
EventDict = load_obj('Event4')
vList = []
tme = []
keyList = []
numEvents = len(EventDict)
eventNo = 1
for event in EventDict:
    ResultsDict = EventDict[event]
    for ele in ResultsDict:
        if ele == 'time':
            #tme = ResultsDict[ele]
            continue
        key = event + '/' + ele
        v = ResultsDict[ele].volt
        vList.append(v)
        keyList.append(key)
    eventNo +=1

print 'Saving list'
save_obj(vList, 'v4')
save_obj(keyList,'key4')
del EventDict
del vList
del keyList



##  dataset 5
print 'Scanning dataset 5'
EventDict = load_obj('Event5')
vList = []
tme = []
keyList = []
numEvents = len(EventDict)
eventNo = 1
for event in EventDict:
    ResultsDict = EventDict[event]
    for ele in ResultsDict:
        if ele == 'time':
            #tme = ResultsDict[ele]
            continue
        key = event + '/' + ele
        v = ResultsDict[ele].volt
        vList.append(v)
        keyList.append(key)
    eventNo +=1

print 'Saving list'
save_obj(vList, 'v5')
save_obj(keyList,'key5')
del EventDict
del vList
del keyList



##  dataset 6
print 'Scanning dataset 6'
EventDict = load_obj('Event6')
vList = []
tme = []
keyList = []
numEvents = len(EventDict)
eventNo = 1
for event in EventDict:
    ResultsDict = EventDict[event]
    for ele in ResultsDict:
        if ele == 'time':
            #tme = ResultsDict[ele]
            continue
        key = event + '/' + ele
        v = ResultsDict[ele].volt
        vList.append(v)
        keyList.append(key)
    eventNo +=1

print 'Saving list'
save_obj(vList, 'v6')
save_obj(keyList,'key6')
del EventDict
del vList
del keyList



##  dataset 7
print 'Scanning dataset 7'
EventDict = load_obj('Event7')
vList = []
tme = []
keyList = []
numEvents = len(EventDict)
eventNo = 1
for event in EventDict:
    ResultsDict = EventDict[event]
    for ele in ResultsDict:
        if ele == 'time':
            #tme = ResultsDict[ele]
            continue
        key = event + '/' + ele
        v = ResultsDict[ele].volt
        vList.append(v)
        keyList.append(key)
    eventNo +=1

print 'Saving list'
save_obj(vList, 'v7')
save_obj(keyList,'key7')
del EventDict
del vList
del keyList



##  dataset 8
print 'Scanning dataset 8'
EventDict = load_obj('Event8')
vList = []
tme = []
keyList = []
numEvents = len(EventDict)
eventNo = 1
for event in EventDict:
    ResultsDict = EventDict[event]
    for ele in ResultsDict:
        if ele == 'time':
            #tme = ResultsDict[ele]
            continue
        key = event + '/' + ele
        v = ResultsDict[ele].volt
        vList.append(v)
        keyList.append(key)
    eventNo +=1

print 'Saving list'
save_obj(vList, 'v8')
save_obj(keyList,'key8')
del EventDict
del vList
del keyList
####################
"""