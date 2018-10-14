# load the voltage data and get the voltage templates from the class 0 and class 1 oscillation data
# then test some of the samples using similarity indices
# implements some of the ideas from RGNCT2010.pdf


# load the voltage data which belongs to stable cases

import pickle
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
import random

# Function to load pickle object
def load_obj(name ):
    # load pickle object
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

# load voltage data
VoltageDataDict = load_obj('VoltageData') # this voltage data dictionary has been generated from the script 'TS3phSimN_2FaultDataSave.py', see it for key format
caseFile = 'Casedvdt.txt' # text file containing classification data


# get the class 0 (no oscillation) and 1 (large oscillation)  event lists
with open(caseFile,'r') as f:
    fileLines = f.read().split('\n')

Class0StartIndex = fileLines.index('Class 0:') + 1
Class0EndIndex = fileLines.index('Class 1:')
Class1StartIndex = Class0EndIndex + 1

class0keys = []
class1keys = []
# get the class 0 events
for i in range(Class0StartIndex, Class0EndIndex):
    line = fileLines[i]
    if line == '':
        continue
    class0keys.append(line)

# get the class 1 events
for i in range(Class1StartIndex, len(fileLines)):
    line = fileLines[i]
    if line == '':
        continue
    class1keys.append(line)


# get time data and get relevant time indices
tme = VoltageDataDict['time']
timestep = abs(tme[1] - tme[0])
ind_fault_clearance = int(0.31/timestep)  + 1 # fault cleared
ind_data_end = ind_fault_clearance + 100 # take 100 time steps

class0Array = np.zeros((len(class0keys),100)) # rows: different samples, columns: data of a sample
class1Array = np.zeros((len(class1keys),100))
j = 0
k = 0
for key in VoltageDataDict:
    if key == 'time':
        continue

    voltage = VoltageDataDict[key]
    croppedV = voltage[ind_fault_clearance:ind_data_end]
    if key in class0keys:
        class0Array[j] = croppedV
        j+=1
    elif key in class1keys:
        class1Array[k] = croppedV
        k+=1


# use fuzzy c-means to generate templates
class0ArrayT = np.transpose(class0Array) # transpose required for fuzzy c-means
ncenters = 1 
tmpltCls0, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    class0ArrayT, ncenters, 2, error=0.005, maxiter=1000, init=None)

tmpltCls0 = np.array(tmpltCls0).reshape(-1)
tmeCropped = tme[ind_fault_clearance:ind_data_end]
meantmpltCls0 = np.mean(tmpltCls0)
#plt.plot(tmeCropped,tmpltCls0, label = 'Class 0 Template')
#plt.title('Class 0 template')
#plt.grid()
#plt.show()
#plt.close()


class1ArrayT = np.transpose(class1Array)
ncenters = 1 
tmpltCls1, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    class1ArrayT, ncenters, 10, error=0.005, maxiter=1000, init=None)

tmpltCls1 = np.array(tmpltCls1).reshape(-1)
tmeCropped = tme[ind_fault_clearance:ind_data_end]
meantmpltCls1 = np.mean(tmpltCls1)
#plt.plot(tmeCropped,tmpltCls1, label = 'Class 1 Template')
#plt.title('Class 0 vs Class 1 templates')
#plt.legend()
#plt.show()
#plt.close()


# test the similarity indices using some class 0 samples
random.seed(42) # uncomment if reproducibility is needed
class0Samples = random.sample(class0keys,10)

#sumEucDist = 0.0
for sample in class0Samples:
    voltage = VoltageDataDict[sample]
    croppedV = voltage[ind_fault_clearance:ind_data_end]
    meanCroppedV = np.mean(croppedV)
    meanOffset0 = (meanCroppedV-meantmpltCls0)*np.ones(croppedV.shape[0])
    meanOffset1 = (meanCroppedV-meantmpltCls1)*np.ones(croppedV.shape[0])
    # using offset
    similarity0 = 1/np.linalg.norm(croppedV-tmpltCls0-meanOffset0)
    similarity1 = 1/np.linalg.norm(croppedV-tmpltCls1-meanOffset1)
    # not using offset
    #similarity0 = 1/np.linalg.norm(croppedV-tmpltCls0)
    #similarity1 = 1/np.linalg.norm(croppedV-tmpltCls1)
    print 'Class 0 Sample: ', sample
    print 'Similarity to template 0: ', similarity0
    print 'Similarity  template 1: ', similarity1



print '\n\n\n'

# test the similarity indices using some class 1 samples
random.seed(42) # uncomment if reproducibility is needed
class1Samples = random.sample(class1keys,10)

#sumEucDist = 0.0
for sample in class1Samples:
    voltage = VoltageDataDict[sample]
    croppedV = voltage[ind_fault_clearance:ind_data_end]
    meanCroppedV = np.mean(croppedV)
    meanOffset0 = (meanCroppedV-meantmpltCls0)*np.ones(croppedV.shape[0])
    meanOffset1 = (meanCroppedV-meantmpltCls1)*np.ones(croppedV.shape[0])

    # using offset
    similarity0 = 1/np.linalg.norm(croppedV-tmpltCls0-meanOffset0)
    similarity1 = 1/np.linalg.norm(croppedV-tmpltCls1-meanOffset1)
    # not using offset
    #similarity0 = 1/np.linalg.norm(croppedV-tmpltCls0)
    #similarity1 = 1/np.linalg.norm(croppedV-tmpltCls1)
    print 'Class 1 Sample: ', sample
    print 'Similarity to template 0: ', similarity0
    print 'Similarity  template 1: ', similarity1



"""
# tests
#key = '152,202,1;3003,3005,2;F3003/211'
#key = '203,205,2;3003,3005,2;F3003/201'
#key = '151,152,2;151,201,1;F201/3001'
# plot the templates and 2 samples of class 0
random.seed(42) # uncomment if reproducibility is needed
class0Samples = random.sample(class0keys,2)
plt.plot(tmeCropped,tmpltCls1, label = 'Class 1 Template')
plt.plot(tmeCropped,tmpltCls0, label = 'Class 0 Template')
for sample in class0Samples:
    voltage = VoltageDataDict[sample]
    croppedV = voltage[ind_fault_clearance:ind_data_end]


    #plt.plot(tmeCropped,croppedV,label = 'Sample')
    plt.plot(tmeCropped,croppedV)
plt.title('Class 0 samples with templates')
plt.grid()
plt.legend()
plt.show()
plt.close()


# plot the templates and 2 samples of class 1
random.seed(42) # uncomment if reproducibility is needed
class1Samples = random.sample(class1keys,2)
plt.plot(tmeCropped,tmpltCls1, label = 'Class 1 Template')
plt.plot(tmeCropped,tmpltCls0, label = 'Class 0 Template')
for sample in class1Samples:
    voltage = VoltageDataDict[sample]
    croppedV = voltage[ind_fault_clearance:ind_data_end]
    plt.plot(tmeCropped,croppedV)
plt.title('Class 1 samples with templates')
plt.grid()
plt.legend()
plt.show()
plt.close()
"""


"""
# just plot a case
#plt.plot(tme,voltage)
#plt.show()
"""



