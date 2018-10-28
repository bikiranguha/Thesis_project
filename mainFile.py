import getListOfSimulations
from simPSSEFn import *
import os
import dill as pickle # used to load more complicated data structures (in this case, a dictionary containing a class)
eventListFile = 'events.txt'

eventsList = []
with open(eventListFile,'r') as f:
    fileLines = f.read().split('\n')
    for line in fileLines:
        if line == '':
            continue
        eventsList.append(line.strip())

dyrFile = 'savnw_dy_sol_0905.dyr'

def save_obj(obj, name ):
    # save as pickle object
    currentdir = os.getcwd()
    objDir = currentdir + '/obj'
    if not os.path.isdir(objDir):
        os.mkdir(objDir)
    with open(objDir+ '/' +  name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL,recurse = 'True')

batch = 0
while True:
    try:
        objectName = 'Events{}'.format(batch)
        startIndex = batch*500
        endIndex = (batch+1) *500
        EventsDict = runPSSESimBatches(eventsList[startIndex:endIndex],dyrFile,objectName) # pass batches of 500 to the function
        save_obj(EventsDict,objectName)
        batch +=1
    except: # reached the end of eventsList
        break