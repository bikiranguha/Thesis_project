import pandas as pd
import numpy as np




####
# get the events which cause angle separation
angleSepEvents = []
tripFile = 'GenTripData.txt' # generated from analyzeOsc.py

####
with open(tripFile,'r') as f:
    fileLines = f.read().split('\n')



i = 0
while i < len(fileLines):
    line = fileLines[i]
    if 'In' in line: # get the trip data
        words = line.split()
        event = words[1].strip()
        angleSepEvents.append(event)
        i+=1
        line = fileLines[i]
        

    else: # blank line, skip
        i+=1

# for event in angleSepEvents:
#     print(event)
##############









#######################
#### get the transient samples (a certain number of samples, starting just after the fault)


# get the start and end time indexes
startTime = 0.31
numSamples = 120

t = pd.read_csv('obj/timeArray.csv',header = None)
#t = t.loc[0].values
t = t[0].values
t = list(t)

startindex = min([ind for ind, val in enumerate(t) if val >=startTime])
#print(startindex)
endindex = startindex+numSamples



#############
class EventData():
    def __init__(self):
        self.busdict ={}


def generateEventDict(readerfile,eventlist,startindex,endindex,skipset,eventdict):
    # get the transient part from the voltage and angle data
    datareader =    csv.reader(readerfile,quoting=csv.QUOTE_NONNUMERIC)

    eventsAddedset = set()
    for idx, row in enumerate(datareader):
        eventID = eventlist[idx]
        eventKeyWords = eventID.split('/')
        bus = eventKeyWords[-1].strip()


        LP = eventKeyWords[0].strip()
        lines = eventKeyWords[1].strip()
        L1 = lines.split(';')[0]
        L2 = lines.split(';')[1]
        faultbus  = eventKeyWords[2].strip()
        event = '{}/{};{}/{}'.format(LP,L1,L2,faultbus)
        if event in skipset:
            continue
        eventsAddedset.add(event)
        if event not in eventdict:
            eventdict[event] = EventData()


        eventdict[event].busdict[bus] = row[startindex:endindex] # only get the number of samples specified




        # if numSamples == -1: # get all the values in the row
        #     eventdict[event].busdict[bus] = datadf.iloc[idx] # to get the full data
        # else:
        #     eventdict[event].busdict[bus] = datadf.iloc[idx][-numSamples:] # only get the last one sec data

    skipset.update(eventsAddedset) # add all the events to the skip (or explored set)
    readerfile.close()
    return eventdict, skipset




def extractEvents(organizedeventlist,buslist,eventdict,eventWriter):

    for event in organizedeventlist:
        currentdatalist = []

        for bus in buslist:
            data = eventdict[event].busdict[bus]
            for v in data:
                currentdatalist.append(v)
        eventWriter.writerow(currentdatalist)



#############

# # get the set of events to skip (gen trip convergence issues)
# skipEventFile = 'skipeventsN_2F.txt'
# eventstoskip = [] # this needs to be consulted always while adding data
# with open(skipEventFile,'r') as f:
#     fileLines = f.read().split('\n')
#     for line in fileLines:
#         if line == '':
#             continue
#         #print(line.strip())
#         eventstoskip.append(line.strip())



# set up two csv files, one for voltage, another for angle
# these will be used to put all the steady state voltage and angle data, one row for one event





# # first add the generator trip data
# # keep track of all the events which are being added
# gentripfilev = 'obj/vGenTripN_2F.csv'
# gentripfilea = 'obj/aGenTripN_2F.csv'
# gentripeventfile = 'obj/eventKeys_GenTripN_2F.csv'
# # vgt = pd.read_csv(gentripfilev,header = None)
# # agt = pd.read_csv(gentripfilea,header = None)
# vreaderfile = open(gentripfilev,'rb')
# areaderfile = open(gentripfilea,'rb')


# eventlist = []
# with open(gentripeventfile,'r') as f:
#     fileLines = f.read().split('\n')
#     for line in fileLines: 
#         if line == '':
#             continue
#         eventlist.append(line.strip())  

# exploredsetv = set(eventstoskip)
# exploredseta = set(eventstoskip)
eventdictv = {}
eventdicta = {}

exploredsetv = set()
exploredseta = set()

# eventdictv, exploredsetv = generateEventDict(vreaderfile,eventlist,startindex,endindex,exploredsetv,eventdictv)
# eventdicta, exploredseta = generateEventDict(areaderfile,eventlist,startindex,endindex,exploredseta,eventdicta)
#now add the remaining data into the event dictionaries
for i in range(1,10):
    # get the event list
    currentEventFile = 'obj/eventKeys_{}.csv'.format(i)
    currentEventList = []
    with open(currentEventFile,'r') as f:
        fileLines = f.read().split('\n')
        for line in fileLines:
            if line == '':
                continue
            currentEventList.append(line)

    # start reading the angle and voltage data
    currentFilea = 'obj/a{}.csv'.format(i)
    currentFilev = 'obj/v{}.csv'.format(i)
    print('Reading files: {} and {}'.format(currentFilea, currentFilev))


    # vgt = pd.read_csv(currentFilev,header = None)
    # agt = pd.read_csv(currentFilea,header = None)
    vreaderfile = open(currentFilev,'rb')
    areaderfile = open(currentFilea,'rb')

    eventdictv, exploredsetv = generateEventDict(vreaderfile,currentEventList,startindex,endindex,exploredsetv,eventdictv)
    eventdicta, exploredseta = generateEventDict(areaderfile,currentEventList,startindex,endindex,exploredseta,eventdicta)




# now extract data from the dictionaries into csv files in an organized fashion

print('Outputting the voltage and angle steady state data')
refRaw = 'savnw.raw'
busdatadict = getBusData(refRaw)

buslist = busdatadict.keys()

# organize the data into new csv files where each row contains an event data
organizedeventlist = [] # list which contains all the event sequence

for event in eventdictv:
    organizedeventlist.append(event)




vfileNew = 'obj/vN_2FTNGT.csv'
vfileObj = open(vfileNew,'wb')
eventWriterv = csv.writer(vfileObj)

afileNew = 'obj/aN_2FTNGT.csv'
afileObj = open(afileNew,'wb')
eventWritera = csv.writer(afileObj)

print('Outputting voltage')
extractEvents(organizedeventlist,buslist,eventdictv,eventWriterv)
print('Outputting angle')
extractEvents(organizedeventlist,buslist,eventdicta,eventWritera)

vfileObj.close()
afileObj.close()


# output the event sequence in the new steady state files

with open('obj/eventN_2FNGT.txt','w') as f:
    for event in organizedeventlist:
        f.write(event)
        f.write('\n')


buslistStr = ','.join(buslist)
with open('obj/buslistN_2FNGT.csv','w') as f:
    f.write(buslistStr)

#####


















########################