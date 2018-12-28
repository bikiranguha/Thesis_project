# write a script which organizes all the generator angle data during (apparent) steady state
# this will help with determining the type of oscillation (none, poorly damped oscillation, angle separation)
from getBusDataFn import getBusData
import csv
class EventGenAngles():
    def __init__(self):
        self.GenDict ={}


organizedEventDict = {}
BusDataDict = getBusData('savnw.raw')
genbustypes = ['2','3']



###
# gather angles for all the simulations (before any gen trip was applied)

# for i in range(1,10):
#     #print('Loop {} out of 9'.format(i))
#     # get the event list
#     currentEventFile = 'obj/eventKeys_{}.csv'.format(i)
#     currentEventList = []
#     with open(currentEventFile,'r') as f:
#         fileLines = f.read().split('\n')
#         for line in fileLines:
#             if line == '':
#                 continue
#             currentEventList.append(line)

#     # start reading the angle data
#     currentFile = 'obj/a{}.csv'.format(i)
#     print('Reading file: {}'.format(currentFile))
#     currentFileObj = open(currentFile,'rb')
#     aReader = csv.reader(currentFileObj,quoting=csv.QUOTE_NONNUMERIC)
#     for idx, row in enumerate(aReader):
#         eventID = currentEventList[idx]
#         eventKeyWords = eventID.split('/')
#         bus = eventKeyWords[-1].strip()
#         # skip if the bus is not a gen bus
#         if BusDataDict[bus].type not in genbustypes: 
#             continue

#         LP = eventKeyWords[0].strip()
#         lines = eventKeyWords[1].strip()
#         L1 = lines.split(';')[0]
#         L2 = lines.split(';')[1]
#         faultbus  = eventKeyWords[2].strip()
#         event = '{}/{};{}/{}'.format(LP,L1,L2,faultbus)
#         if event not in organizedEventDict:
#             organizedEventDict[event] = EventGenAngles()

 
#         #organizedEventDict[event].GenDict[bus] = row[-120:] # only get the last one sec data
#         organizedEventDict[event].GenDict[bus] = row



# # make a gen list which will be used to organize the data
# genlist = []

# for bus in BusDataDict:
#     bustype = BusDataDict[bus].type
#     if bustype in genbustypes:
#         genlist.append(bus)




# eventgenanglefile = 'obj/aGenN_2FAll.csv'
# eventgenanglefileObj = open(eventgenanglefile,'wb')
# eventWriter = csv.writer(eventgenanglefileObj)


# # organize the data into a new csv file where each row contains an event data
# organizedeventlist = []
# for event in organizedEventDict:
#     currentdatalist = []

#     for gen in genlist:
#         data = organizedEventDict[event].GenDict[gen]
#         for v in data:
#             currentdatalist.append(v)
#     eventWriter.writerow(currentdatalist)
#     organizedeventlist.append(event)


# eventgenanglefileObj.close()


# with open('obj/aGenN_2FList.txt','w') as f:
#     for event in organizedeventlist:
#         f.write(event)
#         f.write('\n')


############



############
## gather for the generator trips only


# currentEventFile = 'obj/eventKeys_GenTripN_2F.csv'
# currentEventList = []
# with open(currentEventFile,'r') as f:
#     fileLines = f.read().split('\n')
#     for line in fileLines:
#         if line == '':
#             continue
#         currentEventList.append(line)

# # start reading the angle data
# currentFile = 'obj/aGenTripN_2F.csv'
# print('Reading file: {}'.format(currentFile))
# currentFileObj = open(currentFile,'rb')
# aReader = csv.reader(currentFileObj,quoting=csv.QUOTE_NONNUMERIC)
# for idx, row in enumerate(aReader):
#     eventID = currentEventList[idx]
#     eventKeyWords = eventID.split('/')
#     bus = eventKeyWords[-1].strip()
#     # skip if the bus is not a gen bus
#     if BusDataDict[bus].type not in genbustypes: 
#         continue

#     LP = eventKeyWords[0].strip()
#     lines = eventKeyWords[1].strip()
#     L1 = lines.split(';')[0]
#     L2 = lines.split(';')[1]
#     faultbus  = eventKeyWords[2].strip()
#     event = '{}/{};{}/{}'.format(LP,L1,L2,faultbus)
#     if event not in organizedEventDict:
#         organizedEventDict[event] = EventGenAngles()


#     #organizedEventDict[event].GenDict[bus] = row[-120:] # only get the last one sec data
#     organizedEventDict[event].GenDict[bus] = row



# # make a gen list which will be used to organize the data
# genlist = []

# for bus in BusDataDict:
#     bustype = BusDataDict[bus].type
#     if bustype in genbustypes:
#         genlist.append(bus)




# eventgenanglefile = 'obj/aGenN_2FGenTripAll.csv'
# eventgenanglefileObj = open(eventgenanglefile,'wb')
# eventWriter = csv.writer(eventgenanglefileObj)


# # organize the data into a new csv file where each row contains an event data
# organizedeventlist = []
# for event in organizedEventDict:
#     currentdatalist = []

#     for gen in genlist:
#         data = organizedEventDict[event].GenDict[gen]
#         for v in data:
#             currentdatalist.append(v)
#     eventWriter.writerow(currentdatalist)
#     organizedeventlist.append(event)


# eventgenanglefileObj.close()


# with open('obj/aGenN_2FTripList.txt','w') as f:
#     for event in organizedeventlist:
#         f.write(event)
#         f.write('\n')
##
############
