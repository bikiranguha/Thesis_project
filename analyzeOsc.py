# look at the steady state angles for the N_2 plus fault events and identify 
# events which are oscillations and which are angle separation
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from getBusDataFn import getBusData

# dictionary to help with determining when angle separation happened
# and for which generators
angleComparisonDict = {}
angleComparisonDict['101'] = '211'
angleComparisonDict['102'] = '211'
angleComparisonDict['211'] = '206'
angleComparisonDict['206'] = '3011'
angleComparisonDict['3011'] = '3018'
angleComparisonDict['3018'] = '3011'
#######

class gentripdata():
    def __init__(self,tripdict):
        self.tripdict =tripdict


# get the time info
tFileName = 'obj/timeArray.csv'
tdf = pd.read_csv(tFileName,header = None)
tme = tdf[0].values



print('Reading the gen angle input file')


#######

# these files are generated from gatherGenSteadyAngles.py
file = 'obj/aGenN_2FSteady.csv' # use for steady state values (contains only the last 120 timesteps)
file = 'obj/aGenN_2FAll.csv' # use to visualize the entire angle plot
#file = 'obj/aGenN_2FGenTripAll.csv' # use to visualize those events where a gen trip was necessary
df = pd.read_csv(file,header = None)

eventList = []
eventfile = 'obj/aGenN_2FList.txt' # use for getting angle data for all N-2 f events (without any gen trip)
#eventfile = 'obj/aGenN_2FTripList.txt' # use for getting the gen trip events only
#####



with open(eventfile,'r') as f:
    fileLines = f.read().split('\n')
    for line in fileLines:
        if line == '':
            continue
        eventList.append(line)



### get the gen list arrangement
BusDataDict = getBusData('savnw.raw')
genbustypes = ['2','3']
genlist=[]
for bus in BusDataDict:
    bustype = BusDataDict[bus].type
    if bustype in genbustypes:
        genlist.append(bus)
#####



#### loop through event data to see abnormal events (angle oscillation and angle separation)
# abnormalEvents = []
# angleDevDict = {} 
# for ind, event in enumerate(eventList):
#     data = df.iloc[ind]
#     dataArray = data.values.reshape(-1,120)
#     meanAngles = np.mean(dataArray,axis=0)


#     rangelist = []
#     for j  in range(dataArray.shape[0]):
#         relAngle = abs(meanAngles-dataArray[j])
#         rng = relAngle.max() - relAngle.min()
#         rangelist.append(rng)
#         # if rng > 10.0:
#         #     abnormalEvents.append(event)
#         #     break
#     maxRange = np.max(rangelist)
#     angleDevDict[event] = maxRange

# #print('Number of abnormal events detected: {}'.format(len(abnormalEvents)))


# with open('AbnormalEventData.txt','w') as f:
#     f.write('Event: Max deviation in angle noted')
#     f.write('\n')
#     for key, value in sorted(angleDevDict.iteritems(), key=lambda (k,v): v, reverse = True): # descending order
#         value = '%.3f' % value
#         line = '{}:{}'.format(key,value)
#         f.write(line)
#         f.write('\n')

###############

# ### 
# # loop through each event and determine generator tripping times if angle separation is detected
# print('Analyzing the events for any necessary generator tripping')
# abnormalEvents = []
# gentripdict = {}
# for ind, event in enumerate(eventList):
#     data = df.iloc[ind]
#     dataArray = data.values.reshape(-1,1208) # each generator data occupies 1208 timesteps
#     #meanAngles = np.mean(dataArray,axis=0)


#     rangelist = []
#     d = {}
#     # scan through each generator and see if there is any tripping necessary
#     for j  in range(dataArray.shape[0]):

#         currentgen = genlist[j]
#         refgen = angleComparisonDict[currentgen]
#         refgenindex = genlist.index(refgen)
#         relAngle = abs(dataArray[j]-dataArray[refgenindex])
#         #relAngle = abs(meanAngles-dataArray[j])
#         relAngle = list(relAngle)
#         aboveThreshold = [ind for ind, val in enumerate(relAngle) if val > 90]
#         if len(aboveThreshold) > 0: # tripping necessary
#             tripIndex = min(aboveThreshold)
#             triptme = tme[tripIndex]
#             gen = genlist[j]
#             d[gen] = triptme

#     if len(d) > 0:
#         # put all the tripping data into the event dict
#         gentripdict[event] = gentripdata(d)


# print('Writing output')
# with open('GenTripData.txt','w') as f:
#     #f.write('Event: Max deviation in angle noted')
#     #f.write('\n')
#     for event in gentripdict:
#         f.write('In {}'.format(event))
#         f.write('\n')
#         d = gentripdict[event].tripdict
#         #for gen in d:
#         for gen, value in sorted(d.iteritems(), key=lambda (k,v): v):
#             line = '{}:{}'.format(gen,value)
#             f.write(line)
#             f.write('\n')
#         f.write('\n')



#####









# ##
# visualize any event
event = '104/151,201,1/151,152,1/F152'
event = '103/151,152,1;201,204,1/F201'
event = '106/151,201,1;203,205,2/F203'
event = '103/152,202,1;151,201,1/F201' # clear angle separation but somehow not included
#event = '103/151,152,1;151,201,1/F201' # clear angle separation
#event = '106/151,201,1;152,3004,1/F152' # not so clear angle separation
eventInd = eventList.index(event)
data = df.iloc[eventInd]
#dataArray = data.values.reshape(-1,120) # for the steady state stuff
dataArray = data.values.reshape(-1,1208) # for the entire stuff
meanAngles = np.mean(dataArray,axis=0)
# plt.plot(meanAngles)
# plt.title('Case of angle separation')
# plt.ylabel('Mean of generator angles (degrees)')
# plt.xlabel('Sample number')
for j in range(dataArray.shape[0]):
    # plot the absolute angles
    plt.plot(dataArray[j],label= genlist[j])
    #plt.title('Case of  not so clear angle separation')
    plt.title('Case of angle separation')
    plt.ylabel('Absolute angles (degrees)')
    plt.xlabel('Sample number')


    #plot the relative angles (wrt the mean angle)
    # relAngle = abs(meanAngles-dataArray[j])
    # plt.plot(relAngle,label= genlist[j])
    # #plt.title('Case of angle separation')
    # plt.title('Case of  not so clear angle separation')
    # plt.ylabel('Relative angles wrt mean (degrees)')
    # plt.xlabel('Sample number')
plt.grid()
plt.legend()
plt.show()
# ###
