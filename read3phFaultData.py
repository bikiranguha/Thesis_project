# script to visualize certain event(s) from the fault csv file using event key
import csv
import matplotlib.pyplot as plt
from avgFilterFn import avgFilter
vFileName = 'fault3ph/vData3phL.csv' # csv file containing voltage data (different types of fault)
tFileName = 'fault3ph/tData3ph.csv' # csv file containing the time data
eventKeyFile = 'fault3ph/eventIDFileL.txt'
vFile = open(vFileName,'rb')
tFile = open(tFileName,'rb')
readerV=csv.reader(vFile,quoting=csv.QUOTE_NONNUMERIC) # so that the entries are converted to floats
readerT = csv.reader(tFile,quoting=csv.QUOTE_NONNUMERIC)
tme = [row for idx, row in enumerate(readerT) if idx==0][0]
#interestingrow = [row for idx, row in enumerate(reader) if idx ==1]



# read the event file
eventList = []
with open(eventKeyFile,'r') as f:
    fileLines = f.read().split('\n')
    for line in fileLines[1:]:
        if line == '':
            continue
        eventList.append(line.strip())

# get any event
interestingInd = []

# visualize 3 ph at the fault bus
event = 'R106/F3018/B3018/ABCG'
eventKeyA = '{}/A'.format(event)
eventIndA = eventList.index(eventKeyA)

eventKeyB = '{}/B'.format(event)
eventIndB = eventList.index(eventKeyB)


eventKeyC = '{}/C'.format(event)
eventIndC = eventList.index(eventKeyC)




"""
# visualize 3 ph at some other bus
eventKeyA = 'F151/B3007/AG/A'
eventIndA = eventList.index(eventKeyA)

eventKeyB = 'F151/B3007/AG/B'
eventIndB = eventList.index(eventKeyB)


eventKeyC = 'F151/B3007/AG/C'
eventIndC = eventList.index(eventKeyC)
"""



interestingInd.append(eventIndA)
interestingInd.append(eventIndB)
interestingInd.append(eventIndC)
#print eventInd
interestingrows = [row for idx, row in enumerate(readerV) if idx in interestingInd]
interestingrowA = interestingrows[0]
interestingrowB = interestingrows[1]
interestingrowC = interestingrows[2]


"""
# to visualize filtering
interestingrowAFIL6  = avgFilter(interestingrowA,6)
interestingrowAFIL10  = avgFilter(interestingrowA,10)
plt.plot(tme,interestingrowA,label = 'A')
plt.plot(tme,interestingrowAFIL6,label = 'A filtered 6')
plt.plot(tme,interestingrowAFIL10,label = 'A filtered 10')
plt.grid()
plt.ylim(0,1.5)
plt.legend()
plt.show()
"""



# visualize
plt.plot(tme,interestingrowA,label = 'A')
plt.plot(tme,interestingrowB,label = 'B')
plt.plot(tme,interestingrowC,label = 'C')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (pu)')
plt.title(event)
plt.legend()
plt.grid()
plt.ylim(0,1.5)
plt.show()
"""

# close files
vFile.close()
tFile.close()

