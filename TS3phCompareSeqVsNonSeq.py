# use to visualize TS3ph results with and without seq file
import csv
import matplotlib.pyplot as plt

"""
print 'Reading the csv file containing non-seq simulation'
#vFileName = 'fault3ph/Long/vData3phLI.csv' # csv file containing 5.0 second voltage data (different types of fault)
vFileNameNS = 'fault3ph/vData3phLI.csv' # csv file containing 0.5 second voltage data (different types of fault)
tFileNameNS = 'fault3ph/tData3ph.csv' # csv file containing the time data
eventKeyFileNS = 'fault3ph/eventIDFileLI.txt'
vFileNS = open(vFileNameNS,'rb')
tFileNS = open(tFileNameNS,'rb')
readerVNS=csv.reader(vFileNS,quoting=csv.QUOTE_NONNUMERIC) # so that the entries are converted to floats
readerT = csv.reader(tFileNS,quoting=csv.QUOTE_NONNUMERIC)
tmeNS = [row for idx, row in enumerate(readerT) if idx==0][0]

# read the event file for the non-seq data
print 'Organizing the csv for non-seq data...'
eventListNS = []
with open(eventKeyFileNS,'r') as f:
    fileLines = f.read().split('\n')
    for line in fileLines[1:]:
        if line == '':
            continue
        eventListNS.append(line.strip())


# gathering the non-seq data
SimpleEventDictNS = {} # dictionary containing the non-seq data
for idx, row in enumerate(readerVNS):
    eventKey = eventListNS[idx]
    SimpleEventDictNS[eventKey] = row
"""


print 'Reading the csv file containing seq simulation'
#vFileName = 'fault3ph/Long/vData3phLI.csv' # csv file containing 5.0 second voltage data (different types of fault)
vFileNameS = 'fault3ph/vData3phLISeq.csv' # csv file containing 0.5 second voltage data (different types of fault)
aFileNameS = 'fault3ph/aData3phLISeq.csv'
tFileNameS = 'fault3ph/tData3phLISeq.csv' # csv file containing the time data
eventKeyFileS = 'fault3ph/eventIDFileLISeq.txt'
vFileS = open(vFileNameS,'rb')
aFileS = open(aFileNameS,'rb')
tFileS = open(tFileNameS,'rb')
readerVS=csv.reader(vFileS,quoting=csv.QUOTE_NONNUMERIC) # so that the entries are converted to floats
readerAS = csv.reader(aFileS,quoting=csv.QUOTE_NONNUMERIC)
readerT = csv.reader(tFileS,quoting=csv.QUOTE_NONNUMERIC)
tmeS = [row for idx, row in enumerate(readerT) if idx==0][0]






# read the event file for the seq data
print 'Organizing the csv for seq data...'
eventListS = []
with open(eventKeyFileS,'r') as f:
    fileLines = f.read().split('\n')
    for line in fileLines[1:]:
        if line == '':
            continue
        eventListS.append(line.strip())
# gathering the seq data
SimpleEventDictS = {} # dictionary containing the seq data
for idx, row in enumerate(readerVS):
    eventKey = eventListS[idx]
    SimpleEventDictS[eventKey] = row

# gathering the seq data for the angles
SimpleEventDictAS = {} # dictionary containing the seq data
for idx, row in enumerate(readerAS):
    eventKey = eventListS[idx]
    SimpleEventDictAS[eventKey] = row


# compare voltage data among various phases using seq integrated data
keyA = 'R100/F151/B206/AG/A/1.0e-6'
keyB = 'R100/F151/B206/AG/B/1.0e-6'
keyC = 'R100/F151/B206/AG/C/1.0e-6'


dataSA = SimpleEventDictS[keyA]
dataSB = SimpleEventDictS[keyB]
dataSC = SimpleEventDictS[keyC]

#plt.plot(tmeNS,dataNS,label = 'Without seq data')
plt.plot(tmeS,dataSA, label = 'Phase A')
plt.plot(tmeS,dataSB, label = 'Phase B')
plt.plot(tmeS,dataSC, label = 'Phase C')
plt.ylim(-0.1,1.5)
#plt.title('SLG A Fault at 101,  Voltage monitored bus 151')
plt.title('SLG A Fault at 206,  Voltage monitored bus 205')
plt.legend()
plt.grid()
plt.show()


# compare angle data among various phases using seq integrated data


adataSA = SimpleEventDictAS[keyA]
adataSB = SimpleEventDictAS[keyB]
adataSC = SimpleEventDictAS[keyC]

#plt.plot(tmeNS,dataNS,label = 'Without seq data')
plt.plot(tmeS,adataSA, label = 'Phase A')
plt.plot(tmeS,adataSB, label = 'Phase B')
plt.plot(tmeS,adataSC, label = 'Phase C')
#plt.ylim(-0.1,1.2)
#plt.title('SLG A Fault at 101, Angle monitored bus 151')
plt.title('SLG A Fault at 206, Angle monitored bus 205')
plt.legend()
plt.grid()
plt.show()

"""
# compare voltage data among various phases using seq integrated data
keyA = 'R100/F101/B101/AG/A/1.0e-6'
keyB = 'R100/F101/B101/AG/B/1.0e-6'
keyC = 'R100/F101/B101/AG/C/1.0e-6'


dataSA = SimpleEventDictS[keyA]
dataSB = SimpleEventDictS[keyB]
dataSC = SimpleEventDictS[keyC]

#plt.plot(tmeNS,dataNS,label = 'Without seq data')
plt.plot(tmeS,dataSA, label = 'Phase A')
plt.plot(tmeS,dataSB, label = 'Phase B')
plt.plot(tmeS,dataSC, label = 'Phase C')
#plt.ylim(-0.1,1.5)
#plt.title('SLG A Fault at 101,  Voltage monitored bus 151')
plt.title('SLG A Fault at 101,  Voltage monitored bus 101')
plt.legend()
plt.grid()
plt.show()


# compare angle data among various phases using seq integrated data


adataSA = SimpleEventDictAS[keyA]
adataSB = SimpleEventDictAS[keyB]
adataSC = SimpleEventDictAS[keyC]

#plt.plot(tmeNS,dataNS,label = 'Without seq data')
plt.plot(tmeS,adataSA, label = 'Phase A')
plt.plot(tmeS,adataSB, label = 'Phase B')
plt.plot(tmeS,adataSC, label = 'Phase C')
#plt.ylim(-0.1,1.2)
#plt.title('SLG A Fault at 101, Angle monitored bus 151')
plt.title('SLG A Fault at 101, Angle monitored bus 101')
plt.legend()
plt.grid()
plt.show()
"""





"""
# compare data
compareKey = 'R100/F151/B101/AG/A/1.0e-6'

dataNS = SimpleEventDictNS[compareKey]
dataS = SimpleEventDictS[compareKey]


plt.plot(tmeNS,dataNS,label = 'Without seq data')
plt.plot(tmeS,dataS, label = 'With seq data')
plt.title(' SLG A Fault at 151, monitored bus 101')
plt.legend()
plt.grid()
plt.show()
"""

"""
# compare data
compareKey = 'R100/F101/B102/AG/A/1.0e-6'

dataNS = SimpleEventDictNS[compareKey]
dataS = SimpleEventDictS[compareKey]


plt.plot(tmeNS,dataNS,label = 'Without seq data')
plt.plot(tmeS,dataS, label = 'With seq data')
plt.title('SLG A Fault at 101, monitored bus 102')
plt.legend()
plt.grid()
plt.show()
"""

"""
# compare data
compareKey1 = 'R100/F3005/B204/AG/A/1.0e-6'
compareKey2 = 'R106/F3005/B204/AG/A/1.0e-6'
dataNS1 = SimpleEventDictNS[compareKey1]
dataNS2 = SimpleEventDictNS[compareKey2]


plt.plot(tmeNS,dataNS1,label = '100')
plt.plot(tmeS,dataNS2, label = '106')
#plt.title('SLG A Fault at 101, monitored bus 102')
plt.legend()
plt.grid()
plt.show()
"""