# compile all the data into one single (very large csv file)
import pickle
import csv

largeVFile = 'obj/vN_2F.csv'
largeVFileObj = open(largeVFile,'wb')
vWriter = csv.writer(largeVFileObj)

#largeEventFileObj = open('obj/EventsN_2F.txt','w')
largeEventList = []


for i in range(1,10):
    print('Loop {} out of 9'.format(i))
    # get the event list
    currentEventFile = 'obj/eventKeys_{}.csv'.format(i)
    currentEventList = []
    with open(currentEventFile,'r') as f:
        fileLines = f.read().split('\n')
        for line in fileLines:
            if line == '':
                continue
            currentEventList.append(line)

    # start reading the voltage data
    currentVFile = 'obj/v{}.csv'.format(i)
    currentVFileObj = open(currentVFile,'rb')
    vReader = csv.reader(currentVFileObj,quoting=csv.QUOTE_NONNUMERIC)
    for idx, row in enumerate(vReader):
        event = currentEventList[idx]
        largeEventList.append(event)
        vWriter.writerow(row)

    currentVFileObj.close()

largeVFileObj.close()

# save all the event key data
with open('obj/EventsN_2F.txt','w') as f:
    for event in largeEventList:
        f.write(event)
        f.write('\n')


