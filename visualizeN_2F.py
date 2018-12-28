# plot an event using the event list and the large voltage file
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
event = '100/201,202,1;151,201,1/F151/151'
event = '104/154,3008,1;152,3004,1/F3004/3004'
event = '103/201,204,1;151,152,1/F152/152'
event = '105/151,201,1;201,202,1/F202/211'
event = '106/151,201,1;152,3004,1/F152/101' # not so clear angle separation
#event = '103/151,152,1;201,204,1/F201/151'
#event = '106/151,201,1;203,205,2/F203/203'
event = '100/151,201,1;151,152,1/F151/101' # clear angle separation
#event = '106/151,201,1;152,3004,1/F152/101'
event = '103/151,152,1;151,152,2/F152/211' # clear oscillation
event = '102/201,204,1;152,3004,1/F3004/152'# ' avg steady osc:0.105645805597'
event = '105/152,3004,1;153,3006,1/F3006/3004' #'avg steady osc:0.155613332987'
event = '100/151,201,1;152,202,1/F202/3018' #'avg steady osc:0.209276020527'
event = '106/3005,3006,1;152,3004,1/F152/3008' #'avg steady osc:0.273462384939'
event = '105/201,202,1;152,3004,1/F3004/3004' # damped oscillation
event = '104/152,3004,1;201,204,1/F204/204'
event = '100/3005,3007,1;3007,3008,1/F3008/3008'
####
# scan each of the 9 event files to find out the proper index of the event
eventFileInd = 0 # event file index (can be anything between 1 and 9)
eventLineInd = 0 # (the line index where the event happens, starts from 0)
eventFound = 0 # flag to help break out of the nested loops
for i in range(1,10):
    # get the event list
    currentEventFile = 'obj/eventKeys_{}.csv'.format(i)
    currentEventList = []
    with open(currentEventFile,'r') as f:
        fileLines = f.read().split('\n')
        for line in fileLines:
            if line == event:
                eventLineInd = fileLines.index(line)
                eventFileInd = i
                eventFound = 1
                break
    if eventFound == 1:
        break


# get the time info
tFileName = 'obj/timeArray.csv'
tdf = pd.read_csv(tFileName,header = None)
tme = tdf[0].values

########
## visualize a single voltage
if eventFileInd != 0:
    print('Event file name: eventKeys_{}.csv'.format(eventFileInd))
    print('Event line index: {}'.format(eventLineInd))

    # using the indices above, go to the specific file and the specific line index

    # get the file
    df = pd.read_csv('obj/v{}.csv'.format(eventFileInd),header = None,nrows = eventLineInd + 10)
    # get the line
    data = df.iloc[eventLineInd].values
    # get the plot
    plt.plot(tme,data)
    #plt.title('Bus 152 voltage plot')
    plt.xlabel('t (s)')
    plt.ylabel('V (pu)')
    plt.grid()
    plt.show()
    #######
else:
    print('Event ID not found.')
###########

#############
# # # ## visualize angle
# if eventFileInd != 0:
#     print('Event file name: eventKeys_{}.csv'.format(eventFileInd))
#     print('Event line index: {}'.format(eventLineInd))

#     # using the indices above, go to the specific file and the specific line index

#     # get the file
#     df = pd.read_csv('obj/a{}.csv'.format(eventFileInd),header = None,nrows = eventLineInd + 10)
#     # get the line
#     data = df.iloc[eventLineInd].values
#     # get the plot
#     plt.plot(tme,data)
#     plt.title('Bus 101 angle plot')
#     #plt.title('Bus 201 angle plot before generator tripping')
#     plt.xlabel('t (s)')
#     plt.ylabel('angle (degrees)')
#     plt.grid()
#     plt.show()
#     #######
# else:
#     print('Event ID not found.')
# ###########








