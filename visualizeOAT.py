# script to visualize the transient signals (voltage/angle/frequency)
# has an interactive dialogue which asks for the event, type of variable, transient or steady
# loops till the shell is closed
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# specify the input directory and the patterns
# # pattern for the PSSE simulations
# direc = 'obj'
# TransientFilePattern = 'PFORNLTransient'
# SteadyFilePattern = 'PFORNLSteady'
# eventFilePattern = 'eventKeys_PFORNL'



# pattern for the TS3ph simulations for PF_ORNL
direc = 'PFORNLSim'
TransientFilePattern = 'Transient'
SteadyFilePattern = 'Steady'
eventFilePattern = 'eventKeysPFORNL_'






while True:


    event = raw_input('Event the event id:')
    #event = '1920,2000,1;900,901,1/F900/900'

    ####
    # scan each of the 9 event files to find out the proper index of the event
    eventFileInd = 0 # event file index (can be anything between 1 and 9)
    eventLineInd = 0 # (the line index where the event happens, starts from 0)
    eventFound = 0 # flag to help break out of the nested loops
    for i in range(1,10):
        # get the event list
        currentEventFile = '{}/{}{}.csv'.format(direc,eventFilePattern,i)
        #currentEventList = []
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
    print('What type of variable do you want to visualize?')
    print('Voltage:     1')
    print('Angle:       2')
    print('Frequency:   3')


    vaf = raw_input('Your choice: ').strip()

    if vaf == '1':
        startString = 'v'
    elif vaf == '2':
        startString = 'a'
    elif vaf == '3':
        startString = 'f'
    else:
        print('Invalid input')
        continue



    # if vaf == '1':
    #     df = pd.read_csv('{}/vPFORNLTransient{}.csv'.format(direc,eventFileInd),header = None,nrows = eventLineInd + 10) # transient
    # elif vaf == '2':
    #     df = pd.read_csv('{}/aPFORNLTransient{}.csv'.format(direc,eventFileInd),header = None,nrows = eventLineInd + 10) # transient
    # elif vaf == '3':
    #     df = pd.read_csv('{}/fPFORNLTransient{}.csv'.format(direc,eventFileInd),header = None,nrows = eventLineInd + 10) # transient
    # else:
    #     print('Invalid input')
    #     continue


    print('Transient or steady?')
    print('Transient:       1')
    print('Steady:          2')
    st = raw_input('Your choice: ').strip()




    if st == '1':
        endString = TransientFilePattern
    elif st == '2':
        endString = SteadyFilePattern
    else:
        print('Invalid input')
        continue


    # get the data file name and the time file names
    dataFileName = '{}/{}{}{}.csv'.format(direc,startString,endString,eventFileInd)
    tFileName = '{}/t{}.csv'.format(direc,endString)



    # read the data file
    df = pd.read_csv(dataFileName,header = None, nrows = eventLineInd+1)

    # # read the time data file
    tdf = pd.read_csv(tFileName,header = None)
    tme = tdf[0].values

    # if st == '1':
    #     tFileName = '{}/tPFORNLT.csv'.format(direc) # transient
    # elif st == '2':
    #     tFileName = '{}/tPFORNLS.csv'.format(direc) # steady
    # else:
    #     print('Invalid input')
    #     continue



    ########
    ## visualize a single voltage
    if eventFileInd != 0:
        print('Event file name: eventKeys_{}.csv'.format(eventFileInd))
        print('Event line index: {}'.format(eventLineInd))

        # using the indices above, go to the specific file and the specific line index

        # get the file

        #df = pd.read_csv('{}/vPFORNLTransient{}.csv'.format(eventFileInd),header = None,nrows = eventLineInd + 10) # transient
        #df = pd.read_csv('{}/vPFORNLSteady{}.csv'.format(eventFileInd),header = None,nrows = eventLineInd + 10) # steady

        # get the line
        data = df.iloc[eventLineInd].values
        # get the plot
        plt.plot(tme,data)
        #plt.plot(data)
        #plt.title('Bus 152 voltage plot')
        plt.xlabel('t (s)')
        plt.ylabel('V (pu)')
        if vaf == '1':
            plt.ylim(-0.1, 1.2)
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
#     df = pd.read_csv('{}/a{}.csv'.format(direc,eventFileInd),header = None,nrows = eventLineInd + 10)
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








