# function to generate a plot based on the event key and the data dictionary 
# the output directory (direc) as well the figure name (figName) needs to be specified
# alongwith the steady state samples analyzed to determine class


def plotCase(dataDict,eventKey, direc, figName, steadyStateSamples):


    import matplotlib.pyplot as plt
    import pickle
    # make directories
    import os


    # create directories if they dont exist
    if not os.path.isdir(direc):
        os.mkdir(direc)



    voltage = dataDict[eventKey]
    tme = dataDict['time']
    ind_target_start = len(tme) - steadyStateSamples
    spclTimePts.append(tme[ind_target_start])
    spclVoltPts.append(voltage[ind_target_start])

    plt.plot(tme, voltage)
    plt.plot(spclTimePts,spclVoltPts, ls="", marker="o", label="special points")
    titleStr = eventKey
    plt.title(titleStr)
    plt.ylabel('Voltage (pu)')
    plt.xlabel('Time (s)')
    plt.ticklabel_format(useOffset=False)
    #plt.xlabel('Time step after line clearance')
    #plt.ylim(0.75,1.5)
    plt.grid()
    plt.legend()
    plt.savefig(figName)
    plt.close()


