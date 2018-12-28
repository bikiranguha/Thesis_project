#Function which implements the PCA based event detection proposed in GFKA2015. Given x (time), y (signal) and the steady state time window, it returns a list of 
#time indexes when the difference between the predicted steady state signal and the actual signal deviates beyond a certain threshold.

import csv
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import random # to add some noise





def eventdetectPCA(x,y,steadyStart,steadyEnd,errorThreshold):
    # x and y could be lists or arrays, x: time, y: some signal
    # returns a list of time indices where the difference between the signal and the steady state prediction exceeds the error threshold





    xarr = np.array(x)
    yarr = np.array(y)

    # get the steady state x and y
    xsteady = xarr[steadyStart:steadyEnd]
    ysteady = yarr[steadyStart:steadyEnd]



    # get the means of the steady state
    xm = xsteady.mean()
    ym = ysteady.mean()


    # get the covariance matrix
    X = np.stack((xsteady,ysteady),axis=0)
    COVXY = np.cov(X)
    lmb, P = LA.eig(COVXY) # eig values and eig vectors

    maxlmbInd = list(lmb).index(max(lmb)) # get the index of the max eigenvalue
    Pmax = P[maxlmbInd] # get the corresponding eigen vector

    s = Pmax[1]/Pmax[0] # slope

    # get the predicted steady state y
    yPredSteady = [y[0]] # initial y value
    for i in range(1,xarr.shape[0]):
        yNext = s*(x[i]-xm) + ym
        yPredSteady.append(yNext)


    # compare the actual plots and find out the time when the plots differ

    # get the indices when the difference is greater than the max
    abnormalInd = []
    for i in range(y.shape[0]):
        diff = abs(yPredSteady[i] - y[i])

        if diff > errorThreshold:
            abnormalInd.append(i)

    abnormalTimeInd = [val for idx, val in enumerate(x) if idx in abnormalInd]
    
    return yPredSteady, abnormalTimeInd


# if __name__ == '__main__':
#     vFileName = 'fault3ph/vData3ph.csv' # csv file containing voltage data (different types of fault)
#     tFileName = 'fault3ph/tData3ph.csv' # csv file containing the time data
#     eventKeyFile = 'fault3ph/eventIDFile.txt'

#     vFile = open(vFileName,'rb')
#     tFile = open(tFileName,'rb')
#     readerV=csv.reader(vFile,quoting=csv.QUOTE_NONNUMERIC) # so that the entries are converted to floats
#     readerT = csv.reader(tFile,quoting=csv.QUOTE_NONNUMERIC)
#     tme = [row for idx, row in enumerate(readerT) if idx==0][0]
#     v = [row for idx, row in enumerate(readerV) if idx==0][0]
#     abnormalTimeInd = eventdetect(tme,v,0,10,0.05)
#     print abnormalTimeInd