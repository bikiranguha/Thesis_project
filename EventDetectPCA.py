# pilot implementation and testing of the event detection strategy mentioned in GFKA2015

import csv
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import random # to add some noise

vFileName = 'fault3ph/vData3ph.csv' # csv file containing voltage data (different types of fault)
tFileName = 'fault3ph/tData3ph.csv' # csv file containing the time data
eventKeyFile = 'fault3ph/eventIDFile.txt'

vFile = open(vFileName,'rb')
tFile = open(tFileName,'rb')
readerV=csv.reader(vFile,quoting=csv.QUOTE_NONNUMERIC) # so that the entries are converted to floats
readerT = csv.reader(tFile,quoting=csv.QUOTE_NONNUMERIC)
tme = [row for idx, row in enumerate(readerT) if idx==0][0]
# test with the first element
v = [row for idx, row in enumerate(readerV) if idx==0][0]
vArray = np.array(v)
noise = []
for ind, val in enumerate(v):
    noise.append(random.gauss(0,0.003)) # mu, sigma

vNoisy = np.array(v) + np.array(noise)
#plt.hist(noise,bins='auto')
#plt.show()
# get the values before an event happens
faultOnTime = 0.1
faultOnInd = min([idx for idx,val in enumerate(tme) if val >= faultOnTime])
x = np.array(tme)
#xcropped = np.array(tme[:faultOnInd-1])
#y = np.array(v[:faultOnInd-1])

# use the first 10 cycles as the steady state representation
xcropped = np.array(tme[:10])
y = np.array(vNoisy[:10])
# get the means
xm = xcropped.mean()
ym = y.mean()


# get the covariance matrix
X = np.stack((xcropped,y),axis=0)
COVXY = np.cov(X)
lmb, P = LA.eig(COVXY) # eig values and eig vectors

maxlmbInd = list(lmb).index(max(lmb))
Pmax = P[maxlmbInd]

s = Pmax[1]/Pmax[0] # slope

# get the predicted steady state y
yPredSteady = [y[0]] # initial y value
for i in range(1,x.shape[0]):
    yNext = s*(x[i]-xm) + ym
    yPredSteady.append(yNext)

"""
#####
# See comparison
plt.plot(x,yPredSteady,label = 'Steady')
plt.plot(x,v,label = 'actual')
plt.grid()
plt.ylim(-0.5,1.5)
plt.legend()
plt.show()
######
"""

# compare the actual plots and find out the time when the plots differ

# get the max difference during confirmed steady state
steadydiff = []
for i in range(xcropped.shape[0]):
    steadydiff.append(abs(yPredSteady[i] - vNoisy[i]))

steadydiff = np.array(steadydiff)
steadydiffmax =  steadydiff.max()
errorMargin = 0.1
tolDiff = (1+errorMargin)*steadydiffmax
tolDiff = 0.1
# get the indices when the difference is greater than the max
abnormalInd = []
for i in range(len(v)):
    diff = abs(yPredSteady[i] - v[i])

    if diff > tolDiff:
        abnormalInd.append(i)

abnormalTimeInd = [val for idx, val in enumerate(tme) if idx in abnormalInd]
     


