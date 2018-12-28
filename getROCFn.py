# Function to get the rate of change of a signal
def getROC(data,x,absolute=True):
    import numpy as np
    # calculations for dv_dt
    dataSize = data.shape[0]
    xstep = x[1] - x[0]
    ROC = np.zeros(dataSize) # initialize dv_dt array with all zeros
    for i in range(dataSize):
        try:
            if absolute == True:
                ROC[i] = abs((data[i] - data[i-1])/xstep)
            else:
                ROC[i] = (data[i] - data[i-1])/xstep
        except: # will happen if i = 0, since there is no i-1
            continue
    return ROC