# build a regressor which can output the transient stability index by looking at all the generator 
# angles after a fault
# tries various regressors



# analyze the steady state angle for the N-2 plus fault cases in the savnw case
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from getBusDataFn import getBusData
# these files are generated from getVAST.py
aFileSteady = 'obj/aN_2FSNGT.csv' # contains the steady state angle data, eventwise
aFileInput = 'obj/aN_2FTNGT.csv' # contains the transient angle data, eventwise
eventsFile = 'obj/eventN_2FNGT.txt' # contains the event index
buslistfile = 'obj/buslistN_2FNGT.csv' # contains the bus order within each row

# get the event list
eventwiseList = []
with open(eventsFile,'r') as f:
    fileLines = f.read().split('\n')
    for line in fileLines:
        if line == '':
            continue
        eventwiseList.append(line.strip())

# get the bus list
with open(buslistfile,'r') as f:
    filecontent = f.read().strip()
    buslist = filecontent.split(',')


# get a buswise list
buswiselist = []
for event in eventwiseList:
    for bus in buslist:
        currentstr = '{}/{}'.format(event,bus)
        buswiselist.append(currentstr)

refRaw = 'savnw.raw'
busdatadict = getBusData(refRaw)


# get the inputs
print('Getting the inputs...')
originalColumnSize = 23*120 # 120 timesteps for each of the 23 buses in the savnw case

adfT  = pd.read_csv(aFileInput,header = None)

inputArrayEventWise = adfT.values[:-1] # the last row is incomplete, so take it out


##### only get the rows which correspond to generators
inputArrayGenOnly = []  # each row for an event, where only the gen angles are present
for i in range(len(inputArrayEventWise)):
    currentEvent = inputArrayEventWise[i]
    currentEventArray = currentEvent.reshape(-1,120)
    genAngleDataList = []
    for j in range(currentEventArray.shape[0]):
        bus = buslist[j]
        bustype = busdatadict[bus].type
        if bustype == '2' or bustype == '3':
            genAngleDataList.append(currentEventArray[j])
    genAngleDataArray = np.array(genAngleDataList)
    # flatten array, so that each row represents an event
    genAngleDataArray = genAngleDataArray.reshape(-1)
    inputArrayGenOnly.append(genAngleDataArray)

inputArrayGenOnly = np.array(inputArrayGenOnly)
#####







# if cropping is needed
#inputArrayA = inputArrayEventWise
#inputArrayA = inputArrayEventWise.reshape(-1,120) # reshape so that each row contains only one bus angle
#inputArrayA = inputArrayA[:,:60]

#### Visualizing the covariance matrix
## get the covariance matrix of the input matrix
#import time
#
##start = time.time()
##df = pd.DataFrame(inputArrayEventWise)
##CorrMat = df.corr()
##end = time.time()
##print('Using pandas: {}'.format(end-start))
#
## get the covariance matrix using numpy
## the numpy way is much faster than pandas
#start = time.time()
#CorrMatNumpy = np.corrcoef(inputArrayEventWise,rowvar = False)
##CorrMatNumpy = np.cov(inputArrayEventWise)
#end = time.time()
#print('Using numpy: {}'.format(end-start))
#
#
###### trying to calculate by hand
### comment: takes way too long
##start = time.time()
##X = inputArrayEventWise
##COVX = np.zeros((X.shape[1],X.shape[1]))
##for i in range(X.shape[1]):
##    for j in range(X.shape[1]):
##        Xi = X[:,i]
##        Xj = X[:,j]
##        Xi_m = Xi.mean()
##        Xj_m = Xj.mean()
##        
##        Xi_bar = np.ones(Xi.shape)*Xi_m
##        Xj_bar = np.ones(Xj.shape)*Xj_m
##        s= np.sum((Xi-Xi_bar)*(Xj-Xj_bar))
##        COVX[i,j] =  s/len(Xi)
##end = time.time()
##print('Using hand: {}'.format(end-start))
######
#
######







# #### get the target voltage array

print('Getting the steady state angles and generate targets...')
adfS  = pd.read_csv(aFileSteady,header = None)

targetArrayEventWise = adfS.values[:-1] # the last row is incomplete, so take it out
#aDataBusWise = targetArrayEventWise.reshape(-1,120)

TSIDict = {}
TSIList = [] # this is the target array
# analyze the steady state angle data and generate TSI index
for i in range(targetArrayEventWise.shape[0]):
    currentEvent = targetArrayEventWise[i]
    currentEventID = eventwiseList[i]

    currentEventBusWise = currentEvent.reshape(-1,120)
    event = eventwiseList[i]
    # isolate the gen data
    genAngleDataList = []
    for j in range(currentEventBusWise.shape[0]):
        bus = buslist[j]
        bustype = busdatadict[bus].type
        if bustype == '2' or bustype == '3':
            genAngleDataList.append(currentEventBusWise[j])
    genAngleDataArray = np.array(genAngleDataList)
    # calculate TSI
    deltaMax = genAngleDataArray[:,-1].max() - genAngleDataArray[:,-1].min() # max angle difference between any 2 gen
    TSI = (360-deltaMax)/(360+deltaMax)

    TSIDict[currentEventID] = TSI
    TSIList.append(TSI)

X = inputArrayEventWise # contains all the bus angles
#X = inputArrayGenOnly # contains only gen bus angles
Y = np.array(TSIList)
    

# # save the sorted dict to a text file
# sortedTSIFile = 'TSISorted.txt'
# with open(sortedTSIFile,'w') as f:
#     for currentEventID, TSI in sorted(TSIDict.iteritems(), key=lambda:(k,v): v):
#         line = '{}:{}'.format(currentEventID,TSI)
#         f.write(line)
#         f.write('\n')


#### Performance Analysis Section


# import the required modules for performance analysis
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error
from math import sqrt


# split into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
# lists for boxplots of mse
results_mse = []
names = []


##### apply dimensionality reduction



# # split into training and test sets
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
# # apply PCA reduction to X and see the results
# # Applying PCA
# from sklearn.decomposition import PCA

# #pca = PCA(n_components = None) # use this to see a ranked variance vector using  'explained_variance'
# pca = PCA(n_components = 100) # determine this by looking at explained_variance with PCA(n_components = None)
# X_train = pca.fit_transform(X_train)
# X_test = pca.transform(X_test)
# explained_variance = pca.explained_variance_ratio_
# #############












############ try different regression algorithms on the data

# append a linear regression model





models = []






### Testing Linear Regression
print('Testing various classifiers...')


#### uncomment if you dont want dimensionality reduction
## split into training and test sets
#X = inputArrayEventWise
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
####

from sklearn.linear_model import LinearRegression
LRModel = LinearRegression()

from sklearn.ensemble import RandomForestRegressor
DTmodel = RandomForestRegressor(n_estimators = 10)

# add the different models
#models.append(('LinearReg',LRModel))
models.append(('DT',DTmodel))
# evaluate each model in turn


for name, model in models:

    results = []
    names.append(name)
    print('Model: {}'.format(name))
    # using rmse
    #names.append(name)
    #kfold = StratifiedKFold(n_splits=3, shuffle=True) # for classification
    kfold = KFold(n_splits=3, shuffle=True) # for regression

    for train, test in kfold.split(X_train, Y_train):
        
        model.fit(X_train[train],Y_train[train])
        y_pred = model.predict(X_train[test])

        # calculate mse
        mse = mean_squared_error(Y_train[test], y_pred)
        # calculate rmse
        rmse = sqrt(mse)
        results.append(rmse)


    # print stats about rmse
    print('Results on cross-val set:')
    results_mse.append(results)

    results = np.array(results)
    print('Mean RMSE: {}'.format(results.mean()))
    print('RMSE standard deviation: {}'.format(results.std()))


    y_pred_test = model.predict(X_test)
    # calculate mse
    mse = mean_squared_error(Y_test, y_pred_test)
    # calculate rmse
    rmse = sqrt(mse)
    #rmse = np.array(rmse)
    print('RMSE on test set: {}'.format(rmse))


    # scatter plot of a certain test dataset
    
    
    #y_pred_test = model.predict(X_test)
    plt.scatter(range(len(Y_test)),Y_test,label='actual')
    plt.scatter(range(len(y_pred_test)),y_pred_test,label='predicted')
    plt.xlabel('Samples')
    plt.ylabel('TSI')
    plt.title(name)
    plt.grid()
    plt.legend()
    plt.savefig('Perf{}.png'.format(name))
    plt.close()
    #plt.show()
    #plt.close()
###################









####### Testing SVR
from sklearn.model_selection import StratifiedKFold, KFold
print('Support Vector Regression Performance Test:')
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')

#### uncomment if you dont want dimensionality reduction
## split into training and test sets
#X = inputArrayEventWise
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
####



# #### Quick performance analysis
# X_sc = sc_X.fit_transform(X_train)
# Y_sc = sc_y.fit_transform(Y_train.reshape(-1,1))

# # Fitting SVR to the dataset

# #regressor.fit(X_sc, Y_sc)
# regressor.fit(X_sc, Y_sc.reshape(-1))  # regressor complains when a column vector is passed, so made it into 1D array

# # Predicting a new result
# y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(X_test)))

# # calculate mse
# mse = mean_squared_error(Y_test, y_pred.reshape(-1))
# # calculate rmse
# rmse = sqrt(mse)
# #rmse = np.array(rmse)
# print('RMSE on test set: {}'.format(rmse))
# ###


##### Detailed performance analysis
results = []
kfold = KFold(n_splits=3, shuffle=True) # for regression
names.append('SVR')
for train, test in kfold.split(X_train, Y_train):
    

    X_sc = sc_X.fit_transform(X_train[train])
    Y_sc = sc_y.fit_transform(Y_train[train].reshape(-1,1))


    regressor.fit(X_sc,Y_sc.reshape(-1))
    y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(X_train[test])))

    # calculate mse
    mse = mean_squared_error(Y_train[test], y_pred.reshape(-1))
    # calculate rmse
    rmse = sqrt(mse)
    results.append(rmse)


# print stats about rmse
print('Results on cross-val set:')
results_mse.append(results)
results = np.array(results)
print('Mean RMSE: {}'.format(results.mean()))
print('RMSE standard deviation: {}'.format(results.std()))


y_pred_test = sc_y.inverse_transform(regressor.predict(sc_X.transform(X_test)))
# calculate mse
mse = mean_squared_error(Y_test, y_pred_test)
# calculate rmse
rmse = sqrt(mse)
#rmse = np.array(rmse)
print('RMSE on test set: {}'.format(rmse))



# scatter plot of a certain test dataset


#y_pred_test = model.predict(X_test)
plt.scatter(range(len(Y_test)),Y_test,label='actual')
plt.scatter(range(len(y_pred_test)),y_pred_test,label='predicted')
plt.xlabel('Samples')
plt.ylabel('TSI')
plt.title('Support Vector Regression')
plt.grid()
plt.legend()
plt.savefig('PerfSVR.png')
plt.close()
#plt.show()
#plt.close()
#####



# boxplots of MSE of the various tested classifiers


plt.boxplot(results_mse)
plt.xticks(range(1,len(results_mse)+1), names)
plt.title('Mean Squared Error')
plt.xlabel('Classfier names')
plt.ylabel('MSE')
plt.grid()
plt.savefig('BoxPlotComparison.png')
#plt.show()
#plt.close()




