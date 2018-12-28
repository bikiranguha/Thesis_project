# apply clustering algorithm on the generator angle data and see what happens

# analyze the steady state individual bus voltages for the N-2 plus fault cases
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from getBusDataFn import getBusData
from sklearn.preprocessing import normalize
# these files are generated from integrateN_2VA.py
aFileSteady = 'obj/aN_2FNewSteady.csv'

eventsFile = 'obj/eventN_2FNew.txt'
buslistfile = 'obj/buslistN_2F.csv'
relanglechangefile = 'AbnormalEventDataOsc.txt'


relanglechangedict = {}
with open(relanglechangefile,'r') as f:
    fileLines = f.read().split('\n')
    for line in fileLines[1:]:
        if line == '':
            continue
        words = line.split(':')
        relanglechangedict[words[0].strip()]  = float(words[1].strip())







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


print('Getting the steady state angles and generate targets...')
adfS  = pd.read_csv(aFileSteady,header = None)

targetArrayEventWise = adfS.values[:-1] # the last row is incomplete, so take it out

relAngleArray = [] # rows: events, columns: all the relative gen angles (wrt mean) during steady state
for i in range(targetArrayEventWise.shape[0]):
    currentEvent = targetArrayEventWise[i]
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
    meanAngles = np.mean(genAngleDataArray,axis=0)
    # form a vector containing the relative gen angles for each event
    tmpList = []
    for k in range(genAngleDataArray.shape[0]):
    
        relAngle = abs(meanAngles-genAngleDataArray[k]).reshape(1,-1)
        relAngle = normalize(relAngle)
        relAngle = relAngle.reshape(-1)
        tmpList.append(relAngle)
    tmpList = np.array(tmpList).reshape(-1)
    # append the list
    relAngleArray.append(tmpList)



relAngleArray = np.array(relAngleArray)




#############
# trying k-means
# print('Trying k-means...')
# from sklearn.cluster import KMeans
# wcss = []


# X = relAngleArray


# # Using the elbow method to find the optimal number of clusters
# for i in range(1,11): # to get wcss for 1 to 10 clusters
#     # init: method to initialize the centroids, n_init = Number of times the k-mean algorithm run with different centroid seeds
#     kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0) 
#     kmeans.fit(X)
#     wcss.append(kmeans.inertia_) # kmeans.inertia_ : computes wcss
    
# plt.plot(range(1,11),wcss)
# plt.title('Elbow Method')
# plt.xlabel('No. of clusters')
# plt.ylabel('WCSS')
# plt.grid()
# plt.show()

# print('Applying k-means using 2 clusters')

# # Applying k means with optimal (2) clusters
# kmeans = KMeans(n_clusters = 2, init = 'k-means++', max_iter = 300, n_init = 10) 
# y_kmeans = kmeans.fit_predict(X)



# class1Dict = {}
# class0Dict = {}
# for i in range(y_kmeans.shape[0]):
#     if y_kmeans[i] == 0:
#         #class0List.append(eventwiseList[i])
#         class0Dict[eventwiseList[i]] = relanglechangedict[eventwiseList[i]]
#     else:
#         class1Dict[eventwiseList[i]] = relanglechangedict[eventwiseList[i]]
#         #class1List.append(eventwiseList[i])


# with open('aggclustangle.txt','w') as f:
#     f.write('Class 1:')
#     f.write('\n')
#     for key, value in sorted(class1Dict.iteritems(), key=lambda (k,v): v, reverse = True): # descending order
#         line = '{}:{}'.format(key,value)
#         f.write(line)
#         f.write('\n')



#     f.write('Class 0:')
#     f.write('\n')
#     for key, value in sorted(class0Dict.iteritems(), key=lambda (k,v): v, reverse = True): # descending order
#         line = '{}:{}'.format(key,value)
#         f.write(line)
#         f.write('\n')

    


#################







########
print('Trying agglomerative clustering...')

# # Using the dendrogram to find the optimal number of clusters on the spectrum data
# import scipy.cluster.hierarchy as sch
# X = relAngleArray
# dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
# plt.title('Dendrogram')
# plt.xlabel('Relative angle data')
# plt.ylabel('Euclidean distances')
# plt.show()




# Fitting Hierarchical Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
X = relAngleArray
hc = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

class2Dict = {}
class1Dict = {}
class0Dict = {}
for i in range(y_hc.shape[0]):
    if y_hc[i] == 0:
        #class0List.append(eventwiseList[i])
        class0Dict[eventwiseList[i]] = relanglechangedict[eventwiseList[i]]
    elif y_hc[i] == 1:
        class1Dict[eventwiseList[i]] = relanglechangedict[eventwiseList[i]]
        #class1List.append(eventwiseList[i])
    else:
        class2Dict[eventwiseList[i]] = relanglechangedict[eventwiseList[i]]


with open('aggclustangle2.txt','w') as f:




    f.write('Class 0:')
    f.write('\n')
    for key, value in sorted(class0Dict.iteritems(), key=lambda (k,v): v, reverse = True): # descending order
        line = '{}:{}'.format(key,value)
        f.write(line)
        f.write('\n')

    f.write('Class 1:')
    f.write('\n')
    for key, value in sorted(class1Dict.iteritems(), key=lambda (k,v): v, reverse = True): # descending order
        line = '{}:{}'.format(key,value)
        f.write(line)
        f.write('\n')

    f.write('Class 2:')
    f.write('\n')
    for key, value in sorted(class2Dict.iteritems(), key=lambda (k,v): v, reverse = True): # descending order
        line = '{}:{}'.format(key,value)
        f.write(line)
        f.write('\n')
########
