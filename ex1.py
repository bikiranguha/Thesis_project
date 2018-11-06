# summarize simple information about user movement data
# Just copy-pasting the code from here:
# https://machinelearningmastery.com/indoor-movement-time-series-classification-with-machine-learning-algorithms/
from os import listdir
from numpy import array
from numpy import vstack
from numpy.linalg import lstsq
from pandas import read_csv
from matplotlib import pyplot
from numpy import mean
from numpy import std
from sklearn.model_selection import cross_val_score

# return list of traces, and arrays for targets, groups and paths
def load_dataset(prefix=''):
    grps_dir, data_dir = prefix+'groups/', prefix+'dataset/'
    # load mapping files
    targets = read_csv(data_dir + 'MovementAAL_target.csv', header=0)
    groups = read_csv(grps_dir + 'MovementAAL_DatasetGroup.csv', header=0)
    paths = read_csv(grps_dir + 'MovementAAL_Paths.csv', header=0)
    # load traces
    sequences = list()
    target_mapping = None
    for name in listdir(data_dir):
        filename = data_dir + name
        if filename.endswith('_target.csv'):
            continue
        df = read_csv(filename, header=0)
        values = df.values
        sequences.append(values)
    return sequences, targets.values[:,1], groups.values[:,1], paths.values[:,1]

# load dataset
sequences, targets, groups, paths = load_dataset()
# summarize class breakdown
class1,class2 = len(targets[targets==-1]), len(targets[targets==1])
print('Class=-1: %d %.3f%%' % (class1, class1/len(targets)*100))
print('Class=+1: %d %.3f%%' % (class2, class2/len(targets)*100))

"""
# histogram for each anchor point
all_rows = vstack(sequences)
pyplot.figure()
variables = [0, 1, 2, 3]
for v in variables:
    pyplot.subplot(len(variables), 1, v+1)
    pyplot.hist(all_rows[:, v], bins=20)
pyplot.show()
# histogram for trace lengths
trace_lengths = [len(x) for x in sequences]
pyplot.hist(trace_lengths, bins=50)
pyplot.show()
"""


# group sequences by paths
paths = [1,2,3,4,5,6]
seq_paths = dict()
for path in paths:
    seq_paths[path] = [sequences[j] for j in range(len(paths)) if paths[j]==path]
# plot one example of a trace for each path
pyplot.figure()
for i in paths:
    pyplot.subplot(len(paths), 1, i)
    # line plot each variable
    for j in [0, 1, 2, 3]:
        pyplot.plot(seq_paths[i][0][:, j], label='Anchor ' + str(j+1))
    pyplot.title('Path ' + str(i), y=0, loc='left')
pyplot.show()


# fit a linear regression function and return the predicted values for the series
def regress(y):
    # define input as the time step
    X = array([i for i in range(len(y))]).reshape(len(y), 1)
    # fit linear regression via least squares
    b = lstsq(X, y)[0][0]
    # predict trend on time step
    yhat = b * X[:,0]
    return yhat

# plot series for a single trace with trend
seq = sequences[0]
variables = [0, 1, 2, 3]
pyplot.figure()
for i in variables:
    pyplot.subplot(len(variables), 1, i+1)
    # plot the series
    pyplot.plot(seq[:,i])
    # plot the trend
    pyplot.plot(regress(seq[:,i]))
pyplot.show()


# separate traces
seq1 = [sequences[i] for i in range(len(groups)) if groups[i]==1]
seq2 = [sequences[i] for i in range(len(groups)) if groups[i]==2]
seq3 = [sequences[i] for i in range(len(groups)) if groups[i]==3]
print(len(seq1),len(seq2),len(seq3))
# separate target
targets1 = [targets[i] for i in range(len(groups)) if groups[i]==1]
targets2 = [targets[i] for i in range(len(groups)) if groups[i]==2]
targets3 = [targets[i] for i in range(len(groups)) if groups[i]==3]
print(len(targets1),len(targets2),len(targets3))



# create a fixed 1d vector for each trace with output variable
def create_dataset(sequences, targets):
    # create the transformed dataset
    transformed = list()
    n_vars = 4
    n_steps = 19
    # process each trace in turn
    for i in range(len(sequences)):
        seq = sequences[i]
        vector = list()
        # last n observations
        for row in range(1, n_steps+1):
            for col in range(n_vars):
                vector.append(seq[-row, col])
        # add output
        vector.append(targets[i])
        # store
        transformed.append(vector)
    # prepare array
    transformed = array(transformed)
    transformed = transformed.astype('float32')
    return transformed


# create ES1 dataset
es1 = create_dataset(seq1+seq2, targets1+targets2)
print('ES1: %s' % str(es1.shape))
savetxt('es1.csv', es1, delimiter=',')
# create ES2 dataset
es2_train = create_dataset(seq1+seq2, targets1+targets2)
es2_test = create_dataset(seq3, targets3)
print('ES2 Train: %s' % str(es2_train.shape))
print('ES2 Test: %s' % str(es2_test.shape))
savetxt('es2_train.csv', es2_train, delimiter=',')
savetxt('es2_test.csv', es2_test, delimiter=',')

"""
# evaluate model for ES1
scores = cross_val_score(model, X, y, scoring='accuracy', cv=5, n_jobs=-1)
m, s = mean(scores), std(scores)
"""