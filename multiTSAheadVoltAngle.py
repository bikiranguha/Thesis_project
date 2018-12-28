import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from math import sqrt
from keras.models import model_from_json

#############
# Functions


# split a univariate dataset into train/test sets
def split_dataset(data,testStartInd,numTimeSteps):
    # testStartInd: sample number from which test data starts
    # number of timesteps in each train/test sample fed into the predictor
    train, test = data[:testStartInd], data[testStartInd:-numTimeSteps]
    # restructure into windows with the number of timesteps specified
    train = np.array(np.split(train, len(train)/numTimeSteps))
    test = np.array(np.split(test, len(test)/numTimeSteps))
    return train, test



# convert history into inputs and outputs
def to_supervised(train, n_input, n_output,tsAtaTime):
    # flatten data
    data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
    X, y = list(), list()
    in_start = 0
    # step over the entire history tsAtaTime at a time
    for _ in range(int(len(data)/tsAtaTime)):
        # define the end of the input sequence
        in_end = in_start + n_input
        out_end = in_end + n_output
        # ensure we have enough data for this instance
        if out_end < len(data):
            x_input = data[in_start:in_end, 0]
            x_input = x_input.reshape((len(x_input), 1))
            X.append(x_input)
            y.append(data[in_end:out_end, 0])
        # move along the specified number of time steps
        in_start += tsAtaTime
    return np.array(X), np.array(y)


# train the model
def build_model(train, n_input, n_output):
    # prepare data
    train_x, train_y = to_supervised(train, n_input, 1000, 10)
    # define parameters
    verbose, epochs, batch_size = 2, 70, 16
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    # define model
    model = Sequential()
    model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs))
    model.compile(loss='mse', optimizer='adam')
    # fit network
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return model



# evaluate a single model
def evaluate_model(train, test, n_input, n_output):
    # fit model
    model = build_model(train, n_input, n_output)
    # history is a list of weekly data
    history = [x for x in train]
    # walk-forward validation over each week
    predictions = list()
    for i in range(len(test)):
        # predict the week
        yhat_sequence = forecast(model, history, n_input)
        # store the predictions
        predictions.append(yhat_sequence)
        # get real observation and add to history for predicting the next week
        history.append(test[i, :])
    # evaluate predictions days for each week
    predictions = np.array(predictions)
    score, scores = evaluate_forecasts(test[:, :, 0], predictions)
    return score, scores



# make a forecast
def forecast(model, history, n_input):
    # flatten data
    data = np.array(history)
    data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
    # retrieve last observations for input data
    input_x = data[-n_input:, 0]
    # reshape into [1, n_input, 1]
    input_x = input_x.reshape((1, len(input_x), 1))
    # forecast the next week
    yhat = model.predict(input_x, verbose=0)
    # we only want the vector forecast
    yhat = yhat[0]
    return yhat

#######################



csv_file = '120103,010000000,UT,Austin,3378,Phasor.csv'
df = pd.read_csv(csv_file)

angle1 = np.array(df['UT 3 phase_VALPM_Angle'])
angle2 = np.array(df['McDonald 1P_V1LPM_Angle'])

relangle = abs(np.unwrap(angle1) - np.unwrap(angle2))
drelangle = np.gradient(relangle)

# plt.plot(np.unwrap(angle1))
# plt.show()

angle1_unwrapped = np.unwrap(angle1)
n_input = 100
n_output = 100
train, test = split_dataset(angle1_unwrapped,50000,n_input)
#train, test = split_dataset(drelangle,50000,n_input)

model_name = 'modelangunwrapped'





# 3d array constructed from 2d array, the third dimension representing the timestep
train = train.reshape((train.shape[0],train.shape[1],1))
#test = test.reshape((test.shape[0],test.shape[1],1))


# make more training data by shifting the original training data by a number of 
# time steps
train_x, train_y = to_supervised(train, n_input, n_output, 10)



## build the model
# define parameters
verbose, epochs, batch_size = 2, 100, 100
n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]

#print(n_timesteps,n_features,n_outputs)



# define model
model = Sequential()
#model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
#model.add(LSTM(1, activation='relu', input_shape=(n_timesteps, n_features)))
#model.add(LSTM(100, input_shape=(n_timesteps, n_features)))
model.add(LSTM(1000, input_shape=(n_timesteps, n_features)))
model.add(Dense(1000))
model.add(Dense(n_outputs))
model.compile(loss='mse', optimizer='adam')
early_stop = EarlyStopping(monitor='loss',patience = 3, verbose = 1)
# fit network
model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks = [early_stop])


# serialize model to JSON
model_json = model.to_json()
with open("{}.json".format(model_name), "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("{}.h5".format(model_name))
print("Saved model to disk")




 
#later...
 
# load json and create model
json_file = open('{}.json'.format(model_name), 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("{}.h5".format(model_name))
print("Loaded model from disk")










predList = []
actList = []
# get the forecast rmse
for i in range(len(test)-1):
    in_x = test[i].reshape((1,test[i].shape[0],1))
    pred_x = loaded_model.predict(in_x)
    out_x = test[i+1,:].reshape(1,-1)

    pred_x_vec = pred_x.reshape(-1)
    out_x_vec = out_x.reshape(-1)

    for i in range(len(pred_x_vec)):
        val = pred_x_vec[i]
        predList.append(val)
    for i in range(len(out_x_vec)):
        val = out_x_vec[i]
        actList.append(val)   

    #rmse = sqrt(mean_squared_error(out_x, pred_x))
    #print('For seq {}, rmse = {}'.format(i,rmse))



# Visualizing the performance
f, (ax1, ax2) = plt.subplots(2, 1)
ax1.set_title('Illustration of LSTM using 100 timesteps')
ax1.set_ylabel('Actual')
ax2.set_ylabel('Predicted')
ax2.set_xlabel('Sample No.')
#ax2.set_ylim(0,5)
ax1.plot(actList)
ax2.plot(predList)
ax1.grid(True)
ax2.grid(True)
#ax2.set_ylim(30,40)
#ax2.set_ylim(-0.5,1.5)
plt.show()



















# test_vec = test[4].reshape(-1)
# plt.plot(test_vec)
# plt.grid()
# plt.show()