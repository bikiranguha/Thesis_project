# link: https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
numpy.random.seed(7)


# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.data.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]


# create model
model = Sequential()
model.add(Dense(8, input_dim=8, activation='relu')) # add input layer: the first argument is the dimension of the output of this layer
model.add(Dense(8, activation='relu')) # hidden layer # relu output: 0 (if input <0), else: input
model.add(Dense(1, activation='sigmoid')) # output layer


# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, Y, epochs=150, batch_size=10)

# evaluate the model on the training dataset
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

