# forecast y=x^2

import numpy as np
import pandas as pd
import random

random.seed(2016)
sample_size =50
sample = pd.Series(random.sample(range(-10000 ,10000),sample_size)) # generate 50 samples between +-10,000
x=sample/10000  # scale
y=x**2
"""
# print some values
print "First ten values of x"
print x.head(10)
print "First ten values of y"
print y.head(10)
print x.describe()
print y.describe()
"""
# organize the data to be used by the DNN
count = 0
dataSet = [([x.ix[count]],[y.ix[count]])] # list containing tuples of lists
count =1
while (count<sample_size ):
	#print " Working on data item : ",count
	dataSet = (dataSet +[([x.ix[count]] ,[y.ix[count]])])
	count = count + 1

# Get the neural network library
import neuralpy
fit = neuralpy.Network(1,3,7,1) # each argument is the number of neurons in the layer. The 1st one is the input, last one is the output
epochs = 100
learning_rate = 1
print " fitting model right now "
fit.train( dataSet , epochs , learning_rate )

# Assess model performance
count = 0
pred =[]
while ( count < sample_size ):
	out = fit.forward(x[count]) # the neural network ouput

	print(" Obs : ",count +1," y = ",round(y[ count ] ,4) ,
	 " prediction = ", round(pd.Series(out) ,4))
	pred.append(out)
	count = count + 1

