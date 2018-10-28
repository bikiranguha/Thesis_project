# generate a time series ( yt = alpha + beta*yt-1 + noise) and visualize it
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
seed =2016
np. random.seed(seed)
y_0 = 1 # initial value
alpha = -0.25
beta =0.95
y=pd.Series(y_0) # initialize the time series with y0
num =1000

# iteratively generate the time series
for i in range(num):
	yt = alpha + (beta*y_0) + np.random.uniform(-1,1) # time series equation
	y = y.set_value(i,yt)
	y_0 = yt

y.plot()
plt.show()