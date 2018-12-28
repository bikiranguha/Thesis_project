import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# get the header values
headerfile = pd.read_csv('header.csv')
column_names =[]
for h in list(headerfile.columns):
    column_names.append(h.strip("'"))
    #print(h)


filename = 'Data141123.csv'

testcomedata = pd.read_csv(filename,names = column_names)

angle = testcomedata['VAVPM_Angle'].values
angle = np.unwrap(angle)
dangle = np.gradient(angle)
t = testcomedata.Seconds.values/3600

plt.plot(t,dangle)
plt.grid()
plt.show()