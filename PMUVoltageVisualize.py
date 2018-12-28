import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
csv_file = '120103,010000000,UT,Austin,3378,Phasor.csv'
df = pd.read_csv(csv_file)
v = np.array(df.Austin_V1LPM_Magnitude)
v = v/v.mean()
plt.plot(v)
plt.grid()
plt.ylim(0.7,1.1)
#plt.ylim(70000,80000)
plt.show()