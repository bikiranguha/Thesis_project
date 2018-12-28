import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
csv_file = '120103,010000000,UT,Austin,3378,Phasor.csv'
df = pd.read_csv(csv_file)
sampleV = np.array(df.Austin_V1LPM_Magnitude) #you can also use df['Austin_V1LPM_Magnitude']

plt.plot(sampleV)
plt.grid()
plt.show()
