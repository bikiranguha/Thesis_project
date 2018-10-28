# test whether everything is proper in the events data

import pickle
import matplotlib.pyplot as plt

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


EventDict = load_obj('EventData')

key = EventData.keys()[0]

Results = EventData[key]

vMag = Results['151'].volt
tme = Results['time']
freq = Results['151'].freq

plt.plot(tme,vMag, label = 'Voltage')
plt.plot(tme,freq, label = 'Frequency deviation')
plt.title('Bus 151')
plt.grid()
plt.show()

 


