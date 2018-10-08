# example of how to use hilbert transform to get the envelope of an oscillating signal

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, chirp
import pickle

# Functions
def load_obj(name ):
    # load pickle object
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


VoltageDataDict = load_obj('VoltageData') # this voltage data dictionary has been generated from the script 'TS3phSimN_2FaultDataSave.py', see it for key format
tme = VoltageDataDict['time']
timestep = tme[1] - tme[0]
#key = '154,205,1;3001,3003,1;F3003/154'
key = '151,201,1;151,152,2;F152/201'
voltage = VoltageDataDict[key]
time_2s = int(2.0/timestep)
#voltage = voltage[time_2s:]
voltage -=1 # cancel the offset
#tme = tme[time_2s:]


analytic_signal = hilbert(voltage)
amplitude_envelope = np.abs(analytic_signal)
#test = np.real(analytic_signal)
#test = np.imag(analytic_signal)
plt.plot(tme, voltage, label = 'voltage')
plt.plot(tme, amplitude_envelope, label='envelope')
#plt.plot(tme, analytic_signal, label='envelope')
#plt.plot(tme, test)
plt.legend()
plt.grid()
plt.show()



"""
duration = 1.0
fs = 400.0
samples = int(fs*duration)
t = np.arange(samples) / fs


signal = chirp(t, 20.0, t[-1], 100.0)
signal *= (1.0 + 0.5 * np.sin(2.0*np.pi*3.0*t) )


analytic_signal = hilbert(signal)
amplitude_envelope = np.abs(analytic_signal)
instantaneous_phase = np.unwrap(np.angle(analytic_signal))
instantaneous_frequency = (np.diff(instantaneous_phase)/(2.0*np.pi) * fs)


fig = plt.figure()
ax0 = fig.add_subplot(211)
ax0.plot(t, signal, label='signal')
ax0.plot(t, amplitude_envelope, label='envelope')
ax0.set_xlabel("time in seconds")
ax0.legend()
ax1 = fig.add_subplot(212)
ax1.plot(t[1:], instantaneous_frequency)
ax1.set_xlabel("time in seconds")
ax1.set_ylim(0.0, 120.0)
plt.show()
"""

