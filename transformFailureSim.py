# sinusoidal oscillations within a steady signal
import numpy as np 
import matplotlib.pyplot as plt
import random
from avgFilterFn import avgFilter
totSamples = 10000
x = np.ones(totSamples)
x = x + np.random.normal(0,0.001,x.shape[0])
t= np.linspace(0,1,totSamples)
z = np.zeros(totSamples)

#plt.plot(t,x)
#plt.ylim(0,1.2)
#plt.show()
"""
# generate a short duration sine sample
Fs = 10 # sampling frquency
f = 1
sample = 50
s = np.arange(sample)
y = 0.2*np.sin(2 * np.pi * f * s / Fs)

# inset the sine waveforms here
z[50:100] = y


z[500:550] = y

a = x + z

plt.plot(a)
plt.show()
####
"""

"""
#### play around with tanh function
#pos = np.linspace(0,1,50)
samples = 1000
minval = -0.2
pos = np.linspace(minval,0,samples)
neg = np.linspace(0,minval,samples)
tanhpos = np.tanh(pos)
tanhneg = np.tanh(neg)
minusones = tanhneg[-1]*np.ones(2000)
dip = np.concatenate((tanhneg,minusones,tanhpos))
dip = 0.1*dip + np.random.normal(0,0.001,dip.shape[0])
#plt.plot(dip)
#plt.show()
z[5000:5000+dip.shape[0]] = dip
a = x + z
plt.plot(a)
plt.show()
####
"""

"""
#### play with exponential functions
transitionsamples = 1000
minval = -0.5
down = np.linspace(0,minval,transitionsamples)
up = np.linspace(minval,0,transitionsamples)
drop = np.exp(down)
rise = np.exp(up)
# for minval in np.linspace(-1,-10,10):
#     down = np.linspace(0,minval,transitionsamples)
#     up = np.linspace(minval,0,transitionsamples)
#     drop = np.exp(down)
#     rise = np.exp(up)
#     dip = np.concatenate((drop,rise))
#     plt.plot(dip)
# plt.show()
minusones = drop[-1]*np.ones(3000)
dip = np.concatenate((drop,minusones,rise))
dip = 0.5*(dip + np.random.normal(0,0.001,dip.shape[0])-1)
z[5000:5000+dip.shape[0]] = dip
#plt.plot(dip)
#plt.show()
a = x + z
plt.plot(a)
plt.show()
########
"""


# generating a signal using multiple sine-waves
numSineWaves = 100
suddendrop = 0.01
dropVariance = 0.125 # ratio of variance wrt drop
frequencyRange = np.linspace(1,5,10)
distSamples = 1000
dist = np.zeros(distSamples)
dist = dist - suddendrop 
wave = np.zeros(distSamples)
for i in range(numSineWaves):

    freq = random.choice(frequencyRange)
    Fs = distSamples # sampling frquency
    #sample = 50
    s = np.arange(distSamples)
    y = np.sin(2 * np.pi * freq * s / Fs)
    wave += y

waveRange = wave.max() - wave.min()
wave = wave/waveRange*suddendrop*dropVariance
dist += wave
# plt.plot(signal/signal.max()*0.2)
# plt.grid()
# plt.show()

z[5000:5000+dist.shape[0]] = dist

a = x + z
a  =np.array(avgFilter(a,100))
a += np.random.normal(0,0.001,a.shape[0])
plt.plot(a)
plt.ylim(0.75,1.1)
plt.grid()
plt.show()

