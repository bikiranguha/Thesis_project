import pickle
import matplotlib.pyplot as plt
import h5py
def load_obj(name):
    # load pickle object
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)



event = '100/201,202,1;151,201,1/F151'
eventWords = event.split('/')
rawFileIndicator = eventWords[0].strip()
if rawFileIndicator == '100':
    fileName = 'obj/savnw_conp.h5'
else:
    fileName = 'obj/savnw_conp{}.h5'.format(rawFileIndicator)

fileName = 'obj/savnw_conp.h5'
h5f = h5py.File(fileName,'r')
v = h5f['volt'][:]
a  = h5f['angle'][:]
f = h5f['freq'][:]
tme = h5f['time'][:]
h5f.close()
if rawFileIndicator == '100':
    keyFileName = 'events_savnw_conp'
else:
    keyFileName = 'events_savnw_conp{}'.format(rawFileIndicator)
keyList = load_obj(keyFileName)

# print an event

bus = '151'
key = event + '/' + bus
keyIndex = keyList.index(key)
s = f[keyIndex]
plt.plot(tme,s)
plt.show()
