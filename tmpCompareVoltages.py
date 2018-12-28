import pickle
import matplotlib.pyplot as plt
def load_obj(name ):
    # load pickle object
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)



v151 = load_obj('F151Vmag')
v152 = load_obj('F152Vmag')
tme = v151['time']

# plot the bus 151 and 152 voltage for fault at bus 151
plt.plot(tme,v151[151],label = '151')
plt.plot(tme,v151[152],label = '152')

plt.grid()
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('V (pu)')
plt.title('Fault at bus 151')
plt.show()
plt.close()

# plot the bus 151 and 152 voltage for fault at bus 152
plt.plot(tme,v152[151],label = '151')
plt.plot(tme,v152[152],label = '152')

plt.grid()
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('V (pu)')
plt.title('Fault at bus 152')
plt.show()
plt.close()




