# Load interactive figure
import pickle
# Load figure from disk and display
filename = 'Plot3DAccuracy'

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
fig = load_obj(filename)
fig.show()
