# get the voltage lists, convert them to array and save in an hdf file
import pickle
import h5py
def load_obj(name ):
    # load pickle object
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)




#load_obj('v0')
h5f = h5py.File('v.h5', 'w')
# save numpy array
for i in range(9):
    objName = 'v{}'.format(i)
    print 'Current obj: {}'.format(objName)
    v = load_obj(objName)
    
    h5f.create_dataset(objName, data=v)
    del v

# needs to be done, otherwise wont be able to read it
h5f.close()
