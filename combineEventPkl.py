import dill as pickle
# Functions
def load_obj(name ):
    # load pickle object
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
def save_obj(obj, name ):
    # save as pickle object
    currentdir = os.getcwd()
    objDir = currentdir + '/obj'
    if not os.path.isdir(objDir):
        os.mkdir(objDir)
    with open(objDir+ '/' +  name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL,recurse = 'True')

combinedEventDict = {}


for i in range(9):

    objStr = 'Event{}'.format(i)
    EventDict = load_obj(objStr)

    for event in EventDict:
        combinedEventDict[event] = EventDict[event]


save_obj(combinedEventDict,'combinedEventData')