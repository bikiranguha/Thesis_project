import os
import pickle

# get the ML model
currentdir = os.getcwd()
model_dir = currentdir +  '/MLModels'
#LRFile = model_dir + '/' +  'LR_modeldvdt.sav'
LRFile = model_dir + '/' +  'LR_modelV.sav'
LRModel = pickle.load(open(LRFile, 'rb'))

coeff_array = LRModel.coef_
print coeff_array
