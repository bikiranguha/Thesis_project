# copy all the scripts in the current folder and the sub-folder to the destination folder

import os
import shutil

#dest = 'C:/Users/Bikiran/Documents/Git repositories/Bus Map/Scripts' # destination folder
#dest = 'C:/Users/Bikiran/Documents/Git repositories/Running TS3ph'
dest = 'C:/Users/Bikiran/Documents/Git repositories/Thesis project'


# for copying all scripts in folders and subfolders
#shutil.copyfile(AllMapFileNew,os.path.join(dest, name))
for root, dirs, files in os.walk(".", topdown=True):

   for name in files:
   	if name.endswith('.py'):
   		#print name
   		#os.remove(os.path.join(dest, name))
   		abs_path = os.path.join(root, name) # path of file to be copied
   		shutil.copy(abs_path,dest)

# also copy the current readme file
currentDir = os.getcwd()
readmeFile = currentDir + '/Readme.md'
shutil.copy(readmeFile,dest)




