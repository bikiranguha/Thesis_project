# generate the python scripts to generate chunks of PSSE simulations
from N_2Inputs import eventListFile
import numpy as np
sampleBatchFile = 'simPSSEBatchSample.py'
sampleBatchFile = 'TS3phSimN_2FNewSample.py' # can be used for parallel, not for sequential
#sampleBatchFile = 'TS3phSimN_2FNewSamplewProtection.py' # should be used with protection
##### Functions and classes
def assignOutFileName(currentRow,folderName):   
    # assign the processor number to the TS3ph output file

    for script in currentRow:
        # open the file and change the k value, which changes according to the processor it is assigned to
        fName = folderName +  '/' + script
        with open(fName,'r') as f:
            fileLines = f.read().split('\n')

        kInd = fileLines.index('k=')
        fileLines[kInd] = 'k={}'.format(j)

        with open(fName,'w') as f:
            for l in fileLines:
                f.write(l)
                f.write('\n')



def convertFileLinux(file):
    # function to convert file from crlf to lf (if needed)

    text = open(file, 'rb').read().replace('\r\n', '\n')
    open(file, 'wb').write(text)

########



eventsList = []
with open(eventListFile,'r') as f:
    fileLines = f.read().split('\n')
    for line in fileLines:
        if line == '':
            continue
        eventsList.append(line.strip())

num = len(eventsList)


with open(sampleBatchFile,'r') as f:
    fileLines = f.read().split('\n')

eventSpecInd = fileLines.index('simList = eventsList[x]')
fileInd = eventSpecInd + 1 # line where i is set
outInd = fileInd + 1 # line where k is set

# ###########
# # for PSSE
# startInd = 0
# endInd = 100
# idx = 1

# fileList = []



# while endInd <= num:

    
#     newFileLines = list(fileLines)
#     eventLineStr = 'simList = eventsList[{}:{}]'.format(startInd,endInd)
#     iStr = 'i = {}'.format(idx)
#     newFileLines[eventSpecInd] = eventLineStr
#     newFileLines[fileInd] = iStr

#     #fileName = 'PFORNLBatches/simPSSPFORNLBatch{}.py'.format(idx)
#     fileName = 'simPSSPFORNLBatch{}.py'.format(idx)
#     fileList.append(fileName)
#     with open(fileName,'w') as f:
#         for line in newFileLines:
#             f.write(line)
#             f.write('\n')
#     idx+=1

#     if endInd == num:
#         break

#     if (num - endInd) > 100:
#         startInd = endInd
#         endInd +=100
#     else:
#         startInd = endInd
#         endInd = num



# # write a batch file which will call all the python scripts
# with open('runPSSEBatches.bat','w') as f:
#     for fileName in fileList:
#         line = 'python {}'.format(fileName)
#         f.write(line)
#         f.write('\n')
# ###########



###########
# TS3ph parallel




# get the total number of batches
batchsize = 100 # the number of simulations each script will carry out
# num = 20 # the total number of events to simulate
if num%batchsize == 0:
    batchNum = num/batchsize # number of separate scripts that will be generated
else:
    batchNum = num/batchsize + 1



numCores = 10 # number of cores to use


# initialize
startInd = 0
endInd = batchsize
idx = 1

fileList = []

# generate the batch scripts, as well as .sh scripts for each processor
# and a main .sh script which will run all the .sh scripts in parallel
while endInd <= num:

    newFileLines = list(fileLines)
    eventLineStr = 'simList = eventsList[{}:{}]'.format(startInd,endInd)
    iStr = 'i = {}'.format(idx)
    newFileLines[eventSpecInd] = eventLineStr
    newFileLines[fileInd] = iStr

    #fileName = 'PFORNLBatches/simPSSPFORNLBatch{}.py'.format(idx)
    filePath = 'PFORNLTS3phScripts/simBatch{}.py'.format(idx)
    fileName = filePath.replace('PFORNLTS3phScripts/','')
    fileList.append(fileName)
    with open(filePath,'w') as f:
        for line in newFileLines:
            f.write(line)
            f.write('\n')
    idx+=1

    if endInd == num:
        break

    if (num - endInd) > batchsize:
        startInd = endInd
        endInd +=batchsize
    else:
        startInd = endInd
        endInd = num

fileListArray = np.array(fileList)
lastProcChunk = len(fileListArray)%numCores # remainder after reshaping

if lastProcChunk != 0:
    fileListArray = fileListArray[:-lastProcChunk].reshape(numCores,-1)
else:
    fileListArray = fileListArray.reshape(numCores,-1)



batchFileNames = []
for j in range(fileListArray.shape[0]):
    currentRow = list(fileListArray[j])
    currentFName = 'PFORNLTS3phScripts/runBatch{}.sh'.format(j)
    batchFileNames.append(currentFName.replace('PFORNLTS3phScripts/',''))
    assignOutFileName(currentRow,'PFORNLTS3phScripts')


    with open(currentFName.format(j),'w') as f:
        for script in currentRow[:-1]:
            line = 'python2.7 {}'.format(script)
            f.write(line)
            f.write('\n')
        f.write('python2.7 {}'.format(currentRow[-1]))



if lastProcChunk != 0:
    currentFName = 'PFORNLTS3phScripts/runBatch{}.sh'.format(numCores)
    currentRow = fileList[-lastProcChunk:]
    batchFileNames.append(currentFName.replace('PFORNLTS3phScripts/',''))
    assignOutFileName(currentRow,'PFORNLTS3phScripts')

    with open(currentFName,'w') as f:
        for script in currentRow:
            line = 'python2.7 {}'.format(script)
            f.write(line)
            f.write('\n')
        f.write('python2.7 {}'.format(currentRow[-1]))

with open('PFORNLTS3phScripts/mainBatch.sh','w') as f:
    # give execute permissions for the scripts
    for f1 in batchFileNames:
        string = 'chmod u+x {}'.format(f1)
        f.write(string)
        f.write('\n')



    # write command to run everything in parallel
    string =''
    for fl in batchFileNames:
        string += ' ./{} &'.format(fl)
    f.write(string[:-1])
    f.write('\n')

    # delete all the TS3ph output and output info file
    for c in range(numCores+1):
        f.write('rm TS3phoutput{}.out'.format(c))
        f.write('\n')
        f.write('rm TS3phoutput{}.out.info'.format(c))
        f.write('\n')


    # conversion to linux format needed for all .sh files
    for f1 in batchFileNames:
        convertFileLinux('PFORNLTS3phScripts/'+f1)



convertFileLinux('PFORNLTS3phScripts/mainBatch.sh')
######################










# ###########

# # TS3ph sequential (should be used with protection)
# # get the total number of batches
# batchsize = 100 # the number of simulations each script will carry out
# # num = 20 # the total number of events to simulate
# if num%batchsize == 0:
#     batchNum = num/batchsize # number of separate scripts that will be generated
# else:
#     batchNum = num/batchsize + 1



# #numCores = 4 # number of cores to use


# # initialize
# startInd = 0
# endInd = batchsize
# idx = 1

# outputdir = 'PFORNLTS3phScriptsSeq'

# fileList = []
# while endInd <= num:

#     newFileLines = list(fileLines)
#     eventLineStr = 'simList = eventsList[{}:{}]'.format(startInd,endInd)
#     iStr = 'i = {}'.format(idx)
#     kStr = 'k=0'
#     newFileLines[eventSpecInd] = eventLineStr
#     newFileLines[fileInd] = iStr
#     newFileLines[outInd] = kStr


#     #fileName = 'PFORNLBatches/simPSSPFORNLBatch{}.py'.format(idx)
#     filePath = '{}/simBatch{}.py'.format(outputdir,idx)
#     fileName = filePath.replace('{}/'.format(outputdir),'')
#     fileList.append(fileName)
#     with open(filePath,'w') as f:
#         for line in newFileLines:
#             f.write(line)
#             f.write('\n')
#     idx+=1

#     if endInd == num:
#         break

#     if (num - endInd) > batchsize:
#         startInd = endInd
#         endInd +=batchsize
#     else:
#         startInd = endInd
#         endInd = num

# bashScriptName = '{}/runAllScripts.sh'.format(outputdir)
# with open(bashScriptName,'w') as f:
#     for script in fileList[:-1]:
#         line = 'python2.7 {}'.format(script)
#         f.write(line)
#         f.write('\n')
#     # the last command should not have a newline after it
#     f.write('python2.7 {}'.format(fileList[-1]))

# convertFileLinux(bashScriptName)

# ######################