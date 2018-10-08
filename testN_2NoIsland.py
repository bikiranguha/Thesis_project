# on a given raw file, get all the N-2 line outage combos possible
# test these combos on a power flow and see which ones do not lead to any topology inconsistencies
# List these cases out, so that we can do some TS on these
import sys,os
# The following 2 lines need to be added before redirect and psspy
sys.path.append(r"C:\Program Files (x86)\PTI\PSSE33\PSSBIN")
os.environ['PATH'] = (r"C:\Program Files (x86)\PTI\PSSE33\PSSBIN;"
                       + os.environ['PATH'])
###########################################

import redirect
import psspy

raw = 'savnw_dy_sol_0905.raw'



nonIslandEvents = []
settings = {
# use the same raw data in PSS/E and TS3ph #####################################
    'filename':raw, #use the same raw data in PSS/E and TS3ph
################################################################################
    'dyr_file':'',
    'out_file':'output2.out',
    'pf_options':[
        0,  #disable taps
        0,  #disable area exchange
        0,  #disable phase-shift
        0,  #disable dc-tap
        0,  #disable switched shunts
        0,  #do not flat start
        0,  #apply var limits immediately
        0,  #disable non-div solution
    ]
}

psse_log = 'savnw.log'


redirect.psse2py()
psspy.psseinit(buses=80000)
psspy.report_output(2,psse_log,[0,0])
psspy.progress_output(2,psse_log,[0,0])
psspy.alert_output(2,psse_log,[0,0])
psspy.prompt_output(2,psse_log,[0,0])
_i=psspy.getdefaultint()
_f=psspy.getdefaultreal()
_s=psspy.getdefaultchar()
print "\n Reading raw file:",settings['filename']
ierr = psspy.read(0, settings['filename'])
# get a list of all the non-transformer branches
ierr, brnchs = psspy.abrnint(_i,_i,_i,_i,_i,['FROMNUMBER','TONUMBER']) # page 1789 of API book
ierr, carray = psspy.abrnchar(_i,_i,_i,_i,_i, ['ID']) # get the character ids (page 1798 of API book)
fromBusList = brnchs[0]
toBusList = brnchs[1]
cktIDList = carray[0]
#print '\n\n\n'
#print 'Branch1' + '\t' + 'Branch2' + '\t' + 'Non-convergence?'
# nested loop to simulate N-2 line outages
for i in range(len(toBusList)):
    # disconnect each branch
    
    fromBus1 = fromBusList[i]
    toBus1 = toBusList[i]
    cktID1 = cktIDList[i].strip("'").strip()

    branch1ID = str(fromBus1) + str(toBus1) + cktID1 
    
    for j in range(len(toBusList)):

        if i == j:
            continue
        fromBus2 = fromBusList[j]
        toBus2 = toBusList[j]
        cktID2 = cktIDList[j].strip("'").strip()      
        branch1ID = str(fromBus1) + ',' +  str(toBus1) + ',' + cktID1
        branch2ID = str(fromBus2) + ',' +  str(toBus2) + ',' + cktID2

        # read the original raw file and try to solve power flow
        ierr = psspy.read(0, settings['filename'])
        ierr = psspy.branch_chng(fromBus1,toBus1,cktID1,[0,_i,_i,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f]) # disconnect branch 1
        ierr = psspy.branch_chng(fromBus2,toBus2,cktID2,[0,_i,_i,_i,_i,_i],[_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f]) # disconnect branch 2
        ierr = psspy.fnsl(settings['pf_options'])
        converge =  psspy.solved()
        if converge != 9:
            string =  branch1ID + ';' + branch2ID
            nonIslandEvents.append(string)


with open('OKDoubleBranchOutages.txt','w') as f:
    for line in nonIslandEvents:
        f.write(line)
        f.write('\n')