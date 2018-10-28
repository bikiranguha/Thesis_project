# just read the file and solve power flow
# Please select the working directory

# System imports
import os,sys


from os import listdir
from os.path import isfile, join
sys.path.append(r"C:\Program Files (x86)\PTI\PSSE33\PSSBIN")
os.environ['PATH'] = (r"C:\Program Files (x86)\PTI\PSSE33\PSSBIN;"
                       + os.environ['PATH'])

# Select working path ##########################################################
os.chdir(r"C:\Users\bikiran_remote\Desktop")
################################################################################
# Local imports
import redirect
import psspy
import dyntools
import csv



pf_options = [
    0,  #disable taps
    0,  #disable area exchange
    0,  #disable phase-shift
    1,  #enable dc-tap
    1,  #enable switched shunts
    0,  #do not flat start
    0,  #apply var limits immediately
    0,  #disable non-div solution
]



# Inputs and outputs
filename = 'savnw.raw' # raw file
#raw_new = 'tmpv2_island.raw' # new raw file created after disconnecting all the islands
psse_log = 'output' + filename.replace('.raw','.txt')
######
redirect.psse2py()
psspy.psseinit(buses=80000)
# Redirect any psse outputs to psse_log
psspy.report_output(2,psse_log,[0,0])
psspy.progress_output(2,psse_log,[0,0]) #ignored
psspy.alert_output(2,psse_log,[0,0]) #ignored
psspy.prompt_output(2,psse_log,[0,0]) #ignored
##############################
_i=psspy.getdefaultint()
_f=psspy.getdefaultreal()
_s=psspy.getdefaultchar()

ierr = psspy.read(0, filename)

# Load conversion (multiple-step)
psspy.conl(_i,_i,1,[0,_i],[_f,_f,_f,_f])
psspy.conl(1,1,2,[_i,_i],[0.0, 0.0,100.0, 100.0])
psspy.conl(_i,_i,3,[_i,_i],[_f,_f,_f,_f])



ierr = psspy.rawd_2(0,1,[1,1,1,0,0,0,0],0,'savnw_exp.raw')
#ierr = psspy.fnsl(pf_options) # solve power flow