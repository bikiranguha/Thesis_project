# converts constant power loads to constant impedance loads
import sys,os

# add psspy to the system path
sys.path.append(r"C:\Program Files (x86)\PTI\PSSE33\PSSBIN")
os.environ['PATH'] = (r"C:\Program Files (x86)\PTI\PSSE33\PSSBIN;"
                       + os.environ['PATH'])


# Local imports
import redirect
import psspy
# create constant impedance raw directory
currentdir = os.getcwd()
obj_dir = currentdir +  '/conZRaw'
if not os.path.isdir(obj_dir):
    os.mkdir(obj_dir)
###########

# get the list of raw files to be considered for the simulation
fileList = os.listdir('.')
RawFileList = []
for file in fileList:
    if file.endswith('.raw') and 'savnw_conp' in file:
        #print file
        RawFileList.append(file)




##### Get everything set up on the PSSE side
redirect.psse2py()
psspy.psseinit(buses=80000)
_i=psspy.getdefaultint()
_f=psspy.getdefaultreal()
_s=psspy.getdefaultchar()


psse_log = 'log.log'
# Redirect any psse outputs to psse_log
psspy.report_output(2,psse_log,[0,0])
psspy.progress_output(2,psse_log,[0,0]) #ignored
psspy.alert_output(2,psse_log,[0,0]) #ignored
psspy.prompt_output(2,psse_log,[0,0]) #ignored

for currentRawFile in RawFileList:
	ierr = psspy.read(0, currentRawFile)
	rawFileName = currentRawFile.replace('.raw','')
	if rawFileName == 'savnw_conp':
		PL = ''
	else:
		PL = rawFileName[-3:] # last 3 character contain the percentage loading

	# Load conversion (multiple-step)
	# all constant power load converted to constant admittance load
	psspy.conl(0,1,1,[0,0],[0.0, 100.0,0.0, 100.0])
	psspy.conl(0,1,2,[0,0],[0.0, 100.0,0.0, 100.0])
	psspy.conl(0,1,3,[0,0],[0.0, 100.0,0.0, 100.0])

	newRawFileName = 'conZRaw/savnw_conz{}.raw'.format(PL)
	ierr = psspy.rawd_2(0,1,[1,1,1,0,0,0,0],0,newRawFileName)