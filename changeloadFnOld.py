""" 
Script to generate new raw files with scaled up constant power loads and generation
Also change each generator by that percentage 
After generating new raw files, they are solved and corresponding sav files are also generated
Only works for PSSE version 33 raw file
This is meant to work with constant power loads


"""
import pdb
import os,sys
# The following 2 lines need to be added before redirect and psspy
sys.path.append(r"C:\Program Files (x86)\PTI\PSSE33\PSSBIN")
os.environ['PATH'] = (r"C:\Program Files (x86)\PTI\PSSE33\PSSBIN;"
                       + os.environ['PATH'])
###########################################
import redirect
import psspy
import dyntools
#pdb.set_trace()

#newdir = 'New_Raws_0824'





def changeLoad(raw,start,end,step,newdir):
	""" 
		New raw files are created for each percentage step in [start,end]. 
		The current step defines the percentage scaling up (or down) factor for load and generation
	"""


# create a new directory to put the files in
	currentdir = os.getcwd()
	
	if not os.path.exists(newdir):
		os.mkdir(newdir)
	output_dir = currentdir+'/'+newdir
	genDiscount = 1.0
	lossRatio = 0.0  # gen scale-up factor: (scalePercent + (scalePercent-100)*lossRatio)/100
############################################


	for scalePercent in range(start,end+step,step):
		scalePercent = float(scalePercent) # float is needed, otherwise 101/100 returns 1


		scalePercentInt = int(scalePercent) # integer value needed to append to filename
		scalePercentStr = str(scalePercentInt)


		# variables to store load data
		loadBusList = [] # list of load buses (string)
		loadPList = [] # list of Pload values (string)
		loadQList = [] # list of Qload values (string)
		loadPListInt = [] # list of Pload values (float)
		loadQListInt = [] # list of Qload values (float)
		#loadBusListInt = [] # list of load buses (int)

		# variables to store gen data
		genBusList = []
		#genBusListInt = []
		genPList = []
		genMVAList = []
		genMVAListInt = []
		genPListInt = []


        raw_name = raw.replace('.raw','')
        out_file = 'pf_ornl' + scalePercentStr + '.raw'  # output file
		out_path = output_dir+'/'+out_file
		impLoadBuses = [] # enter specified load buses to scale, if empty all loads are scaled
		incLoss = (scalePercent-100)*lossRatio  # Additional percentage increase in Pgen (to account for losses)
		#############################################


		#Read raw file
		with open(org_file,'r') as f:
			filecontent = f.read()
			filelines = filecontent.split('\n')

			## Get start and end indices of load and gen info
			#########################################
			loadStartIndex = filelines.index('0 / END OF BUS DATA, BEGIN LOAD DATA')+1
			loadEndIndex = filelines.index('0 / END OF LOAD DATA, BEGIN FIXED SHUNT DATA')

			genStartIndex = filelines.index('0 / END OF FIXED SHUNT DATA, BEGIN GENERATOR DATA')+1
			genEndIndex = filelines.index('0 / END OF GENERATOR DATA, BEGIN BRANCH DATA')
			##############################################################################

			# Extract load info
			for i in range(loadStartIndex,loadEndIndex):
				words = filelines[i].split(',')
				#lenLoadP = len(words[5])  # needed to align the updated values properly in the output file
				#lenloadQ = len(words[6])
				loadBusList.append(words[0].strip())
				loadPList.append(words[5].strip()) # adding P value (constant power)
				loadQList.append(words[6].strip()) # adding Q value (constant power)
                loadIPList.append(words[7].strip()) # constant current P
                loadIQList.append(words[7].strip()) # constant current Q
				loadZPList.append(words[9].strip()) # adding P value (constant admittance)
				loadZQList.append(words[10].strip()) # adding Q value (constant admittance)


			# Extract gen info
			for i in range(genStartIndex, genEndIndex):
				words = filelines[i].split(',')
				lenGenP = len(words[2]) 
				genBusList.append(words[0].strip())
				genPList.append(words[2].strip())
				genMVAList.append(words[8].strip())  # needed to make sure the new Pgen value does not exceed genMVA

		# for i in range(len(loadBusList)):
		# 	loadBusListInt.append(int(loadBusList[i]))
			


		# convert Pload and Qload values to float so they can be multiplied
		for i in range(len(loadPList)):
			loadPListInt.append(float(loadPList[i]))
			loadQListInt.append(float(loadQList[i]))

		# convert Pgen values to float so they can be multiplied
		for i in range(len(genPList)):
			genPListInt.append(float(genPList[i]))
			genMVAListInt.append(float(genMVAList[i]))




		# scale loads by specified percentage
		for i in range(len(loadPListInt)):
			#if len(impLoadBuses) ==0: # empty list means that all loads need to be scaled up
			loadPListInt[i] *= scalePercent/100
			loadQListInt[i] *= scalePercent/100
			# else:
			# 	if loadBusListInt[i] in impLoadBuses:
			# 		loadPListInt[i] *= scalePercent/100
			# 		loadQListInt[i] *= scalePercent/100


		# scale generators by specified percentage (plus incLoss) 
		# if scaled value is beyond genMVA, then set updated value to genMVA

		for i in range(len(genPListInt)):
			genNewP = genPListInt[i]*(genDiscount*scalePercent+incLoss)/100
			if genNewP <= genMVAListInt[i]:
				genPListInt[i] = genNewP
			else:
				genPListInt[i] = genMVAListInt[i]





		with open(out_path,'w') as f:
			# copy everything before load data
			for i in range(loadStartIndex):
				f.write(filelines[i])
				f.write('\n')

			# modify the load data
			for i in range(loadStartIndex,loadEndIndex):
				words = filelines[i].split(',')
				loadBusIndex = loadBusList.index(words[0].strip()) # find index of the data to be copied over
				words[5] = '%.3f' % loadPListInt[loadBusIndex] # constant power
				words[6] = '%.3f' % loadQListInt[loadBusIndex] # constant power
#				words[9] = '%.3f' % loadPListInt[loadBusIndex] # constant admittance
#				words[10] = '%.3f' % loadQListInt[loadBusIndex] # constant admittance

				lendiffP = lenLoadP - len(words[5])  # constant power
				lendiffQ = lenloadQ - len(words[6])  # constant power
#				lendiffP = lenLoadP - len(words[9]) # constant admittance
#				lendiffQ = lenloadQ - len(words[10]) # constant admittance

				words[5] = lendiffP*' ' + words[5] # constant power
				words[6]= lendiffQ*' ' + words[6]  # constant power
#				words[9] = lendiffP*' ' + words[9]  # constant admittance
#				words[10] = lendiffQ*' ' + words[10] # constant admittance

				# construct a whole string by inserting commas between the words list
				tempLine = ''
				for i in range(len(words)):
					tempLine+=words[i]
					if i != len(words) - 1:
						tempLine+=','
				filelines[i] = tempLine
				f.write(filelines[i])
				f.write('\n')

			# copy the shunt data, which is in between the load and gen data	
			for i in range(loadEndIndex,genStartIndex):
				f.write(filelines[i])
				f.write('\n')

			# update and write the gen data
			for i in range(genStartIndex,genEndIndex):
				words = filelines[i].split(',')
				genBusIndex = genBusList.index(words[0].strip())
				words[2] = '%.3f' % genPListInt[genBusIndex]

				# add spaces for alignment
				lendiffP = lenGenP - len(words[2])
				words[2] = lendiffP*' ' + words[2]

				# construct a whole string by inserting commas between the words list
				tempLine = ''
				for i in range(len(words)):
					tempLine+= words[i]
					if i != len(words) - 1:
						tempLine+=',' # add commas between words unless its the last word
				filelines[i] = tempLine
				f.write(filelines[i])
				f.write('\n')


				# copy the rest of the raw data
			for i in range(genEndIndex,len(filelines)):
				f.write(filelines[i])
				f.write('\n')



    #currentdir = os.getcwd()
    output_dir = currentdir + '/' + newdir
    NewRawFiles = os.listdir(output_dir)
    PathList = [(output_dir + '/' + f) for f in NewRawFiles]


    for i in range(len(PathList)):
    	#Settings. CONFIGURE THIS
    	settings = {
    	# use the same raw data in PSS/E and TS3ph #####################################
    	    'filename':PathList[i], #use the same raw data in PSS/E and TS3ph
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

    	psse_log = output_dir + '/' + 'log' + NewRawFiles[i].replace('.raw','.txt')


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
    	ierr = psspy.fnsl(settings['pf_options'])
    	ierr = psspy.rawd_2(0,1,[1,1,1,0,0,0,0],0,PathList[i])
    	"""
        # Uncomment if you want to save .sav files as well
    	# Load conversion (multiple-step)
        ierr = psspy.cong(0) #converting generators
    	psspy.conl(_i,_i,1,[0,_i],[_f,_f,_f,_f])
    	psspy.conl(1,1,2,[_i,_i],[0.0, 0.0,100.0, 100.0])
    	psspy.conl(_i,_i,3,[_i,_i],[_f,_f,_f,_f])

    	ierr = psspy.save(PathList[i].replace('.raw','.sav'))
        """



