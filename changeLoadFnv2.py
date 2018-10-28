""" 
Script to generate new raw files with scaled up constant power loads and generation
Also change each generator by that percentage 
After generating new raw files, they are solved and corresponding sav files are also generated
Only works for PSSE version 33 raw file
This is meant to work with constant power loads


"""
#import pdb
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
from getBusDataFn import getBusData


def reconstructLine2(words):
    currentLine = ''
    for word in words:
        currentLine += word
        currentLine += ','
    return currentLine[:-1]



def changeLoad(raw,start,end,step,newdir):
    """ 
        New raw files are created for each percentage step in [start,end]. 
        The current step defines the percentage scaling up (or down) factor for load and generation
    """

    # convert the raw file to another one where all the load is constant power
    raw_conp = raw.replace('.raw','') + '_conp.raw'
    redirect.psse2py()
    psspy.psseinit(buses=80000)

    # ignore the output
    psspy.report_output(6,'',[0,0])
    psspy.progress_output(6,'',[0,0])
    psspy.alert_output(6,'',[0,0])
    psspy.prompt_output(6,'',[0,0])

    # read the raw file and convert all the loads to constant power
    ierr = psspy.read(0, raw)
   
    # multi-line command to convert the loads to 100% constant power
    psspy.conl(0,1,1,[1,0],[0.0,0.0,0.0,0.0])
    psspy.conl(0,1,2,[1,0],[0.0,0.0,0.0,0.0])
    psspy.conl(0,1,3,[1,0],[0.0,0.0,0.0,0.0])
    ierr = psspy.rawd_2(0,1,[1,1,1,0,0,0,0],0,raw_conp)
    # run change Load on the constant power load raw file






    rawBusDataDict = getBusData(raw_conp)
    # create a new directory to put the files in
    currentdir = os.getcwd()
    
    if not os.path.exists(newdir):
        os.mkdir(newdir)
    output_dir = currentdir+'/'+newdir
    #genDiscount = 0.90 # ratio of the actual increase in generation
    genDiscount = 1.0
    lossRatio = 0.0  # gen scale-up factor: (scalePercent + (scalePercent-100)*lossRatio)/100
    ############################################

    # create new raw files with scaled up loads and generation
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


        raw_name = raw_conp.replace('.raw','')
        out_file = raw_name + scalePercentStr + '.raw'  # output file
        out_path = output_dir+'/'+out_file
        impLoadBuses = [] # enter specified load buses to scale, if empty all loads are scaled
        incLoss = (scalePercent-100)*lossRatio  # Additional percentage increase in Pgen (to account for losses)
        #############################################


        #Read raw file
        with open(raw_conp,'r') as f:
            filecontent = f.read()
            filelines = filecontent.split('\n')

            ## Get start and end indices of load and gen info
            #########################################
            loadStartIndex = filelines.index('0 / END OF BUS DATA, BEGIN LOAD DATA')+1
            loadEndIndex = filelines.index('0 / END OF LOAD DATA, BEGIN FIXED SHUNT DATA')

            genStartIndex = filelines.index('0 / END OF FIXED SHUNT DATA, BEGIN GENERATOR DATA')+1
            genEndIndex = filelines.index('0 / END OF GENERATOR DATA, BEGIN BRANCH DATA')
            ##############################################################################

            totalPincr = 0.0
            totalQincr = 0.0
            percentIncr = (scalePercent - 100.0)/100 # increment in percentage

            newPConList=[]
            newQConList=[]
            newIPList=[]
            newIQList=[]
            newZPList=[]
            newZQList=[]
            # Extract load info
            for i in range(loadStartIndex,loadEndIndex):
                words = filelines[i].split(',')
                loadBus = words[0].strip()
                #loadBusList.append(words[0].strip())
                loadPCon = float(words[5].strip())
                loadQCon = float(words[6].strip())
                loadIP = float(words[7].strip())
                loadIQ = float(words[8].strip())
                loadZP = float(words[9].strip())
                loadZQ = float(words[10].strip())

                # calculate the total MW (MVAr) increase in load
                loadBusVolt = float(rawBusDataDict[loadBus].voltpu)
                
                Pincr = percentIncr*(loadPCon + loadIP*loadBusVolt + loadZP*loadBusVolt**2) # this equation is provided in PAGV1 page 293
                Qincr = percentIncr*(loadQCon + loadIQ*loadBusVolt + loadZQ*loadBusVolt**2)
                totalPincr += Pincr
                totalQincr += Qincr
                ###


                # new load values
                newPConList.append(loadPCon*scalePercent/100)
                newQConList.append(loadQCon*scalePercent/100)
                newIPList.append(loadIP*scalePercent/100)
                newIQList.append(loadIQ*scalePercent/100)
                newZPList.append(loadZP*scalePercent/100)
                newZQList.append(loadZQ*scalePercent/100)

                """
                loadPList.append(words[5].strip()) # adding P value (constant power)
                loadQList.append(words[6].strip()) # adding Q value (constant power)
                loadIPList.append(words[7].strip()) # constant current P
                loadIQList.append(words[7].strip()) # constant current Q
                loadZPList.append(words[9].strip()) # adding P value (constant admittance)
                loadZQList.append(words[10].strip()) # adding Q value (constant admittance)
                """


            # get total MW gen
            totalGenMW = 0.0 # total generation excluding the swing bus
            for i in range(genStartIndex, genEndIndex):
                words = filelines[i].split(',')
                GenBus = words[0].strip()
                if rawBusDataDict[GenBus].type == '3':
                    continue
                PGen = float(words[2].strip())
                totalGenMW += PGen

            # get new MW Gen
            GenMWDict = {} # dictionary to hold new PGen values
            for i in range(genStartIndex, genEndIndex):
                words = filelines[i].split(',')
                Bus = words[0].strip()
                if rawBusDataDict[Bus].type == '3':
                    continue
                macID = words[1].strip()
                key = Bus + macID
                PGen = float(words[2].strip())
                genIncr = PGen/totalGenMW*totalPincr
                newPGen = (PGen + genIncr)*genDiscount
                GenMVA = float(words[8].strip())
                if newPGen < GenMVA:
                    GenMWDict[key] = newPGen
                else:
                    GenMWDict[key] = GenMVA

        #  generate the new raw file
        with open(out_path,'w') as f:
            # copy everything before load data
            for i in range(loadStartIndex):
                f.write(filelines[i])
                f.write('\n')

            # modify the load data
            j=0
            for i in range(loadStartIndex,loadEndIndex):
                words = filelines[i].split(',')

                # change the constant MVA values
                words[5] = '%.3f' %newPConList[j] 
                words[6] = '%.3f' %newQConList[j] 
                words[5] = words[5].rjust(10)
                words[6]=  words[6].rjust(10)  

                # change the constant current values
                words[7] = '%.3f' %newIPList[j] 
                words[8] = '%.3f' %newIQList[j] 
                words[7] = words[7].rjust(10)
                words[8]=  words[8].rjust(10)

                # change the constant impedance values
                words[9] = '%.3f' %newZPList[j] 
                words[10] = '%.3f' %newZQList[j] 
                words[9] = words[9].rjust(10)
                words[10]=  words[10].rjust(10)

                # construct a whole string by inserting commas between the words list
                filelines[i] = reconstructLine2(words)
                f.write(filelines[i])
                f.write('\n')
                # increment the load list index
                j+=1

            # copy the shunt data, which is in between the load and gen data    
            for i in range(loadEndIndex,genStartIndex):
                f.write(filelines[i])
                f.write('\n')

            # update and write the gen data
            for i in range(genStartIndex,genEndIndex):
                words = filelines[i].split(',')
                Bus = words[0].strip()
                
                if rawBusDataDict[Bus].type == '3':
                    f.write(filelines[i])
                    f.write('\n')
                    continue
                macID = words[1].strip()
                key = Bus + macID
                newPGen = GenMWDict[key]

                words[2] = '%.3f' %newPGen
                words[2] =words[2].rjust(10)


                # construct a whole string by inserting commas between the words list
                filelines[i] = reconstructLine2(words)
                f.write(filelines[i])
                f.write('\n')


                # copy the rest of the raw data
            for i in range(genEndIndex,len(filelines)):
                f.write(filelines[i])
                f.write('\n')



    # solves each of the newly generated raw files and saves them 
    output_dir = currentdir + '/' + newdir
    NewRawFiles = os.listdir(output_dir)
    PathList = [(output_dir + '/' + f) for f in NewRawFiles]


    redirect.psse2py()
    psspy.psseinit(buses=80000)

    _i=psspy.getdefaultint()
    _f=psspy.getdefaultreal()
    _s=psspy.getdefaultchar()

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
        psspy.report_output(2,psse_log,[0,0])
        psspy.progress_output(2,psse_log,[0,0])
        psspy.alert_output(2,psse_log,[0,0])
        psspy.prompt_output(2,psse_log,[0,0])



        print "\n Reading raw file:",settings['filename']
        ierr = psspy.read(0, settings['filename'])
        ierr = psspy.fnsl(settings['pf_options'])
        converge = psspy.solved()
        if converge == 0:
            ierr = psspy.rawd_2(0,1,[1,1,1,0,0,0,0],0,PathList[i])
        else: # file does not converge, remove raw file, keep log file
            os.remove(PathList[i])
        """
        # Uncomment if you want to save .sav files as well
        # Load conversion (multiple-step)
        ierr = psspy.cong(0) #converting generators
        psspy.conl(_i,_i,1,[0,_i],[_f,_f,_f,_f])
        psspy.conl(1,1,2,[_i,_i],[0.0, 0.0,100.0, 100.0])
        psspy.conl(_i,_i,3,[_i,_i],[_f,_f,_f,_f])

        ierr = psspy.save(PathList[i].replace('.raw','.sav'))
        """




if __name__ == '__main__':
    raw = 'savnw.raw'
    changeLoad(raw,101,110,1,'test_chLoad')