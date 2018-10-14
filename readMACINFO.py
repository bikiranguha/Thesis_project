


def getIndFromMacInfo(MacInfoFile):


#MacInfoFile =  'MAC_INFO.txt'
    class GenInd():
        def __init__(self,EqpInd,EfdInd,PmInd):
            self.EqpInd = EqpInd
            self.EfdInd = EfdInd
            self.PmInd = PmInd

    GenIndDict = {}

    with open(MacInfoFile,'r') as f:
        fileLines = f.read().split('\n')
        for line in fileLines[1:]: # the first line is the header, so skip
            if line == '':
                continue
            words = line.split(',')
            Bus = words[0].strip()
            cktID = words[1].strip()
            key = Bus + ',' + cktID

            EqpInd = int(words[11].strip()) + 1 # +1 since, the 1st index in the output file is time
            EfdInd = int(words[12].strip()) + 1
            PmInd = int(words[13].strip()) + 1
            GenIndDict[key] = GenInd(EqpInd, EfdInd, PmInd)
            #lst = [Bus, cktID, EqpInd, EfdInd, PmInd]
            #lstStr = ','.join(lst)
            #print lstStr
    return GenIndDict
