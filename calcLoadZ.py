# calculate load impedances for constant impedance load
import math

raw = 'savnw.raw'
Sbase = 100.0
with open(raw,'r') as f:
    fileLines = f.read().split('\n')

loadStartIndex = fileLines.index('0 / END OF BUS DATA, BEGIN LOAD DATA')  + 1
loadEndIndex = fileLines.index('0 / END OF LOAD DATA, BEGIN FIXED SHUNT DATA')

print 'Load Bus: Load Impedance (pu)'
for i in range(loadStartIndex,loadEndIndex):
    line =fileLines[i]
    words = line.split(',')
    bus = words[0].strip()
    ZP = abs(float(words[9].strip()))
    ZQ = abs(float(words[10].strip()))
    ZS = math.sqrt(ZP**2 + ZQ**2)
    ZSpu = ZS/Sbase
    LoadZ = 1/ZSpu # the voltage is set to unity for constant impedance load power calculation
    print '{}:{}'.format(bus,LoadZ)

# if you need to calculate impedances on constant current and constant power loads,
# then you need to also get the voltage magnitude