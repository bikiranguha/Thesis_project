# script to convert all tabs to 4 spaces
file = 'changeLoadFnOld.py' # specify the file to convert here
newfile = 'changeLoadFn.py'
newLines = []
with open(file,'r') as f:
    fileLines = f.read().split('\n')
    for line in fileLines:
        nLine = line.replace('\t','    ') # replace all tabs with 4 spaces
        newLines.append(nLine)

with open(newfile,'w') as f:
    for line in newLines:
        f.write(line)
        f.write('\n')
