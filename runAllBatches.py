# script to run all batch python scripts
#exec('import simPSSPFORNLBatch1.py')
i  = 1
while True:
    try:
    	print('Running file: simPSSPFORNLBatch{}.py'.format(i))
    	print('\n')
        exec('import simPSSPFORNLBatch{}.py'.format(i))
        i+=1
    except: # reached the end of simulation batches
    	print('No such file: simPSSPFORNLBatch{}.py'.format(i))
        break