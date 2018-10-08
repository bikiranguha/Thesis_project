"""
Search for something in fileName or within py or txt files
"""

import os

while True:

	searchTerm = raw_input('Please enter search term:')
	cwd = os.getcwd()
	print 'Search Term appears as a file in the following locations:'
	for root, dirs, files in os.walk(cwd, topdown=True):
	   for name in files:
	   	if searchTerm in name:
	   		currentdir =  os.path.join(cwd,root)
	   		print os.path.join(currentdir, name)

	print 'Search Term appears inside the following files:'


	for root, dirs, files in os.walk(cwd, topdown=True):
		for name in files:
			if name.endswith('.py') or name.endswith('.txt'):
				currentFile = os.path.join(root, name)
				with open(currentFile,'r') as f:
					filecontent = f.read()
					if searchTerm in filecontent:
						currentdir =  os.path.join(cwd,root)
						print os.path.join(currentdir, name)
