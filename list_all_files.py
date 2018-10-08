"""
	Get a list of all files in the folder you are searching for
"""

import os

currentdir = raw_input('Please enter absolute path of the directory whose files you want to list:')

fileList = os.listdir(currentdir)
print 'Scripts:'
for file in fileList:
	if file.endswith('py'):
		print file

print '\n\n\n'

print 'Non-script files:'
for file in fileList:
	if not file.endswith('py'):
		print file
