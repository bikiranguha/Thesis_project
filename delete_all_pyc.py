"""
Deletes any .pyc file
Use before commit
"""

import os
for root, dirs, files in os.walk(".", topdown=True):

   for name in files:
   	if name.endswith('.pyc') or 'desktop.ini' in name:
   		#print name
   		os.remove(os.path.join(root, name))
   	"""
      print(os.path.join(root, name))
   for name in dirs:
      print(os.path.join(root, name))
    """
