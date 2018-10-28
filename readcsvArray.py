

import csv
from numpy import genfromtxt

# to read line by line
with open("savnw_conpv.csv") as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
    # read line by line
    for row in reader: # each row is a list
        pass # code to perform with the row

# to get array directly
my_data = genfromtxt('savnw_conpv.csv', delimiter=',')
