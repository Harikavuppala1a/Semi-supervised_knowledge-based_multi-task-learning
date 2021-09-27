import csv
import operator
import re
import random
from loadPreProc import *

filename = "data/data_trans.csv"
opfilename = "data/small_finegrained.csv"
sampleSize = 10

with open(filename) as csvfile:
	reader = csv.reader(csvfile, delimiter='\t')
	header = next(reader)
	origList = list(reader)

sList = random.sample(origList, sampleSize)

with open(opfilename, 'w') as opfile:
    wr = csv.writer(opfile, delimiter = '\t')
    wr.writerow(header)
    for entry in sList:
    	t = random.sample(LABEL_MAP.keys(), int(len(LABEL_MAP.keys())/2))
    	entry[1] = ','.join(t)
    	wr.writerow(entry)
