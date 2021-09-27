import os
import csv

rootdir = 'C:/Users/HARIKA/Downloads/sexismClassification_selftraining/datasets/Sarcasm/sarcasm_v2/sarcasm_v2'
header = ['ID', 'post','label']
f = open('sarcasm.txt', 'w', encoding='UTF8')
writer = csv.writer(f,  delimiter = '\t')
writer.writerow(header)

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
    	print (os.path.join(subdir, file))
    	filename = os.path.join(subdir, file)
    	with open(filename, 'r',encoding='UTF8') as csvfile:
    		reader = csv.DictReader(csvfile)
    		for row in reader:
                    if row['class'] == "notsarc":
                        label = 0
                    else:
                        label = 1
                    writer.writerow([row['id'], row['text'], label])
with open('C:/Users/HARIKA/Downloads/sexismClassification_selftraining/datasets/Sarcasm/sarcasm_v2/sarcasm_v2/GEN-sarc-notsarc.csv', 'r',encoding='UTF8') as csvfile:
    reader = csv.DictReader(csvfile)
    for ind, row in enumerate(reader):
        if ind <900 or (ind > 5100 and ind <6000):
            if row['class'] == "notsarc":
                label = 0
            else:
                label = 1
            writer.writerow([row['id'], row['text'], label])
    		# print (os.path.join(subdir, file))