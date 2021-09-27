import os
import csv

rootdir = 'C:/Users/HARIKA/Downloads/sexismClassification_selftraining/datasets/SemEval2018-Task1-all-data/SemEval2018-Task1-all-data/English/E-c'   
header = ['ID',	'post','label']
f = open('emotion.txt', 'w', encoding='UTF8')
writer = csv.writer(f,  delimiter = '\t')
writer.writerow(header)

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        print (os.path.join(subdir, file))
        filename = os.path.join(subdir, file)
        with open(filename, 'r',encoding='UTF8') as csvfile:
            reader = csv.DictReader(csvfile, delimiter = '\t')
            for row in reader:
                    label = [row['anger'], row['anticipation'], row['disgust'], row['fear'], row['joy'], row['love'], row['optimism'], row['pessimism'], row['sadness'], row['surprise'], row['trust']]
                    writer.writerow([row['ID'], row['Tweet'], label])
        with open(filename, 'r',encoding='UTF8') as csvfile:
            reader = csv.DictReader(csvfile, delimiter = '\t')
            for ind,row in enumerate(reader):
                if ind < 35 :
                    label = [row['anger'], row['anticipation'], row['disgust'], row['fear'], row['joy'], row['love'], row['optimism'], row['pessimism'], row['sadness'], row['surprise'], row['trust']]
                    writer.writerow([row['ID'], row['Tweet'], label])
                else:
                    break