import csv

dictionary ={}
header = ['post','labels']
f = open('Unlabeled_data_weaklabels.txt', 'w', encoding='UTF8')
writer = csv.writer(f,  delimiter = '\t')
writer.writerow(header)

def select_data():

    with open('entiredata.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile, delimiter = '\t')
        for row in reader:
            listnode = []
            label_list = []
            listnode.append(row['post'])
            if row['tag'] != "":
                tags = row['tag'].split(',')
                for i in range(len(tags)):
                    if tags[i] == 'Home' or tags[i] == 'School' or  tags[i] == 'Public space' or tags[i] == 'Workplace' or tags[i] == 'University' or tags[i] == 'Media' or tags[i] == 'Public Transport':
                        if tags[i] == 'Public Transport':
                            print (tags[i])
                            tags[i] = 'Public space'
                            print (tags[i])
                        label_list.append(tags[i])
                    else:
                        label_list.append("Other")

                labels = list(set(label_list))

                listnode.append(','.join(labels))
                writer.writerow(listnode)


def data_stats():

    with open('entiredata.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile, delimiter = '\t')
        count = 0
        fullcount = 0
        for row in reader:
            listnode = []
            fullcount = fullcount +1
            listnode.append(row['post'])
            if row['tag'] != "":
                count = count + 1
                tags = row['tag'].split(',')
                for i in range(len(tags)):
                    if tags[i] not in dictionary:
                        dictionary[tags[i]] = 0
                    dictionary[tags[i]] = dictionary[tags[i]] + 1 

        for ind,i in enumerate(sorted (dictionary.items(), key=lambda item: item[1], reverse=True)):
            if ind <=30:
                print (i)

data_stats()
select_data()
            