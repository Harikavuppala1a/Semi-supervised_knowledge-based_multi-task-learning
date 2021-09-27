
import csv
from bert_serving.client import ConcurrentBertClient
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle

header = ["Home", "School", "Public space", "Workplace", "University", "Media", "Other"]
f = open('cosine_similarity_scores.txt', 'w', encoding='UTF8')
writer = csv.writer(f,  delimiter = '\t')
writer.writerow(header)

weaklabeled_datadict = dict([(key, []) for key in header])

with open('data/Unlabeled_data_weaklabels.txt', 'r') as csvfile:
  reader = csv.DictReader(csvfile, delimiter = '\t')
  for row in reader:
    post = str(row['post'])
    labels = row['labels'].split(',')
    for i in range(len(labels)):
    	if labels[i] == "Public Transport":
    		weaklabeled_datadict["Public space"] = weaklabeled_datadict["Public space"] + [post]
    	else:
    		weaklabeled_datadict[labels[i]] = weaklabeled_datadict[labels[i]] + [post]

  for key in header:
  	weaklabeled_datadict[key] = ' '.join(weaklabeled_datadict[key])

header_labeleddata = ["Role stereotyping", "Attribute stereotyping", "Body shaming", "Hyper-sexualization (excluding body shaming)","Internalized sexism", "Pay gap", "Hostile work environment (excluding pay gap)", "Denial or trivialization of sexist misconduct",
"Threats", "Rape", "Sexual assault (excluding rape)", "Sexual harassment (excluding assault)", "Tone policing", "Moral policing (excluding tone policing)", "Victim blaming", "Slut shaming", "Motherhood-related discrimination", 
"Menstruation-related discrimination", "Religion-based sexism", "Physical violence (excluding sexual violence)", "Mansplaining", "Gaslighting", "Other"]
labeled_data = dict([(key, []) for key in header_labeleddata])


with open('data/sexismClassi.csv', 'r') as csvfile:
  reader = csv.DictReader(csvfile, delimiter = '\t')
  for row in reader:
    post = str(row['post'])
    labels = row['labels'].split(',')
    for i in range(len(labels)):
    	labeled_data[labels[i]] = labeled_data[labels[i]] + [post]

  for key in header_labeleddata:
  	labeled_data[key] = ' '.join(labeled_data[key])

posts_arr_weak = np.zeros((len(header), 768))
bc = ConcurrentBertClient()
for ind,key in enumerate(weaklabeled_datadict.keys()):
	posts_arr_weak[ind] = bc.encode([weaklabeled_datadict[key]])

posts_arr_lab = np.zeros((len(header_labeleddata), 768))
for ind,key in enumerate(labeled_data.keys()):
	posts_arr_lab[ind] = bc.encode([labeled_data[key]])


co_mat = np.zeros((len(header_labeleddata), len(header)))
for i in range(len(header_labeleddata)):
	for j in range(len(header)):
		co_mat[i,j] = cosine_similarity(np.array([posts_arr_lab[i]]),np.array([posts_arr_weak[j]]))
print (co_mat)

filename = "similarity_scores.pickle"
with open(filename, 'wb') as f:
    pickle.dump(co_mat, f)




 



	