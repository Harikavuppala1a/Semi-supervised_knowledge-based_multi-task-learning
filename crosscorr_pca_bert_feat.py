from __future__ import division
from statistics import mean

import csv
import h5py
from loadPreProc import *
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

datadict = {new_list: [] for new_list in range(23)}

with h5py.File('sent_enc_feat~bert~False.h5', "r") as hf:
        bert_feat = hf['feats'][:]
with open('data/data.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile, delimiter = '\t')
        for num,row in enumerate(reader):
          post = str(row['post'])
          cat_list = str(row['category_2']).lower().split(',')
          label_ids = list(set([LABEL_MAP[cat] for cat in cat_list]))
          for lids in label_ids:
          	datadict[lids].append(bert_feat[num])


### cross correlation
csvfile = open('bert_feat.csv', 'w')
filewriter = csv.writer(csvfile, delimiter=' ')
mean_data = []
for i in range(len(datadict)):
	mean_data.append(np.mean(datadict[i], axis=0))
dfObj = pd.DataFrame(mean_data) 

trans_dfObj = dfObj.T
corr_dfObj = trans_dfObj.corr()


####PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(dfObj)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
print (principalDf)

export_csv = corr_dfObj.to_csv ('cross_corr.csv', sep='\t', mode='w')
