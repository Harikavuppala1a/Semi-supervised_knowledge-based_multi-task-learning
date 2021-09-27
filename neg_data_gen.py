import csv
import codecs
import sys
import nltk
import re
import os
csv.field_size_limit(sys.maxsize)
import numpy as np

# 'random' or 'other oppression'
neg_data_type = 'other oppression'
blogtext_prepared_data = "blogtext_prepared_data.csv"
no_of_sent = 1
min_words = 7
max_words = 288
no_of_VBD = 1
count_blog =0
count_unlab = 0
total_unlab_data = 70000


if neg_data_type == 'other oppression':
    csvfile_data_other_oppression = open('blogtext_selected_other_oppression.csv','w')
    words_list = []
    with open('keywords_slurs.txt') as f:
        words_list.extend([line.strip().lower() for line in f])
    with open('keywords_slurs_LGBTQ.txt') as f:
        words_list.extend([line.strip().lower() for line in f])
    with open('keywords_slurs_radhika_pulkit.txt') as f:
        words_list.extend([line.strip().lower() for line in f])
    words_list = list(set(word_list))
elif neg_data_type == "random":
    csvfile_data_random = open('blogtext_selected_random.csv','w')


def get_count(post):
	sent_count = 0
	post_sent = nltk.tokenize.sent_tokenize(post)
	for sent in post_sent:
		count_VBD = 0
		count_VBN = 0
		words = nltk.word_tokenize(sent)
		if "i" in words or "we" in words or "he" in words:
			pos_tags = nltk.pos_tag(words)
			for tags in pos_tags:
				if tags[1] == "VBD":
					count_VBD = count_VBD + 1
				elif tags[1] == "VBN":
					count_VBN = count_VBN + 1					
			if count_VBD >=no_of_VBD :
				sent_count = sent_count + 1
	return sent_count

if not os.path.isfile(blogtext_prepared_data):
	csvfile_data = open(blogtext_prepared_data,'w')
	writer = csv.writer(csvfile_data)
	writer.writerow(["id","gender","age","topic","sign","date","text"])
	with open('blogtext.csv') as csvfile:
		reader = csv.reader( (line.lower().replace('\0','') for line in csvfile),delimiter = ',' )
		next(reader)
		for row in reader:
			writer.writerow(row)
	csvfile_data.close()

with open(blogtext_prepared_data, 'r') as csvfile:
	reader = csv.DictReader(csvfile,delimiter= ',')
	for row in reader:
		no_of_words = len(nltk.word_tokenize(row['text']))
		if no_of_words >=min_words and no_of_words <=max_words:
			sent_count = get_count(row['text'])
			if sent_count >=no_of_sent:
        		if neg_data_type == 'other oppression' and any(word in row['text'].split() for word in words_list):
           			csvfile_data_other_oppression.write(row['text'])
           			csvfile_data_other_oppression.write('\n')
           			count_blog = count_blog + 1
                elif neg_data_type == 'random':
                    csvfile_data_random.write(row['text'])
                    csvfile_data_random.write('\n')
                    count_blog = count_blog + 1

csvfile_data_other_oppression.close()
csvfile_data_random.close()
# with open('unlab_minus_lab_shortest_n.txt', 'r') as unlabfile:
# 	reader_unlab = unlabfile.readlines()
# 	for row in reader_unlab:
# 		sent_count = get_count(row)
# 		if sent_count >=no_of_sent:
# 			count_unlab = count_unlab + 1

print (count_blog)
