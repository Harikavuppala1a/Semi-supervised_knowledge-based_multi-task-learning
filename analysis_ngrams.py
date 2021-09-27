import csv
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.util import ngrams
from collections import Counter

# set of stop words
stop_words = set(stopwords.words('english')) 
text = []; text_unlab =[]
r_anum = re.compile(r'([^\sa-z0-9.(?)!])+')
r_white = re.compile(r'[\s.(?)!]+')
with open('data.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile, delimiter = '\t')
        for row in reader:
          post = str(row['post'])
          row_clean = r_white.sub(' ', r_anum.sub('', post.lower())).strip()
          word_tokens = word_tokenize(row_clean)
          filtered_sentence = [] 
          for w in word_tokens: 
          	if w not in stop_words: 
          		filtered_sentence.append(w)
          text.append(" ".join(filtered_sentence))

with open('unlab_minus_lab_shortest_n.txt', 'r') as unlabfile:

		reader_unlab = unlabfile.readlines()
		for row_unlab in reader_unlab:
			post_unlab = str(row_unlab)
			row_clean = r_white.sub(' ', r_anum.sub('', post_unlab.lower())).strip()
			word_tokens = word_tokenize(row_clean)
			filtered_sentence = [] 
			for w in word_tokens: 
				if w not in stop_words: 
					filtered_sentence.append(w)
			text_unlab.append(" ".join(filtered_sentence))


def get_ngrams(input_data, num):
	final_ngrams = []
	for post in input_data:
		n_grams = ngrams(nltk.word_tokenize(post), num)
		for grams in n_grams:
			final_ngrams.append(' '.join(grams))
	counter = Counter(final_ngrams)
	print (counter.most_common(25))

print ("Top 25 unigrams for labeled data")
get_ngrams(text,1)
print ("Top 25 unigrams for unlabeled data")
get_ngrams(text_unlab,1)
print ("Top 25 bigrams for labeled data")
get_ngrams(text, 2)
print ("Top 25 bigrams for unlabeled data")
get_ngrams(text_unlab,2)
print ("Top 25 trigrams for labeled data")
get_ngrams(text,3)
print ("Top 25 trigrams for unlabeled data")
get_ngrams(text_unlab,3)
