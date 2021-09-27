from PyDictionary import PyDictionary
from loadPreProc import *
from nltk.stem import PorterStemmer 
import sys
from nltk import pos_tag
import pickle

dictionary=PyDictionary()
ps = PorterStemmer()
r_anum = re.compile(r'([^\sa-z0-9.(?)!])+')
r_white = re.compile(r'[\s.(?)!]+')

conf_dict_list, conf_dict_com = load_config(sys.argv[1])
list_of_keywords =[]
list_of_stemwords = []
created_text = []

def save_meanings(list_of_keywords,dict_filename):
    meaning_dict ={}
    for keyword in list_of_keywords:
        print (keyword)
        meaning_dict[keyword.strip()] = dictionary.meaning(keyword.strip())
    with open(dict_filename, 'wb') as f_data:
      pickle.dump(meaning_dict, f_data)

with open('data/domain_specific_keywords.txt', 'r') as file:
    reader_keywords = file.readlines()
    for word in reader_keywords:
        list_of_keywords.append(str(word).lower().strip())
        list_of_stemwords.append(ps.stem(str(word).lower().strip()))

counts_count =0
with open('saved/raw_data~sexismClassi~0.15~0.15~22~35~1~True~23_class_maps.pickle', 'rb') as f_data:
        data_dict = pickle.load(f_data)

for ID, text in enumerate(data_dict['text']):
    print (ID)
    word_len_post = len(data_dict['text'][ID].split(' '))
    feats_ID = {}
    count = 0

    tokens_tag = pos_tag(text.split(' '))

    for tokens in tokens_tag:
        word,tag = tokens
        stemword = ps.stem(word)
        if stemword in list_of_stemwords or stemword in list_of_keywords or word in list_of_keywords:
            count = count +1
            print (word)

            meaning_word = dictionary.meaning(word,disable_errors=True)
            if meaning_word == None:
                meaning_word = dictionary.meaning(stemword)
            if len(meaning_word) >1:
                if tag == "NN" or tag == "NNS" or tag == "NNP" or tag =="NNPS":
                    tag_name = "Noun"
                elif tag == "JJ" or tag == "JJR" or tag =="JJS" :
                    tag_name = "Adjective"
                elif tag == "VB" or tag == "VBG" or tag =="VBD" or tag == "VBN" or tag == "VBP" or tag == "VBZ":
                        tag_name = "Verb"
                for k,v in meaning_word.items():
                    if k ==tag_name:
                        feats_ID[word] = word + ' ' + '(' + ' '.join(v) + ')'
                        break
            else:
                for k,v in meaning_word.items():
                    feats_ID[word] =  word +' '+ '(' + ' '.join(v) + ')'
        else:
            feats_ID[word] = word  

    updated_post = []
    for k,v in feats_ID.items():
            updated_post.append(v)

    post = ' '.join(updated_post)
    row_clean = r_white.sub(' ', r_anum.sub('', post.lower())).strip()
    created_text.append(row_clean)
    if count >0:
        counts_count = counts_count + 1
print (counts_count)
with open('wordmeanings.pickle', 'wb') as f_data:
    pickle.dump(created_text, f_data)
