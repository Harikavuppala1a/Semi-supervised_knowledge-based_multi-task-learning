from bert_serving.client import ConcurrentBertClient
import numpy as np
import pickle
import h5py


def save_postlevel_knowledge_rep(s_filename, meaning_pickle_filename):
    bc = ConcurrentBertClient()
    with open("saved/"+ meaning_pickle_filename , 'rb') as f_data:
        posts = pickle.load(f_data)
    posts_arr = np.zeros((len(posts), 768))
    bc = ConcurrentBertClient()
    bert_batch_size = 64
    for ind in range(0, len(posts), bert_batch_size):
        end_ind = min(ind+bert_batch_size, len(posts))
        posts_arr[ind:end_ind, :] = bc.encode(posts[ind:end_ind])
    with h5py.File(s_filename, "w") as hf:
        hf.create_dataset('feats', data=posts_arr)

def wordlevel_elmo(text_sen,text,meaning_dict):
    for ID, sentences in enumerate(text_sen):
        word_len_post = len(text[ID].split(' '))
        feats_ID = np.zeros((word_len_post, emb_size))
        w_ind = 0
        count = 0
        for k,v in meaning_dict[ID].items():
            if v != "":
                count = count + 1
                break
        if count >0:          
            for ind_sen, sent in enumerate(sentences):
                sent_words = sent.split(' ')
                for word in sent_words:
                    if meaning_dict[ID][word] !="":                       
                        vectors = elmo.embed_sentence(meaning_dict[ID][word].split(' '))
                        feats_ID[w_ind] = np.mean(vectors,axis=0)
                    w_ind += 1

        np.save(dir_filepath + str(ID) + '.npy', feats_ID)

def wordlevel_glove(text_sen,text,meaning_dict):
    for ID, sentences in enumerate(text_sen):
        feats_ID = np.zeros((max_num_sent, max_word_count_per_sent, embed_size))
        l = min(len(sentences), max_num_sent)
        count = 0
        for k,v in meaning_dict[ID].items():
            if v != "":
                count = count + 1
                break
        if count >0:          
            for ind_sen, sen in enumerate(sentences[:l]):
                words = sen.split(' ')
                l = min(len(words), max_word_count_per_sent)
                for ind_w, w in enumerate(words[:l]):
                    list_mean =[]
                    if meaning_dict[ID][w] !="":
                        mean_w =  meaning_dict[ID][w].split(' ')
                        for word in mean_w:
                            list_mean.append(comb_vocab[word])
                        feats_ID[ind_sen, ind_w, :] = np.mean(np.asarray(list_mean),axis =0)
        np.save(dir_filepath + str(ID) + '.npy', feats_ID)

def save_wordlevel_knowledge_rep(s_filename, meaning_pickle_filename):
    with open("saved/"+ meaning_pickle_filename , 'rb') as f_data:
        meaning_dict = pickle.load(f_data)

    wordlevel_elmo (data_dict['text_sen'], data_dict['text'],meaning_dict)
    wordlevel_glove (data_dict['text_sen'], data_dict['text'],meaning_dict)

