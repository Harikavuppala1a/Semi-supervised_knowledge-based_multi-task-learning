import torch
import os
import sys
import numpy as np
import h5py
#sys.path.insert(0, '/home/pulkit/research/InferSent')
#from models import InferSent
import tensorflow as tf
import tensorflow_hub as hub
from bert_serving.client import ConcurrentBertClient
from keras.models import load_model
from loadPreProc import is_model_hier 
from bert_features import *
import csv
from bert_modeling import *
import time, requests

def bert_tune_embed_posts(bert_path,bert_max_seq_len,posts,max_sent_cnt,feat_name,tbert_paths):
	posts_arr = np.zeros((len(posts), max_sent_cnt, 3*bert_max_seq_len))
	print (posts_arr.shape)
	if feat_name == "trainable_tbert" and not os.path.isdir(bert_path):
            build_bert_module(tbert_paths['config_path'],tbert_paths['vocab_path'], tbert_paths['ckpt_path'], bert_path)
	tokenizer = create_tokenizer_from_hub_module(bert_path)
	for ind_post, sentences in enumerate(posts):
		l = min(max_sent_cnt,len(sentences))
		for ind_sen, sen in enumerate(sentences[:l]):
			in_examp = InputExample(guid=None,text_a = sen,text_b = None)
			feat = convert_single_example(ind_sen, in_examp, bert_max_seq_len, tokenizer)
			posts_arr[ind_post, ind_sen, :] = np.concatenate((np.array(feat.input_ids),np.array(feat.input_mask),np.array(feat.segment_ids)))

	print (posts_arr.shape)
	return posts_arr

def bert_flat_embed_posts(posts, embed_dim):
	posts_arr = np.zeros((len(posts), embed_dim))
	bc = ConcurrentBertClient()
	bert_batch_size = 64
	for ind in range(0, len(posts), bert_batch_size):
		end_ind = min(ind+bert_batch_size, len(posts))
		posts_arr[ind:end_ind, :] = bc.encode(posts[ind:end_ind])
	return posts_arr


def tuned_embed_posts(posts, max_sent_cnt, org_embed_dim, embed_dim, mod_name, base_feat_Name, base_feat_filename, func_name, use_saved_sent_enc_feats, data_fold_path, save_fold_path):
	posts_arr = np.zeros((len(posts), max_sent_cnt, embed_dim))
	if use_saved_sent_enc_feats and os.path.isfile(base_feat_filename):
		print("loading %s sent feats" % base_feat_Name)
		with h5py.File(base_feat_filename, "r") as hf:
			feats = hf['feats'][:]
	else:
		print("computing %s sent feats" % base_feat_Name)
		feats = func_name(posts, max_sent_cnt, org_embed_dim, data_fold_path)

	encoder = load_model(save_fold_path + 'mod_aut~' + mod_name + '.h5')	
	for ind, sentences in enumerate(posts):
		l = min(max_sent_cnt,len(sentences))
		embeddings = feats[ind, :l, :]
		posts_arr[ind, :l, :] = encoder.predict(embeddings)
	return posts_arr

# run the command below on a seperate terminal before running the function
# screen -L bert-serving-start -model_dir ../bert/uncased_L-24_H-1024_A-16/  -num_worker=1 -max_seq_len=None -device_map=0
# screen -L bert-serving-start -model_dir=../bert/uncased_L-12_H-768_A-12 -tuned_model_dir=../bert/tmp/pretraining_output/ -ckpt_name=model.ckpt-100000 -num_worker=1 -max_seq_len=None -device_map=0

def bert_embed_posts(posts, max_sent_cnt, embed_dim, data_fold_path):
	posts_arr = np.zeros((len(posts), max_sent_cnt, embed_dim))
	bc = ConcurrentBertClient()
	for ind, sentences in enumerate(posts):
		embeddings = bc.encode(sentences)
		l = min(max_sent_cnt,len(sentences))
		posts_arr[ind, :l, :] = embeddings[:l]
		if ind % 1000 == 0:
			print("batch %s of %s done" % (ind, len(posts)))
	return posts_arr

def perspective_features(text,attr_dict,attributes,url):
	data_dict = {
		'comment': {'text': text},
		'languages': "en",
		'requestedAttributes': attr_dict
	}
	# print (attributes)
	time.sleep(1*5)
	response = requests.post(url=url, data=json.dumps(data_dict)) 
	response_dict = json.loads(response.content)
	# print (response_dict)
	# print (response_dict)
	pers_dict = {"summary": {}, "span": {}}
	for attr in attributes:
		pers_dict["summary"][attr] = response_dict["attributeScores"][attr]["summaryScore"]["value"]
		curr_span = []
		spanScores = response_dict["attributeScores"][attr]["spanScores"]
		for span in spanScores:
			curr_span.append({'begin': span['begin'], 'end': span['end'], 'score': span['score']['value']})
		pers_dict["span"][attr] = curr_span

	return pers_dict

def perspective_embed_posts(posts, max_sent_cnt, embed_dim, data_fold_path):
	api_key = 'AIzaSyDAbzyQanYnq_b5XO4zJX9QrlA-aHO3iTQ'
	url = ('https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze' +'?key=' + api_key)
	attributes = ['TOXICITY', 'SEVERE_TOXICITY', 'IDENTITY_ATTACK', 'INSULT', 'PROFANITY', 'THREAT', 'TOXICITY_FAST', 'SEXUALLY_EXPLICIT', 'OBSCENE','FLIRTATION']
	attr_dict = {}
	for attr in attributes:
		attr_dict[attr] = {} 
	posts_arr = np.zeros((len(posts), max_sent_cnt, embed_dim))
	for ind, sentences in enumerate(posts):
		print (ind)
		l = min(max_sent_cnt,len(sentences))
		for in_ind,sent in enumerate(sentences[:l]):
			perspective_dict = perspective_features(sent,attr_dict,attributes,url)
			for items in perspective_dict.items():
				summary_key,summary_value = items
				for ind_summary, tup in enumerate(summary_value.items()):
					k,v = tup
					posts_arr[ind,in_ind, ind_summary] = v
				break
	print (posts_arr.shape)
	return posts_arr


def read_lexicon(lexicon_filename,LEN,categories):
	lexicon = dict()
	with open("data/" + lexicon_filename) as f:
		reader = csv.DictReader(f, delimiter="\t")
		for row in reader:
			if not row["lemma"] in lexicon:
				lexicon[row["lemma"].strip()] = np.zeros(2*LEN)
			if row["category"] in categories:
				if row["level"] == "inclusive":
					lexicon[row["lemma"].strip()][LEN + categories.index(row["category"])] += 1
				else:
					lexicon[row["lemma"].strip()][categories.index(row["category"])] += 1
	return lexicon

def check_presence(text, lexicon,LEN):
    final_features = np.zeros(2*LEN)
    for k,v in lexicon.items():
        string = r"\b" + k+ r"\b"                 
        all_matches = re.findall(string, text.strip())
        for match in all_matches:
            if match.strip() == "refuses" or match.strip() == "what the fuck":
                    continue
            final_features = np.add(final_features, lexicon[match.strip()])
    return final_features

def hurtlex_embed_posts(posts, max_sent_cnt, embed_dim, data_fold_path,LEN,categories):
	lexicon = read_lexicon('hurtlex_en.tsv',LEN,categories)
	posts_arr = np.zeros((len(posts), max_sent_cnt, embed_dim))
	for ind, sentences in enumerate(posts):
		l = min(max_sent_cnt,len(sentences))
		for in_ind,sent in enumerate(sentences[:l]):
			posts_arr[ind, in_ind,:] = check_presence(sent.lower(), lexicon,LEN)
	return posts_arr

def use_embed_posts(posts, max_sent_cnt, embed_dim, data_fold_path):
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	posts_arr = np.zeros((len(posts), max_sent_cnt, embed_dim))
	with tf.Graph().as_default():
		embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3")
		messages = tf.placeholder(dtype=tf.string)
		output = embed(messages)

		with tf.Session(config=config) as session:
			session.run([tf.global_variables_initializer(), tf.tables_initializer()])
			for ind, sentences in enumerate(posts):
				embeddings = session.run(output, feed_dict={messages: sentences})
				l = min(max_sent_cnt,len(sentences))
				posts_arr[ind, :l, :] = embeddings[:l]
	return posts_arr

def infersent_embed_posts(posts, max_sent_cnt, embed_dim, data_fold_path):
	model_path = data_fold_path + 'word_sent_embed/infersent2.pickle'
	word_emb_path = data_fold_path + 'word_sent_embed/fasttext.vec'
	posts_arr = np.zeros((len(posts), max_sent_cnt, embed_dim))

	params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
	                'pool_type': 'max', 'dpout_model': 0.0, 'version': 2}
	model = InferSent(params_model)
	model.load_state_dict(torch.load(model_path))
	model.set_w2v_path(word_emb_path)

	all_sents = []
	for sens in posts:
		all_sents.extend(sens)
	
	model.build_vocab(all_sents, tokenize=False)

	for ind, sentences in enumerate(posts):
		embeddings = model.encode(sentences, tokenize=False, verbose=False)
		l = min(max_sent_cnt,len(sentences))
		posts_arr[ind, :l, :] = embeddings[:l]

	return posts_arr

def sent_enc_featurize(sent_enc_feats_raw, model_type, data_dict, poss_sent_enc_feats_emb_dict, use_saved_sent_enc_feats, save_sent_enc_feats, data_fold_path, save_fold_path, test_mode,filename,bert_path,bert_max_seq_len,tbert_paths):
	max_num_sent_enc_feats = 3
	max_num_attributes = 2
	sent_enc_feats = []
	var_model_hier = is_model_hier(model_type)
	sent_enc_feat_str = ''
	for sent_enc_feat_raw_dict in sent_enc_feats_raw:
		feat_name = sent_enc_feat_raw_dict['emb']
		sent_enc_feat_str += ("%s~%s" % (sent_enc_feat_raw_dict['emb'], sent_enc_feat_raw_dict['m_id']))

		sent_enc_feat_dict ={}
		for sent_enc_feat_attr_name, sent_enc_feat_attr_val in sent_enc_feat_raw_dict.items():
			sent_enc_feat_dict[sent_enc_feat_attr_name] = sent_enc_feat_attr_val

		print("computing %s sent feats; hier model: %s; test_mode = %s; filename = %s" % (feat_name, var_model_hier, test_mode,filename))

		s_filename = ("%ssent_enc_feat~%s~%s~%s.h5" % (save_fold_path, feat_name, var_model_hier,filename))
		if use_saved_sent_enc_feats and os.path.isfile(s_filename):
			print("loading %s sent feats" % feat_name)
			with h5py.File(s_filename, "r") as hf:
				sent_enc_feat_dict['feats'] = hf['feats'][:data_dict['test_en_ind']]
		else:
			if var_model_hier:
				if feat_name.startswith('bert'):
					if feat_name == 'bert' or feat_name.startswith('bert_pre'):
						feats = bert_embed_posts(data_dict['text_sen'], data_dict['max_num_sent'], poss_sent_enc_feats_emb_dict[feat_name], data_fold_path)
					else:
						feats = tuned_embed_posts(data_dict['text_sen'], data_dict['max_num_sent'], poss_sent_enc_feats_emb_dict['bert'], poss_sent_enc_feats_emb_dict[feat_name], feat_name, 'bert', ("%ssent_enc_feat~bert.h5" % (save_fold_path)), bert_embed_posts, use_saved_sent_enc_feats, data_fold_path, save_fold_path)
				elif feat_name.startswith('use'):
					if feat_name == 'use':
						feats = use_embed_posts(data_dict['text_sen'], data_dict['max_num_sent'], poss_sent_enc_feats_emb_dict[feat_name], data_fold_path)
					else:
						feats = tuned_embed_posts(data_dict['text_sen'], data_dict['max_num_sent'], poss_sent_enc_feats_emb_dict['use'], poss_sent_enc_feats_emb_dict[feat_name], feat_name, 'use', ("%ssent_enc_feat~use.h5" % (save_fold_path)), use_embed_posts, use_saved_sent_enc_feats, data_fold_path, save_fold_path)
				elif feat_name.startswith('infersent'):
					if feat_name == 'infersent':
						feats = infersent_embed_posts(data_dict['text_sen'], data_dict['max_num_sent'], poss_sent_enc_feats_emb_dict[feat_name], data_fold_path)
					else:
						feats = tuned_embed_posts(data_dict['text_sen'], data_dict['max_num_sent'], poss_sent_enc_feats_emb_dict['infersent'], poss_sent_enc_feats_emb_dict[feat_name], feat_name, 'infersent', ("%ssent_enc_feat~infersent.h5" % (save_fold_path)), infersent_embed_posts, use_saved_sent_enc_feats, data_fold_path, save_fold_path)
				elif feat_name.startswith('trainable'):
						feats = bert_tune_embed_posts(bert_path,bert_max_seq_len,data_dict['text_sen'],data_dict['max_num_sent'],feat_name,tbert_paths)
				elif feat_name.startswith('perspective'):
						feats = perspective_embed_posts(data_dict['text_sen'], data_dict['max_num_sent'], poss_sent_enc_feats_emb_dict[feat_name], data_fold_path)
				elif feat_name.startswith('hurtlex'):
						categories = ["asm","ps", "pa", "ddf", "ddp", "asf", "pr", "om", "qas", "cds"]
						LEN = len(categories)
						feats = hurtlex_embed_posts(data_dict['text_sen'], data_dict['max_num_sent'], poss_sent_enc_feats_emb_dict[feat_name], data_fold_path,LEN,categories)
			else:
				print("Error! Check the config file.")

			sent_enc_feat_dict['feats'] = feats[:data_dict['test_en_ind']]

			if save_sent_enc_feats:
				print("saving %s sent feats" % feat_name)
				with h5py.File(s_filename, "w") as hf:
					hf.create_dataset('feats', data=feats)
		print (sent_enc_feat_dict['feats'].shape)
		sent_enc_feats.append(sent_enc_feat_dict)

	sent_enc_feat_str += "~" * ((max_num_sent_enc_feats - len(sent_enc_feats_raw)) * max_num_attributes)

	return sent_enc_feats, sent_enc_feat_str[:-1]