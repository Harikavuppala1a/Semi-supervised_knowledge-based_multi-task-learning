import csv 
from sklearn.utils import shuffle
from ast import literal_eval
import numpy as np   
import os
import re
import pickle
from nltk import sent_tokenize


def is_model_hier(model_type):
  if model_type.startswith('hier') or model_type.startswith('uni'):
    return True
  return False

def sep_load_data(filename, data_path, save_path, test_ratio, valid_ratio, rand_state, max_words_sent, test_mode,sep_task_dict,data_dict):
  cl_in_filename = ("%sraw_data~%s~%s.pickle" % (save_path, filename[:-4], max_words_sent))
  if os.path.isfile(cl_in_filename):
    print("loading cleaned unshuffled input")
    with open(cl_in_filename, 'rb') as f_cl_in:
        text, text_sen, label_lists = pickle.load(f_cl_in)
  else:
    r_anum = re.compile(r'([^\sa-z0-9.(?)!])+')
    r_white = re.compile(r'[\s.(?)!]+')
    text = []; label_lists = []; text_sen = []
    with open(data_path + filename, 'r') as csvfile:
      reader = csv.DictReader(csvfile, delimiter = '\t')
      for row in reader:
        post = str(row['post'])
        row_clean = r_white.sub(' ', r_anum.sub('', post.lower())).strip()
        text.append(row_clean)
        se_list = []
        for se in sent_tokenize(post):
          se_cl = r_white.sub(' ', r_anum.sub('', str(se).lower())).strip()
          if se_cl == "":
            continue
          words = se_cl.split(' ')
          while len(words) > max_words_sent:
            se_list.append(' '.join(words[:max_words_sent]))
            words = words[max_words_sent:]
          se_list.append(' '.join(words))
        text_sen.append(se_list)
        label_lists.append(literal_eval(row['label']))

    print("saving cleaned unshuffled input")
    with open(cl_in_filename, 'wb') as f_cl_in:
      pickle.dump([text, text_sen, label_lists], f_cl_in)

  sep_task_dict['text'], sep_task_dict['text_sen'], sep_task_dict['lab'] = text, text_sen, label_lists

  sep_task_dict['max_num_sent'] = data_dict['max_num_sent']
  sep_task_dict['max_post_length'] = data_dict['max_post_length']
  sep_task_dict['max_words_sent'] = max_words_sent

  sep_task_dict['train_en_ind'] = sep_task_dict['test_en_ind'] = len(text)

def update_data_dict(data_path, filename_map,data_dict, data_dict_original):
  conf_map = load_map(data_path + filename_map)
  data_dict['FOR_LMAP'] = conf_map['FOR_LMAP']
  data_dict['LABEL_MAP'] = conf_map['LABEL_MAP']
  data_dict['NUM_CLASSES'] = len(data_dict['FOR_LMAP'])
  data_dict['prob_type'] = conf_map['prob_type']

  for lab_list_ind,label_list in enumerate(data_dict_original['lab']):
    merge_categories_list = []
    for label in label_list:
      orig_label = data_dict_original['FOR_LMAP'][label]
      merge_categories_list.append(data_dict['LABEL_MAP'][orig_label])
    data_dict['lab'][lab_list_ind] = list(set(merge_categories_list))

def load_data(filename, data_path, save_path, test_ratio, valid_ratio, rand_state, max_words_sent, test_mode,train_ratio,filename_map):
  data_dict_filename = ("%sraw_data~%s~%s~%s~%s~%s~%s~%s~%s.pickle" % (save_path, filename[:-4], test_ratio, valid_ratio, rand_state, max_words_sent, train_ratio, test_mode, filename_map[:-4]))
  if os.path.isfile(data_dict_filename):
    print("loading input data")
    with open(data_dict_filename, 'rb') as f_data:
        data_dict = pickle.load(f_data)
  else:      
    cl_in_filename = ("%sraw_data~%s~%s~%s.pickle" % (save_path, filename[:-4], max_words_sent,filename_map[:-4]))
    if os.path.isfile(cl_in_filename):
      print("loading cleaned unshuffled input")
      with open(cl_in_filename, 'rb') as f_cl_in:
          text, text_sen, label_lists, conf_map = pickle.load(f_cl_in)
    else:
      conf_map = load_map(data_path + filename_map)
      r_anum = re.compile(r'([^\sa-z0-9.(?)!])+')
      r_white = re.compile(r'[\s.(?)!]+')
      text = []; label_lists = []; text_sen = []
      with open(data_path + filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile, delimiter = '\t')
        for row in reader:
          post = str(row['post'])
          row_clean = r_white.sub(' ', r_anum.sub('', post.lower())).strip()
          text.append(row_clean)

          se_list = []
          for se in sent_tokenize(post):
            se_cl = r_white.sub(' ', r_anum.sub('', str(se).lower())).strip()
            if se_cl == "":
              continue
            words = se_cl.split(' ')
            while len(words) > max_words_sent:
              se_list.append(' '.join(words[:max_words_sent]))
              words = words[max_words_sent:]
            se_list.append(' '.join(words))
          text_sen.append(se_list)

          cat_list = str(row['labels']).split(',')
          label_ids = list(set([conf_map['LABEL_MAP'][cat] for cat in cat_list]))
          label_lists.append(label_ids)

      print("saving cleaned unshuffled input")
      with open(cl_in_filename, 'wb') as f_cl_in:
        pickle.dump([text, text_sen, label_lists, conf_map], f_cl_in)

    data_dict = {}  
    data_dict['text'], data_dict['text_sen'], data_dict['lab'] = shuffle(text, text_sen, label_lists, random_state = rand_state)
    train_all_index = int((1 - test_ratio - valid_ratio)*len(text)+0.5)
    val_index = int((1 - test_ratio)*len(text)+0.5)

    data_dict['max_num_sent'] = max([len(post_sen) for post_sen in data_dict['text_sen'][:val_index]])
    data_dict['max_post_length'] = max([len(post.split(' ')) for post in data_dict['text'][:val_index]])
    data_dict['max_words_sent'] = max_words_sent

    if test_mode:
      data_dict['train_en_ind'] = int((1 - test_ratio)*train_ratio*len(text)+0.5)
      print (data_dict['train_en_ind'])
      data_dict['test_st_ind'] = val_index
      data_dict['test_en_ind'] = len(text)
    else:
      data_dict['train_en_ind'] = train_all_index
      data_dict['test_st_ind'] = train_all_index
      data_dict['test_en_ind'] = val_index

    data_dict['FOR_LMAP'] = conf_map['FOR_LMAP']
    data_dict['LABEL_MAP'] = conf_map['LABEL_MAP']
    data_dict['NUM_CLASSES'] = len(data_dict['FOR_LMAP'])
    data_dict['prob_type'] = conf_map['prob_type']

    print("saving input data")
    with open(data_dict_filename, 'wb') as f_data:
      pickle.dump(data_dict, f_data)

  return data_dict

def load_map(filename):
    conf_sep = "----------"
    content = ''
    with open(filename, 'r') as f:
      for line in f:
        line = line.strip()
        if line != '' and line[0] != '#':
          content += line

    items = content.split(conf_sep)
    conf_map = {}
    for item in items:
      parts = [x.strip() for x in item.split('=')]
      conf_map[parts[0]] = literal_eval(parts[1])
    # print(conf_map)
    return conf_map

def load_config(filename):
  print("loading config")
  conf_sep_1 = "----------\n"
  conf_sep_2 = "**********\n"
  conf_dict_list = []
  conf_dict_com = {}
  with open(filename, 'r') as f:
    content = f.read()
  break_ind = content.find(conf_sep_2)  

  nested_comps = content[:break_ind].split(conf_sep_1)
  for comp in nested_comps:
    pairs = comp.split(';')
    conf_dict = {}
    for pair in pairs:
      pair = ''.join(pair.split())
      if pair == "" or pair[0] == '#': 
        continue
      parts = pair.split('=')
      conf_dict[parts[0]] = literal_eval(parts[1])
    conf_dict_list.append(conf_dict)

  lines = content[break_ind+len(conf_sep_2):].split('\n')
  for pair in lines:
    pair = ''.join(pair.split())
    if pair == "" or pair[0] == '#': 
      continue
    parts = pair.split('=')
    conf_dict_com[parts[0]] = literal_eval(parts[1])

  print("config loaded")
  return conf_dict_list, conf_dict_com


def binary_to_decimal(b_list):
  out1 = 0
  for bit in b_list:
    out1 = (out1 << 1) | bit
  return out1

def num_to_bin_array(num):
  f_str = ("0%sb" % NUM_CLASSES)
  return [int(x) for x in format(num, f_str)]

def num_to_label_list(num):
  f_str = ("%sb" % NUM_CLASSES)
  return [ind for ind, x in enumerate(format(num, f_str)) if x == '1']

def powerset_vec_to_bin_arrays(vec):
  return [num_to_bin_array(x) for x in vec]    

def powerset_vec_to_label_lists(vec, bac_map):
  return [num_to_label_list(bac_map[x]) for x in vec]    

def br_op_to_label_lists(vecs):
  op_list = []
  for i in range(len(vecs[0])):
    cat_l = []
    for j in range(NUM_CLASSES):
      if vecs[j][i] == 1:
        cat_l.append(j)  
    op_list.append(cat_l)    
  return op_list

def di_op_to_label_lists(vecs,NUM_CLASSES):
  op_list = []
  for i in range(len(vecs)):
    cat_l = []
    for j in range(NUM_CLASSES):
      if vecs[i][j] == 1:
        cat_l.append(j)  
    op_list.append(cat_l)    
  return op_list

def map_labels_to_num(label_ids):
  arr = [0] * NUM_CLASSES
  for label_id in label_ids:
    arr[label_id] = 1
  num = binary_to_decimal(arr) 
  return num

def fit_trans_labels_powerset(org_lables,NUM_CLASSES):
  ind = 0
  for_map = {}
  bac_map = {}
  new_labels = np.empty(len(org_lables), dtype=np.int64)
  for s_ind, label_ids in enumerate(org_lables):
    l = map_labels_to_num(label_ids)
    if l not in for_map:
      for_map[l] = ind
      bac_map[ind] = l
      ind += 1
    new_labels[s_ind] = for_map[l]
  num_lp_classes = ind
  return new_labels, num_lp_classes, bac_map, for_map

def trans_labels_powerset(org_lables, for_map, num_lp_classes):
  new_labels = np.empty(len(org_lables), dtype=np.int64)
  for ind, label_ids in enumerate(org_lables):
    l = map_labels_to_num(label_ids)
    if l not in for_map:
      new_labels[ind] = np.random.randint(0, num_lp_classes)
    else:
      new_labels[ind] = for_map[l]  
  return new_labels

def load_label_powerset(filename, test_ratio, valid_ratio, rand_state):
  data_dict = load_data(filename, test_ratio, valid_ratio, rand_state)
  lp_trainY, num_lp_classes, bac_map = fit_trans_labels_powerset(data_dict['trainY'])
  return data_dict, lp_trainY, num_lp_classes, bac_map

def trans_labels_multi_hot(org_lables,NUM_CLASSES):
  label_arr = np.zeros([len(org_lables), NUM_CLASSES], dtype=np.int64)
  for sample_ind, label_ids in enumerate(org_lables):
    for label_id in label_ids:
      label_arr[sample_ind][label_id] = 1
  return label_arr

def trans_labels_multi_hot_emotion(org_lables,NUM_CLASSES):
  print (len(org_lables))
  label_arr = np.zeros([len(org_lables), NUM_CLASSES], dtype=np.int64)
  for sample_ind, label_ids in enumerate(org_lables):
    for ind, label_id in enumerate(label_ids):
      if int(label_id) == 1:
        label_arr[sample_ind][ind] = 1
  return label_arr 

def load_multi_hot(filename, test_ratio, valid_ratio, rand_state):
  data_dict = load_data(filename, test_ratio, valid_ratio, rand_state)
  label_arr = trans_labels_multi_hot(data_dict['trainY'])
  return data_dict, label_arr

def trans_labels_BR(org_lables,NUM_CLASSES):
  label_lists_br = [np.zeros(len(org_lables), dtype=np.int64) for i in range(NUM_CLASSES)]
  for sample_ind, label_ids in enumerate(org_lables):
    for label_id in label_ids:
      label_lists_br[label_id][sample_ind] = 1
  return label_lists_br

def load_bin_relevance(filename, test_ratio, valid_ratio, rand_state):
  data_dict = load_data(filename, test_ratio, valid_ratio, rand_state)
  label_lists_br = trans_labels_BR(data_dict['trainY'])
  return data_dict, label_lists_br
  
def trans_labels_bin_classi(org_lables):
  print (len(org_lables))
  return [np.array([l for l in org_lables], dtype=np.int64)]

def trans_labels_multi_classi(org_lables):
  return np.array([l for l in org_lables], dtype=np.int64)

def weights_cat(org_lables):
  w_arr = np.empty(NUM_CLASSES)
  scores = np.zeros(NUM_CLASSES)
  for lab_ids in org_lables:
    row_sum = 0
    inds_to_update = []
    for lab_id, lab_val in enumerate(lab_ids):
      if lab_val == 1:
        row_sum += 1
        inds_to_update.append(lab_id)
    for lab_id in inds_to_update:
      scores[lab_id] += 1/row_sum
  for i in range(NUM_CLASSES):
    w_arr[i] = len(org_lables)/scores[i]
  return w_arr

