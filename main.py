import os
import time
import sys
from sent_enc_embed import sent_enc_featurize
from word_embed import word_featurize
from neuralApproaches import *
from arranging import *
from sklearn.preprocessing import MinMaxScaler
from loadPreProc import *
import numpy as np
import copy
from knowledge_rep import *

sys.setrecursionlimit(10000)
conf_dict_list, conf_dict_com = load_config(sys.argv[1])
os.environ["CUDA_VISIBLE_DEVICES"] = conf_dict_com['GPU_ID']
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

os.makedirs(conf_dict_com["output_folder_name"], exist_ok=True)
os.makedirs(conf_dict_com["save_folder_name"], exist_ok=True)
res_path = conf_dict_com["output_folder_name"] + conf_dict_com["res_filename"]
if os.path.isfile(res_path):
    f_res = open(res_path, 'a')
else:
    f_res = open(res_path, 'w')

tsv_path = conf_dict_com["output_folder_name"] + conf_dict_com["res_tsv_filename"]
if os.path.isfile(tsv_path):
    f_tsv = open(tsv_path, 'a')
else:
    f_tsv = open(tsv_path, 'w')
    f_tsv.write("multi_task_tl\tshare_weights_sep_mt\tprime filename\taux filename\tmerge\tcascade\tcorr setting\tbeta\taugment data\tst_variant\tconfidence_thr\tretaining_ratio\tscaling_confscores\tscale_value\tmodel\tword feats\tsent feats\ttrain ratio\ttrans\tclass imb\tcnn fils\tcnn kernerls\tthresh\trnn dim\tatt dim\tpool k\tstack RNN\tf_I+f_Ma\tstd_d\tf1-Inst\tf1-Macro\tsum_4\tJaccard\tf1-Micro\tExact\tI-Ham\trnn type\tl rate\tb size\tdr1\tdr2\ttest mode\n") 

def get_filename(filename):
    return filename.split('.')[0]

if conf_dict_com['augment_data']:
    with open(conf_dict_com['save_folder_name'] + conf_dict_com['augment_data_filename'], 'rb') as f_data:
        data_dict = pickle.load(f_data)
    if not os.path.isfile(conf_dict_com['augment_data_elmo_folderpath'] + '0.npy'):
        get_elmo_embeddings(conf_dict_com['original_elmo_folderpath'],conf_dict_com['augment_data_elmo_folderpath'], conf_dict_com['augment_index_filename'],data_dict['test_en_ind'])
        get_bert_embeddings(conf_dict_com['original_bert_file'],conf_dict_com['augment_data_bert_filename'], conf_dict_com['augment_index_filename'],data_dict)
    if conf_dict_com["test_mode"]:
        data_dict['train_en_ind'] = data_dict['val_en_ind']
    else:
        data_dict['test_st_ind'] = data_dict['val_st_ind']
        data_dict['test_en_ind'] = data_dict['val_en_ind']
    prime_filename = get_filename(conf_dict_com ['augment_data_filename'])
else:
    data_dict = load_data(conf_dict_com["filename"], conf_dict_com["data_folder_name"], conf_dict_com["save_folder_name"], conf_dict_com['TEST_RATIO'], conf_dict_com['VALID_RATIO'], conf_dict_com['RANDOM_STATE'], conf_dict_com['MAX_WORDS_SENT'], conf_dict_com["test_mode"],conf_dict_com['train_ratio'], conf_dict_com["filename_map_list"][-1])
    prime_filename = get_filename(conf_dict_com['filename'])

conf_scores_array = []
aux_str =""
max_num_tasks = 4
aux_filename_str = ""

def get_aux_str(task_dict):
    aux_str = ''   
    sorted_dict = sorted(task_dict.items(),key = lambda dict:(dict[0]))
    for sort_keys in sorted_dict:
        if sort_keys[0] == "filename":
            aux_str  = aux_str + '+' + get_filename(sort_keys[1])
        else:
            aux_str  = aux_str + '+' + str(sort_keys[1])
    return aux_str[1:]

for single_task_tup in conf_dict_com['single_inp_tasks_list']:
    single_task_name, single_task_dict = single_task_tup
    aux_filename_str = aux_filename_str + get_filename(single_task_dict['filename'])
    aux_str += '~' + single_task_name + '~' + get_aux_str(single_task_dict)
for sep_task_tup in conf_dict_com['sep_inp_tasks_list']:
    sep_task_name, sep_task_dict = sep_task_tup
    aux_filename_str = aux_filename_str + '+' + get_filename(sep_task_dict['filename'])
    aux_str += '~' + sep_task_name + '~' + get_aux_str(sep_task_dict)
aux_str += "~" * ((max_num_tasks - (len(conf_dict_com['single_inp_tasks_list'])+len(conf_dict_com['sep_inp_tasks_list']))) *2)
aux_str = aux_str[1:]

for sep_task_tup in conf_dict_com['sep_inp_tasks_list']:
    sep_task_name, sep_task_dict = sep_task_tup
    sep_load_data(sep_task_dict['filename'], conf_dict_com["data_folder_name"], conf_dict_com["save_folder_name"], conf_dict_com['TEST_RATIO'], conf_dict_com['VALID_RATIO'], conf_dict_com['RANDOM_STATE'], conf_dict_com['MAX_WORDS_SENT'], conf_dict_com["test_mode"],sep_task_dict,data_dict)
for single_task_tup in conf_dict_com['single_inp_tasks_list']:
    single_task_name, single_task_dict = single_task_tup
    if single_task_name == "topic":
        get_vecs(conf_dict_com['augment_data_filename'],conf_dict_com['supervised_data_filename'],conf_dict_com['totalraw_data_filename'],conf_dict_com['augment_index_filename'],conf_dict_com,data_dict,single_task_dict,conf_dict_com['train_ratio'])
    elif single_task_name == "kmeans":
        get_vecs(conf_dict_com['augment_data_filename'],conf_dict_com['supervised_data_filename'],conf_dict_com['totalraw_data_filename'],conf_dict_com['augment_index_filename'],conf_dict_com,data_dict,single_task_dict,conf_dict_com['train_ratio'])

if conf_dict_com['use_knowledge']:
    s_filename = ("%sknowledge_enc_feat.h5~%s" % (conf_dict_com['save_folder_name'],conf_dict_com['test_mode']))
    if not os.path.isfile(s_filename):
        save_postlevel_knowledge_rep(s_filename, conf_dict_com['meaning_pickle_filename'])
    with h5py.File(s_filename, "r") as hf:
        knowledge_array = hf['feats'][:data_dict['test_en_ind']]
        print (knowledge_array)
else:
    knowledge_array = []


print("max # sentences: %d, max # words per sentence: %d, max # words per post: %d" % (data_dict['max_num_sent'], data_dict['max_words_sent'], data_dict['max_post_length']))

numclasses_maplist = []
data_dict_original = copy.deepcopy(data_dict)
for  map_ind,map_filename in enumerate(conf_dict_com["filename_map_list"]):
    update_data_dict(conf_dict_com["data_folder_name"],map_filename,data_dict,data_dict_original)
    numclasses_maplist.append(data_dict['NUM_CLASSES'])
    if conf_dict_com['use_conf_scores']:
        if not os.path.isfile(conf_dict_com['conf_score_filename_list'][map_ind]):
            derive_confscores(data_dict,data_dict_original, conf_dict_com['conf_score_filename_list'],map_ind,conf_dict_com['save_folder_name'])
        with open(conf_dict_com['conf_score_filename_list'][map_ind], 'rb') as conf_data:
            conf_scores_array,initial_train_en_ind,semisup_train_end_ind = pickle.load(conf_data)
            flat_scores=conf_scores_array[initial_train_en_ind:semisup_train_end_ind].flatten()
            max_value = np.amax(flat_scores)
            if conf_dict_com['scaling_confscores'] == "minmax":                   
                scaler = MinMaxScaler(feature_range=(conf_dict_com['scale_value'],1), copy=True)
                scaler.fit(conf_scores_array[initial_train_en_ind:semisup_train_end_ind])
                min_max_scores = scaler.transform(conf_scores_array[initial_train_en_ind:semisup_train_end_ind])
                conf_scores_array[initial_train_en_ind:semisup_train_end_ind] = min_max_scores
            elif conf_dict_com['scaling_confscores'] == "scaleup":
                scale_confscore_array = np.zeros([(semisup_train_end_ind-initial_train_en_ind),conf_scores_array.shape[-1]])
                for ind,confscore in enumerate(conf_scores_array[initial_train_en_ind:semisup_train_end_ind]):
                    for sub_ind,score in enumerate(confscore):
                        scale_confscore_array[ind][sub_ind] = (1/max_value)*score
                conf_scores_array[initial_train_en_ind:semisup_train_end_ind] = scale_confscore_array
            elif conf_dict_com['scaling_confscores'] == "additive":
                additive_confscore_array = np.zeros([(semisup_train_end_ind-initial_train_en_ind),conf_scores_array.shape[-1]])
                for ind,confscore in enumerate(conf_scores_array[initial_train_en_ind:semisup_train_end_ind]):
                    for sub_ind,score in enumerate(confscore):
                        additive_confscore_array[ind][sub_ind] = (1-max_value)+score
                conf_scores_array[initial_train_en_ind:semisup_train_end_ind] = additive_confscore_array
        if conf_dict_com["test_mode"]:
            dummy_conf_scores_array = np.ones([(data_dict['test_en_ind']-data_dict['test_st_ind']),conf_scores_array.shape[-1]])
            conf_scores_array = np.insert(conf_scores_array,len(conf_scores_array),dummy_conf_scores_array,axis=0)

    metr_dict = init_metr_dict()
    for conf_dict in conf_dict_list:
        for prob_trans_type in conf_dict["prob_trans_types"]:
            trainY_list, trainY_noncat_list, num_classes_var, bac_map = transform_labels(data_dict['lab'][:data_dict['train_en_ind']], prob_trans_type, conf_dict_com['test_mode'], conf_dict_com["save_folder_name"],prime_filename,None,conf_dict_com['train_ratio'],data_dict['NUM_CLASSES'],get_filename(map_filename))
            for sep_task_tup in conf_dict_com['sep_inp_tasks_list']:
                sep_task_name, sep_task_dict = sep_task_tup
                if sep_task_name == "sd":
                    sep_task_dict['trainY_list'], sep_task_dict['trainY_noncat_list'], sep_task_dict['num_classes_var'], sep_task_dict['bac_map'] = transform_labels(sep_task_dict['lab'][:sep_task_dict['train_en_ind']], 'binary', conf_dict_com['test_mode'], conf_dict_com["save_folder_name"],get_filename(sep_task_dict['filename']), None,conf_dict_com['train_ratio'],data_dict['NUM_CLASSES'],"NA")
                elif sep_task_name == "sepkmeans":
                    sep_task_dict['trainY_list'], sep_task_dict['trainY_noncat_list'], sep_task_dict['num_classes_var'], sep_task_dict['bac_map'] = transform_labels(sep_task_dict['lab'][:sep_task_dict['train_en_ind']], 'multi_class', conf_dict_com['test_mode'], conf_dict_com["save_folder_name"],get_filename(sep_task_dict['filename']),sep_task_dict['n_clusters'],conf_dict_com['train_ratio'],data_dict['NUM_CLASSES'],"NA")
                elif sep_task_name == "septopics":
                    sep_task_dict['trainY_list'], sep_task_dict['trainY_noncat_list'], sep_task_dict['num_classes_var'], sep_task_dict['bac_map'] = transform_labels(sep_task_dict['lab'][:sep_task_dict['train_en_ind']], None, conf_dict_com['test_mode'], conf_dict_com["save_folder_name"],get_filename(sep_task_dict['filename']),None,conf_dict_com['train_ratio'],data_dict['NUM_CLASSES'], "NA")
                elif sep_task_name == "emotion":
                    sep_task_dict['trainY_list'], sep_task_dict['trainY_noncat_list'], sep_task_dict['num_classes_var'], sep_task_dict['bac_map'] = transform_labels(sep_task_dict['lab'][:sep_task_dict['train_en_ind']], 'multi-label', conf_dict_com['test_mode'], conf_dict_com["save_folder_name"],get_filename(sep_task_dict['filename']),sep_task_dict['n_labels'],conf_dict_com['train_ratio'],data_dict['NUM_CLASSES'], "NA")
                elif sep_task_name == "sarcasm":
                    sep_task_dict['trainY_list'], sep_task_dict['trainY_noncat_list'], sep_task_dict['num_classes_var'], sep_task_dict['bac_map'] = transform_labels(sep_task_dict['lab'][:sep_task_dict['train_en_ind']], 'binary', conf_dict_com['test_mode'], conf_dict_com["save_folder_name"],get_filename(sep_task_dict['filename']),None,conf_dict_com['train_ratio'],data_dict['NUM_CLASSES'], "NA")
                if sep_task_name == "misogyny":
                    sep_task_dict['trainY_list'], sep_task_dict['trainY_noncat_list'], sep_task_dict['num_classes_var'], sep_task_dict['bac_map'] = transform_labels(sep_task_dict['lab'][:sep_task_dict['train_en_ind']], 'binary', conf_dict_com['test_mode'], conf_dict_com["save_folder_name"],get_filename(sep_task_dict['filename']), None,conf_dict_com['train_ratio'],data_dict['NUM_CLASSES'],"NA")
            for class_imb_flag in conf_dict["class_imb_flags"]:
                loss_func_list, nonlin, out_vec_size, cw_list = class_imb_loss_nonlin(trainY_noncat_list, class_imb_flag, num_classes_var, prob_trans_type, conf_dict_com['test_mode'], conf_dict_com["save_folder_name"],conf_dict_com['use_conf_scores'],conf_dict_com['multi_task_tl'],conf_dict_com['classi_loss_weight'], prime_filename, aux_filename_str,conf_dict_com['augment_data'],conf_dict_com['single_inp_tasks_list'],conf_dict_com['sep_inp_tasks_list'],conf_dict_com['uncorr_c_pairs_filename'],conf_dict_com['beta'],conf_dict_com['label_corr_setting'],conf_dict_com['train_ratio'],get_filename(map_filename))
                for model_type in conf_dict["model_types"]:
                    for word_feats_raw in conf_dict["word_feats_l"]:
                        word_feats, word_feat_str = word_featurize(word_feats_raw, model_type, data_dict, conf_dict_com['poss_word_feats_emb_dict'], conf_dict_com['use_saved_word_feats'], conf_dict_com['save_word_feats'], conf_dict_com["data_folder_name"], conf_dict_com["save_folder_name"], conf_dict_com["test_mode"],prime_filename)
                        for sep_task_tup in conf_dict_com['sep_inp_tasks_list']:
                            sep_task_name, sep_task_dict = sep_task_tup 
                            sep_task_dict['word_feats'], sep_task_dict['word_feat_str'] = word_featurize(word_feats_raw, model_type, sep_task_dict, conf_dict_com['poss_word_feats_emb_dict'], conf_dict_com['use_saved_word_feats'], conf_dict_com['save_word_feats'], conf_dict_com["data_folder_name"], conf_dict_com["save_folder_name"], conf_dict_com["test_mode"],get_filename(sep_task_dict['filename']))
                        for sent_enc_feats_raw in conf_dict["sent_enc_feats_l"]:
                            sent_enc_feats, sent_enc_feat_str = sent_enc_featurize(sent_enc_feats_raw, model_type, data_dict, conf_dict_com['poss_sent_enc_feats_emb_dict'], conf_dict_com['use_saved_sent_enc_feats'], conf_dict_com['save_sent_enc_feats'], conf_dict_com["data_folder_name"], conf_dict_com["save_folder_name"], conf_dict_com["test_mode"],prime_filename,conf_dict_com['bert_path'],conf_dict_com['bert_max_seq_len'],conf_dict_com["tbert_paths"])
                            for sep_task_tup in conf_dict_com['sep_inp_tasks_list']:
                                sep_task_name, sep_task_dict = sep_task_tup
                                sep_task_dict['sent_enc_feats'], sep_task_dict['sent_enc_feat_str'] = sent_enc_featurize(sent_enc_feats_raw, model_type, sep_task_dict, conf_dict_com['poss_sent_enc_feats_emb_dict'], conf_dict_com['use_saved_sent_enc_feats'], conf_dict_com['save_sent_enc_feats'], conf_dict_com["data_folder_name"], conf_dict_com["save_folder_name"], conf_dict_com["test_mode"],get_filename(sep_task_dict['filename']),conf_dict_com['bert_path'],conf_dict_com['bert_max_seq_len'],conf_dict_com["tbert_paths"]) 
                            for num_cnn_filters in conf_dict["num_cnn_filters"]:
                                for max_pool_k_val in conf_dict["max_pool_k_vals"]:
                                    for cnn_kernel_set in conf_dict["cnn_kernel_sets"]:
                                        cnn_kernel_set_str = str(cnn_kernel_set)[1:-1].replace(',','').replace(' ', '')
                                        for rnn_type in conf_dict["rnn_types"]:
                                            for rnn_dim in conf_dict["rnn_dims"]:
                                                for att_dim in conf_dict["att_dims"]:
                                                    for stack_rnn_flag in conf_dict["stack_rnn_flags"]:
                                                        mod_op_list_save_list = []
                                                        for thresh in conf_dict["threshes"]:
                                                            startTime = time.time()
                                                            info_str = "model: %s, word_feats = %s, sent_enc_feats = %s, prob_trans_type = %s, class_imb_flag = %s, num_cnn_filters = %s, cnn_kernel_set = %s, rnn_type = %s, rnn_dim = %s, att_dim = %s, max_pool_k_val = %s, stack_rnn_flag = %s, thresh = %s, multi_task_tl = %s, share_weights_sep_mt =%s, prime filename = %s, aux filename = %s, augment data = %s, scaling_confscores = %s, scale_value = %s, beta = %s, use label corr = %s, train ratio = %s, map filename = %s, merge = %s, cascade = %s, knowledge = %s, test mode = %s" % (model_type,word_feat_str,sent_enc_feat_str,prob_trans_type,class_imb_flag,num_cnn_filters,cnn_kernel_set,rnn_type,rnn_dim,att_dim,max_pool_k_val,stack_rnn_flag, thresh, conf_dict_com['multi_task_tl'], conf_dict_com['share_weights_sep_mt'], prime_filename, aux_str, conf_dict_com['augment_data'], conf_dict_com['scaling_confscores'], conf_dict_com['scale_value'],conf_dict_com['beta'],conf_dict_com['label_corr_setting'],conf_dict_com['train_ratio'],map_filename, conf_dict_com['merge'], conf_dict_com['cascade'], conf_dict_com['use_knowledge'], conf_dict_com["test_mode"])
                                                            fname_part = ("%s~%s~%s~%s~%s~%s~%s~%s~%s~%s~%s~%s~%s~%s~%s~%s~%s~%s~%s~%s~%s~%s~%s~%s~%s~%s~%s" % (model_type,word_feat_str,sent_enc_feat_str,prob_trans_type,class_imb_flag,num_cnn_filters,cnn_kernel_set_str,rnn_type,rnn_dim,att_dim,max_pool_k_val,stack_rnn_flag, conf_dict_com['multi_task_tl'],conf_dict_com['share_weights_sep_mt'], prime_filename, aux_str,conf_dict_com['augment_data'],conf_dict_com['scaling_confscores'], conf_dict_com['scale_value'],conf_dict_com['beta'],conf_dict_com['label_corr_setting'],conf_dict_com['train_ratio'],get_filename(map_filename), conf_dict_com["merge"], conf_dict_com['cascade'], conf_dict_com['use_knowledge'], conf_dict_com["test_mode"]))
                                                            pred_vals_total = []
                                                            true_vals_total = []
                                                            for run_ind in range(conf_dict_com["num_runs_list"][map_ind]):
                                                                print('run: %s; %s\n' % (run_ind, info_str))
                                                                if run_ind < len(mod_op_list_save_list):
                                                                    mod_op_list = mod_op_list_save_list[run_ind]   
                                                                else:
                                                                    mod_op_list = []
                                                                    for m_ind, (loss_func, cw, trainY) in enumerate(zip(loss_func_list, cw_list, trainY_list)):
                                                                        mod_op, att_op= train_predict(word_feats,sent_enc_feats, trainY,data_dict, model_type, num_cnn_filters, rnn_type, fname_part, loss_func, cw,nonlin, out_vec_size, rnn_dim, att_dim, max_pool_k_val,stack_rnn_flag,cnn_kernel_set, m_ind, run_ind, conf_dict_com["save_folder_name"], conf_dict_com["use_saved_model"], conf_dict_com["gen_att"], conf_dict_com["LEARN_RATE"], conf_dict_com["dropO1"], conf_dict_com["dropO2"], conf_dict_com["BATCH_SIZE"], conf_dict_com["EPOCHS"], conf_dict_com["save_model"],conf_dict_com["st_variant"],conf_scores_array,conf_dict_com['use_conf_scores'],conf_dict_com['multi_task_tl'],conf_dict_com['share_weights_sep_mt'],conf_dict_com['single_inp_tasks_list'],conf_dict_com['sep_inp_tasks_list'],conf_dict_com['filename_map_list'],map_ind,numclasses_maplist,conf_dict_com['n_fine_tune_layers'],conf_dict_com['bert_path'],conf_dict_com['output_representation'],conf_dict_com['bert_trainable'],conf_dict_com['cascade'],conf_dict_com['merge'], conf_dict_com['slightcascade'],conf_dict_com['use_knowledge'], knowledge_array)
                                                                        if conf_dict_com['use_conf_scores']:
                                                                            mod_op_list.append((mod_op[:,:out_vec_size], att_op))
                                                                        else:
                                                                            mod_op_list.append((mod_op, att_op))
                                                                    mod_op_list_save_list.append(mod_op_list)
                                                                if  map_ind == len(conf_dict_com["filename_map_list"])-1:                                                                 
                                                                    pred_vals, true_vals, metr_dict = evaluate_model(mod_op_list, data_dict, bac_map, prob_trans_type, metr_dict, thresh, conf_dict_com['gen_att'], conf_dict_com["output_folder_name"], ("%s~%d" % (fname_part,run_ind)),data_dict['NUM_CLASSES'])
                                                                    pred_vals_total.append(pred_vals)
                                                                    true_vals_total.append(true_vals)
                                                                    if conf_dict_com['gen_inst_res'] and run_ind == 0:                                                               
                                                                        insights_results(pred_vals, true_vals, data_dict['text'][data_dict['test_st_ind']:data_dict['test_en_ind']], data_dict['text_sen'][data_dict['test_st_ind']:data_dict['test_en_ind']], data_dict['lab'][0:data_dict['train_en_ind']], fname_part, conf_dict_com["output_folder_name"])
                                                            if  map_ind == len(conf_dict_com["filename_map_list"])-1:         
                                                                f_res.write("%s\n\n" % info_str)
                                                                print("%s\n" % info_str)
                                                                metr_dict = aggregate_metr(metr_dict, conf_dict_com["num_runs_list"][map_ind])
                                                                insights_results_lab(pred_vals_total, true_vals_total, data_dict['lab'][0:data_dict['train_en_ind']], fname_part, conf_dict_com["output_folder_name"],conf_dict_com["num_runs_list"][map_ind],data_dict['NUM_CLASSES'],data_dict['FOR_LMAP'] )
                                                                f_tsv.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%.3f\t%.3f\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%s\t%s\t%s\t%s\t%s\t%s\n" % (conf_dict_com['multi_task_tl'],conf_dict_com['share_weights_sep_mt'], prime_filename, aux_str, conf_dict_com['merge'],conf_dict_com['label_corr_setting'],conf_dict_com['beta'], conf_dict_com['augment_data'],conf_dict_com['st_variant'],conf_dict_com['confidence_thr'],conf_dict_com['retaining_ratio'],conf_dict_com['scaling_confscores'],conf_dict_com['scale_value'],model_type,word_feat_str,sent_enc_feat_str,conf_dict_com['train_ratio'],prob_trans_type,class_imb_flag,num_cnn_filters,cnn_kernel_set_str,thresh,rnn_dim,att_dim,max_pool_k_val,stack_rnn_flag,(metr_dict['avg_fl_ma']+metr_dict['avg_fi'])/2,(metr_dict['std_fl_ma']+metr_dict['std_fi'])/2,metr_dict['avg_fi'],metr_dict['avg_fl_ma'],(metr_dict['avg_fl_ma']+metr_dict['avg_fi']+metr_dict['avg_ji']+metr_dict['avg_fl_mi'])/4,metr_dict['avg_ji'],metr_dict['avg_fl_mi'],metr_dict['avg_em'],metr_dict['avg_ihl'],rnn_type,conf_dict_com["LEARN_RATE"],conf_dict_com["BATCH_SIZE"],conf_dict_com["dropO1"],conf_dict_com["dropO2"], conf_dict_com["test_mode"]))
                                                                write_results(metr_dict, f_res)                                                            
                                                                timeLapsed = int(time.time() - startTime + 0.5)
                                                                hrs = timeLapsed/3600.
                                                                t_str = "%.1f hours = %.1f minutes over %d hours\n" % (hrs, (timeLapsed % 3600)/60.0, int(hrs))
                                                                print(t_str)                
                                                                f_res.write("%s\n" % t_str)                                                       
f_res.close()
f_tsv.close()




