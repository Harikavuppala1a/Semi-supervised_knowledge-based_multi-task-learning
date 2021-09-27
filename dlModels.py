import keras
import tensorflow as tf
from keras import backend as K
from keras.layers import TimeDistributed, Embedding, Dense, Input, Flatten, Conv1D, GlobalMaxPooling1D, Dropout, LSTM, GRU, Bidirectional, concatenate,Lambda, average
from keras.models import Model
from keras import optimizers
from keras.engine.topology import Layer
from keras import initializers
import numpy as np
import tensorflow_hub as hub
from keras.utils import multi_gpu_model

def sen_embed(enc_algo, sen_word_emb, word_cnt_sent, word_emb_len, dropO1, num_cnn_filters, max_pool_k_val, kernel_sizes, rnn_dim, att_dim, rnn_type, att_outputs, use_saved_layers, saved_layers_dict):
    if enc_algo == "rnn":
        rnn_sen_mod, att_mod, = rnn_sen_embed(word_cnt_sent, word_emb_len, dropO1, rnn_dim, att_dim, rnn_type, use_saved_layers, saved_layers_dict)
        rnn_sen_emb_output = TimeDistributed(rnn_sen_mod)(sen_word_emb)    
        if att_dim > 0:
            att_outputs.append(TimeDistributed(att_mod)(sen_word_emb))
        return [rnn_sen_emb_output], att_outputs
    elif enc_algo == "cnn":
        cnn_sen_mod = cnn_sen_embed(word_cnt_sent, word_emb_len, dropO1, num_cnn_filters, max_pool_k_val, kernel_sizes) 
        return [TimeDistributed(cnn_sen_mod)(sen_word_emb)], att_outputs
    else:
        rnn_sen_mod, att_mod = rnn_sen_embed(word_cnt_sent, word_emb_len, dropO1, rnn_dim, att_dim, rnn_type, use_saved_layers, saved_layers_dict)
        rnn_sen_emb_output = TimeDistributed(rnn_sen_mod)(sen_word_emb)    
        if att_dim > 0:
            att_outputs.append(TimeDistributed(att_mod)(sen_word_emb))
        cnn_sen_emb_output = TimeDistributed(cnn_sen_embed(word_cnt_sent, word_emb_len, dropO1, num_cnn_filters, max_pool_k_val, kernel_sizes))(sen_word_emb)
        
        if enc_algo == "comb_cnn_rnn":
            return [concatenate([cnn_sen_emb_output, rnn_sen_emb_output])], att_outputs
        elif enc_algo == "sep_cnn_rnn":
            return [cnn_sen_emb_output, rnn_sen_emb_output], att_outputs

def flat_embed(enc_algo, word_emb_seq, word_cnt_post, word_emb_len, dropO1, num_cnn_filters, max_pool_k_val, kernel_sizes, rnn_dim, att_dim, rnn_type, att_outputs,use_saved_layers, saved_layers_dict):
    if enc_algo == "rnn":
        if att_dim > 0:
            rnn_mod, att_mod = rnn_sen_embed(word_cnt_post, word_emb_len, dropO1, rnn_dim, att_dim, rnn_type,use_saved_layers, saved_layers_dict)
            att_outputs.append(att_mod(word_emb_seq))
        else:
            rnn_mod = rnn_sen_embed(word_cnt_post, word_emb_len, dropO1, rnn_dim, att_dim, rnn_type,use_saved_layers, saved_layers_dict)
        return rnn_mod(word_emb_seq), att_outputs
    elif enc_algo == "cnn":
        cnn_mod = cnn_sen_embed(word_cnt_post, word_emb_len, dropO1, num_cnn_filters, max_pool_k_val, kernel_sizes)
        return cnn_mod(word_emb_seq), att_outputs
    elif enc_algo == "comb_cnn_rnn":
        if att_dim > 0:
            rnn_mod, att_mod = rnn_sen_embed(word_cnt_post, word_emb_len, dropO1, rnn_dim, att_dim, rnn_type,use_saved_layers, saved_layers_dict)
            rnn_emb_output = rnn_mod(word_emb_seq)
            att_outputs.append(att_mod(word_emb_seq))
        else:
            rnn_emb_output = rnn_sen_embed(word_cnt_post, word_emb_len, dropO1, rnn_dim, att_dim, rnn_type,use_saved_layers, saved_layers_dict)(word_emb_seq)
        cnn_emb_output = cnn_sen_embed(word_cnt_post, word_emb_len, dropO1, num_cnn_filters, max_pool_k_val, kernel_sizes)(word_emb_seq)

        return concatenate([cnn_emb_output, rnn_emb_output]), att_outputs

def return_and_point_to_next(saved_layers_dict):
    v = saved_layers_dict['val'][saved_layers_dict['ind']]
    saved_layers_dict['ind'] += 1
    return v

def rnn_sen_embed(word_cnt_sent, word_emb_len, dropO1, rnn_dim, att_dim, rnn_type, use_saved_layers, saved_layers_dict):
    if use_saved_layers:
        blstm_layer = return_and_point_to_next(saved_layers_dict)
        if att_dim > 0:
            a_layer = return_and_point_to_next(saved_layers_dict)
    else:
        if rnn_type == 'lstm':
            blstm_layer = Bidirectional(LSTM(rnn_dim, return_sequences=(att_dim > 0)))
        else:
            blstm_layer = Bidirectional(GRU(rnn_dim, return_sequences=(att_dim > 0)))
        saved_layers_dict['val'].append(blstm_layer)
        if att_dim > 0:
            a_layer = attLayer_hier(att_dim)
            saved_layers_dict['val'].append(a_layer)

    w_emb_input_seq = Input(shape=(word_cnt_sent, word_emb_len), name='emb_input')
    blstm_l = blstm_layer(w_emb_input_seq)
    if att_dim > 0:
        blstm_l, att_w = a_layer(blstm_l)
        return Model(w_emb_input_seq, blstm_l), Model(w_emb_input_seq, att_w)
    else:
        return Model(w_emb_input_seq, blstm_l), None

def cnn_sen_embed(word_cnt_sent, word_emb_len, dropO1, num_cnn_filters, max_pool_k_val, kernel_sizes):
    w_emb_input_seq = Input(shape=(word_cnt_sent, word_emb_len), name='emb_input')
    conv_l_list = []
    for k in kernel_sizes:
        conv_t = Conv1D(num_cnn_filters, k, padding='same', activation='relu')(w_emb_input_seq)
        if max_pool_k_val == 1:
            pool_t = GlobalMaxPooling1D()(conv_t)
        else:
            pool_t = kmax_pooling(max_pool_k_val)(conv_t)
        conv_l_list.append(pool_t)
    feat_vec = concatenate(conv_l_list)
    return Model(w_emb_input_seq, feat_vec)

def post_embed(sen_emb, rnn_dim, att_dim, rnn_type, stack_rnn_flag, att_outputs, use_saved_layers, saved_layers_dict):
    if use_saved_layers:
        blstm_layer = return_and_point_to_next(saved_layers_dict)
        if att_dim > 0:
            a_layer = return_and_point_to_next(saved_layers_dict)
    else:
        if rnn_type == 'lstm':
            blstm_layer = Bidirectional(LSTM(rnn_dim, return_sequences=(att_dim > 0)))
        else:
            blstm_layer = Bidirectional(GRU(rnn_dim, return_sequences=(att_dim > 0)))
        saved_layers_dict['val'].append(blstm_layer)
        if att_dim > 0:
            a_layer = attLayer_hier(att_dim)
            saved_layers_dict['val'].append(a_layer)

    blstm_l = blstm_layer(sen_emb)

    if att_dim > 0:
        blstm_l, att_w = a_layer(blstm_l)
        att_outputs.append(att_w)
    return blstm_l, att_outputs

def add_word_sen_emb_p1(model_inputs, word_emb_input, word_f_sen_word_emb, word_f_sen_emb_size, enc_algo, stage1_id, stage2_id, p1_dict):
        model_inputs.append(word_emb_input)
        if stage1_id in p1_dict:
            p1_dict[stage1_id]["comb_feature_list"].append(word_f_sen_word_emb)
            p1_dict[stage1_id]["word_emb_len"] += word_f_sen_emb_size
            p1_dict[stage1_id]["enc_algo"] = enc_algo
        else:
            p1_dict[stage1_id] = {}
            p1_dict[stage1_id]["comb_feature_list"] = [word_f_sen_word_emb]
            p1_dict[stage1_id]["word_emb_len"] = word_f_sen_emb_size
            p1_dict[stage1_id]["stage2"] = stage2_id 
            p1_dict[stage1_id]["enc_algo"] = enc_algo

def add_sen_emb_p2(sen_emb, stage2_id, p2_dict):
        if stage2_id in p2_dict:
            p2_dict[stage2_id]["comb_feature_list"].append(sen_emb)
        else:
            p2_dict[stage2_id] = {}
            p2_dict[stage2_id]["comb_feature_list"] = [sen_emb]

def add_word_emb_p_flat(model_inputs, word_emb_input, word_f_word_emb, word_f_emb_size, enc_algo, m_id, p_dict):
        model_inputs.append(word_emb_input)
        if m_id in p_dict:
            p_dict[m_id]["comb_feature_list"].append(word_f_word_emb)
            p_dict[m_id]["word_emb_len"] += word_f_emb_size
            p_dict[m_id]["enc_algo"] = enc_algo
        else:
            p_dict[m_id] = {}
            p_dict[m_id]["comb_feature_list"] = [word_f_word_emb]
            p_dict[m_id]["word_emb_len"] = word_f_emb_size 
            p_dict[m_id]["enc_algo"] = enc_algo

def multi_inputs_slightcascade(sent_cnt, word_cnt_sent, rnn_dim, att_dim, word_feats, sen_enc_feats, dropO1, dropO2, nonlin, out_vec_size, rnn_type, stack_rnn_flag, num_cnn_filters, max_pool_k_val, kernel_sizes,st_variant,use_conf_scores,multi_task_tl, share_weights_sep_mt, single_inp_tasks_list, sep_inp_tasks_list,n_fine_tune_layers,tf_hub_path, output_representation, bert_trainable,cascade,use_knowledge, knowledge_array):
    post_vec, model_inputs, att_mod, post_mod, saved_layers, classi_name_main= hier_fuse(sent_cnt, word_cnt_sent, rnn_dim, att_dim, word_feats, sen_enc_feats, dropO1, dropO2, nonlin, out_vec_size, single_inp_tasks_list, rnn_type, stack_rnn_flag, num_cnn_filters, max_pool_k_val, kernel_sizes,st_variant,use_conf_scores,multi_task_tl,"classi", False, [],n_fine_tune_layers,tf_hub_path, output_representation, bert_trainable,cascade,use_knowledge, knowledge_array)
    main_inputs = model_inputs.copy()

    for sep_task_tup in sep_inp_tasks_list:
        sep_task_name, sep_task_dict = sep_task_tup
        if multi_task_tl =="tr_learn" or share_weights_sep_mt:
            aux_post_vec, aux_model_inputs, aux_att_mod, aux_post_mod, unused_saved_layers, classi_name_sd= hier_fuse(sep_task_dict['max_num_sent'], sep_task_dict['max_words_sent'], rnn_dim, att_dim, sep_task_dict['word_feats'], sep_task_dict['sent_enc_feats'], dropO1, dropO2, sep_task_dict['nonlin'], sep_task_dict['out_vec_size'], [], rnn_type, stack_rnn_flag, num_cnn_filters, max_pool_k_val, kernel_sizes,st_variant,use_conf_scores,multi_task_tl,sep_task_name, True, saved_layers,n_fine_tune_layers,tf_hub_path, output_representation, bert_trainable,cascade,use_knowledge, knowledge_array)
        else:
            aux_post_vec, aux_model_inputs, aux_att_mod, aux_post_mod, unused_saved_layers,classi_name_sd = hier_fuse(sep_task_dict['max_num_sent'], sep_task_dict['max_words_sent'], rnn_dim, att_dim, sep_task_dict['word_feats'], sep_task_dict['sent_enc_feats'], dropO1, dropO2, sep_task_dict['nonlin'], sep_task_dict['out_vec_size'], [], rnn_type, stack_rnn_flag, num_cnn_filters, max_pool_k_val, kernel_sizes,st_variant,use_conf_scores,multi_task_tl,sep_task_name, False, [],n_fine_tune_layers,tf_hub_path, output_representation, bert_trainable,cascade,use_knowledge, knowledge_array)
        model_inputs.extend(aux_model_inputs)

    hidden_lay_arr = []
    for d_ind in range(2):
        hidden_lay_arr.append(Dense(int(post_vec.shape[1]),activation='relu'))

    dense_postvec = hidden_lay_arr[0](post_vec)
   
    out_vec_list = []
    dr2_l = Dropout(dropO2)(dense_postvec)
    for single_task_tup in single_inp_tasks_list:
        single_task_name, single_task_dict = single_task_tup
        out_vec_list.append(Dense(single_task_dict['out_vec_size'], activation=single_task_dict['nonlin'], name = single_task_name)(dr2_l))

    for sep_task_tup in sep_inp_tasks_list:
        sep_task_name, sep_task_dict = sep_task_tup
        out_vec_list.append(Dense(sep_task_dict['out_vec_size'], activation=sep_task_dict['nonlin'], name = sep_task_name)(aux_post_vec))

    dense_single = hidden_lay_arr[1](dense_postvec)
    fin_dense_lay = Dense(out_vec_size, activation=nonlin, name=classi_name_main)
    out_vec_list.insert(0,fin_dense_lay(dense_single))

    main_output = fin_dense_lay(hidden_lay_arr[1](dense_postvec))

    if multi_task_tl == "multi_task":   
        return Model(main_inputs, main_output),Model(model_inputs,out_vec_list),None,None
    # elif multi_task_tl == "tr_learn":
    #     return Model(, ),aux_model,None, None

def multi_inputs_cascade(sent_cnt, word_cnt_sent, rnn_dim, att_dim, word_feats, sen_enc_feats, dropO1, dropO2, nonlin, out_vec_size, rnn_type, stack_rnn_flag, num_cnn_filters, max_pool_k_val, kernel_sizes,st_variant,use_conf_scores,multi_task_tl, share_weights_sep_mt, single_inp_tasks_list, sep_inp_tasks_list,n_fine_tune_layers,tf_hub_path, output_representation, bert_trainable,cascade,use_knowledge, knowledge_array):
    post_vec, model_inputs, att_mod, post_mod, saved_layers, classi_name_main= hier_fuse(sent_cnt, word_cnt_sent, rnn_dim, att_dim, word_feats, sen_enc_feats, dropO1, dropO2, nonlin, out_vec_size, single_inp_tasks_list, rnn_type, stack_rnn_flag, num_cnn_filters, max_pool_k_val, kernel_sizes,st_variant,use_conf_scores,multi_task_tl,"classi", False, [],n_fine_tune_layers,tf_hub_path, output_representation, bert_trainable,cascade,use_knowledge, knowledge_array)
    main_inputs = model_inputs.copy()

    final_pv_list = [Dense(int(post_vec.shape[1]),activation='relu',name ='dense1')(post_vec)]
    
    for sep_task_tup in sep_inp_tasks_list:
        sep_task_name, sep_task_dict = sep_task_tup
        if multi_task_tl =="tr_learn" or share_weights_sep_mt:
            aux_post_vec, aux_model_inputs, aux_att_mod, aux_post_mod, unused_saved_layers, classi_name_sd= hier_fuse(sep_task_dict['max_num_sent'], sep_task_dict['max_words_sent'], rnn_dim, att_dim, sep_task_dict['word_feats'], sep_task_dict['sent_enc_feats'], dropO1, dropO2, sep_task_dict['nonlin'], sep_task_dict['out_vec_size'], [], rnn_type, stack_rnn_flag, num_cnn_filters, max_pool_k_val, kernel_sizes,st_variant,use_conf_scores,multi_task_tl,sep_task_name, True, saved_layers,n_fine_tune_layers,tf_hub_path, output_representation, bert_trainable,cascade, use_knowledge, knowledge_array)
        else:
            aux_post_vec, aux_model_inputs, aux_att_mod, aux_post_mod, unused_saved_layers,classi_name_sd = hier_fuse(sep_task_dict['max_num_sent'], sep_task_dict['max_words_sent'], rnn_dim, att_dim, sep_task_dict['word_feats'], sep_task_dict['sent_enc_feats'], dropO1, dropO2, sep_task_dict['nonlin'], sep_task_dict['out_vec_size'], [], rnn_type, stack_rnn_flag, num_cnn_filters, max_pool_k_val, kernel_sizes,st_variant,use_conf_scores,multi_task_tl,sep_task_name, False, [],n_fine_tune_layers,tf_hub_path, output_representation, bert_trainable,cascade,use_knowledge, knowledge_array)
        final_pv_list.append(Dense(int(aux_post_vec.shape[1]),activation='relu')(aux_post_vec))
        model_inputs.extend(aux_model_inputs)
    final_pv = concatenate(final_pv_list)

    hidden_lay_arr = []
    for d_ind in range(2):
        hidden_lay_arr.append(Dense(int(final_pv.shape[1]),activation='relu'))
   
    out_vec_list = []
    dr2_l = Dropout(dropO2)(final_pv)
    for single_task_tup in single_inp_tasks_list:
        single_task_name, single_task_dict = single_task_tup
        out_vec_list.append(Dense(single_task_dict['out_vec_size'], activation=single_task_dict['nonlin'], name = single_task_name)(dr2_l))
    
    dense_single = hidden_lay_arr[0](final_pv)

    dr2_l = Dropout(dropO2)(dense_single)
    for sep_task_tup in sep_inp_tasks_list:
        sep_task_name, sep_task_dict = sep_task_tup
        out_vec_list.append(Dense(sep_task_dict['out_vec_size'], activation=sep_task_dict['nonlin'], name = sep_task_name)(dr2_l))

    dense_sep = hidden_lay_arr[1](dense_single)
    fin_dense_lay = Dense(out_vec_size, activation=nonlin, name=classi_name_main)
    out_vec_list.insert(0,fin_dense_lay(dense_sep))

    main_output = fin_dense_lay(hidden_lay_arr[1](hidden_lay_arr[0](concatenate([final_pv_list[0] , final_pv_list[0]]))))
    # print (main_output.shape)
    if multi_task_tl == "multi_task":   
        return Model(main_inputs, main_output),Model(model_inputs,out_vec_list),None,None
    # elif multi_task_tl == "tr_learn":
    #     return Model(, ),aux_model,None, None

def multi_inputs(sent_cnt, word_cnt_sent, rnn_dim, att_dim, word_feats, sen_enc_feats, dropO1, dropO2, nonlin, out_vec_size, rnn_type, stack_rnn_flag, num_cnn_filters, max_pool_k_val, kernel_sizes,st_variant,use_conf_scores,multi_task_tl, share_weights_sep_mt, single_inp_tasks_list, sep_inp_tasks_list,n_fine_tune_layers,tf_hub_path, output_representation, bert_trainable,cascade,use_knowledge, knowledge_array):
    model, model_multi_task_or_tr, att_mod, post_vec, model_inputs, model_outputs, saved_layers = hier_fuse(sent_cnt, word_cnt_sent, rnn_dim, att_dim, word_feats, sen_enc_feats, dropO1, dropO2, nonlin, out_vec_size, single_inp_tasks_list, rnn_type, stack_rnn_flag, num_cnn_filters, max_pool_k_val, kernel_sizes,st_variant,use_conf_scores,multi_task_tl,"classi", False, [],n_fine_tune_layers,tf_hub_path, output_representation, bert_trainable,cascade,use_knowledge, knowledge_array)
    for sep_task_tup in sep_inp_tasks_list:
        sep_task_name, sep_task_dict = sep_task_tup
        if multi_task_tl =="tr_learn" or share_weights_sep_mt:
            aux_model, aux_model_multi_task_or_tr, aux_att_mod, aux_post_vec, aux_model_inputs,aux_model_outputs, unused_saved_layers = hier_fuse(sep_task_dict['max_num_sent'], sep_task_dict['max_words_sent'], rnn_dim, att_dim, sep_task_dict['word_feats'], sep_task_dict['sent_enc_feats'], dropO1, dropO2, sep_task_dict['nonlin'], sep_task_dict['out_vec_size'], [], rnn_type, stack_rnn_flag, num_cnn_filters, max_pool_k_val, kernel_sizes,st_variant,use_conf_scores,multi_task_tl,sep_task_name, True, saved_layers,n_fine_tune_layers,tf_hub_path, output_representation, bert_trainable,cascade,use_knowledge, knowledge_array)
        else:
            aux_model, aux_model_multi_task_or_tr, aux_att_mod, aux_post_vec, aux_model_inputs,aux_model_outputs, unused_saved_layers = hier_fuse(sep_task_dict['max_num_sent'], sep_task_dict['max_words_sent'], rnn_dim, att_dim, sep_task_dict['word_feats'], sep_task_dict['sent_enc_feats'], dropO1, dropO2, sep_task_dict['nonlin'], sep_task_dict['out_vec_size'], [], rnn_type, stack_rnn_flag, num_cnn_filters, max_pool_k_val, kernel_sizes,st_variant,use_conf_scores,multi_task_tl,sep_task_name, False, [],n_fine_tune_layers,tf_hub_path, output_representation, bert_trainable,cascade,use_knowledge, knowledge_array)
        model_outputs.extend(aux_model_outputs)
        model_inputs.extend(aux_model_inputs)

    if multi_task_tl == "multi_task":   
        return model,Model(model_inputs,model_outputs),None,None
    elif multi_task_tl == "tr_learn":
        return model,aux_model,None, None

def gen_tune_bert_model(comp_dim,rnn_dim,att_dim,n_fine_tune_layers,tf_hub_path, output_representation, bert_trainable):
      
    b_input = Input(shape=(comp_dim,))
    ind_len = int(comp_dim/3)
    in_id = Lambda(lambda x: x[:,0:ind_len], output_shape=(ind_len,))(b_input)
    in_mask = Lambda(lambda x: x[:,ind_len:2*ind_len], output_shape=(ind_len,))(b_input)
    in_segment = Lambda(lambda x: x[:,2*ind_len:3*ind_len], output_shape=(ind_len,))(b_input)
    bert_inputs = [in_id, in_mask, in_segment]
    bert_output = BertLayer(n_fine_tune_layers,tf_hub_path, output_representation, bert_trainable)(bert_inputs)

    blstm_l = Bidirectional(LSTM(rnn_dim, return_sequences=(att_dim > 0)))(bert_output)
    if att_dim > 0:
        blstm_l, att_w = attLayer_hier(att_dim)(blstm_l)
    m = Model(b_input,blstm_l)
    m.summary()
    return m

def hier_fuse(sent_cnt, word_cnt_sent, rnn_dim, att_dim, word_feats, sen_enc_feats, dropO1, dropO2, nonlin, out_vec_size, single_inp_tasks_list, rnn_type, stack_rnn_flag, num_cnn_filters, max_pool_k_val, kernel_sizes,st_variant,use_conf_scores,multi_task_tl,classi_name, use_saved_layers, saved_layers,n_fine_tune_layers,tf_hub_path, output_representation, bert_trainable,cascade,use_knowledge, knowledge_array):
    p1_dict = {}
    p2_dict = {}
    model_inputs = []
    att_outputs = []
    saved_layers_dict = {'ind': 0, 'val': saved_layers}
    for word_feat in word_feats:
        if 'embed_mat' in word_feat:
            word_f_input, word_f_sen_word_emb = tunable_embed_hier_embed(sent_cnt, word_cnt_sent, len(word_feat['embed_mat']), word_feat['embed_mat'], word_feat['emb'], dropO1)
            add_word_sen_emb_p1(model_inputs, word_f_input, word_f_sen_word_emb, word_feat['embed_mat'].shape[-1], word_feat['s_enc'], word_feat['m_id'][0], word_feat['m_id'][1:], p1_dict)
        else:
            word_f_input = Input(shape=(sent_cnt, word_cnt_sent, word_feat['dim_shape'][-1]))
            word_f_sen_word_emb = Dropout(dropO1)(word_f_input)
            add_word_sen_emb_p1(model_inputs, word_f_input, word_f_sen_word_emb, word_feat['dim_shape'][-1], word_feat['s_enc'], word_feat['m_id'][0], word_feat['m_id'][1:], p1_dict)
    
    for my_dict in p1_dict.values():
        my_dict["sen_word_emb"] = concatenate(my_dict["comb_feature_list"]) if len(my_dict["comb_feature_list"]) > 1 else my_dict["comb_feature_list"][0]
        sen_emb_list, att_outputs = sen_embed(my_dict["enc_algo"], my_dict["sen_word_emb"], word_cnt_sent, my_dict["word_emb_len"], dropO1, num_cnn_filters, max_pool_k_val, kernel_sizes, rnn_dim, att_dim, rnn_type, att_outputs, use_saved_layers, saved_layers_dict)
        for ind, sen_emb in enumerate(sen_emb_list):
            add_sen_emb_p2(sen_emb, my_dict["stage2"][ind], p2_dict)

    for sen_enc_feat in sen_enc_feats:
        sen_f_input = Input(shape=(sent_cnt, sen_enc_feat['feats'].shape[-1]))
        model_inputs.append(sen_f_input)
        if sen_enc_feat['emb'].startswith('trainable'):
            tune_bert_model = gen_tune_bert_model(sen_enc_feat['feats'].shape[-1],rnn_dim,att_dim,n_fine_tune_layers,tf_hub_path, output_representation, bert_trainable)
            sen_f_dr1 = TimeDistributed(tune_bert_model)(sen_f_input)
            sen_f_dr1 = Dropout(dropO1)(sen_f_dr1)
        else:
            sen_f_dr1 = Dropout(dropO1)(sen_f_input)               
        add_sen_emb_p2(sen_f_dr1, sen_enc_feat['m_id'], p2_dict)

    post_vec_list = []    
    for stage2_val, my_dict in p2_dict.items():
        my_dict["sen_emb"] = concatenate(my_dict["comb_feature_list"]) if len(my_dict["comb_feature_list"]) > 1 else my_dict["comb_feature_list"][0]
        post_emb, att_outputs = post_embed(my_dict["sen_emb"], rnn_dim, att_dim, rnn_type, stack_rnn_flag, att_outputs, use_saved_layers, saved_layers_dict)
        post_vec_list.append(post_emb)
    print (len(post_vec_list))
    
    if len(model_inputs) == 1:
        model_inputs = model_inputs[0]
    att_mod = Model(model_inputs, att_outputs) if att_outputs else None
    if st_variant.startswith('diversity') or st_variant.startswith('intersection'):
        post_mod = Model(model_inputs,post_vec)
    else: 
        post_mod = None

    if use_conf_scores:
        conf_input = Input(shape=(out_vec_size,))
        model_inputs.append(conf_input)
    else:
        conf_input = None
    if use_knowledge and classi_name == "classi":
        knowl_input = Input(shape=(knowledge_array.shape[-1],))
        model_inputs.append(knowl_input)
        
    post_vec = concatenate(post_vec_list) if len(post_vec_list) > 1 else post_vec_list[0]
    if cascade:
        return post_vec ,model_inputs,att_mod,post_mod, saved_layers_dict['val'], classi_name

    else:
        mod, mod_multi_task_or_tr, out_vec_list = apply_dense(model_inputs, dropO2, post_vec, nonlin, out_vec_size, single_inp_tasks_list, conf_input,multi_task_tl,classi_name)    
        return mod, mod_multi_task_or_tr, att_mod, post_mod, model_inputs, out_vec_list, saved_layers_dict['val']

def flat_fuse(word_cnt_post, rnn_dim, att_dim, word_feats, sen_enc_feats, dropO1, dropO2, nonlin, out_vec_size, single_inp_tasks_list, rnn_type, stack_rnn_flag, num_cnn_filters, max_pool_k_val, kernel_sizes, multi_task_tl, classi_name):
    p_dict = {}
    model_inputs = []
    att_outputs = []
    conf_input = None
    saved_layers_dict = {'ind': 0, 'val': []}
    for word_feat in word_feats:
        if 'embed_mat' in word_feat:
            word_f_input, word_f_word_emb_raw = tunable_embed_apply(word_cnt_post, len(word_feat['embed_mat']), word_feat['embed_mat'], word_feat['emb'])
            word_f_word_emb = Dropout(dropO1)(word_f_word_emb_raw)
            add_word_emb_p_flat(model_inputs, word_f_input, word_f_word_emb, word_feat['embed_mat'].shape[-1], word_feat['s_enc'], word_feat['m_id'], p_dict)
        else:
            word_f_input = Input(shape=(word_cnt_post, word_feat['dim_shape'][-1]), name=word_feat['emb'])
            word_f_word_emb = Dropout(dropO1)(word_f_input)
            add_word_emb_p_flat(model_inputs, word_f_input, word_f_word_emb, word_feat['dim_shape'][-1], word_feat['s_enc'], word_feat['m_id'], p_dict)

    post_vec_list = []    
    for my_dict in p_dict.values():
        my_dict["word_emb"] = concatenate(my_dict["comb_feature_list"]) if len(my_dict["comb_feature_list"]) > 1 else my_dict["comb_feature_list"][0]
        flat_emb, att_outputs = flat_embed(my_dict["enc_algo"], my_dict["word_emb"], word_cnt_post, my_dict["word_emb_len"], dropO1, num_cnn_filters, max_pool_k_val, kernel_sizes, rnn_dim, att_dim, rnn_type, att_outputs,False, saved_layers_dict)
        post_vec_list.append(flat_emb)

    post_vec = concatenate(post_vec_list) if len(post_vec_list) > 1 else post_vec_list[0]
    if len(model_inputs) == 1:
        model_inputs = model_inputs[0]
    att_mod = Model(model_inputs, att_outputs) if att_outputs else None
    mod, mod_multi_task_or_tr, out_vec_list = apply_dense(model_inputs, dropO2, post_vec, nonlin, out_vec_size, single_inp_tasks_list, conf_input,multi_task_tl,classi_name)    
    return mod, mod_multi_task_or_tr, att_mod, None

def apply_dense(input_seq, dropO2, post_vec, nonlin, out_vec_size,single_inp_tasks_list,conf_output,multi_task_tl,classi_name):
    dr2_l = Dropout(dropO2)(post_vec)
    if conf_output !=None:
        out_vec = concatenate([Dense(out_vec_size, activation=nonlin)(dr2_l),conf_output], name=classi_name)
    else:
        out_vec = Dense(out_vec_size, activation=nonlin, name=classi_name)(dr2_l)
    # print (out_vec_size)
    out_vec_list = [out_vec]
    for single_task_tup in single_inp_tasks_list:
        single_task_name, single_task_dict = single_task_tup
        out_vec_list.append(Dense(single_task_dict['out_vec_size'], activation=single_task_dict['nonlin'], name = single_task_name)(dr2_l))
    if multi_task_tl == "multi_task":
        return Model(input_seq, out_vec), Model(input_seq,out_vec_list), out_vec_list
    elif multi_task_tl == "tr_learn" and len(out_vec_list) > 1:
        return Model(input_seq, out_vec), Model(input_seq, out_vec_list[1]), out_vec_list
    return Model(input_seq, out_vec), None, out_vec_list

def c_bilstm(word_cnt_post, word_f, rnn_dim, att_dim, dropO1, dropO2, nonlin, out_vec_size,rnn_type, num_cnn_filters, kernel_sizes,single_inp_tasks_list,multi_task_tl,classi_name):
    conf_input= None
    if 'embed_mat' in word_f:
        input_seq, embedded_seq = tunable_embed_apply(word_cnt_post, len(word_f['embed_mat']), word_f['embed_mat'])
        dr1_l = Dropout(dropO1)(embedded_seq)
    else:
        input_seq = Input(shape=(word_cnt_post, word_f['dim_shape'][-1]))
        dr1_l = Dropout(dropO1)(input_seq)

    conv_l_list = []
    for k in kernel_sizes:
        conv_t = Conv1D(num_cnn_filters, k, padding='same', activation='relu')(dr1_l)
        conv_l_list.append(conv_t)
    conc_mat = concatenate(conv_l_list)
    mod, mod_multi_task_or_tr, out_vec_list = rnn_dense_apply(conc_mat, input_seq, rnn_dim, att_dim, dropO2, nonlin, out_vec_size, rnn_type,single_inp_tasks_list,conf_input,multi_task_tl,classi_name)
    return mod, mod_multi_task_or_tr, None, None

def uni_sent(sent_cnt, rnn_dim, att_dim, dropO1, dropO2, nonlin, out_vec_size, single_inp_tasks_list,rnn_type, given_sen_enc_feat,multi_task_tl,classi_name):
    conf_input= None
    aux_input_seq = Input(shape=(sent_cnt, given_sen_enc_feat['feats'].shape[-1]), name='sen_input')
    aux_dr1 = Dropout(dropO1)(aux_input_seq)
    mod, mod_multi_task_or_tr, out_vec_list = rnn_dense_apply(aux_dr1, aux_input_seq, rnn_dim, att_dim, dropO2, nonlin, out_vec_size, rnn_type,single_inp_tasks_list,conf_input,multi_task_tl,classi_name)
    return mod, mod_multi_task_or_tr, None, None

def rnn_dense_apply(rnn_seq, input_seq, rnn_dim, att_dim, dropO2, nonlin, out_vec_size, rnn_type,single_inp_tasks_list,conf_input,multi_task_tl,classi_name):
    if rnn_type == 'lstm':
        blstm_l = Bidirectional(LSTM(rnn_dim, return_sequences=(att_dim > 0)))(rnn_seq)
    else:
        blstm_l = Bidirectional(GRU(rnn_dim, return_sequences=(att_dim > 0)))(rnn_seq)
    if att_dim > 0:
        blstm_l, att_w = attLayer_hier(att_dim)(blstm_l)
    return apply_dense(input_seq, dropO2, blstm_l, nonlin, out_vec_size,single_inp_tasks_list, conf_input,multi_task_tl,classi_name)

def tunable_embed_apply(word_cnt_post, vocab_size, embed_mat, word_feat_name):
    input_seq = Input(shape=(word_cnt_post,), name=word_feat_name+'_t')
    embed_layer = Embedding(vocab_size, embed_mat.shape[1], embeddings_initializer=initializers.Constant(embed_mat), input_length=word_cnt_post, name=word_feat_name)
    embed_layer.trainable = True
    embed_l = embed_layer(input_seq)
    return input_seq, embed_l

def tunable_embed_hier_embed(sent_cnt, word_cnt_sent, vocab_size, embed_mat, word_feat_name, dropO1):
    word_input_seq, embed_l = tunable_embed_apply(word_cnt_sent, vocab_size, embed_mat, word_feat_name)
    emb_model = Model(word_input_seq, embed_l)
    sen_input_seq = Input(shape=(sent_cnt, word_cnt_sent), name=word_feat_name+'_t')
    time_l = TimeDistributed(emb_model)(sen_input_seq)
    dr1_l = Dropout(dropO1)(time_l)
    return sen_input_seq, dr1_l


class BertLayer(Layer):
    
    '''BertLayer which support next output_representation param:
    
    pooled_output: the first CLS token after adding projection layer () with shape [batch_size, 768]. 
    sequence_output: all tokens output with shape [batch_size, max_length, 768].
    mean_pooling: mean pooling of all tokens output [batch_size, max_length, 768].
    
    
    You can simple fine-tune last n layers in BERT with n_fine_tune_layers parameter. For view trainable parameters call model.trainable_weights after creating model.
    
    '''
    
    def __init__(self, n_fine_tune_layers, tf_hub_path, output_representation,trainable,**kwargs):
        
        self.n_fine_tune_layers = n_fine_tune_layers
        self.is_trainble = trainable
        self.output_size = 768
        self.tf_hub = tf_hub_path
        self.output_representation = output_representation
        
        super(BertLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        self.bert = hub.Module(
            self.tf_hub,
            trainable=self.is_trainble,
            name="{}_module".format(self.name)
        )
        
        
        variables = list(self.bert.variable_map.values())
        # for var in variables:
        #     print(var)
        # print("------------------")
        # for var in self.bert.variables:
        #     print(var)
        # exit()
        if self.is_trainble:
            # 1 first remove unused layers
            trainable_vars = [var for var in variables if not "/cls/" in var.name]
            
            
            if self.output_representation == "sequence_output" or self.output_representation == "mean_pooling":
                # 1 first remove unused pooled layers
                trainable_vars = [var for var in trainable_vars if not "/pooler/" in var.name]
                
            # Select how many layers to fine tune
            trainable_vars = trainable_vars[-self.n_fine_tune_layers :]
            
            # Add to trainable weights
            for var in trainable_vars:
                self._trainable_weights.append(var)

            # Add non-trainable weights
            for var in self.bert.variables:
                if var not in self._trainable_weights:
                    self._non_trainable_weights.append(var)
                
        else:
             for var in variables:
                self._non_trainable_weights.append(var)
                

        super(BertLayer, self).build(input_shape)

    def call(self, inputs):
        inputs = [K.cast(x, dtype="int32") for x in inputs]
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(
            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
        )
        result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)
        
        if self.output_representation == "pooled_output":
            pooled = result["pooled_output"]
            
        elif self.output_representation == "mean_pooling":
            result_tmp = result["sequence_output"]
        
            mul_mask = lambda x, m: x * tf.expand_dims(m, axis=-1)
            masked_reduce_mean = lambda x, m: tf.reduce_sum(mul_mask(x, m), axis=1) / (
                    tf.reduce_sum(m, axis=1, keepdims=True) + 1e-10)
            input_mask = tf.cast(input_mask, tf.float32)
            pooled = masked_reduce_mean(result_tmp, input_mask)
            
        elif self.output_representation == "sequence_output":
            
            pooled = result["sequence_output"]
       
        return pooled
        
        
    def compute_output_shape(self, input_shape):
        if self.output_representation == "sequence_output":
            return (input_shape[0][0], input_shape[0][1], self.output_size)
        else:
            return (input_shape[0][0], self.output_size)

# adapted from https://github.com/richliao/textClassifier
class attLayer_hier(Layer):
    def __init__(self, attention_dim, **kwargs):
        self.init = initializers.get('glorot_uniform')
        self.attention_dim = attention_dim
        super(attLayer_hier, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name = 'W', shape = (input_shape[-1], self.attention_dim), initializer=self.init, trainable=True)
        self.b = self.add_weight(name = 'b', shape = (self.attention_dim, ), initializer=self.init, trainable=True)
        self.u = self.add_weight(name = 'u', shape = (self.attention_dim, 1), initializer=self.init, trainable=True)
        super(attLayer_hier, self).build(input_shape)

    def call(self, x, mask=None):
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)
        ait = K.exp(ait)

        if mask is not None:
            ait *= K.cast(mask, K.floatx())

        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        exp_ait = K.expand_dims(ait)
        weighted_input = x * exp_ait
        output = K.sum(weighted_input, axis=1)

        return [output, ait]

    def compute_output_shape(self, input_shape):
        return [(input_shape[0], input_shape[-1]), (input_shape[0], input_shape[1])]

    def get_config(self):
        config = {'attention_dim': self.attention_dim}
        base_config = super(attLayer_hier, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class kmax_pooling(Layer):
    def __init__(self, k_val, **kwargs):
        self.k_val = k_val
        super(kmax_pooling, self).__init__(**kwargs)

    def call(self, inputs):
        shifted_input = tf.transpose(inputs, [0, 2, 1])
        top_k_var = tf.nn.top_k(shifted_input, k=self.k_val, sorted=True, name=None)[0]
        return tf.reshape(top_k_var, [tf.shape(top_k_var)[0], -1])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1]* self.k_val)

    def get_config(self):
        config = {'k_val': self.k_val}
        base_config = super(kmax_pooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def lp_categ_loss(weights):
    def lp_categ_of(y_true, y_pred):
        return K.sum(weights*y_true,axis=1)*K.categorical_crossentropy(y_true, y_pred)
    return lp_categ_of

def br_binary_loss(weights):
    def br_binary_of(y_true, y_pred):
        return ((weights[0]*(1-y_true))+(weights[1]*y_true))*K.binary_crossentropy(y_true, y_pred)
    return br_binary_of

def kl_divergence_loss():
    def kl_div(y_true, y_pred):
        return keras.losses.kullback_leibler_divergence(y_true, y_pred)
    return kl_div

def mse_loss():
    def mse(y_true, y_pred):
        return keras.losses.mean_squared_error(y_true, y_pred)
    return mse

# norm correlation binary loss - L-unc
def corr_multi_binary_loss(weights, uncorrelated_c_pairs, beta):
    def corr_multi_binary_of(y_true, y_pred):
        corr_loss = 0
        for uncorr_c_pair in uncorrelated_c_pairs:
            c1, c2 = uncorr_c_pair
            m_c1 = tf.slice(y_pred, [0,c1],[-1,1])
            m_c2 = tf.slice(y_pred, [0,c2],[-1,1])
            sum_m_c1 = K.sum(m_c1)
            sum_m_c2 = K.sum(m_c2)
            corr_loss += K.sum(m_c1*m_c2)*(sum_m_c1+sum_m_c2)/(sum_m_c1*sum_m_c2)
        return K.mean(((weights[:,0]*(1-y_true))+(weights[:,1]*y_true))*K.binary_crossentropy(y_true, y_pred), axis=1) + beta*corr_loss/len(uncorrelated_c_pairs)
    return corr_multi_binary_of

def unnorm_corr_multi_binary_loss(weights, uncorrelated_c_pairs, beta):
    def unnorm_corr_multi_binary_of(y_true, y_pred):
        corr_loss = 0
        for uncorr_c_pair in uncorrelated_c_pairs:
            c1, c2 = uncorr_c_pair
            m_c1 = tf.slice(y_pred, [0,c1],[-1,1])
            m_c2 = tf.slice(y_pred, [0,c2],[-1,1])
            corr_loss += K.mean(m_c1*m_c2)
        return K.mean(((weights[:,0]*(1-y_true))+(weights[:,1]*y_true))*K.binary_crossentropy(y_true, y_pred), axis=1) + beta*corr_loss/len(uncorrelated_c_pairs)
    return unnorm_corr_multi_binary_of

def corr_basic_calc(weights, y_pred):
    num_classes = len(weights[:,0])
    m_c = []
    sum_m_c = []
    for i in range(num_classes):
        m_c.append(tf.slice(y_pred, [0,i],[-1,1]))
        sum_m_c.append(K.sum(m_c[i]))
    return num_classes, m_c, sum_m_c

#use corr_fuzzy_0.pickle with this "sep_norm_bin_pairs"
# sep correlation binary loss - L-cor, t=0
def sep_corr_multi_binary_loss(weights, uncorrelated_c_pairs, beta):
    def sep_corr_multi_binary_of(y_true, y_pred):
        num_classes, m_c, sum_m_c = corr_basic_calc(weights, y_pred)
        corr_loss = 0
        for uncorr_c_pair in uncorrelated_c_pairs:
            c1, c2, v = uncorr_c_pair
            corr_loss += K.sum(m_c[c1]*m_c[c2])/sum_m_c[c1]
        return K.mean(((weights[:,0]*(1-y_true))+(weights[:,1]*y_true))*K.binary_crossentropy(y_true, y_pred), axis=1) + beta*corr_loss/len(uncorrelated_c_pairs)
    return sep_corr_multi_binary_of

def fuzzy_corr_multi_binary_loss(weights, uncorrelated_c_pairs, beta):
    def fuzzy_corr_multi_binary_of(y_true, y_pred):
        num_classes, m_c, sum_m_c = corr_basic_calc(weights, y_pred)
        corr_loss = 0
        for uncorr_c_pair in uncorrelated_c_pairs:
            c1, c2, v = uncorr_c_pair
            corr_loss += pow((K.sum(m_c[c1]*m_c[c2])/sum_m_c[c1] - v),2)
        return K.mean(((weights[:,0]*(1-y_true))+(weights[:,1]*y_true))*K.binary_crossentropy(y_true, y_pred), axis=1) + beta*corr_loss/len(uncorrelated_c_pairs)
    return fuzzy_corr_multi_binary_of

# l1 fuzzy correlation - L-cor
def fuzzy_l1_corr_multi_binary_loss(weights, uncorrelated_c_pairs, beta):
    def fuzzy_l1_corr_multi_binary_of(y_true, y_pred):
        num_classes, m_c, sum_m_c = corr_basic_calc(weights, y_pred)
        corr_loss = 0
        for uncorr_c_pair in uncorrelated_c_pairs:
            c1, c2, v = uncorr_c_pair
            corr_loss += abs(K.sum(m_c[c1]*m_c[c2])/sum_m_c[c1] - v)
        return K.mean(((weights[:,0]*(1-y_true))+(weights[:,1]*y_true))*K.binary_crossentropy(y_true, y_pred), axis=1) + beta*corr_loss/len(uncorrelated_c_pairs)
    return fuzzy_l1_corr_multi_binary_of

def multi_binary_loss(weights):
    def multi_binary_of(y_true, y_pred):
        return K.mean(((weights[:,0]*(1-y_true))+(weights[:,1]*y_true))*K.binary_crossentropy(y_true, y_pred), axis=1)
    return multi_binary_of

def multi_binary_loss_conf(weights,out_vec_size):
    def multi_binary_of_conf(y_true, y_pred):
        y_pred_score = tf.slice(y_pred,[0,0],[-1,out_vec_size])
        y_pred_conf_score = tf.slice(y_pred,[0,out_vec_size],[-1,out_vec_size])
        return K.mean(((weights[:,0]*(1-y_true))+(weights[:,1]*y_true))* y_pred_conf_score*K.binary_crossentropy(y_true, y_pred_score), axis=1)
    return multi_binary_of_conf

def multi_cat_w_loss(weights):
    def multi_cat_w_of(y_true, y_pred):
        return -K.sum(weights*y_true*K.log(y_pred),axis = 1)/K.sum(y_true,axis = 1)
    return multi_cat_w_of

def multi_cat_loss():
    def multi_cat_of(y_true, y_pred):
        return -K.sum(y_true*K.log(y_pred),axis = 1)/K.sum(y_true,axis = 1)
    return multi_cat_of

def get_model(m_type, word_cnt_post, sent_cnt, word_cnt_sent, word_feats, sen_enc_feats, learn_rate, dropO1, dropO2, num_cnn_filters, rnn_type, loss_func, nonlin, out_vec_size, rnn_dim, att_dim, max_pool_k_val, stack_rnn_flag, kernel_sizes,st_variant,use_conf_scores,multi_task_tl,share_weights_sep_mt, single_inp_tasks_list, sep_inp_tasks_list,compile_flag,n_fine_tune_layers,tf_hub_path, output_representation, bert_trainable,cascade,slightcascade,use_knowledge, knowledge_array):
    post_vec = None
    if m_type == 'flat_fuse':
        model, model_multi_task_or_tr, att_mod, post_vec = flat_fuse(word_cnt_post, rnn_dim, att_dim, word_feats, sen_enc_feats, dropO1, dropO2, nonlin, out_vec_size, single_inp_tasks_list, rnn_type, stack_rnn_flag, num_cnn_filters, max_pool_k_val, kernel_sizes,multi_task_tl, "classi")
    elif m_type == 'hier_fuse':
        if len(sep_inp_tasks_list) == 0:
            model, model_multi_task_or_tr, att_mod, post_vec,model_inputs,out_vec,saved_layers = hier_fuse(sent_cnt, word_cnt_sent, rnn_dim, att_dim, word_feats, sen_enc_feats, dropO1, dropO2, nonlin, out_vec_size,single_inp_tasks_list, rnn_type, stack_rnn_flag, num_cnn_filters, max_pool_k_val, kernel_sizes,st_variant,use_conf_scores,multi_task_tl,"classi",False,[],n_fine_tune_layers,tf_hub_path, output_representation, bert_trainable, cascade,use_knowledge, knowledge_array)
        elif len(sep_inp_tasks_list) >0 and not cascade and not slightcascade:
            model, model_multi_task_or_tr, att_mod, post_vec = multi_inputs(sent_cnt, word_cnt_sent, rnn_dim, att_dim, word_feats, sen_enc_feats, dropO1, dropO2, nonlin, out_vec_size, rnn_type, stack_rnn_flag, num_cnn_filters, max_pool_k_val, kernel_sizes,st_variant,use_conf_scores,multi_task_tl,share_weights_sep_mt, single_inp_tasks_list, sep_inp_tasks_list,n_fine_tune_layers,tf_hub_path, output_representation, bert_trainable, cascade,use_knowledge, knowledge_array)
        elif len(sep_inp_tasks_list) >0 and cascade:
            model, model_multi_task_or_tr, att_mod, post_vec = multi_inputs_cascade(sent_cnt, word_cnt_sent, rnn_dim, att_dim, word_feats, sen_enc_feats, dropO1, dropO2, nonlin, out_vec_size, rnn_type, stack_rnn_flag, num_cnn_filters, max_pool_k_val, kernel_sizes,st_variant,use_conf_scores,multi_task_tl,share_weights_sep_mt, single_inp_tasks_list, sep_inp_tasks_list,n_fine_tune_layers,tf_hub_path, output_representation, bert_trainable, cascade,use_knowledge, knowledge_array)
        elif len(sep_inp_tasks_list) >0 and slightcascade:
            model, model_multi_task_or_tr, att_mod, post_vec = multi_inputs_slightcascade(sent_cnt, word_cnt_sent, rnn_dim, att_dim, word_feats, sen_enc_feats, dropO1, dropO2, nonlin, out_vec_size, rnn_type, stack_rnn_flag, num_cnn_filters, max_pool_k_val, kernel_sizes,st_variant,use_conf_scores,multi_task_tl,share_weights_sep_mt, single_inp_tasks_list, sep_inp_tasks_list,n_fine_tune_layers,tf_hub_path, output_representation, bert_trainable, slightcascade,use_knowledge, knowledge_array)
    elif m_type == 'c_bilstm':
        model, model_multi_task_or_tr, att_mod, post_vec = c_bilstm(word_cnt_post, word_feats[0], rnn_dim, 0, dropO1, dropO2, nonlin, out_vec_size,rnn_type, num_cnn_filters, kernel_sizes,single_inp_tasks_list,multi_task_tl,"classi")
    elif m_type == 'uni_sent':
        model, model_multi_task_or_tr, att_mod, post_vec= uni_sent(sent_cnt, rnn_dim, att_dim, dropO1, dropO2, nonlin, out_vec_size,single_inp_tasks_list,rnn_type, sen_enc_feats[0],multi_task_tl,"classi")
    else:
        print("ERROR: No model named %s" % m_type)
        return None, None

    adam = optimizers.Adam(lr=learn_rate)
    # model = multi_gpu_model(model, gpus=2)
    model_multi_task_or_tr = multi_gpu_model(model_multi_task_or_tr, gpus=2)
    if compile_flag:
        if multi_task_tl == "multi_task":
            model_multi_task_or_tr.compile(loss=loss_func[0], optimizer=adam, loss_weights = loss_func[1])
        elif multi_task_tl == "tr_learn":
            if len(single_inp_tasks_list) == 0:
                aux_task_name = sep_inp_tasks_list[0][0]
            else:
                aux_task_name = single_inp_tasks_list[0][0]
            model_multi_task_or_tr.compile(loss=loss_func[0][aux_task_name], optimizer=adam)
            model.compile(loss=loss_func[0]["classi"], optimizer=adam)
        else:
            model.compile(loss=loss_func, optimizer=adam)
    return model, model_multi_task_or_tr, att_mod, post_vec

# import numpy as np

# word_feats = [{'emb': 'elmo', 's_enc': 'rnn', 'm_id': '11', 'dim_shape': (100,3000)}, {'emb': 'glove', 's_enc': 'rnn', 'm_id': '21','dim_shape': [100,300]}]
# sen_enc_feats = [{'emb': 'bert', 'm_id': '1', 'feats': np.zeros((100, 1024))}]
# single_inp_tasks_list = [("topic",{"filename":"topic_vecs.pickle","loss_func":"mean_squared_error","loss_weight":0.3,"out_vec_size":10,"nonlin":"softmax"})]
# sep_inp_tasks_list = [("sd",{"filename":"sexismDet.txt","loss_weight":0.3,"max_num_sent" :20,"max_words_sent":20 ,"word_feats":word_feats,"sent_enc_feats":sen_enc_feats,"nonlin":"sigmoid","out_vec_size":1})]

# m_org, m_mul, d1, d2 = get_model('hier_fuse', 200, 20, 100, word_feats, sen_enc_feats, 0.001, 0.1, 0.1, 80, 'lstm', 'binary_crossentropy', 'sigmoid', 23, 100, 200, 1, False, [2,3,4], "", False,"multi_task",True,single_inp_tasks_list,sep_inp_tasks_list,True,"","","","",False,True)
# m_mul.summary()
# # input()
# m_org.summary()