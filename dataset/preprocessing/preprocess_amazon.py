import numpy as np
import torch
import pandas as pd
from scipy.stats import kendalltau
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt
import random
import os


def read_behaviors_amazon(behaviors_path, before_item_id_to_dic, before_item_name_to_id, before_item_id_to_name, max_seq_len, min_seq_len, Log_file):
    #Log_file.info("##### news number {} {} (before clearing)#####".format(len(before_item_id_to_dic), len(before_item_name_to_id)))
    #Log_file.info("##### min seq len {}, max seq len {}#####".format(min_seq_len, max_seq_len))
    item_num = len(before_item_name_to_id)
    #before_item_counts = [0] * (before_item_num + 1)
    user_seq_dic = {}
    #seq_num = 0
    #pairs_num = 0
    Log_file.info('rebuild user seqs...')
    c=pd.read_csv('/dataset/amazon/Clothing_Shoes_and_Jewelry_cleaned_history.csv',converters={'history':eval})
    for i in range(c.shape[0]):
        user_name = c.iloc[i,0]
        history_item_name = c.iloc[i,1]
        #print(history_item_name)
        #print(type(history_item_name))
        #print(len(history_item_name))
        item_ids_sub_seq = [before_item_name_to_id[i] for i in history_item_name]
        user_seq_dic[user_name] = item_ids_sub_seq

    item_id_to_dic = {}
    item_name_to_id = before_item_name_to_id
    #item_id_before_to_now = {}
    #Log_file.info("##### items after clearing {}, {}, {} #####".format(item_num, len(item_id_before_to_now), len(item_id_to_dic)))

    users_train = {}
    users_valid = {}
    users_test = {}
    users_history_for_valid = {}
    users_history_for_test = {}
    user_id = 0
    for user_name, item_seqs in user_seq_dic.items():
        user_seq = [i for i in item_seqs]
        train = user_seq[:-2]
        valid = user_seq[-(max_seq_len+2):-1]
        test = user_seq[-(max_seq_len+1):]
        users_train[user_id] = train
        users_valid[user_id] = valid
        users_test[user_id] = test

        users_history_for_valid[user_id] = torch.LongTensor(np.array(train))
        users_history_for_test[user_id] = torch.LongTensor(np.array(user_seq[:-1]))
        user_id += 1
    #Log_file.info("##### user seqs after clearing {}, {}, {}, {}, {}#####".
    #              format(seq_num, len(user_seq_dic), len(users_train), len(users_valid), len(users_test)))
    return item_num, item_id_to_dic, users_train, users_valid, users_test, \
           users_history_for_valid, users_history_for_test, item_name_to_id




def read_news_bert_amazon(news_path, args, tokenizer):
    item_id_to_dic = {}
    item_id_to_name = {}
    item_name_to_id = {}
    c=pd.read_csv('/dataset/amazon/metadata_Clothing_Shoes_and_Jewelry.csv')
    #c=c.drop('title',axis=1)
    c['index']=c.index+1
    item_name_to_id = c.set_index('asin')['index'].to_dict()
    return item_id_to_dic, item_name_to_id, item_id_to_name
