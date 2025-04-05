import numpy as np
import torch
import pandas as pd
from scipy.stats import kendalltau
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt
import random
import os

def read_behaviors_twotower_mmd2(behaviors_path, before_item_id_to_dic, before_item_name_to_id, before_item_id_to_name, max_seq_len, min_seq_len, Log_file,num_k):
    random.seed(12345)
    np.random.seed(12345)
    Log_file.info("##### news number {} {} (before clearing)#####".format(len(before_item_id_to_dic), len(before_item_name_to_id)))
    Log_file.info("##### min seq len {}, max seq len {}#####".format(min_seq_len, max_seq_len))
    before_item_num = len(before_item_name_to_id)
    before_item_counts = [0] * (before_item_num + 1)
    user_seq_dic = {}
    seq_num = 0
    pairs_num = 0
    Log_file.info('rebuild user seqs...')
    with open(behaviors_path, "r") as f:
        for line in f:
            splited = line.strip('\n').split('\t')
            user_name = splited[0]
            history_item_name = splited[1].split(' ')
            if len(history_item_name) < min_seq_len:
                continue
            history_item_name = history_item_name[-(max_seq_len+3):]
            item_ids_sub_seq = [before_item_name_to_id[i] for i in history_item_name]
            user_seq_dic[user_name] = item_ids_sub_seq
            for item_id in item_ids_sub_seq:
                before_item_counts[item_id] += 1
                pairs_num += 1
            seq_num += 1
    Log_file.info("##### pairs_num {}".format(pairs_num))

    item_id = 1
    item_id_to_dic = {}
    item_name_to_id = {}
    item_id_before_to_now = {}
    for before_item_id in range(1, before_item_num + 1):
        if before_item_counts[before_item_id] != 0:
            item_id_before_to_now[before_item_id] = item_id
            item_id_to_dic[item_id] = before_item_id_to_dic[before_item_id]
            item_name_to_id[before_item_id_to_name[before_item_id]] = item_id
            item_id += 1
    item_num = len(item_id_before_to_now)
    Log_file.info("##### items after clearing {}, {}, {} #####".format(item_num, len(item_id_before_to_now), len(item_id_to_dic)))

    users_train = {}
    users_valid = {}
    users_test = {}
    users_history_for_valid = {}
    users_history_for_test = {}
    user_id = 0
    ee=pd.read_csv('/dataset/fre.csv')
    ee.columns = ['id', 'fre']
    ee.drop(0,inplace=True)
    e=ee.sort_values('fre',ascending=False)
    eee=e[['id']]
    eee=list(eee['id'])
    num=10
    #l = [0]
    #for k in range(num):
    #    l.append(int(79707*(1+k)/num))
    l = [0, 41, 127, 276, 517, 889, 1467, 2413, 4070, 7618, 79707, 79707]
    i=num_k###################################
    valid_id=0
    target=[] 

    for user_name, item_seqs in user_seq_dic.items():
        user_seq = [item_id_before_to_now[i] for i in item_seqs]
        train = user_seq[:-2] ####################################################################
        users_train[user_id] = train

        if (eee.index(user_seq[-2])<l[i+1] and eee.index(user_seq[-2])>=l[i] and i<10) or (i==10):
            valid = user_seq[-(max_seq_len+2):-1]
            users_valid[valid_id] = valid
            users_history_for_valid[valid_id] = torch.LongTensor(np.array(train))
            valid_id+=1
            target.append(user_seq[-2])
        test = user_seq[-(max_seq_len+1):]
        users_test[user_id] = test
        users_history_for_test[user_id] = torch.LongTensor(np.array(user_seq[:-1]))
        user_id += 1
    Log_file.info("##### user seqs after clearing {}, {}, {}, {}, {}#####".
                  format(seq_num, len(user_seq_dic), len(users_train), len(users_valid), len(users_test)))
    
    a = torch.load('/dataset/bce_text/id_plus_mo-2stage/tensor/llm2vec_0.pt')
    b = torch.load('/dataset/bce_text/id_plus_mo-2stage/tensor/llm2vec_1.pt')
    c = torch.load('/dataset/bce_text/id_plus_mo-2stage/tensor/llm2vec_2.pt')
    d = torch.load('/dataset/bce_text/id_plus_mo-2stage/tensor/llm2vec_3.pt')
    id_emb = torch.cat((a,b,c,d),0)
    content_emb = torch.load('./tensor/twotower_aftertrain_content_128.pt')
    
    id_emb = id_emb.detach().numpy()
    content_emb = content_emb.detach().numpy()
    distance_llm = np.array([])
    distance_rs_cos = np.array([])
    distance_rs_dot = np.array([])
    path = '/dataset/bce_text/id_plus_mo-2stage/tensor/twotower_aftertrain_content_'+str(num_k)+'.dat'
    if os.path.exists(path):
        distance_llm = np.fromfile(path, dtype=np.half)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/single_beforetrain_content_'+str(num_k)+'.dat'
        distance_rs_cos = np.fromfile(path, dtype=np.half)
    else:
        for k in range(valid_id):
            seq=users_history_for_valid[k].tolist()
            seq.append(target[k])
            neg_items = []
            seq = [eee.index(j) for j in seq]
            if i==10:
                left=0
                right=79706
            else:
                left=l[i]
                right=l[i+1]-1
            for j in range(1):
                sam_neg = random.randint(left,right)
                while sam_neg in seq:
                    sam_neg = random.randint(left,right)
                neg_items.append(eee[sam_neg])
            neg_items=torch.LongTensor(np.array(neg_items))

            term_1 = content_emb[users_history_for_valid[k]]
            term_2 = content_emb[target[k]].reshape(1,-1)
            dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            distance_llm = np.concatenate((distance_llm,dis),0)
            
            term_3 = content_emb[neg_items].reshape(1,-1)
            dis = pairwise_distances(term_1, term_3, metric='cosine').reshape(term_1.shape[0])
            distance_llm = np.concatenate((distance_llm,dis),0)

            term_1 = id_emb[users_history_for_valid[k]]
            term_2 = id_emb[target[k]].reshape(1,-1)
            dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            distance_rs_cos= np.concatenate((distance_rs_cos,dis),0)
            
            term_3 = id_emb[neg_items].reshape(1,-1)
            dis = pairwise_distances(term_1, term_3, metric='cosine').reshape(term_1.shape[0])
            distance_rs_cos = np.concatenate((distance_rs_cos,dis),0)

        distance_llm = np.half(distance_llm)
        distance_rs_cos = np.half(distance_rs_cos)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/twotower_aftertrain_content_'+str(num_k)+'.dat'
        distance_llm.tofile(path)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/single_beforetrain_content_'+str(num_k)+'.dat'
        distance_rs_cos.tofile(path)


    Kendallta,p_value = kendalltau(distance_rs_cos,distance_llm)
    print(Kendallta,p_value)
    print('done twotower task 3-2')

    
    return item_num, item_id_to_dic, users_train, users_valid, users_test, \
           users_history_for_valid, users_history_for_test, item_name_to_id


def read_behaviors_twotower_mmd(behaviors_path, before_item_id_to_dic, before_item_name_to_id, before_item_id_to_name, max_seq_len, min_seq_len, Log_file,num_k):
    random.seed(12345)
    np.random.seed(12345)
    Log_file.info("##### news number {} {} (before clearing)#####".format(len(before_item_id_to_dic), len(before_item_name_to_id)))
    Log_file.info("##### min seq len {}, max seq len {}#####".format(min_seq_len, max_seq_len))
    before_item_num = len(before_item_name_to_id)
    before_item_counts = [0] * (before_item_num + 1)
    user_seq_dic = {}
    seq_num = 0
    pairs_num = 0
    Log_file.info('rebuild user seqs...')
    with open(behaviors_path, "r") as f:
        for line in f:
            splited = line.strip('\n').split('\t')
            user_name = splited[0]
            history_item_name = splited[1].split(' ')
            if len(history_item_name) < min_seq_len:
                continue
            history_item_name = history_item_name[-(max_seq_len+3):]
            item_ids_sub_seq = [before_item_name_to_id[i] for i in history_item_name]
            user_seq_dic[user_name] = item_ids_sub_seq
            for item_id in item_ids_sub_seq:
                before_item_counts[item_id] += 1
                pairs_num += 1
            seq_num += 1
    Log_file.info("##### pairs_num {}".format(pairs_num))

    item_id = 1
    item_id_to_dic = {}
    item_name_to_id = {}
    item_id_before_to_now = {}
    for before_item_id in range(1, before_item_num + 1):
        if before_item_counts[before_item_id] != 0:
            item_id_before_to_now[before_item_id] = item_id
            item_id_to_dic[item_id] = before_item_id_to_dic[before_item_id]
            item_name_to_id[before_item_id_to_name[before_item_id]] = item_id
            item_id += 1
    item_num = len(item_id_before_to_now)
    Log_file.info("##### items after clearing {}, {}, {} #####".format(item_num, len(item_id_before_to_now), len(item_id_to_dic)))

    users_train = {}
    users_valid = {}
    users_test = {}
    users_history_for_valid = {}
    users_history_for_test = {}
    user_id = 0
    ee=pd.read_csv('/dataset/fre.csv')
    ee.columns = ['id', 'fre']
    ee.drop(0,inplace=True)
    e=ee.sort_values('fre',ascending=False)
    eee=e[['id']]
    eee=list(eee['id'])
    num=10
    #l = [0]
    #for k in range(num):
    #    l.append(int(79707*(1+k)/num))
    l = [0, 41, 127, 276, 517, 889, 1467, 2413, 4070, 7618, 79707, 79707]
    i=num_k###################################
    valid_id=0
    target=[] 

    for user_name, item_seqs in user_seq_dic.items():
        user_seq = [item_id_before_to_now[i] for i in item_seqs]
        train = user_seq[:-2] ####################################################################
        users_train[user_id] = train

        if (eee.index(user_seq[-2])<l[i+1] and eee.index(user_seq[-2])>=l[i] and i<10) or (i==10):
            valid = user_seq[-(max_seq_len+2):-1]
            users_valid[valid_id] = valid
            users_history_for_valid[valid_id] = torch.LongTensor(np.array(train))
            valid_id+=1
            target.append(user_seq[-2])
        test = user_seq[-(max_seq_len+1):]
        users_test[user_id] = test
        users_history_for_test[user_id] = torch.LongTensor(np.array(user_seq[:-1]))
        user_id += 1
    Log_file.info("##### user seqs after clearing {}, {}, {}, {}, {}#####".
                  format(seq_num, len(user_seq_dic), len(users_train), len(users_valid), len(users_test)))
    
    id_emb = torch.load('./tensor/single_beforetrain_id_128.pt')
    content_emb = torch.load('./tensor/twotower_aftertrain_id_128.pt')
    
    id_emb = id_emb.detach().numpy()
    content_emb = content_emb.detach().numpy()
    distance_llm = np.array([])
    distance_rs_cos = np.array([])
    distance_rs_dot = np.array([])
    path = '/dataset/bce_text/id_plus_mo-2stage/tensor/twotower_aftertrain_id_'+str(num_k)+'.dat'
    if os.path.exists(path):
        distance_llm = np.fromfile(path, dtype=np.half)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/single_beforetrain_id_'+str(num_k)+'.dat'
        distance_rs_cos = np.fromfile(path, dtype=np.half)
    else:
        for k in range(valid_id):
            seq=users_history_for_valid[k].tolist()
            seq.append(target[k])
            neg_items = []
            seq = [eee.index(j) for j in seq]
            if i==10:
                left=0
                right=79706
            else:
                left=l[i]
                right=l[i+1]-1
            for j in range(1):
                sam_neg = random.randint(left,right)
                while sam_neg in seq:
                    sam_neg = random.randint(left,right)
                neg_items.append(eee[sam_neg])
            neg_items=torch.LongTensor(np.array(neg_items))

            term_1 = content_emb[users_history_for_valid[k]]
            term_2 = content_emb[target[k]].reshape(1,-1)
            dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            distance_llm = np.concatenate((distance_llm,dis),0)
            
            term_3 = content_emb[neg_items].reshape(1,-1)
            dis = pairwise_distances(term_1, term_3, metric='cosine').reshape(term_1.shape[0])
            distance_llm = np.concatenate((distance_llm,dis),0)

            term_1 = id_emb[users_history_for_valid[k]]
            term_2 = id_emb[target[k]].reshape(1,-1)
            dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            distance_rs_cos= np.concatenate((distance_rs_cos,dis),0)
            
            term_3 = id_emb[neg_items].reshape(1,-1)
            dis = pairwise_distances(term_1, term_3, metric='cosine').reshape(term_1.shape[0])
            distance_rs_cos = np.concatenate((distance_rs_cos,dis),0)

        distance_llm = np.half(distance_llm)
        distance_rs_cos = np.half(distance_rs_cos)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/twotower_aftertrain_id_'+str(num_k)+'.dat'
        distance_llm.tofile(path)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/single_beforetrain_id_'+str(num_k)+'.dat'
        distance_rs_cos.tofile(path)


    Kendallta,p_value = kendalltau(distance_rs_cos,distance_llm)
    print(Kendallta,p_value)
    print('done twotower task 3-1')

    id_emb = torch.load('./tensor/single_beforetrain_content_128.pt')
    content_emb = torch.load('./tensor/twotower_aftertrain_content_128.pt')
    
    id_emb = id_emb.detach().numpy()
    content_emb = content_emb.detach().numpy()
    distance_llm = np.array([])
    distance_rs_cos = np.array([])
    distance_rs_dot = np.array([])
    path = '/dataset/bce_text/id_plus_mo-2stage/tensor/twotower_aftertrain_content_'+str(num_k)+'.dat'
    if os.path.exists(path):
        distance_llm = np.fromfile(path, dtype=np.half)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/single_beforetrain_content_'+str(num_k)+'.dat'
        distance_rs_cos = np.fromfile(path, dtype=np.half)
    else:
        for k in range(valid_id):
            seq=users_history_for_valid[k].tolist()
            seq.append(target[k])
            neg_items = []
            seq = [eee.index(j) for j in seq]
            if i==10:
                left=0
                right=79706
            else:
                left=l[i]
                right=l[i+1]-1
            for j in range(1):
                sam_neg = random.randint(left,right)
                while sam_neg in seq:
                    sam_neg = random.randint(left,right)
                neg_items.append(eee[sam_neg])
            neg_items=torch.LongTensor(np.array(neg_items))

            term_1 = content_emb[users_history_for_valid[k]]
            term_2 = content_emb[target[k]].reshape(1,-1)
            dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            distance_llm = np.concatenate((distance_llm,dis),0)
            
            term_3 = content_emb[neg_items].reshape(1,-1)
            dis = pairwise_distances(term_1, term_3, metric='cosine').reshape(term_1.shape[0])
            distance_llm = np.concatenate((distance_llm,dis),0)

            term_1 = id_emb[users_history_for_valid[k]]
            term_2 = id_emb[target[k]].reshape(1,-1)
            dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            distance_rs_cos= np.concatenate((distance_rs_cos,dis),0)
            
            term_3 = id_emb[neg_items].reshape(1,-1)
            dis = pairwise_distances(term_1, term_3, metric='cosine').reshape(term_1.shape[0])
            distance_rs_cos = np.concatenate((distance_rs_cos,dis),0)

        distance_llm = np.half(distance_llm)
        distance_rs_cos = np.half(distance_rs_cos)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/twotower_aftertrain_content_'+str(num_k)+'.dat'
        distance_llm.tofile(path)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/single_beforetrain_content_'+str(num_k)+'.dat'
        distance_rs_cos.tofile(path)


    Kendallta,p_value = kendalltau(distance_rs_cos,distance_llm)
    print(Kendallta,p_value)
    print('done twotower task 3-2')

    id_emb = torch.load('./tensor/single_beforetrain_id_128.pt')
    content_emb = torch.load('./tensor/twotower_aftertrain_idtower_128.pt')
    
    id_emb = id_emb.detach().numpy()
    content_emb = content_emb.detach().numpy()
    distance_llm = np.array([])
    distance_rs_cos = np.array([])
    distance_rs_dot = np.array([])
    path = '/dataset/bce_text/id_plus_mo-2stage/tensor/single_beforetrain_id_'+str(num_k)+'.dat'
    distance_rs_cos = np.fromfile(path, dtype=np.half)
    path = '/dataset/bce_text/id_plus_mo-2stage/tensor/twotower_aftertrain_idtower_'+str(num_k)+'.dat'
    if os.path.exists(path):
        distance_llm = np.fromfile(path, dtype=np.half)
    else:
        for k in range(valid_id):
            seq=users_history_for_valid[k].tolist()
            seq.append(target[k])
            neg_items = []
            seq = [eee.index(j) for j in seq]
            if i==10:
                left=0
                right=79706
            else:
                left=l[i]
                right=l[i+1]-1
            for j in range(1):
                sam_neg = random.randint(left,right)
                while sam_neg in seq:
                    sam_neg = random.randint(left,right)
                neg_items.append(eee[sam_neg])
            neg_items=torch.LongTensor(np.array(neg_items))

            term_1 = content_emb[users_history_for_valid[k]]
            term_2 = content_emb[target[k]].reshape(1,-1)
            dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            distance_llm = np.concatenate((distance_llm,dis),0)
            
            term_3 = content_emb[neg_items].reshape(1,-1)
            dis = pairwise_distances(term_1, term_3, metric='cosine').reshape(term_1.shape[0])
            distance_llm = np.concatenate((distance_llm,dis),0)

        distance_llm = np.half(distance_llm)
        distance_rs_cos = np.half(distance_rs_cos)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/twotower_aftertrain_idtower_'+str(num_k)+'.dat'
        distance_llm.tofile(path)


    Kendallta,p_value = kendalltau(distance_rs_cos,distance_llm)
    print(Kendallta,p_value)
    print('done twotower task 3-3')

    return item_num, item_id_to_dic, users_train, users_valid, users_test, \
           users_history_for_valid, users_history_for_test, item_name_to_id

def read_behaviors_single_mmd2(behaviors_path, before_item_id_to_dic, before_item_name_to_id, before_item_id_to_name, max_seq_len, min_seq_len, Log_file,num_k):
    random.seed(12345)
    np.random.seed(12345)
    Log_file.info("##### news number {} {} (before clearing)#####".format(len(before_item_id_to_dic), len(before_item_name_to_id)))
    Log_file.info("##### min seq len {}, max seq len {}#####".format(min_seq_len, max_seq_len))
    before_item_num = len(before_item_name_to_id)
    before_item_counts = [0] * (before_item_num + 1)
    user_seq_dic = {}
    seq_num = 0
    pairs_num = 0
    Log_file.info('rebuild user seqs...')
    with open(behaviors_path, "r") as f:
        for line in f:
            splited = line.strip('\n').split('\t')
            user_name = splited[0]
            history_item_name = splited[1].split(' ')
            if len(history_item_name) < min_seq_len:
                continue
            history_item_name = history_item_name[-(max_seq_len+3):]
            item_ids_sub_seq = [before_item_name_to_id[i] for i in history_item_name]
            user_seq_dic[user_name] = item_ids_sub_seq
            for item_id in item_ids_sub_seq:
                before_item_counts[item_id] += 1
                pairs_num += 1
            seq_num += 1
    Log_file.info("##### pairs_num {}".format(pairs_num))

    item_id = 1
    item_id_to_dic = {}
    item_name_to_id = {}
    item_id_before_to_now = {}
    for before_item_id in range(1, before_item_num + 1):
        if before_item_counts[before_item_id] != 0:
            item_id_before_to_now[before_item_id] = item_id
            item_id_to_dic[item_id] = before_item_id_to_dic[before_item_id]
            item_name_to_id[before_item_id_to_name[before_item_id]] = item_id
            item_id += 1
    item_num = len(item_id_before_to_now)
    Log_file.info("##### items after clearing {}, {}, {} #####".format(item_num, len(item_id_before_to_now), len(item_id_to_dic)))

    users_train = {}
    users_valid = {}
    users_test = {}
    users_history_for_valid = {}
    users_history_for_test = {}
    user_id = 0
    ee=pd.read_csv('/dataset/fre.csv')
    ee.columns = ['id', 'fre']
    ee.drop(0,inplace=True)
    e=ee.sort_values('fre',ascending=False)
    eee=e[['id']]
    eee=list(eee['id'])
    num=10
    #l = [0]
    #for k in range(num):
    #    l.append(int(79707*(1+k)/num))
    l = [0, 41, 127, 276, 517, 889, 1467, 2413, 4070, 7618, 79707, 79707]
    i=num_k###################################
    valid_id=0
    target=[] 

    for user_name, item_seqs in user_seq_dic.items():
        user_seq = [item_id_before_to_now[i] for i in item_seqs]
        train = user_seq[:-2] ####################################################################
        users_train[user_id] = train

        if (eee.index(user_seq[-2])<l[i+1] and eee.index(user_seq[-2])>=l[i] and i<10) or (i==10):
            valid = user_seq[-(max_seq_len+2):-1]
            users_valid[valid_id] = valid
            users_history_for_valid[valid_id] = torch.LongTensor(np.array(train))
            valid_id+=1
            target.append(user_seq[-2])
        test = user_seq[-(max_seq_len+1):]
        users_test[user_id] = test
        users_history_for_test[user_id] = torch.LongTensor(np.array(user_seq[:-1]))
        user_id += 1
    Log_file.info("##### user seqs after clearing {}, {}, {}, {}, {}#####".
                  format(seq_num, len(user_seq_dic), len(users_train), len(users_valid), len(users_test)))
    
    
    a = torch.load('/dataset/bce_text/id_plus_mo-2stage/tensor/llm2vec_0.pt')
    b = torch.load('/dataset/bce_text/id_plus_mo-2stage/tensor/llm2vec_1.pt')
    c = torch.load('/dataset/bce_text/id_plus_mo-2stage/tensor/llm2vec_2.pt')
    d = torch.load('/dataset/bce_text/id_plus_mo-2stage/tensor/llm2vec_3.pt')
    id_emb = torch.cat((a,b,c,d),0)
    content_emb = torch.load('./tensor/single_aftertrain_content_128.pt')
    
    id_emb = id_emb.detach().numpy()
    content_emb = content_emb.detach().numpy()
    distance_llm = np.array([])
    distance_rs_cos = np.array([])
    distance_rs_dot = np.array([])
    path = '/dataset/bce_text/id_plus_mo-2stage/tensor/single111_aftertrain_content_'+str(num_k)+'.dat'
    if os.path.exists(path):
        distance_llm = np.fromfile(path, dtype=np.half)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/single_beforetrain_content_'+str(num_k)+'.dat'
        distance_rs_cos = np.fromfile(path, dtype=np.half)
    else:
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/single_aftertrain_content_'+str(num_k)+'.dat'
        distance_llm = np.fromfile(path, dtype=np.half)
        for k in range(valid_id):
            seq=users_history_for_valid[k].tolist()
            seq.append(target[k])
            neg_items = []
            seq = [eee.index(j) for j in seq]
            if i==10:
                left=0
                right=79706
            else:
                left=l[i]
                right=l[i+1]-1
            for j in range(1):
                sam_neg = random.randint(left,right)
                while sam_neg in seq:
                    sam_neg = random.randint(left,right)
                neg_items.append(eee[sam_neg])
            neg_items=torch.LongTensor(np.array(neg_items))

            term_1 = id_emb[users_history_for_valid[k]]
            term_2 = id_emb[target[k]].reshape(1,-1)
            dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            distance_rs_cos= np.concatenate((distance_rs_cos,dis),0)
            
            term_3 = id_emb[neg_items].reshape(1,-1)
            dis = pairwise_distances(term_1, term_3, metric='cosine').reshape(term_1.shape[0])
            distance_rs_cos = np.concatenate((distance_rs_cos,dis),0)

        distance_rs_cos = np.half(distance_rs_cos)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/single_beforetrain_content_'+str(num_k)+'.dat'
        distance_rs_cos.tofile(path)


    Kendallta,p_value = kendalltau(distance_rs_cos,distance_llm)
    print(Kendallta,p_value)
    print('done twotower task 3-2')


    return item_num, item_id_to_dic, users_train, users_valid, users_test, \
           users_history_for_valid, users_history_for_test, item_name_to_id



def read_behaviors_single_mmd(behaviors_path, before_item_id_to_dic, before_item_name_to_id, before_item_id_to_name, max_seq_len, min_seq_len, Log_file,num_k):
    random.seed(12345)
    np.random.seed(12345)
    Log_file.info("##### news number {} {} (before clearing)#####".format(len(before_item_id_to_dic), len(before_item_name_to_id)))
    Log_file.info("##### min seq len {}, max seq len {}#####".format(min_seq_len, max_seq_len))
    before_item_num = len(before_item_name_to_id)
    before_item_counts = [0] * (before_item_num + 1)
    user_seq_dic = {}
    seq_num = 0
    pairs_num = 0
    Log_file.info('rebuild user seqs...')
    with open(behaviors_path, "r") as f:
        for line in f:
            splited = line.strip('\n').split('\t')
            user_name = splited[0]
            history_item_name = splited[1].split(' ')
            if len(history_item_name) < min_seq_len:
                continue
            history_item_name = history_item_name[-(max_seq_len+3):]
            item_ids_sub_seq = [before_item_name_to_id[i] for i in history_item_name]
            user_seq_dic[user_name] = item_ids_sub_seq
            for item_id in item_ids_sub_seq:
                before_item_counts[item_id] += 1
                pairs_num += 1
            seq_num += 1
    Log_file.info("##### pairs_num {}".format(pairs_num))

    item_id = 1
    item_id_to_dic = {}
    item_name_to_id = {}
    item_id_before_to_now = {}
    for before_item_id in range(1, before_item_num + 1):
        if before_item_counts[before_item_id] != 0:
            item_id_before_to_now[before_item_id] = item_id
            item_id_to_dic[item_id] = before_item_id_to_dic[before_item_id]
            item_name_to_id[before_item_id_to_name[before_item_id]] = item_id
            item_id += 1
    item_num = len(item_id_before_to_now)
    Log_file.info("##### items after clearing {}, {}, {} #####".format(item_num, len(item_id_before_to_now), len(item_id_to_dic)))

    users_train = {}
    users_valid = {}
    users_test = {}
    users_history_for_valid = {}
    users_history_for_test = {}
    user_id = 0
    ee=pd.read_csv('/dataset/fre.csv')
    ee.columns = ['id', 'fre']
    ee.drop(0,inplace=True)
    e=ee.sort_values('fre',ascending=False)
    eee=e[['id']]
    eee=list(eee['id'])
    num=10
    #l = [0]
    #for k in range(num):
    #    l.append(int(79707*(1+k)/num))
    l = [0, 41, 127, 276, 517, 889, 1467, 2413, 4070, 7618, 79707, 79707]
    i=num_k###################################
    valid_id=0
    target=[] 

    for user_name, item_seqs in user_seq_dic.items():
        user_seq = [item_id_before_to_now[i] for i in item_seqs]
        train = user_seq[:-2] ####################################################################
        users_train[user_id] = train

        if (eee.index(user_seq[-2])<l[i+1] and eee.index(user_seq[-2])>=l[i] and i<10) or (i==10):
            valid = user_seq[-(max_seq_len+2):-1]
            users_valid[valid_id] = valid
            users_history_for_valid[valid_id] = torch.LongTensor(np.array(train))
            valid_id+=1
            target.append(user_seq[-2])
        test = user_seq[-(max_seq_len+1):]
        users_test[user_id] = test
        users_history_for_test[user_id] = torch.LongTensor(np.array(user_seq[:-1]))
        user_id += 1
    Log_file.info("##### user seqs after clearing {}, {}, {}, {}, {}#####".
                  format(seq_num, len(user_seq_dic), len(users_train), len(users_valid), len(users_test)))
    
    id_emb = torch.load('./tensor/single_beforetrain_id_128.pt')
    #content_emb = torch.load('./tensor/single_aftertrain_id_128.pt')
    content_emb = torch.load('./tensor/single_aftertune_id_128.pt')
    
    id_emb = id_emb.detach().numpy()
    content_emb = content_emb.detach().numpy()
    distance_llm = np.array([])
    distance_rs_cos = np.array([])
    distance_rs_dot = np.array([])
    #path = '/dataset/bce_text/id_plus_mo-2stage/tensor/single_aftertrain_id_'+str(num_k)+'.dat'
    path = '/dataset/bce_text/id_plus_mo-2stage/tensor/single_aftertune_id_'+str(num_k)+'.dat'
    if os.path.exists(path):
        distance_llm = np.fromfile(path, dtype=np.half)
        #path = '/dataset/bce_text/id_plus_mo-2stage/tensor/single_beforetrain_id_'+str(num_k)+'.dat'
        #distance_rs_cos = np.fromfile(path, dtype=np.half)
        #print(distance_rs_cos.shape)
        #print(distance_llm.shape)
        #else:
        for k in range(valid_id):
            seq=users_history_for_valid[k].tolist()
            seq.append(target[k])
            neg_items = []
            seq = [eee.index(j) for j in seq]
            if i==10:
                left=0
                right=79706
            else:
                left=l[i]
                right=l[i+1]-1
            for j in range(1):
                sam_neg = random.randint(left,right)
                while sam_neg in seq:
                    sam_neg = random.randint(left,right)
                neg_items.append(eee[sam_neg])
            neg_items=torch.LongTensor(np.array(neg_items))
            '''
            term_1 = content_emb[users_history_for_valid[k]]
            term_2 = content_emb[target[k]].reshape(1,-1)
            dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            distance_llm = np.concatenate((distance_llm,dis),0)
            
            term_3 = content_emb[neg_items].reshape(1,-1)
            dis = pairwise_distances(term_1, term_3, metric='cosine').reshape(term_1.shape[0])
            distance_llm = np.concatenate((distance_llm,dis),0)
            '''
            if num_k==10:
                term_1 = id_emb[users_history_for_valid[k]]
                term_2 = id_emb[target[k]].reshape(1,-1)
                dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
                distance_rs_cos= np.concatenate((distance_rs_cos,dis),0)
                
                term_3 = id_emb[neg_items].reshape(1,-1)
                dis = pairwise_distances(term_1, term_3, metric='cosine').reshape(term_1.shape[0])
                distance_rs_cos = np.concatenate((distance_rs_cos,dis),0)
            
        #distance_llm = np.half(distance_llm)
        distance_rs_cos = np.half(distance_rs_cos)
        #path = '/dataset/bce_text/id_plus_mo-2stage/tensor/single_aftertune_id_'+str(num_k)+'.dat'
        #distance_llm.tofile(path)
        #path = '/dataset/bce_text/id_plus_mo-2stage/tensor/single_beforetrain_id_'+str(num_k)+'.dat'
        #distance_rs_cos.tofile(path)
        #if num_k<10:
        #    path = '/dataset/bce_text/id_plus_mo-2stage/tensor/single_beforetrain_id_'+str(num_k)+'.dat'
        #    distance_rs_cos = np.fromfile(path, dtype=np.half)
        #else:
            
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/single_beforetrain_id_'+str(num_k)+'.dat'
        #if os.path.exists(path):
        distance_rs_cos.tofile(path)

    Kendallta,p_value = kendalltau(distance_rs_cos,distance_llm)
    print(Kendallta,p_value)
    print('done twotower task 3-1')

    #id_emb = torch.load('./tensor/single_beforetrain_content_128.pt')
    id_emb = torch.load('./tensor/single_beforetrain_id_128.pt')
    #content_emb = torch.load('./tensor/single_aftertrain_content_128.pt')
    content_emb = torch.load('./tensor/single_afteralignandtune_id_128.pt')

    id_emb = id_emb.detach().numpy()
    content_emb = content_emb.detach().numpy()
    distance_llm = np.array([])
    #distance_rs_cos = np.array([])
    distance_rs_dot = np.array([])
    path = '/dataset/bce_text/id_plus_mo-2stage/tensor/single_afteralignandtune_id_'+str(num_k)+'.dat'
    if os.path.exists(path):
        distance_llm = np.fromfile(path, dtype=np.half)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/single_beforetrain_id_'+str(num_k)+'.dat'
        #distance_rs_cos = np.fromfile(path, dtype=np.half)
        print(distance_rs_cos.shape)
        print(distance_llm.shape)
    else:
        for k in range(valid_id):
            seq=users_history_for_valid[k].tolist()
            seq.append(target[k])
            neg_items = []
            seq = [eee.index(j) for j in seq]
            if i==10:
                left=0
                right=79706
            else:
                left=l[i]
                right=l[i+1]-1
            for j in range(1):
                sam_neg = random.randint(left,right)
                while sam_neg in seq:
                    sam_neg = random.randint(left,right)
                neg_items.append(eee[sam_neg])
            neg_items=torch.LongTensor(np.array(neg_items))

            term_1 = content_emb[users_history_for_valid[k]]
            term_2 = content_emb[target[k]].reshape(1,-1)
            dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            distance_llm = np.concatenate((distance_llm,dis),0)
            
            term_3 = content_emb[neg_items].reshape(1,-1)
            dis = pairwise_distances(term_1, term_3, metric='cosine').reshape(term_1.shape[0])
            distance_llm = np.concatenate((distance_llm,dis),0)
            '''
            if num_k==10:
                term_1 = id_emb[users_history_for_valid[k]]
                term_2 = id_emb[target[k]].reshape(1,-1)
                dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
                distance_rs_cos= np.concatenate((distance_rs_cos,dis),0)
                
                term_3 = id_emb[neg_items].reshape(1,-1)
                dis = pairwise_distances(term_1, term_3, metric='cosine').reshape(term_1.shape[0])
                distance_rs_cos = np.concatenate((distance_rs_cos,dis),0)
            '''
        distance_llm = np.half(distance_llm)
        #distance_rs_cos = np.half(distance_rs_cos)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/single_afteralignandtune_id_'+str(num_k)+'.dat'
        distance_llm.tofile(path)
        #path = '/dataset/bce_text/id_plus_mo-2stage/tensor/single_beforetrain_id_'+str(num_k)+'.dat'
        #distance_rs_cos.tofile(path)
        #distance_rs_cos = np.fromfile(path, dtype=np.half)
            

    #print(distance_rs_cos.shape)
    #print(distance_llm.shape)
    Kendallta,p_value = kendalltau(distance_rs_cos,distance_llm)
    print(Kendallta,p_value)
    print('done twotower task 3-2')


    return item_num, item_id_to_dic, users_train, users_valid, users_test, \
           users_history_for_valid, users_history_for_test, item_name_to_id



def read_behaviors_fre_target_llm_neg_top_twotower_mmd(behaviors_path, before_item_id_to_dic, before_item_name_to_id, before_item_id_to_name, max_seq_len, min_seq_len, Log_file,num_k):
    random.seed(12345)
    np.random.seed(12345)
    Log_file.info("##### news number {} {} (before clearing)#####".format(len(before_item_id_to_dic), len(before_item_name_to_id)))
    Log_file.info("##### min seq len {}, max seq len {}#####".format(min_seq_len, max_seq_len))
    before_item_num = len(before_item_name_to_id)
    before_item_counts = [0] * (before_item_num + 1)
    user_seq_dic = {}
    seq_num = 0
    pairs_num = 0
    Log_file.info('rebuild user seqs...')
    with open(behaviors_path, "r") as f:
        for line in f:
            splited = line.strip('\n').split('\t')
            user_name = splited[0]
            history_item_name = splited[1].split(' ')
            if len(history_item_name) < min_seq_len:
                continue
            history_item_name = history_item_name[-(max_seq_len+3):]
            item_ids_sub_seq = [before_item_name_to_id[i] for i in history_item_name]
            user_seq_dic[user_name] = item_ids_sub_seq
            for item_id in item_ids_sub_seq:
                before_item_counts[item_id] += 1
                pairs_num += 1
            seq_num += 1
    Log_file.info("##### pairs_num {}".format(pairs_num))

    item_id = 1
    item_id_to_dic = {}
    item_name_to_id = {}
    item_id_before_to_now = {}
    for before_item_id in range(1, before_item_num + 1):
        if before_item_counts[before_item_id] != 0:
            item_id_before_to_now[before_item_id] = item_id
            item_id_to_dic[item_id] = before_item_id_to_dic[before_item_id]
            item_name_to_id[before_item_id_to_name[before_item_id]] = item_id
            item_id += 1
    item_num = len(item_id_before_to_now)
    Log_file.info("##### items after clearing {}, {}, {} #####".format(item_num, len(item_id_before_to_now), len(item_id_to_dic)))

    users_train = {}
    users_valid = {}
    users_test = {}
    users_history_for_valid = {}
    users_history_for_test = {}
    user_id = 0
    ee=pd.read_csv('/dataset/fre.csv')
    ee.columns = ['id', 'fre']
    ee.drop(0,inplace=True)
    e=ee.sort_values('fre',ascending=False)
    eee=e[['id']]
    eee=list(eee['id'])
    num=10
    #l = [0]
    #for k in range(num):
    #    l.append(int(79707*(1+k)/num))
    l = [0, 41, 127, 276, 517, 889, 1467, 2413, 4070, 7618, 79707, 79707]
    i=num_k###################################
    valid_id=0
    target=[] 

    for user_name, item_seqs in user_seq_dic.items():
        user_seq = [item_id_before_to_now[i] for i in item_seqs]
        train = user_seq[:-2] ####################################################################
        users_train[user_id] = train

        if (eee.index(user_seq[-2])<l[i+1] and eee.index(user_seq[-2])>=l[i] and i<10) or (i==10):
            valid = user_seq[-(max_seq_len+2):-1]
            users_valid[valid_id] = valid
            users_history_for_valid[valid_id] = torch.LongTensor(np.array(train))
            valid_id+=1
            target.append(user_seq[-2])
        test = user_seq[-(max_seq_len+1):]
        users_test[user_id] = test
        users_history_for_test[user_id] = torch.LongTensor(np.array(user_seq[:-1]))
        user_id += 1
    Log_file.info("##### user seqs after clearing {}, {}, {}, {}, {}#####".
                  format(seq_num, len(user_seq_dic), len(users_train), len(users_valid), len(users_test)))
    
    id_emb = torch.load('./tensor/order_twotower_mmd_other_128.pt')
    content_emb = torch.load('./tensor/order_twotower_mmd_content_128.pt')
    
    id_emb = id_emb.detach().numpy()
    content_emb = content_emb.detach().numpy()
    distance_llm = np.array([])
    distance_rs_cos = np.array([])
    distance_rs_dot = np.array([])
    path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_twotower_mmd_othercontent_llm_cos_'+str(num_k)+'.dat'
    if os.path.exists(path):
        distance_llm = np.fromfile(path, dtype=np.half)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_twotower_mmd_othercontent_rs_cos_'+str(num_k)+'.dat'
        distance_rs_cos = np.fromfile(path, dtype=np.half)
        #path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_mmd_rs_dot_'+str(num_k)+'.dat'
        #distance_rs_dot = np.fromfile(path, dtype=np.half)
    else:
        for k in range(valid_id):
            seq=users_history_for_valid[k].tolist()
            seq.append(target[k])
            neg_items = []
            seq = [eee.index(j) for j in seq]
            if i==10:
                left=0
                right=79706
            else:
                left=l[i]
                right=l[i+1]-1
            for j in range(1):
                sam_neg = random.randint(left,right)
                while sam_neg in seq:
                    sam_neg = random.randint(left,right)
                neg_items.append(eee[sam_neg])
            neg_items=torch.LongTensor(np.array(neg_items))

            term_1 = content_emb[users_history_for_valid[k]]
            term_2 = content_emb[target[k]].reshape(1,-1)
            dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            distance_llm = np.concatenate((distance_llm,dis),0)
            #a=pairwise_distances(term_1,metric="cosine")
            #uptri_idx = np.triu_indices_from(a, k=1)
            #dis=a[(uptri_idx)]
            #distance_llm = np.concatenate((distance_llm,dis),0)
            
            term_3 = content_emb[neg_items].reshape(1,-1)
            dis = pairwise_distances(term_1, term_3, metric='cosine').reshape(term_1.shape[0])
            distance_llm = np.concatenate((distance_llm,dis),0)

            term_1 = id_emb[users_history_for_valid[k]]
            term_2 = id_emb[target[k]].reshape(1,-1)
            dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            #distance_rs= np.concatenate((distance_rs,dis),0)
            #a=pairwise_distances(term_1,metric="cosine")
            #dis=a[(uptri_idx)]
            distance_rs_cos= np.concatenate((distance_rs_cos,dis),0)
            
            term_3 = id_emb[neg_items].reshape(1,-1)
            dis = pairwise_distances(term_1, term_3, metric='cosine').reshape(term_1.shape[0])
            distance_rs_cos = np.concatenate((distance_rs_cos,dis),0)
            '''
            term_1 = id_emb[users_history_for_valid[k]]
            term_2 = id_emb[target[k]].reshape(1,-1)
            #dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            #distance_rs= np.concatenate((distance_rs,dis),0)
            dis = np.dot(term_1,term_2.T).reshape(term_1.shape[0])
            #dis=a[(uptri_idx)]
            distance_rs_dot= np.concatenate((distance_rs_dot,dis),0)
            '''
        distance_llm = np.half(distance_llm)
        distance_rs_cos = np.half(distance_rs_cos)
        #distance_rs_dot = np.half(distance_rs_dot)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_twotower_mmd_othercontent_llm_cos_'+str(num_k)+'.dat'
        distance_llm.tofile(path)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_twotower_mmd_othercontent_rs_cos_'+str(num_k)+'.dat'
        distance_rs_cos.tofile(path)
        #path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_mmd_rs_dot_'+str(num_k)+'.dat'
        #distance_rs_dot.tofile(path)


    #print((distance_llm.shape))
    Kendallta,p_value = kendalltau(distance_rs_cos,distance_llm)
    print(Kendallta,p_value)
    c1 = distance_rs_cos
    d1 = distance_llm
    #e=c1
    #c1=d1
    #d1=e
    length = c1.shape[0]
    left = int(length/num)
    right = int(length/num*(num-1))
    indices = np.argsort(d1)[::-1]
    #plt.hist(d1, bins=60, alpha=0.5, color='grey')  # bins参数定义了直方图的柱子数量
    top_k_indices = indices[:left]
    pairs_1 = d1[top_k_indices]
    #plt.hist(pairs, bins=60, alpha=0.5, color='red')
    btm_k_indices = indices[right:]
    pairs_2 = d1[btm_k_indices]
    #plt.hist(pairs, bins=60, alpha=0.5, color='blue')
    num_bins = 80
    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    n, bins, patches = ax[0].hist(d1, bins=num_bins, edgecolor='skyblue', color='white', hatch='/', label='distance_distribution')
    n_set, bins_set, patches_set = ax[0].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_'+str(int(100/num)))
    n_set_2, bins_set_2, patches_set_2 = ax[0].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_'+str(int(100/num)))
    ax[0].set_title('llm2vec Embedding Distance Distribution', fontsize=16)
    ax[0].set_xlabel('distance', fontsize=14)
    ax[0].set_ylabel('num', fontsize=14)
    ax[0].legend()

    pairs_1 = c1[top_k_indices]
    pairs_2 = c1[btm_k_indices]
    n, bins, patches = ax[1].hist(c1, bins=num_bins, edgecolor='orange', color='white', hatch='/', label='sasrec_distribution')
    n_set, bins_set, patches_set = ax[1].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_'+str(int(100/num)), alpha=1)
    n_set_2, bins_set_2, patches_set_2 = ax[1].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_'+str(int(100/num)), alpha=0.4)
    ax[1].set_title('llm2vec Top/bottom '+str(int(100/num))+'% in SASRec', fontsize=16)
    ax[1].set_xlabel('distance', fontsize=14)
    ax[1].set_ylabel('num', fontsize=14)
    ax[1].legend()
    
    plt.savefig('./figure/target_llm_neg_top_distance-twotower_mmd-othercontent-cos-'+str(i)+'.png', dpi=600)  # 保存图为png格式
    plt.show()
    plt.clf()
    print('done twotower task 3-2')

    return item_num, item_id_to_dic, users_train, users_valid, users_test, \
           users_history_for_valid, users_history_for_test, item_name_to_id


def read_behaviors_fre_target_llm_neg_top_twotower_origin(behaviors_path, before_item_id_to_dic, before_item_name_to_id, before_item_id_to_name, max_seq_len, min_seq_len, Log_file,num_k):
    random.seed(12345)
    np.random.seed(12345)
    Log_file.info("##### news number {} {} (before clearing)#####".format(len(before_item_id_to_dic), len(before_item_name_to_id)))
    Log_file.info("##### min seq len {}, max seq len {}#####".format(min_seq_len, max_seq_len))
    before_item_num = len(before_item_name_to_id)
    before_item_counts = [0] * (before_item_num + 1)
    user_seq_dic = {}
    seq_num = 0
    pairs_num = 0
    Log_file.info('rebuild user seqs...')
    with open(behaviors_path, "r") as f:
        for line in f:
            splited = line.strip('\n').split('\t')
            user_name = splited[0]
            history_item_name = splited[1].split(' ')
            if len(history_item_name) < min_seq_len:
                continue
            history_item_name = history_item_name[-(max_seq_len+3):]
            item_ids_sub_seq = [before_item_name_to_id[i] for i in history_item_name]
            user_seq_dic[user_name] = item_ids_sub_seq
            for item_id in item_ids_sub_seq:
                before_item_counts[item_id] += 1
                pairs_num += 1
            seq_num += 1
    Log_file.info("##### pairs_num {}".format(pairs_num))

    item_id = 1
    item_id_to_dic = {}
    item_name_to_id = {}
    item_id_before_to_now = {}
    for before_item_id in range(1, before_item_num + 1):
        if before_item_counts[before_item_id] != 0:
            item_id_before_to_now[before_item_id] = item_id
            item_id_to_dic[item_id] = before_item_id_to_dic[before_item_id]
            item_name_to_id[before_item_id_to_name[before_item_id]] = item_id
            item_id += 1
    item_num = len(item_id_before_to_now)
    Log_file.info("##### items after clearing {}, {}, {} #####".format(item_num, len(item_id_before_to_now), len(item_id_to_dic)))

    users_train = {}
    users_valid = {}
    users_test = {}
    users_history_for_valid = {}
    users_history_for_test = {}
    user_id = 0
    ee=pd.read_csv('/dataset/fre.csv')
    ee.columns = ['id', 'fre']
    ee.drop(0,inplace=True)
    e=ee.sort_values('fre',ascending=False)
    eee=e[['id']]
    eee=list(eee['id'])
    num=10
    #l = [0]
    #for k in range(num):
    #    l.append(int(79707*(1+k)/num))
    l = [0, 41, 127, 276, 517, 889, 1467, 2413, 4070, 7618, 79707, 79707]
    i=num_k###################################
    valid_id=0
    target=[] 

    for user_name, item_seqs in user_seq_dic.items():
        user_seq = [item_id_before_to_now[i] for i in item_seqs]
        train = user_seq[:-2] ####################################################################
        users_train[user_id] = train

        if (eee.index(user_seq[-2])<l[i+1] and eee.index(user_seq[-2])>=l[i] and i<10) or (i==10):
            valid = user_seq[-(max_seq_len+2):-1]
            users_valid[valid_id] = valid
            users_history_for_valid[valid_id] = torch.LongTensor(np.array(train))
            valid_id+=1
            target.append(user_seq[-2])
        test = user_seq[-(max_seq_len+1):]
        users_test[user_id] = test
        users_history_for_test[user_id] = torch.LongTensor(np.array(user_seq[:-1]))
        user_id += 1
    Log_file.info("##### user seqs after clearing {}, {}, {}, {}, {}#####".
                  format(seq_num, len(user_seq_dic), len(users_train), len(users_valid), len(users_test)))
    
    id_emb = torch.load('./tensor/order_twotower_origin_other_128.pt')
    content_emb = torch.load('./tensor/order_twotower_origin_content_128.pt')
    
    id_emb = id_emb.detach().numpy()
    content_emb = content_emb.detach().numpy()
    distance_llm = np.array([])
    distance_rs_cos = np.array([])
    distance_rs_dot = np.array([])
    path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_twotower_origin_othercontent_llm_cos_'+str(num_k)+'.dat'
    if os.path.exists(path):
        distance_llm = np.fromfile(path, dtype=np.half)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_twotower_origin_othercontent_rs_cos_'+str(num_k)+'.dat'
        distance_rs_cos = np.fromfile(path, dtype=np.half)
        #path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_mmd_rs_dot_'+str(num_k)+'.dat'
        #distance_rs_dot = np.fromfile(path, dtype=np.half)
    else:
        for k in range(valid_id):
            seq=users_history_for_valid[k].tolist()
            seq.append(target[k])
            neg_items = []
            seq = [eee.index(j) for j in seq]
            if i==10:
                left=0
                right=79706
            else:
                left=l[i]
                right=l[i+1]-1
            for j in range(1):
                sam_neg = random.randint(left,right)
                while sam_neg in seq:
                    sam_neg = random.randint(left,right)
                neg_items.append(eee[sam_neg])
            neg_items=torch.LongTensor(np.array(neg_items))

            term_1 = content_emb[users_history_for_valid[k]]
            term_2 = content_emb[target[k]].reshape(1,-1)
            dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            distance_llm = np.concatenate((distance_llm,dis),0)
            #a=pairwise_distances(term_1,metric="cosine")
            #uptri_idx = np.triu_indices_from(a, k=1)
            #dis=a[(uptri_idx)]
            #distance_llm = np.concatenate((distance_llm,dis),0)
            
            term_3 = content_emb[neg_items].reshape(1,-1)
            dis = pairwise_distances(term_1, term_3, metric='cosine').reshape(term_1.shape[0])
            distance_llm = np.concatenate((distance_llm,dis),0)

            term_1 = id_emb[users_history_for_valid[k]]
            term_2 = id_emb[target[k]].reshape(1,-1)
            dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            #distance_rs= np.concatenate((distance_rs,dis),0)
            #a=pairwise_distances(term_1,metric="cosine")
            #dis=a[(uptri_idx)]
            distance_rs_cos= np.concatenate((distance_rs_cos,dis),0)
            
            term_3 = id_emb[neg_items].reshape(1,-1)
            dis = pairwise_distances(term_1, term_3, metric='cosine').reshape(term_1.shape[0])
            distance_rs_cos = np.concatenate((distance_rs_cos,dis),0)
            '''
            term_1 = id_emb[users_history_for_valid[k]]
            term_2 = id_emb[target[k]].reshape(1,-1)
            #dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            #distance_rs= np.concatenate((distance_rs,dis),0)
            dis = np.dot(term_1,term_2.T).reshape(term_1.shape[0])
            #dis=a[(uptri_idx)]
            distance_rs_dot= np.concatenate((distance_rs_dot,dis),0)
            '''
        distance_llm = np.half(distance_llm)
        distance_rs_cos = np.half(distance_rs_cos)
        #distance_rs_dot = np.half(distance_rs_dot)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_twotower_origin_othercontent_llm_cos_'+str(num_k)+'.dat'
        distance_llm.tofile(path)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_twotower_origin_othercontent_rs_cos_'+str(num_k)+'.dat'
        distance_rs_cos.tofile(path)
        #path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_mmd_rs_dot_'+str(num_k)+'.dat'
        #distance_rs_dot.tofile(path)


    #print((distance_llm.shape))
    Kendallta,p_value = kendalltau(distance_rs_cos,distance_llm)
    print(Kendallta,p_value)
    c1 = distance_rs_cos
    d1 = distance_llm
    #e=c1
    #c1=d1
    #d1=e
    length = c1.shape[0]
    left = int(length/num)
    right = int(length/num*(num-1))
    indices = np.argsort(d1)[::-1]
    #plt.hist(d1, bins=60, alpha=0.5, color='grey')  # bins参数定义了直方图的柱子数量
    top_k_indices = indices[:left]
    pairs_1 = d1[top_k_indices]
    #plt.hist(pairs, bins=60, alpha=0.5, color='red')
    btm_k_indices = indices[right:]
    pairs_2 = d1[btm_k_indices]
    #plt.hist(pairs, bins=60, alpha=0.5, color='blue')
    num_bins = 80
    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    n, bins, patches = ax[0].hist(d1, bins=num_bins, edgecolor='skyblue', color='white', hatch='/', label='distance_distribution')
    n_set, bins_set, patches_set = ax[0].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_'+str(int(100/num)))
    n_set_2, bins_set_2, patches_set_2 = ax[0].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_'+str(int(100/num)))
    ax[0].set_title('llm2vec Embedding Distance Distribution', fontsize=16)
    ax[0].set_xlabel('distance', fontsize=14)
    ax[0].set_ylabel('num', fontsize=14)
    ax[0].legend()

    pairs_1 = c1[top_k_indices]
    pairs_2 = c1[btm_k_indices]
    n, bins, patches = ax[1].hist(c1, bins=num_bins, edgecolor='orange', color='white', hatch='/', label='sasrec_distribution')
    n_set, bins_set, patches_set = ax[1].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_'+str(int(100/num)), alpha=1)
    n_set_2, bins_set_2, patches_set_2 = ax[1].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_'+str(int(100/num)), alpha=0.4)
    ax[1].set_title('llm2vec Top/bottom '+str(int(100/num))+'% in SASRec', fontsize=16)
    ax[1].set_xlabel('distance', fontsize=14)
    ax[1].set_ylabel('num', fontsize=14)
    ax[1].legend()
    
    plt.savefig('./figure/target_llm_neg_top_distance-twotower_origin-othercontent-cos-'+str(i)+'.png', dpi=600)  # 保存图为png格式
    plt.show()
    plt.clf()
    print('done twotower task 3-2')

    return item_num, item_id_to_dic, users_train, users_valid, users_test, \
           users_history_for_valid, users_history_for_test, item_name_to_id



def read_behaviors_fre_target_llm_neg_top_twotower_infonce(behaviors_path, before_item_id_to_dic, before_item_name_to_id, before_item_id_to_name, max_seq_len, min_seq_len, Log_file,num_k):
    random.seed(12345)
    np.random.seed(12345)
    Log_file.info("##### news number {} {} (before clearing)#####".format(len(before_item_id_to_dic), len(before_item_name_to_id)))
    Log_file.info("##### min seq len {}, max seq len {}#####".format(min_seq_len, max_seq_len))
    before_item_num = len(before_item_name_to_id)
    before_item_counts = [0] * (before_item_num + 1)
    user_seq_dic = {}
    seq_num = 0
    pairs_num = 0
    Log_file.info('rebuild user seqs...')
    with open(behaviors_path, "r") as f:
        for line in f:
            splited = line.strip('\n').split('\t')
            user_name = splited[0]
            history_item_name = splited[1].split(' ')
            if len(history_item_name) < min_seq_len:
                continue
            history_item_name = history_item_name[-(max_seq_len+3):]
            item_ids_sub_seq = [before_item_name_to_id[i] for i in history_item_name]
            user_seq_dic[user_name] = item_ids_sub_seq
            for item_id in item_ids_sub_seq:
                before_item_counts[item_id] += 1
                pairs_num += 1
            seq_num += 1
    Log_file.info("##### pairs_num {}".format(pairs_num))

    item_id = 1
    item_id_to_dic = {}
    item_name_to_id = {}
    item_id_before_to_now = {}
    for before_item_id in range(1, before_item_num + 1):
        if before_item_counts[before_item_id] != 0:
            item_id_before_to_now[before_item_id] = item_id
            item_id_to_dic[item_id] = before_item_id_to_dic[before_item_id]
            item_name_to_id[before_item_id_to_name[before_item_id]] = item_id
            item_id += 1
    item_num = len(item_id_before_to_now)
    Log_file.info("##### items after clearing {}, {}, {} #####".format(item_num, len(item_id_before_to_now), len(item_id_to_dic)))

    users_train = {}
    users_valid = {}
    users_test = {}
    users_history_for_valid = {}
    users_history_for_test = {}
    user_id = 0
    ee=pd.read_csv('/dataset/fre.csv')
    ee.columns = ['id', 'fre']
    ee.drop(0,inplace=True)
    e=ee.sort_values('fre',ascending=False)
    eee=e[['id']]
    eee=list(eee['id'])
    num=10
    #l = [0]
    #for k in range(num):
    #    l.append(int(79707*(1+k)/num))
    l = [0, 41, 127, 276, 517, 889, 1467, 2413, 4070, 7618, 79707, 79707]
    i=num_k###################################
    valid_id=0
    target=[] 

    for user_name, item_seqs in user_seq_dic.items():
        user_seq = [item_id_before_to_now[i] for i in item_seqs]
        train = user_seq[:-2] ####################################################################
        users_train[user_id] = train

        if (eee.index(user_seq[-2])<l[i+1] and eee.index(user_seq[-2])>=l[i] and i<10) or (i==10):
            valid = user_seq[-(max_seq_len+2):-1]
            users_valid[valid_id] = valid
            users_history_for_valid[valid_id] = torch.LongTensor(np.array(train))
            valid_id+=1
            target.append(user_seq[-2])
        test = user_seq[-(max_seq_len+1):]
        users_test[user_id] = test
        users_history_for_test[user_id] = torch.LongTensor(np.array(user_seq[:-1]))
        user_id += 1
    Log_file.info("##### user seqs after clearing {}, {}, {}, {}, {}#####".
                  format(seq_num, len(user_seq_dic), len(users_train), len(users_valid), len(users_test)))
    
    id_emb = torch.load('./tensor/order_twotower_infonce_other_128.pt')
    content_emb = torch.load('./tensor/order_twotower_infonce_content_128.pt')
    
    id_emb = id_emb.detach().numpy()
    content_emb = content_emb.detach().numpy()
    distance_llm = np.array([])
    distance_rs_cos = np.array([])
    distance_rs_dot = np.array([])
    path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_twotower_infonce_othercontent_llm_cos_'+str(num_k)+'.dat'
    if os.path.exists(path):
        distance_llm = np.fromfile(path, dtype=np.half)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_twotower_infonce_othercontent_rs_cos_'+str(num_k)+'.dat'
        distance_rs_cos = np.fromfile(path, dtype=np.half)
        #path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_mmd_rs_dot_'+str(num_k)+'.dat'
        #distance_rs_dot = np.fromfile(path, dtype=np.half)
    else:
        for k in range(valid_id):
            seq=users_history_for_valid[k].tolist()
            seq.append(target[k])
            neg_items = []
            seq = [eee.index(j) for j in seq]
            if i==10:
                left=0
                right=79706
            else:
                left=l[i]
                right=l[i+1]-1
            for j in range(1):
                sam_neg = random.randint(left,right)
                while sam_neg in seq:
                    sam_neg = random.randint(left,right)
                neg_items.append(eee[sam_neg])
            neg_items=torch.LongTensor(np.array(neg_items))

            term_1 = content_emb[users_history_for_valid[k]]
            term_2 = content_emb[target[k]].reshape(1,-1)
            dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            distance_llm = np.concatenate((distance_llm,dis),0)
            #a=pairwise_distances(term_1,metric="cosine")
            #uptri_idx = np.triu_indices_from(a, k=1)
            #dis=a[(uptri_idx)]
            #distance_llm = np.concatenate((distance_llm,dis),0)
            
            term_3 = content_emb[neg_items].reshape(1,-1)
            dis = pairwise_distances(term_1, term_3, metric='cosine').reshape(term_1.shape[0])
            distance_llm = np.concatenate((distance_llm,dis),0)

            term_1 = id_emb[users_history_for_valid[k]]
            term_2 = id_emb[target[k]].reshape(1,-1)
            dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            #distance_rs= np.concatenate((distance_rs,dis),0)
            #a=pairwise_distances(term_1,metric="cosine")
            #dis=a[(uptri_idx)]
            distance_rs_cos= np.concatenate((distance_rs_cos,dis),0)
            
            term_3 = id_emb[neg_items].reshape(1,-1)
            dis = pairwise_distances(term_1, term_3, metric='cosine').reshape(term_1.shape[0])
            distance_rs_cos = np.concatenate((distance_rs_cos,dis),0)
            '''
            term_1 = id_emb[users_history_for_valid[k]]
            term_2 = id_emb[target[k]].reshape(1,-1)
            #dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            #distance_rs= np.concatenate((distance_rs,dis),0)
            dis = np.dot(term_1,term_2.T).reshape(term_1.shape[0])
            #dis=a[(uptri_idx)]
            distance_rs_dot= np.concatenate((distance_rs_dot,dis),0)
            '''
        distance_llm = np.half(distance_llm)
        distance_rs_cos = np.half(distance_rs_cos)
        #distance_rs_dot = np.half(distance_rs_dot)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_twotower_infonce_othercontent_llm_cos_'+str(num_k)+'.dat'
        distance_llm.tofile(path)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_twotower_infonce_othercontent_rs_cos_'+str(num_k)+'.dat'
        distance_rs_cos.tofile(path)
        #path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_mmd_rs_dot_'+str(num_k)+'.dat'
        #distance_rs_dot.tofile(path)


    #print((distance_llm.shape))
    Kendallta,p_value = kendalltau(distance_rs_cos,distance_llm)
    print(Kendallta,p_value)
    c1 = distance_rs_cos
    d1 = distance_llm
    #e=c1
    #c1=d1
    #d1=e
    length = c1.shape[0]
    left = int(length/num)
    right = int(length/num*(num-1))
    indices = np.argsort(d1)[::-1]
    #plt.hist(d1, bins=60, alpha=0.5, color='grey')  # bins参数定义了直方图的柱子数量
    top_k_indices = indices[:left]
    pairs_1 = d1[top_k_indices]
    #plt.hist(pairs, bins=60, alpha=0.5, color='red')
    btm_k_indices = indices[right:]
    pairs_2 = d1[btm_k_indices]
    #plt.hist(pairs, bins=60, alpha=0.5, color='blue')
    num_bins = 80
    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    n, bins, patches = ax[0].hist(d1, bins=num_bins, edgecolor='skyblue', color='white', hatch='/', label='distance_distribution')
    n_set, bins_set, patches_set = ax[0].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_'+str(int(100/num)))
    n_set_2, bins_set_2, patches_set_2 = ax[0].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_'+str(int(100/num)))
    ax[0].set_title('llm2vec Embedding Distance Distribution', fontsize=16)
    ax[0].set_xlabel('distance', fontsize=14)
    ax[0].set_ylabel('num', fontsize=14)
    ax[0].legend()

    pairs_1 = c1[top_k_indices]
    pairs_2 = c1[btm_k_indices]
    n, bins, patches = ax[1].hist(c1, bins=num_bins, edgecolor='orange', color='white', hatch='/', label='sasrec_distribution')
    n_set, bins_set, patches_set = ax[1].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_'+str(int(100/num)), alpha=1)
    n_set_2, bins_set_2, patches_set_2 = ax[1].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_'+str(int(100/num)), alpha=0.4)
    ax[1].set_title('llm2vec Top/bottom '+str(int(100/num))+'% in SASRec', fontsize=16)
    ax[1].set_xlabel('distance', fontsize=14)
    ax[1].set_ylabel('num', fontsize=14)
    ax[1].legend()
    
    plt.savefig('./figure/target_llm_neg_top_distance-twotower_infonce-othercontent-cos-'+str(i)+'.png', dpi=600)  # 保存图为png格式
    plt.show()
    plt.clf()
    print('done twotower task 3-2')

    return item_num, item_id_to_dic, users_train, users_valid, users_test, \
           users_history_for_valid, users_history_for_test, item_name_to_id


def read_behaviors_fre_target_llm_neg_top_twotower_cosine(behaviors_path, before_item_id_to_dic, before_item_name_to_id, before_item_id_to_name, max_seq_len, min_seq_len, Log_file,num_k):
    random.seed(12345)
    np.random.seed(12345)
    Log_file.info("##### news number {} {} (before clearing)#####".format(len(before_item_id_to_dic), len(before_item_name_to_id)))
    Log_file.info("##### min seq len {}, max seq len {}#####".format(min_seq_len, max_seq_len))
    before_item_num = len(before_item_name_to_id)
    before_item_counts = [0] * (before_item_num + 1)
    user_seq_dic = {}
    seq_num = 0
    pairs_num = 0
    Log_file.info('rebuild user seqs...')
    with open(behaviors_path, "r") as f:
        for line in f:
            splited = line.strip('\n').split('\t')
            user_name = splited[0]
            history_item_name = splited[1].split(' ')
            if len(history_item_name) < min_seq_len:
                continue
            history_item_name = history_item_name[-(max_seq_len+3):]
            item_ids_sub_seq = [before_item_name_to_id[i] for i in history_item_name]
            user_seq_dic[user_name] = item_ids_sub_seq
            for item_id in item_ids_sub_seq:
                before_item_counts[item_id] += 1
                pairs_num += 1
            seq_num += 1
    Log_file.info("##### pairs_num {}".format(pairs_num))

    item_id = 1
    item_id_to_dic = {}
    item_name_to_id = {}
    item_id_before_to_now = {}
    for before_item_id in range(1, before_item_num + 1):
        if before_item_counts[before_item_id] != 0:
            item_id_before_to_now[before_item_id] = item_id
            item_id_to_dic[item_id] = before_item_id_to_dic[before_item_id]
            item_name_to_id[before_item_id_to_name[before_item_id]] = item_id
            item_id += 1
    item_num = len(item_id_before_to_now)
    Log_file.info("##### items after clearing {}, {}, {} #####".format(item_num, len(item_id_before_to_now), len(item_id_to_dic)))

    users_train = {}
    users_valid = {}
    users_test = {}
    users_history_for_valid = {}
    users_history_for_test = {}
    user_id = 0
    ee=pd.read_csv('/dataset/fre.csv')
    ee.columns = ['id', 'fre']
    ee.drop(0,inplace=True)
    e=ee.sort_values('fre',ascending=False)
    eee=e[['id']]
    eee=list(eee['id'])
    num=10
    #l = [0]
    #for k in range(num):
    #    l.append(int(79707*(1+k)/num))
    l = [0, 41, 127, 276, 517, 889, 1467, 2413, 4070, 7618, 79707, 79707]
    i=num_k###################################
    valid_id=0
    target=[] 

    for user_name, item_seqs in user_seq_dic.items():
        user_seq = [item_id_before_to_now[i] for i in item_seqs]
        train = user_seq[:-2] ####################################################################
        users_train[user_id] = train

        if (eee.index(user_seq[-2])<l[i+1] and eee.index(user_seq[-2])>=l[i] and i<10) or (i==10):
            valid = user_seq[-(max_seq_len+2):-1]
            users_valid[valid_id] = valid
            users_history_for_valid[valid_id] = torch.LongTensor(np.array(train))
            valid_id+=1
            target.append(user_seq[-2])
        test = user_seq[-(max_seq_len+1):]
        users_test[user_id] = test
        users_history_for_test[user_id] = torch.LongTensor(np.array(user_seq[:-1]))
        user_id += 1
    Log_file.info("##### user seqs after clearing {}, {}, {}, {}, {}#####".
                  format(seq_num, len(user_seq_dic), len(users_train), len(users_valid), len(users_test)))
    
    id_emb = torch.load('./tensor/order_twotower_cosine_other_128.pt')
    content_emb = torch.load('./tensor/order_twotower_cosine_content_128.pt')
    
    id_emb = id_emb.detach().numpy()
    content_emb = content_emb.detach().numpy()
    distance_llm = np.array([])
    distance_rs_cos = np.array([])
    distance_rs_dot = np.array([])
    path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_twotower_cosine_othercontent_llm_cos_'+str(num_k)+'.dat'
    if os.path.exists(path):
        distance_llm = np.fromfile(path, dtype=np.half)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_twotower_cosine_othercontent_rs_cos_'+str(num_k)+'.dat'
        distance_rs_cos = np.fromfile(path, dtype=np.half)
        #path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_mmd_rs_dot_'+str(num_k)+'.dat'
        #distance_rs_dot = np.fromfile(path, dtype=np.half)
    else:
        for k in range(valid_id):
            seq=users_history_for_valid[k].tolist()
            seq.append(target[k])
            neg_items = []
            seq = [eee.index(j) for j in seq]
            if i==10:
                left=0
                right=79706
            else:
                left=l[i]
                right=l[i+1]-1
            for j in range(1):
                sam_neg = random.randint(left,right)
                while sam_neg in seq:
                    sam_neg = random.randint(left,right)
                neg_items.append(eee[sam_neg])
            neg_items=torch.LongTensor(np.array(neg_items))

            term_1 = content_emb[users_history_for_valid[k]]
            term_2 = content_emb[target[k]].reshape(1,-1)
            dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            distance_llm = np.concatenate((distance_llm,dis),0)
            #a=pairwise_distances(term_1,metric="cosine")
            #uptri_idx = np.triu_indices_from(a, k=1)
            #dis=a[(uptri_idx)]
            #distance_llm = np.concatenate((distance_llm,dis),0)
            
            term_3 = content_emb[neg_items].reshape(1,-1)
            dis = pairwise_distances(term_1, term_3, metric='cosine').reshape(term_1.shape[0])
            distance_llm = np.concatenate((distance_llm,dis),0)

            term_1 = id_emb[users_history_for_valid[k]]
            term_2 = id_emb[target[k]].reshape(1,-1)
            dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            #distance_rs= np.concatenate((distance_rs,dis),0)
            #a=pairwise_distances(term_1,metric="cosine")
            #dis=a[(uptri_idx)]
            distance_rs_cos= np.concatenate((distance_rs_cos,dis),0)
            
            term_3 = id_emb[neg_items].reshape(1,-1)
            dis = pairwise_distances(term_1, term_3, metric='cosine').reshape(term_1.shape[0])
            distance_rs_cos = np.concatenate((distance_rs_cos,dis),0)
            '''
            term_1 = id_emb[users_history_for_valid[k]]
            term_2 = id_emb[target[k]].reshape(1,-1)
            #dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            #distance_rs= np.concatenate((distance_rs,dis),0)
            dis = np.dot(term_1,term_2.T).reshape(term_1.shape[0])
            #dis=a[(uptri_idx)]
            distance_rs_dot= np.concatenate((distance_rs_dot,dis),0)
            '''
        distance_llm = np.half(distance_llm)
        distance_rs_cos = np.half(distance_rs_cos)
        #distance_rs_dot = np.half(distance_rs_dot)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_twotower_cosine_othercontent_llm_cos_'+str(num_k)+'.dat'
        distance_llm.tofile(path)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_twotower_cosine_othercontent_rs_cos_'+str(num_k)+'.dat'
        distance_rs_cos.tofile(path)
        #path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_mmd_rs_dot_'+str(num_k)+'.dat'
        #distance_rs_dot.tofile(path)


    #print((distance_llm.shape))
    Kendallta,p_value = kendalltau(distance_rs_cos,distance_llm)
    print(Kendallta,p_value)
    c1 = distance_rs_cos
    d1 = distance_llm
    #e=c1
    #c1=d1
    #d1=e
    length = c1.shape[0]
    left = int(length/num)
    right = int(length/num*(num-1))
    indices = np.argsort(d1)[::-1]
    #plt.hist(d1, bins=60, alpha=0.5, color='grey')  # bins参数定义了直方图的柱子数量
    top_k_indices = indices[:left]
    pairs_1 = d1[top_k_indices]
    #plt.hist(pairs, bins=60, alpha=0.5, color='red')
    btm_k_indices = indices[right:]
    pairs_2 = d1[btm_k_indices]
    #plt.hist(pairs, bins=60, alpha=0.5, color='blue')
    num_bins = 80
    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    n, bins, patches = ax[0].hist(d1, bins=num_bins, edgecolor='skyblue', color='white', hatch='/', label='distance_distribution')
    n_set, bins_set, patches_set = ax[0].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_'+str(int(100/num)))
    n_set_2, bins_set_2, patches_set_2 = ax[0].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_'+str(int(100/num)))
    ax[0].set_title('llm2vec Embedding Distance Distribution', fontsize=16)
    ax[0].set_xlabel('distance', fontsize=14)
    ax[0].set_ylabel('num', fontsize=14)
    ax[0].legend()

    pairs_1 = c1[top_k_indices]
    pairs_2 = c1[btm_k_indices]
    n, bins, patches = ax[1].hist(c1, bins=num_bins, edgecolor='orange', color='white', hatch='/', label='sasrec_distribution')
    n_set, bins_set, patches_set = ax[1].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_'+str(int(100/num)), alpha=1)
    n_set_2, bins_set_2, patches_set_2 = ax[1].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_'+str(int(100/num)), alpha=0.4)
    ax[1].set_title('llm2vec Top/bottom '+str(int(100/num))+'% in SASRec', fontsize=16)
    ax[1].set_xlabel('distance', fontsize=14)
    ax[1].set_ylabel('num', fontsize=14)
    ax[1].legend()
    
    plt.savefig('./figure/target_llm_neg_top_distance-twotower_cosine-othercontent-cos-'+str(i)+'.png', dpi=600)  # 保存图为png格式
    plt.show()
    plt.clf()
    print('done twotower task 3-2')

    return item_num, item_id_to_dic, users_train, users_valid, users_test, \
           users_history_for_valid, users_history_for_test, item_name_to_id
def read_behaviors_fre_target_llm_neg_top_twotower_infonce(behaviors_path, before_item_id_to_dic, before_item_name_to_id, before_item_id_to_name, max_seq_len, min_seq_len, Log_file,num_k):
    random.seed(12345)
    np.random.seed(12345)
    Log_file.info("##### news number {} {} (before clearing)#####".format(len(before_item_id_to_dic), len(before_item_name_to_id)))
    Log_file.info("##### min seq len {}, max seq len {}#####".format(min_seq_len, max_seq_len))
    before_item_num = len(before_item_name_to_id)
    before_item_counts = [0] * (before_item_num + 1)
    user_seq_dic = {}
    seq_num = 0
    pairs_num = 0
    Log_file.info('rebuild user seqs...')
    with open(behaviors_path, "r") as f:
        for line in f:
            splited = line.strip('\n').split('\t')
            user_name = splited[0]
            history_item_name = splited[1].split(' ')
            if len(history_item_name) < min_seq_len:
                continue
            history_item_name = history_item_name[-(max_seq_len+3):]
            item_ids_sub_seq = [before_item_name_to_id[i] for i in history_item_name]
            user_seq_dic[user_name] = item_ids_sub_seq
            for item_id in item_ids_sub_seq:
                before_item_counts[item_id] += 1
                pairs_num += 1
            seq_num += 1
    Log_file.info("##### pairs_num {}".format(pairs_num))

    item_id = 1
    item_id_to_dic = {}
    item_name_to_id = {}
    item_id_before_to_now = {}
    for before_item_id in range(1, before_item_num + 1):
        if before_item_counts[before_item_id] != 0:
            item_id_before_to_now[before_item_id] = item_id
            item_id_to_dic[item_id] = before_item_id_to_dic[before_item_id]
            item_name_to_id[before_item_id_to_name[before_item_id]] = item_id
            item_id += 1
    item_num = len(item_id_before_to_now)
    Log_file.info("##### items after clearing {}, {}, {} #####".format(item_num, len(item_id_before_to_now), len(item_id_to_dic)))

    users_train = {}
    users_valid = {}
    users_test = {}
    users_history_for_valid = {}
    users_history_for_test = {}
    user_id = 0
    ee=pd.read_csv('/dataset/fre.csv')
    ee.columns = ['id', 'fre']
    ee.drop(0,inplace=True)
    e=ee.sort_values('fre',ascending=False)
    eee=e[['id']]
    eee=list(eee['id'])
    num=10
    #l = [0]
    #for k in range(num):
    #    l.append(int(79707*(1+k)/num))
    l = [0, 41, 127, 276, 517, 889, 1467, 2413, 4070, 7618, 79707, 79707]
    i=num_k###################################
    valid_id=0
    target=[] 

    for user_name, item_seqs in user_seq_dic.items():
        user_seq = [item_id_before_to_now[i] for i in item_seqs]
        train = user_seq[:-2] ####################################################################
        users_train[user_id] = train

        if (eee.index(user_seq[-2])<l[i+1] and eee.index(user_seq[-2])>=l[i] and i<10) or (i==10):
            valid = user_seq[-(max_seq_len+2):-1]
            users_valid[valid_id] = valid
            users_history_for_valid[valid_id] = torch.LongTensor(np.array(train))
            valid_id+=1
            target.append(user_seq[-2])
        test = user_seq[-(max_seq_len+1):]
        users_test[user_id] = test
        users_history_for_test[user_id] = torch.LongTensor(np.array(user_seq[:-1]))
        user_id += 1
    Log_file.info("##### user seqs after clearing {}, {}, {}, {}, {}#####".
                  format(seq_num, len(user_seq_dic), len(users_train), len(users_valid), len(users_test)))
    
    id_emb = torch.load('./tensor/order_twotower_infonce_other_128.pt')
    content_emb = torch.load('./tensor/order_twotower_infonce_content_128.pt')
    
    id_emb = id_emb.detach().numpy()
    content_emb = content_emb.detach().numpy()
    distance_llm = np.array([])
    distance_rs_cos = np.array([])
    distance_rs_dot = np.array([])
    path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_twotower_infonce_othercontent_llm_cos_'+str(num_k)+'.dat'
    if os.path.exists(path):
        distance_llm = np.fromfile(path, dtype=np.half)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_twotower_infonce_othercontent_rs_cos_'+str(num_k)+'.dat'
        distance_rs_cos = np.fromfile(path, dtype=np.half)
        #path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_mmd_rs_dot_'+str(num_k)+'.dat'
        #distance_rs_dot = np.fromfile(path, dtype=np.half)
    else:
        for k in range(valid_id):
            seq=users_history_for_valid[k].tolist()
            seq.append(target[k])
            neg_items = []
            seq = [eee.index(j) for j in seq]
            if i==10:
                left=0
                right=79706
            else:
                left=l[i]
                right=l[i+1]-1
            for j in range(1):
                sam_neg = random.randint(left,right)
                while sam_neg in seq:
                    sam_neg = random.randint(left,right)
                neg_items.append(eee[sam_neg])
            neg_items=torch.LongTensor(np.array(neg_items))

            term_1 = content_emb[users_history_for_valid[k]]
            term_2 = content_emb[target[k]].reshape(1,-1)
            dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            distance_llm = np.concatenate((distance_llm,dis),0)
            #a=pairwise_distances(term_1,metric="cosine")
            #uptri_idx = np.triu_indices_from(a, k=1)
            #dis=a[(uptri_idx)]
            #distance_llm = np.concatenate((distance_llm,dis),0)
            
            term_3 = content_emb[neg_items].reshape(1,-1)
            dis = pairwise_distances(term_1, term_3, metric='cosine').reshape(term_1.shape[0])
            distance_llm = np.concatenate((distance_llm,dis),0)

            term_1 = id_emb[users_history_for_valid[k]]
            term_2 = id_emb[target[k]].reshape(1,-1)
            dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            #distance_rs= np.concatenate((distance_rs,dis),0)
            #a=pairwise_distances(term_1,metric="cosine")
            #dis=a[(uptri_idx)]
            distance_rs_cos= np.concatenate((distance_rs_cos,dis),0)
            
            term_3 = id_emb[neg_items].reshape(1,-1)
            dis = pairwise_distances(term_1, term_3, metric='cosine').reshape(term_1.shape[0])
            distance_rs_cos = np.concatenate((distance_rs_cos,dis),0)
            '''
            term_1 = id_emb[users_history_for_valid[k]]
            term_2 = id_emb[target[k]].reshape(1,-1)
            #dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            #distance_rs= np.concatenate((distance_rs,dis),0)
            dis = np.dot(term_1,term_2.T).reshape(term_1.shape[0])
            #dis=a[(uptri_idx)]
            distance_rs_dot= np.concatenate((distance_rs_dot,dis),0)
            '''
        distance_llm = np.half(distance_llm)
        distance_rs_cos = np.half(distance_rs_cos)
        #distance_rs_dot = np.half(distance_rs_dot)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_twotower_infonce_othercontent_llm_cos_'+str(num_k)+'.dat'
        distance_llm.tofile(path)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_twotower_infonce_othercontent_rs_cos_'+str(num_k)+'.dat'
        distance_rs_cos.tofile(path)
        #path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_mmd_rs_dot_'+str(num_k)+'.dat'
        #distance_rs_dot.tofile(path)


    #print((distance_llm.shape))
    Kendallta,p_value = kendalltau(distance_rs_cos,distance_llm)
    print(Kendallta,p_value)
    c1 = distance_rs_cos
    d1 = distance_llm
    #e=c1
    #c1=d1
    #d1=e
    length = c1.shape[0]
    left = int(length/num)
    right = int(length/num*(num-1))
    indices = np.argsort(d1)[::-1]
    #plt.hist(d1, bins=60, alpha=0.5, color='grey')  # bins参数定义了直方图的柱子数量
    top_k_indices = indices[:left]
    pairs_1 = d1[top_k_indices]
    #plt.hist(pairs, bins=60, alpha=0.5, color='red')
    btm_k_indices = indices[right:]
    pairs_2 = d1[btm_k_indices]
    #plt.hist(pairs, bins=60, alpha=0.5, color='blue')
    num_bins = 80
    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    n, bins, patches = ax[0].hist(d1, bins=num_bins, edgecolor='skyblue', color='white', hatch='/', label='distance_distribution')
    n_set, bins_set, patches_set = ax[0].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_'+str(int(100/num)))
    n_set_2, bins_set_2, patches_set_2 = ax[0].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_'+str(int(100/num)))
    ax[0].set_title('llm2vec Embedding Distance Distribution', fontsize=16)
    ax[0].set_xlabel('distance', fontsize=14)
    ax[0].set_ylabel('num', fontsize=14)
    ax[0].legend()

    pairs_1 = c1[top_k_indices]
    pairs_2 = c1[btm_k_indices]
    n, bins, patches = ax[1].hist(c1, bins=num_bins, edgecolor='orange', color='white', hatch='/', label='sasrec_distribution')
    n_set, bins_set, patches_set = ax[1].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_'+str(int(100/num)), alpha=1)
    n_set_2, bins_set_2, patches_set_2 = ax[1].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_'+str(int(100/num)), alpha=0.4)
    ax[1].set_title('llm2vec Top/bottom '+str(int(100/num))+'% in SASRec', fontsize=16)
    ax[1].set_xlabel('distance', fontsize=14)
    ax[1].set_ylabel('num', fontsize=14)
    ax[1].legend()
    
    plt.savefig('./figure/target_llm_neg_top_distance-twotower_infonce-othercontent-cos-'+str(i)+'.png', dpi=600)  # 保存图为png格式
    plt.show()
    plt.clf()
    print('done twotower task 3-2')

    return item_num, item_id_to_dic, users_train, users_valid, users_test, \
           users_history_for_valid, users_history_for_test, item_name_to_id


def read_behaviors_fre_target_llm_neg_top_twotower_cosine(behaviors_path, before_item_id_to_dic, before_item_name_to_id, before_item_id_to_name, max_seq_len, min_seq_len, Log_file,num_k):
    random.seed(12345)
    np.random.seed(12345)
    Log_file.info("##### news number {} {} (before clearing)#####".format(len(before_item_id_to_dic), len(before_item_name_to_id)))
    Log_file.info("##### min seq len {}, max seq len {}#####".format(min_seq_len, max_seq_len))
    before_item_num = len(before_item_name_to_id)
    before_item_counts = [0] * (before_item_num + 1)
    user_seq_dic = {}
    seq_num = 0
    pairs_num = 0
    Log_file.info('rebuild user seqs...')
    with open(behaviors_path, "r") as f:
        for line in f:
            splited = line.strip('\n').split('\t')
            user_name = splited[0]
            history_item_name = splited[1].split(' ')
            if len(history_item_name) < min_seq_len:
                continue
            history_item_name = history_item_name[-(max_seq_len+3):]
            item_ids_sub_seq = [before_item_name_to_id[i] for i in history_item_name]
            user_seq_dic[user_name] = item_ids_sub_seq
            for item_id in item_ids_sub_seq:
                before_item_counts[item_id] += 1
                pairs_num += 1
            seq_num += 1
    Log_file.info("##### pairs_num {}".format(pairs_num))

    item_id = 1
    item_id_to_dic = {}
    item_name_to_id = {}
    item_id_before_to_now = {}
    for before_item_id in range(1, before_item_num + 1):
        if before_item_counts[before_item_id] != 0:
            item_id_before_to_now[before_item_id] = item_id
            item_id_to_dic[item_id] = before_item_id_to_dic[before_item_id]
            item_name_to_id[before_item_id_to_name[before_item_id]] = item_id
            item_id += 1
    item_num = len(item_id_before_to_now)
    Log_file.info("##### items after clearing {}, {}, {} #####".format(item_num, len(item_id_before_to_now), len(item_id_to_dic)))

    users_train = {}
    users_valid = {}
    users_test = {}
    users_history_for_valid = {}
    users_history_for_test = {}
    user_id = 0
    ee=pd.read_csv('/dataset/fre.csv')
    ee.columns = ['id', 'fre']
    ee.drop(0,inplace=True)
    e=ee.sort_values('fre',ascending=False)
    eee=e[['id']]
    eee=list(eee['id'])
    num=10
    #l = [0]
    #for k in range(num):
    #    l.append(int(79707*(1+k)/num))
    l = [0, 41, 127, 276, 517, 889, 1467, 2413, 4070, 7618, 79707, 79707]
    i=num_k###################################
    valid_id=0
    target=[] 

    for user_name, item_seqs in user_seq_dic.items():
        user_seq = [item_id_before_to_now[i] for i in item_seqs]
        train = user_seq[:-2] ####################################################################
        users_train[user_id] = train

        if (eee.index(user_seq[-2])<l[i+1] and eee.index(user_seq[-2])>=l[i] and i<10) or (i==10):
            valid = user_seq[-(max_seq_len+2):-1]
            users_valid[valid_id] = valid
            users_history_for_valid[valid_id] = torch.LongTensor(np.array(train))
            valid_id+=1
            target.append(user_seq[-2])
        test = user_seq[-(max_seq_len+1):]
        users_test[user_id] = test
        users_history_for_test[user_id] = torch.LongTensor(np.array(user_seq[:-1]))
        user_id += 1
    Log_file.info("##### user seqs after clearing {}, {}, {}, {}, {}#####".
                  format(seq_num, len(user_seq_dic), len(users_train), len(users_valid), len(users_test)))
    
    id_emb = torch.load('./tensor/order_twotower_cosine_other_128.pt')
    content_emb = torch.load('./tensor/order_twotower_cosine_content_128.pt')
    
    id_emb = id_emb.detach().numpy()
    content_emb = content_emb.detach().numpy()
    distance_llm = np.array([])
    distance_rs_cos = np.array([])
    distance_rs_dot = np.array([])
    path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_twotower_cosine_othercontent_llm_cos_'+str(num_k)+'.dat'
    if os.path.exists(path):
        distance_llm = np.fromfile(path, dtype=np.half)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_twotower_cosine_othercontent_rs_cos_'+str(num_k)+'.dat'
        distance_rs_cos = np.fromfile(path, dtype=np.half)
        #path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_mmd_rs_dot_'+str(num_k)+'.dat'
        #distance_rs_dot = np.fromfile(path, dtype=np.half)
    else:
        for k in range(valid_id):
            seq=users_history_for_valid[k].tolist()
            seq.append(target[k])
            neg_items = []
            seq = [eee.index(j) for j in seq]
            if i==10:
                left=0
                right=79706
            else:
                left=l[i]
                right=l[i+1]-1
            for j in range(1):
                sam_neg = random.randint(left,right)
                while sam_neg in seq:
                    sam_neg = random.randint(left,right)
                neg_items.append(eee[sam_neg])
            neg_items=torch.LongTensor(np.array(neg_items))

            term_1 = content_emb[users_history_for_valid[k]]
            term_2 = content_emb[target[k]].reshape(1,-1)
            dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            distance_llm = np.concatenate((distance_llm,dis),0)
            #a=pairwise_distances(term_1,metric="cosine")
            #uptri_idx = np.triu_indices_from(a, k=1)
            #dis=a[(uptri_idx)]
            #distance_llm = np.concatenate((distance_llm,dis),0)
            
            term_3 = content_emb[neg_items].reshape(1,-1)
            dis = pairwise_distances(term_1, term_3, metric='cosine').reshape(term_1.shape[0])
            distance_llm = np.concatenate((distance_llm,dis),0)

            term_1 = id_emb[users_history_for_valid[k]]
            term_2 = id_emb[target[k]].reshape(1,-1)
            dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            #distance_rs= np.concatenate((distance_rs,dis),0)
            #a=pairwise_distances(term_1,metric="cosine")
            #dis=a[(uptri_idx)]
            distance_rs_cos= np.concatenate((distance_rs_cos,dis),0)
            
            term_3 = id_emb[neg_items].reshape(1,-1)
            dis = pairwise_distances(term_1, term_3, metric='cosine').reshape(term_1.shape[0])
            distance_rs_cos = np.concatenate((distance_rs_cos,dis),0)
            '''
            term_1 = id_emb[users_history_for_valid[k]]
            term_2 = id_emb[target[k]].reshape(1,-1)
            #dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            #distance_rs= np.concatenate((distance_rs,dis),0)
            dis = np.dot(term_1,term_2.T).reshape(term_1.shape[0])
            #dis=a[(uptri_idx)]
            distance_rs_dot= np.concatenate((distance_rs_dot,dis),0)
            '''
        distance_llm = np.half(distance_llm)
        distance_rs_cos = np.half(distance_rs_cos)
        #distance_rs_dot = np.half(distance_rs_dot)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_twotower_cosine_othercontent_llm_cos_'+str(num_k)+'.dat'
        distance_llm.tofile(path)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_twotower_cosine_othercontent_rs_cos_'+str(num_k)+'.dat'
        distance_rs_cos.tofile(path)
        #path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_mmd_rs_dot_'+str(num_k)+'.dat'
        #distance_rs_dot.tofile(path)


    #print((distance_llm.shape))
    Kendallta,p_value = kendalltau(distance_rs_cos,distance_llm)
    print(Kendallta,p_value)
    c1 = distance_rs_cos
    d1 = distance_llm
    #e=c1
    #c1=d1
    #d1=e
    length = c1.shape[0]
    left = int(length/num)
    right = int(length/num*(num-1))
    indices = np.argsort(d1)[::-1]
    #plt.hist(d1, bins=60, alpha=0.5, color='grey')  # bins参数定义了直方图的柱子数量
    top_k_indices = indices[:left]
    pairs_1 = d1[top_k_indices]
    #plt.hist(pairs, bins=60, alpha=0.5, color='red')
    btm_k_indices = indices[right:]
    pairs_2 = d1[btm_k_indices]
    #plt.hist(pairs, bins=60, alpha=0.5, color='blue')
    num_bins = 80
    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    n, bins, patches = ax[0].hist(d1, bins=num_bins, edgecolor='skyblue', color='white', hatch='/', label='distance_distribution')
    n_set, bins_set, patches_set = ax[0].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_'+str(int(100/num)))
    n_set_2, bins_set_2, patches_set_2 = ax[0].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_'+str(int(100/num)))
    ax[0].set_title('llm2vec Embedding Distance Distribution', fontsize=16)
    ax[0].set_xlabel('distance', fontsize=14)
    ax[0].set_ylabel('num', fontsize=14)
    ax[0].legend()

    pairs_1 = c1[top_k_indices]
    pairs_2 = c1[btm_k_indices]
    n, bins, patches = ax[1].hist(c1, bins=num_bins, edgecolor='orange', color='white', hatch='/', label='sasrec_distribution')
    n_set, bins_set, patches_set = ax[1].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_'+str(int(100/num)), alpha=1)
    n_set_2, bins_set_2, patches_set_2 = ax[1].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_'+str(int(100/num)), alpha=0.4)
    ax[1].set_title('llm2vec Top/bottom '+str(int(100/num))+'% in SASRec', fontsize=16)
    ax[1].set_xlabel('distance', fontsize=14)
    ax[1].set_ylabel('num', fontsize=14)
    ax[1].legend()
    
    plt.savefig('./figure/target_llm_neg_top_distance-twotower_cosine-othercontent-cos-'+str(i)+'.png', dpi=600)  # 保存图为png格式
    plt.show()
    plt.clf()
    print('done twotower task 3-2')

    return item_num, item_id_to_dic, users_train, users_valid, users_test, \
           users_history_for_valid, users_history_for_test, item_name_to_id


def read_behaviors_fre_target_llm_neg_top_twotower_linear(behaviors_path, before_item_id_to_dic, before_item_name_to_id, before_item_id_to_name, max_seq_len, min_seq_len, Log_file,num_k):
    random.seed(12345)
    np.random.seed(12345)
    Log_file.info("##### news number {} {} (before clearing)#####".format(len(before_item_id_to_dic), len(before_item_name_to_id)))
    Log_file.info("##### min seq len {}, max seq len {}#####".format(min_seq_len, max_seq_len))
    before_item_num = len(before_item_name_to_id)
    before_item_counts = [0] * (before_item_num + 1)
    user_seq_dic = {}
    seq_num = 0
    pairs_num = 0
    Log_file.info('rebuild user seqs...')
    with open(behaviors_path, "r") as f:
        for line in f:
            splited = line.strip('\n').split('\t')
            user_name = splited[0]
            history_item_name = splited[1].split(' ')
            if len(history_item_name) < min_seq_len:
                continue
            history_item_name = history_item_name[-(max_seq_len+3):]
            item_ids_sub_seq = [before_item_name_to_id[i] for i in history_item_name]
            user_seq_dic[user_name] = item_ids_sub_seq
            for item_id in item_ids_sub_seq:
                before_item_counts[item_id] += 1
                pairs_num += 1
            seq_num += 1
    Log_file.info("##### pairs_num {}".format(pairs_num))

    item_id = 1
    item_id_to_dic = {}
    item_name_to_id = {}
    item_id_before_to_now = {}
    for before_item_id in range(1, before_item_num + 1):
        if before_item_counts[before_item_id] != 0:
            item_id_before_to_now[before_item_id] = item_id
            item_id_to_dic[item_id] = before_item_id_to_dic[before_item_id]
            item_name_to_id[before_item_id_to_name[before_item_id]] = item_id
            item_id += 1
    item_num = len(item_id_before_to_now)
    Log_file.info("##### items after clearing {}, {}, {} #####".format(item_num, len(item_id_before_to_now), len(item_id_to_dic)))

    users_train = {}
    users_valid = {}
    users_test = {}
    users_history_for_valid = {}
    users_history_for_test = {}
    user_id = 0
    ee=pd.read_csv('/dataset/fre.csv')
    ee.columns = ['id', 'fre']
    ee.drop(0,inplace=True)
    e=ee.sort_values('fre',ascending=False)
    eee=e[['id']]
    eee=list(eee['id'])
    num=10
    #l = [0]
    #for k in range(num):
    #    l.append(int(79707*(1+k)/num))
    l = [0, 41, 127, 276, 517, 889, 1467, 2413, 4070, 7618, 79707, 79707]
    i=num_k###################################
    valid_id=0
    target=[] 

    for user_name, item_seqs in user_seq_dic.items():
        user_seq = [item_id_before_to_now[i] for i in item_seqs]
        train = user_seq[:-2] ####################################################################
        users_train[user_id] = train

        if (eee.index(user_seq[-2])<l[i+1] and eee.index(user_seq[-2])>=l[i] and i<10) or (i==10):
            valid = user_seq[-(max_seq_len+2):-1]
            users_valid[valid_id] = valid
            users_history_for_valid[valid_id] = torch.LongTensor(np.array(train))
            valid_id+=1
            target.append(user_seq[-2])
        test = user_seq[-(max_seq_len+1):]
        users_test[user_id] = test
        users_history_for_test[user_id] = torch.LongTensor(np.array(user_seq[:-1]))
        user_id += 1
    Log_file.info("##### user seqs after clearing {}, {}, {}, {}, {}#####".
                  format(seq_num, len(user_seq_dic), len(users_train), len(users_valid), len(users_test)))
    '''
    id_emb = torch.load('./tensor/order_twotower_linear_item_id_128.pt')
    content_emb = torch.load('./tensor/order_twotower_linear_content_128.pt')
    
    id_emb = id_emb.detach().numpy()
    content_emb = content_emb.detach().numpy()
    distance_llm = np.array([])
    distance_rs_cos = np.array([])
    distance_rs_dot = np.array([])
    path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_twotower_linear_idcontent_llm_cos_'+str(num_k)+'.dat'
    if os.path.exists(path):
        distance_llm = np.fromfile(path, dtype=np.half)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_twotower_linear_idcontent_rs_cos_'+str(num_k)+'.dat'
        distance_rs_cos = np.fromfile(path, dtype=np.half)
        #path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_mmd_rs_dot_'+str(num_k)+'.dat'
        #distance_rs_dot = np.fromfile(path, dtype=np.half)
    else:
        for k in range(valid_id):
            seq=users_history_for_valid[k].tolist()
            seq.append(target[k])
            neg_items = []
            seq = [eee.index(j) for j in seq]
            if i==10:
                left=0
                right=79706
            else:
                left=l[i]
                right=l[i+1]-1
            for j in range(1):
                sam_neg = random.randint(left,right)
                while sam_neg in seq:
                    sam_neg = random.randint(left,right)
                neg_items.append(eee[sam_neg])
            neg_items=torch.LongTensor(np.array(neg_items))

            term_1 = content_emb[users_history_for_valid[k]]
            term_2 = content_emb[target[k]].reshape(1,-1)
            dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            distance_llm = np.concatenate((distance_llm,dis),0)
            #a=pairwise_distances(term_1,metric="cosine")
            #uptri_idx = np.triu_indices_from(a, k=1)
            #dis=a[(uptri_idx)]
            #distance_llm = np.concatenate((distance_llm,dis),0)
            
            term_3 = content_emb[neg_items].reshape(1,-1)
            dis = pairwise_distances(term_1, term_3, metric='cosine').reshape(term_1.shape[0])
            distance_llm = np.concatenate((distance_llm,dis),0)

            term_1 = id_emb[users_history_for_valid[k]]
            term_2 = id_emb[target[k]].reshape(1,-1)
            dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            #distance_rs= np.concatenate((distance_rs,dis),0)
            #a=pairwise_distances(term_1,metric="cosine")
            #dis=a[(uptri_idx)]
            distance_rs_cos= np.concatenate((distance_rs_cos,dis),0)
            
            term_3 = id_emb[neg_items].reshape(1,-1)
            dis = pairwise_distances(term_1, term_3, metric='cosine').reshape(term_1.shape[0])
            distance_rs_cos = np.concatenate((distance_rs_cos,dis),0)
            
            
        distance_llm = np.half(distance_llm)
        distance_rs_cos = np.half(distance_rs_cos)
        #distance_rs_dot = np.half(distance_rs_dot)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_twotower_linear_idcontent_llm_cos_'+str(num_k)+'.dat'
        distance_llm.tofile(path)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_twotower_linear_idcontent_rs_cos_'+str(num_k)+'.dat'
        distance_rs_cos.tofile(path)
        #path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_mmd_rs_dot_'+str(num_k)+'.dat'
        #distance_rs_dot.tofile(path)


    #print((distance_llm.shape))
    Kendallta,p_value = kendalltau(distance_rs_cos,distance_llm)
    print(Kendallta,p_value)
    c1 = distance_rs_cos
    d1 = distance_llm
    #e=c1
    #c1=d1
    #d1=e
    length = c1.shape[0]
    left = int(length/num)
    right = int(length/num*(num-1))
    indices = np.argsort(d1)[::-1]
    #plt.hist(d1, bins=60, alpha=0.5, color='grey')  # bins参数定义了直方图的柱子数量
    top_k_indices = indices[:left]
    pairs_1 = d1[top_k_indices]
    #plt.hist(pairs, bins=60, alpha=0.5, color='red')
    btm_k_indices = indices[right:]
    pairs_2 = d1[btm_k_indices]
    #plt.hist(pairs, bins=60, alpha=0.5, color='blue')
    num_bins = 80
    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    n, bins, patches = ax[0].hist(d1, bins=num_bins, edgecolor='skyblue', color='white', hatch='/', label='distance_distribution')
    n_set, bins_set, patches_set = ax[0].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_'+str(int(100/num)))
    n_set_2, bins_set_2, patches_set_2 = ax[0].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_'+str(int(100/num)))
    ax[0].set_title('llm2vec Embedding Distance Distribution', fontsize=16)
    ax[0].set_xlabel('distance', fontsize=14)
    ax[0].set_ylabel('num', fontsize=14)
    ax[0].legend()

    pairs_1 = c1[top_k_indices]
    pairs_2 = c1[btm_k_indices]
    n, bins, patches = ax[1].hist(c1, bins=num_bins, edgecolor='orange', color='white', hatch='/', label='sasrec_distribution')
    n_set, bins_set, patches_set = ax[1].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_'+str(int(100/num)), alpha=1)
    n_set_2, bins_set_2, patches_set_2 = ax[1].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_'+str(int(100/num)), alpha=0.4)
    ax[1].set_title('llm2vec Top/bottom '+str(int(100/num))+'% in SASRec', fontsize=16)
    ax[1].set_xlabel('distance', fontsize=14)
    ax[1].set_ylabel('num', fontsize=14)
    ax[1].legend()
    
    plt.savefig('./figure/target_llm_neg_top_distance-twotower_linear-idcontent-cos-'+str(i)+'.png', dpi=600)  # 保存图为png格式
    plt.show()
    plt.clf()
    print('done twotower task 3-1')
    '''
    
    
    
    id_emb = torch.load('./tensor/order_twotower_linear_other_128.pt')
    content_emb = torch.load('./tensor/order_twotower_linear_content_128.pt')
    
    id_emb = id_emb.detach().numpy()
    content_emb = content_emb.detach().numpy()
    distance_llm = np.array([])
    distance_rs_cos = np.array([])
    distance_rs_dot = np.array([])
    path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_twotower_linear_othercontent_llm_cos_'+str(num_k)+'.dat'
    if os.path.exists(path):
        distance_llm = np.fromfile(path, dtype=np.half)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_twotower_linear_othercontent_rs_cos_'+str(num_k)+'.dat'
        distance_rs_cos = np.fromfile(path, dtype=np.half)
        #path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_mmd_rs_dot_'+str(num_k)+'.dat'
        #distance_rs_dot = np.fromfile(path, dtype=np.half)
    else:
        for k in range(valid_id):
            seq=users_history_for_valid[k].tolist()
            seq.append(target[k])
            neg_items = []
            seq = [eee.index(j) for j in seq]
            if i==10:
                left=0
                right=79706
            else:
                left=l[i]
                right=l[i+1]-1
            for j in range(1):
                sam_neg = random.randint(left,right)
                while sam_neg in seq:
                    sam_neg = random.randint(left,right)
                neg_items.append(eee[sam_neg])
            neg_items=torch.LongTensor(np.array(neg_items))

            term_1 = content_emb[users_history_for_valid[k]]
            term_2 = content_emb[target[k]].reshape(1,-1)
            dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            distance_llm = np.concatenate((distance_llm,dis),0)
            #a=pairwise_distances(term_1,metric="cosine")
            #uptri_idx = np.triu_indices_from(a, k=1)
            #dis=a[(uptri_idx)]
            #distance_llm = np.concatenate((distance_llm,dis),0)
            
            term_3 = content_emb[neg_items].reshape(1,-1)
            dis = pairwise_distances(term_1, term_3, metric='cosine').reshape(term_1.shape[0])
            distance_llm = np.concatenate((distance_llm,dis),0)

            term_1 = id_emb[users_history_for_valid[k]]
            term_2 = id_emb[target[k]].reshape(1,-1)
            dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            #distance_rs= np.concatenate((distance_rs,dis),0)
            #a=pairwise_distances(term_1,metric="cosine")
            #dis=a[(uptri_idx)]
            distance_rs_cos= np.concatenate((distance_rs_cos,dis),0)
            
            term_3 = id_emb[neg_items].reshape(1,-1)
            dis = pairwise_distances(term_1, term_3, metric='cosine').reshape(term_1.shape[0])
            distance_rs_cos = np.concatenate((distance_rs_cos,dis),0)
            '''
            term_1 = id_emb[users_history_for_valid[k]]
            term_2 = id_emb[target[k]].reshape(1,-1)
            #dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            #distance_rs= np.concatenate((distance_rs,dis),0)
            dis = np.dot(term_1,term_2.T).reshape(term_1.shape[0])
            #dis=a[(uptri_idx)]
            distance_rs_dot= np.concatenate((distance_rs_dot,dis),0)
            '''
        distance_llm = np.half(distance_llm)
        distance_rs_cos = np.half(distance_rs_cos)
        #distance_rs_dot = np.half(distance_rs_dot)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_twotower_linear_othercontent_llm_cos_'+str(num_k)+'.dat'
        distance_llm.tofile(path)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_twotower_linear_othercontent_rs_cos_'+str(num_k)+'.dat'
        distance_rs_cos.tofile(path)
        #path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_mmd_rs_dot_'+str(num_k)+'.dat'
        #distance_rs_dot.tofile(path)


    #print((distance_llm.shape))
    Kendallta,p_value = kendalltau(distance_rs_cos,distance_llm)
    print(Kendallta,p_value)
    c1 = distance_rs_cos
    d1 = distance_llm
    #e=c1
    #c1=d1
    #d1=e
    length = c1.shape[0]
    left = int(length/num)
    right = int(length/num*(num-1))
    indices = np.argsort(d1)[::-1]
    #plt.hist(d1, bins=60, alpha=0.5, color='grey')  # bins参数定义了直方图的柱子数量
    top_k_indices = indices[:left]
    pairs_1 = d1[top_k_indices]
    #plt.hist(pairs, bins=60, alpha=0.5, color='red')
    btm_k_indices = indices[right:]
    pairs_2 = d1[btm_k_indices]
    #plt.hist(pairs, bins=60, alpha=0.5, color='blue')
    num_bins = 80
    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    n, bins, patches = ax[0].hist(d1, bins=num_bins, edgecolor='skyblue', color='white', hatch='/', label='distance_distribution')
    n_set, bins_set, patches_set = ax[0].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_'+str(int(100/num)))
    n_set_2, bins_set_2, patches_set_2 = ax[0].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_'+str(int(100/num)))
    ax[0].set_title('llm2vec Embedding Distance Distribution', fontsize=16)
    ax[0].set_xlabel('distance', fontsize=14)
    ax[0].set_ylabel('num', fontsize=14)
    ax[0].legend()

    pairs_1 = c1[top_k_indices]
    pairs_2 = c1[btm_k_indices]
    n, bins, patches = ax[1].hist(c1, bins=num_bins, edgecolor='orange', color='white', hatch='/', label='sasrec_distribution')
    n_set, bins_set, patches_set = ax[1].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_'+str(int(100/num)), alpha=1)
    n_set_2, bins_set_2, patches_set_2 = ax[1].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_'+str(int(100/num)), alpha=0.4)
    ax[1].set_title('llm2vec Top/bottom '+str(int(100/num))+'% in SASRec', fontsize=16)
    ax[1].set_xlabel('distance', fontsize=14)
    ax[1].set_ylabel('num', fontsize=14)
    ax[1].legend()
    
    plt.savefig('./figure/target_llm_neg_top_distance-twotower_linear-othercontent-cos-'+str(i)+'.png', dpi=600)  # 保存图为png格式
    plt.show()
    plt.clf()
    print('done twotower task 3-2')

    return item_num, item_id_to_dic, users_train, users_valid, users_test, \
           users_history_for_valid, users_history_for_test, item_name_to_id


def read_behaviors_fre_target_llm_neg_top_twotower(behaviors_path, before_item_id_to_dic, before_item_name_to_id, before_item_id_to_name, max_seq_len, min_seq_len, Log_file,num_k):
    random.seed(12345)
    np.random.seed(12345)
    Log_file.info("##### news number {} {} (before clearing)#####".format(len(before_item_id_to_dic), len(before_item_name_to_id)))
    Log_file.info("##### min seq len {}, max seq len {}#####".format(min_seq_len, max_seq_len))
    before_item_num = len(before_item_name_to_id)
    before_item_counts = [0] * (before_item_num + 1)
    user_seq_dic = {}
    seq_num = 0
    pairs_num = 0
    Log_file.info('rebuild user seqs...')
    with open(behaviors_path, "r") as f:
        for line in f:
            splited = line.strip('\n').split('\t')
            user_name = splited[0]
            history_item_name = splited[1].split(' ')
            if len(history_item_name) < min_seq_len:
                continue
            history_item_name = history_item_name[-(max_seq_len+3):]
            item_ids_sub_seq = [before_item_name_to_id[i] for i in history_item_name]
            user_seq_dic[user_name] = item_ids_sub_seq
            for item_id in item_ids_sub_seq:
                before_item_counts[item_id] += 1
                pairs_num += 1
            seq_num += 1
    Log_file.info("##### pairs_num {}".format(pairs_num))

    item_id = 1
    item_id_to_dic = {}
    item_name_to_id = {}
    item_id_before_to_now = {}
    for before_item_id in range(1, before_item_num + 1):
        if before_item_counts[before_item_id] != 0:
            item_id_before_to_now[before_item_id] = item_id
            item_id_to_dic[item_id] = before_item_id_to_dic[before_item_id]
            item_name_to_id[before_item_id_to_name[before_item_id]] = item_id
            item_id += 1
    item_num = len(item_id_before_to_now)
    Log_file.info("##### items after clearing {}, {}, {} #####".format(item_num, len(item_id_before_to_now), len(item_id_to_dic)))

    users_train = {}
    users_valid = {}
    users_test = {}
    users_history_for_valid = {}
    users_history_for_test = {}
    user_id = 0
    ee=pd.read_csv('/dataset/fre.csv')
    ee.columns = ['id', 'fre']
    ee.drop(0,inplace=True)
    e=ee.sort_values('fre',ascending=False)
    eee=e[['id']]
    eee=list(eee['id'])
    num=10
    #l = [0]
    #for k in range(num):
    #    l.append(int(79707*(1+k)/num))
    l = [0, 41, 127, 276, 517, 889, 1467, 2413, 4070, 7618, 79707, 79707]
    i=num_k###################################
    valid_id=0
    target=[] 

    for user_name, item_seqs in user_seq_dic.items():
        user_seq = [item_id_before_to_now[i] for i in item_seqs]
        train = user_seq[:-2] ####################################################################
        users_train[user_id] = train

        if (eee.index(user_seq[-2])<l[i+1] and eee.index(user_seq[-2])>=l[i] and i<10) or (i==10):
            valid = user_seq[-(max_seq_len+2):-1]
            users_valid[valid_id] = valid
            users_history_for_valid[valid_id] = torch.LongTensor(np.array(train))
            valid_id+=1
            target.append(user_seq[-2])
        test = user_seq[-(max_seq_len+1):]
        users_test[user_id] = test
        users_history_for_test[user_id] = torch.LongTensor(np.array(user_seq[:-1]))
        user_id += 1
    Log_file.info("##### user seqs after clearing {}, {}, {}, {}, {}#####".
                  format(seq_num, len(user_seq_dic), len(users_train), len(users_valid), len(users_test)))

    id_emb = torch.load('./tensor/order_twotower_item_id_128.pt')
    content_emb = torch.load('./tensor/order_twotower_content_128.pt')
    
    id_emb = id_emb.detach().numpy()
    content_emb = content_emb.detach().numpy()
    distance_llm = np.array([])
    distance_rs_cos = np.array([])
    distance_rs_dot = np.array([])
    path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_twotower_idcontent_llm_cos_'+str(num_k)+'.dat'
    if os.path.exists(path):
        distance_llm = np.fromfile(path, dtype=np.half)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_twotower_idcontent_rs_cos_'+str(num_k)+'.dat'
        distance_rs_cos = np.fromfile(path, dtype=np.half)
        #path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_mmd_rs_dot_'+str(num_k)+'.dat'
        #distance_rs_dot = np.fromfile(path, dtype=np.half)
    else:
        for k in range(valid_id):
            seq=users_history_for_valid[k].tolist()
            seq.append(target[k])
            neg_items = []
            seq = [eee.index(j) for j in seq]
            if i==10:
                left=0
                right=79706
            else:
                left=l[i]
                right=l[i+1]-1
            for j in range(1):
                sam_neg = random.randint(left,right)
                while sam_neg in seq:
                    sam_neg = random.randint(left,right)
                neg_items.append(eee[sam_neg])
            neg_items=torch.LongTensor(np.array(neg_items))

            term_1 = content_emb[users_history_for_valid[k]]
            term_2 = content_emb[target[k]].reshape(1,-1)
            dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            distance_llm = np.concatenate((distance_llm,dis),0)
            #a=pairwise_distances(term_1,metric="cosine")
            #uptri_idx = np.triu_indices_from(a, k=1)
            #dis=a[(uptri_idx)]
            #distance_llm = np.concatenate((distance_llm,dis),0)
            
            term_3 = content_emb[neg_items].reshape(1,-1)
            dis = pairwise_distances(term_1, term_3, metric='cosine').reshape(term_1.shape[0])
            distance_llm = np.concatenate((distance_llm,dis),0)

            term_1 = id_emb[users_history_for_valid[k]]
            term_2 = id_emb[target[k]].reshape(1,-1)
            dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            #distance_rs= np.concatenate((distance_rs,dis),0)
            #a=pairwise_distances(term_1,metric="cosine")
            #dis=a[(uptri_idx)]
            distance_rs_cos= np.concatenate((distance_rs_cos,dis),0)
            
            term_3 = id_emb[neg_items].reshape(1,-1)
            dis = pairwise_distances(term_1, term_3, metric='cosine').reshape(term_1.shape[0])
            distance_rs_cos = np.concatenate((distance_rs_cos,dis),0)
            '''
            term_1 = id_emb[users_history_for_valid[k]]
            term_2 = id_emb[target[k]].reshape(1,-1)
            #dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            #distance_rs= np.concatenate((distance_rs,dis),0)
            dis = np.dot(term_1,term_2.T).reshape(term_1.shape[0])
            #dis=a[(uptri_idx)]
            distance_rs_dot= np.concatenate((distance_rs_dot,dis),0)
            '''
        distance_llm = np.half(distance_llm)
        distance_rs_cos = np.half(distance_rs_cos)
        #distance_rs_dot = np.half(distance_rs_dot)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_twotower_idcontent_llm_cos_'+str(num_k)+'.dat'
        distance_llm.tofile(path)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_twotower_idcontent_rs_cos_'+str(num_k)+'.dat'
        distance_rs_cos.tofile(path)
        #path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_mmd_rs_dot_'+str(num_k)+'.dat'
        #distance_rs_dot.tofile(path)


    #print((distance_llm.shape))
    Kendallta,p_value = kendalltau(distance_rs_cos,distance_llm)
    print(Kendallta,p_value)
    c1 = distance_rs_cos
    d1 = distance_llm
    #e=c1
    #c1=d1
    #d1=e
    length = c1.shape[0]
    left = int(length/num)
    right = int(length/num*(num-1))
    indices = np.argsort(d1)[::-1]
    #plt.hist(d1, bins=60, alpha=0.5, color='grey')  # bins参数定义了直方图的柱子数量
    top_k_indices = indices[:left]
    pairs_1 = d1[top_k_indices]
    #plt.hist(pairs, bins=60, alpha=0.5, color='red')
    btm_k_indices = indices[right:]
    pairs_2 = d1[btm_k_indices]
    #plt.hist(pairs, bins=60, alpha=0.5, color='blue')
    num_bins = 80
    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    n, bins, patches = ax[0].hist(d1, bins=num_bins, edgecolor='skyblue', color='white', hatch='/', label='distance_distribution')
    n_set, bins_set, patches_set = ax[0].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_'+str(int(100/num)))
    n_set_2, bins_set_2, patches_set_2 = ax[0].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_'+str(int(100/num)))
    ax[0].set_title('llm2vec Embedding Distance Distribution', fontsize=16)
    ax[0].set_xlabel('distance', fontsize=14)
    ax[0].set_ylabel('num', fontsize=14)
    ax[0].legend()

    pairs_1 = c1[top_k_indices]
    pairs_2 = c1[btm_k_indices]
    n, bins, patches = ax[1].hist(c1, bins=num_bins, edgecolor='orange', color='white', hatch='/', label='sasrec_distribution')
    n_set, bins_set, patches_set = ax[1].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_'+str(int(100/num)), alpha=1)
    n_set_2, bins_set_2, patches_set_2 = ax[1].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_'+str(int(100/num)), alpha=0.4)
    ax[1].set_title('llm2vec Top/bottom '+str(int(100/num))+'% in SASRec', fontsize=16)
    ax[1].set_xlabel('distance', fontsize=14)
    ax[1].set_ylabel('num', fontsize=14)
    ax[1].legend()
    
    plt.savefig('./figure/target_llm_neg_top_distance-twotower-idcontent-cos-'+str(i)+'.png', dpi=600)  # 保存图为png格式
    plt.show()
    plt.clf()
    print('done twotower task 3-1')
    
    
    
    
    id_emb = torch.load('./tensor/order_twotower_other_128.pt')
    content_emb = torch.load('./tensor/order_twotower_content_128.pt')
    
    id_emb = id_emb.detach().numpy()
    content_emb = content_emb.detach().numpy()
    distance_llm = np.array([])
    distance_rs_cos = np.array([])
    distance_rs_dot = np.array([])
    path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_twotower_othercontent_llm_cos_'+str(num_k)+'.dat'
    if os.path.exists(path):
        distance_llm = np.fromfile(path, dtype=np.half)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_twotower_othercontent_rs_cos_'+str(num_k)+'.dat'
        distance_rs_cos = np.fromfile(path, dtype=np.half)
        #path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_mmd_rs_dot_'+str(num_k)+'.dat'
        #distance_rs_dot = np.fromfile(path, dtype=np.half)
    else:
        for k in range(valid_id):
            seq=users_history_for_valid[k].tolist()
            seq.append(target[k])
            neg_items = []
            seq = [eee.index(j) for j in seq]
            if i==10:
                left=0
                right=79706
            else:
                left=l[i]
                right=l[i+1]-1
            for j in range(1):
                sam_neg = random.randint(left,right)
                while sam_neg in seq:
                    sam_neg = random.randint(left,right)
                neg_items.append(eee[sam_neg])
            neg_items=torch.LongTensor(np.array(neg_items))

            term_1 = content_emb[users_history_for_valid[k]]
            term_2 = content_emb[target[k]].reshape(1,-1)
            dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            distance_llm = np.concatenate((distance_llm,dis),0)
            #a=pairwise_distances(term_1,metric="cosine")
            #uptri_idx = np.triu_indices_from(a, k=1)
            #dis=a[(uptri_idx)]
            #distance_llm = np.concatenate((distance_llm,dis),0)
            
            term_3 = content_emb[neg_items].reshape(1,-1)
            dis = pairwise_distances(term_1, term_3, metric='cosine').reshape(term_1.shape[0])
            distance_llm = np.concatenate((distance_llm,dis),0)

            term_1 = id_emb[users_history_for_valid[k]]
            term_2 = id_emb[target[k]].reshape(1,-1)
            dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            #distance_rs= np.concatenate((distance_rs,dis),0)
            #a=pairwise_distances(term_1,metric="cosine")
            #dis=a[(uptri_idx)]
            distance_rs_cos= np.concatenate((distance_rs_cos,dis),0)
            
            term_3 = id_emb[neg_items].reshape(1,-1)
            dis = pairwise_distances(term_1, term_3, metric='cosine').reshape(term_1.shape[0])
            distance_rs_cos = np.concatenate((distance_rs_cos,dis),0)
            '''
            term_1 = id_emb[users_history_for_valid[k]]
            term_2 = id_emb[target[k]].reshape(1,-1)
            #dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            #distance_rs= np.concatenate((distance_rs,dis),0)
            dis = np.dot(term_1,term_2.T).reshape(term_1.shape[0])
            #dis=a[(uptri_idx)]
            distance_rs_dot= np.concatenate((distance_rs_dot,dis),0)
            '''
        distance_llm = np.half(distance_llm)
        distance_rs_cos = np.half(distance_rs_cos)
        #distance_rs_dot = np.half(distance_rs_dot)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_twotower_othercontent_llm_cos_'+str(num_k)+'.dat'
        distance_llm.tofile(path)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_twotower_othercontent_rs_cos_'+str(num_k)+'.dat'
        distance_rs_cos.tofile(path)
        #path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_mmd_rs_dot_'+str(num_k)+'.dat'
        #distance_rs_dot.tofile(path)


    #print((distance_llm.shape))
    Kendallta,p_value = kendalltau(distance_rs_cos,distance_llm)
    print(Kendallta,p_value)
    c1 = distance_rs_cos
    d1 = distance_llm
    #e=c1
    #c1=d1
    #d1=e
    length = c1.shape[0]
    left = int(length/num)
    right = int(length/num*(num-1))
    indices = np.argsort(d1)[::-1]
    #plt.hist(d1, bins=60, alpha=0.5, color='grey')  # bins参数定义了直方图的柱子数量
    top_k_indices = indices[:left]
    pairs_1 = d1[top_k_indices]
    #plt.hist(pairs, bins=60, alpha=0.5, color='red')
    btm_k_indices = indices[right:]
    pairs_2 = d1[btm_k_indices]
    #plt.hist(pairs, bins=60, alpha=0.5, color='blue')
    num_bins = 80
    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    n, bins, patches = ax[0].hist(d1, bins=num_bins, edgecolor='skyblue', color='white', hatch='/', label='distance_distribution')
    n_set, bins_set, patches_set = ax[0].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_'+str(int(100/num)))
    n_set_2, bins_set_2, patches_set_2 = ax[0].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_'+str(int(100/num)))
    ax[0].set_title('llm2vec Embedding Distance Distribution', fontsize=16)
    ax[0].set_xlabel('distance', fontsize=14)
    ax[0].set_ylabel('num', fontsize=14)
    ax[0].legend()

    pairs_1 = c1[top_k_indices]
    pairs_2 = c1[btm_k_indices]
    n, bins, patches = ax[1].hist(c1, bins=num_bins, edgecolor='orange', color='white', hatch='/', label='sasrec_distribution')
    n_set, bins_set, patches_set = ax[1].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_'+str(int(100/num)), alpha=1)
    n_set_2, bins_set_2, patches_set_2 = ax[1].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_'+str(int(100/num)), alpha=0.4)
    ax[1].set_title('llm2vec Top/bottom '+str(int(100/num))+'% in SASRec', fontsize=16)
    ax[1].set_xlabel('distance', fontsize=14)
    ax[1].set_ylabel('num', fontsize=14)
    ax[1].legend()
    
    plt.savefig('./figure/target_llm_neg_top_distance-twotower-othercontent-cos-'+str(i)+'.png', dpi=600)  # 保存图为png格式
    plt.show()
    plt.clf()
    print('done twotower task 3-2')

    return item_num, item_id_to_dic, users_train, users_valid, users_test, \
           users_history_for_valid, users_history_for_test, item_name_to_id



def read_behaviors_fre_target_llm_neg_top_cosine(behaviors_path, before_item_id_to_dic, before_item_name_to_id, before_item_id_to_name, max_seq_len, min_seq_len, Log_file,num_k):
    random.seed(12345)
    np.random.seed(12345)
    Log_file.info("##### news number {} {} (before clearing)#####".format(len(before_item_id_to_dic), len(before_item_name_to_id)))
    Log_file.info("##### min seq len {}, max seq len {}#####".format(min_seq_len, max_seq_len))
    before_item_num = len(before_item_name_to_id)
    before_item_counts = [0] * (before_item_num + 1)
    user_seq_dic = {}
    seq_num = 0
    pairs_num = 0
    Log_file.info('rebuild user seqs...')
    with open(behaviors_path, "r") as f:
        for line in f:
            splited = line.strip('\n').split('\t')
            user_name = splited[0]
            history_item_name = splited[1].split(' ')
            if len(history_item_name) < min_seq_len:
                continue
            history_item_name = history_item_name[-(max_seq_len+3):]
            item_ids_sub_seq = [before_item_name_to_id[i] for i in history_item_name]
            user_seq_dic[user_name] = item_ids_sub_seq
            for item_id in item_ids_sub_seq:
                before_item_counts[item_id] += 1
                pairs_num += 1
            seq_num += 1
    Log_file.info("##### pairs_num {}".format(pairs_num))

    item_id = 1
    item_id_to_dic = {}
    item_name_to_id = {}
    item_id_before_to_now = {}
    for before_item_id in range(1, before_item_num + 1):
        if before_item_counts[before_item_id] != 0:
            item_id_before_to_now[before_item_id] = item_id
            item_id_to_dic[item_id] = before_item_id_to_dic[before_item_id]
            item_name_to_id[before_item_id_to_name[before_item_id]] = item_id
            item_id += 1
    item_num = len(item_id_before_to_now)
    Log_file.info("##### items after clearing {}, {}, {} #####".format(item_num, len(item_id_before_to_now), len(item_id_to_dic)))

    users_train = {}
    users_valid = {}
    users_test = {}
    users_history_for_valid = {}
    users_history_for_test = {}
    user_id = 0
    ee=pd.read_csv('/dataset/fre.csv')
    ee.columns = ['id', 'fre']
    ee.drop(0,inplace=True)
    e=ee.sort_values('fre',ascending=False)
    eee=e[['id']]
    eee=list(eee['id'])
    num=10
    #l = [0]
    #for k in range(num):
    #    l.append(int(79707*(1+k)/num))
    l = [0, 41, 127, 276, 517, 889, 1467, 2413, 4070, 7618, 79707, 79707]
    i=num_k###################################
    valid_id=0
    target = []

    for user_name, item_seqs in user_seq_dic.items():
        user_seq = [item_id_before_to_now[i] for i in item_seqs]
        train = user_seq[:-2] ####################################################################
        users_train[user_id] = train

        if (eee.index(user_seq[-2])<l[i+1] and eee.index(user_seq[-2])>=l[i] and i<10) or (i==10):
            valid = user_seq[-(max_seq_len+2):-1]
            users_valid[valid_id] = valid
            users_history_for_valid[valid_id] = torch.LongTensor(np.array(train))
            valid_id+=1
            target.append(user_seq[-2])
        test = user_seq[-(max_seq_len+1):]
        users_test[user_id] = test
        users_history_for_test[user_id] = torch.LongTensor(np.array(user_seq[:-1]))
        user_id += 1
    Log_file.info("##### user seqs after clearing {}, {}, {}, {}, {}#####".
                  format(seq_num, len(user_seq_dic), len(users_train), len(users_valid), len(users_test)))

    id_emb = torch.load('./tensor/order_cosine_item_id_128.pt')
    content_emb = torch.load('./tensor/order_cosine_content_128.pt')
    
    id_emb = id_emb.detach().numpy()
    content_emb = content_emb.detach().numpy()
    distance_llm = np.array([])
    distance_rs_cos = np.array([])
    distance_rs_dot = np.array([])
    path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_cosine_llm_cos_'+str(num_k)+'.dat'
    if os.path.exists(path):
        distance_llm = np.fromfile(path, dtype=np.half)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_cosine_rs_cos_'+str(num_k)+'.dat'
        distance_rs_cos = np.fromfile(path, dtype=np.half)
        #path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_mmd_rs_dot_'+str(num_k)+'.dat'
        #distance_rs_dot = np.fromfile(path, dtype=np.half)
    else:
        for k in range(valid_id):
            seq=users_history_for_valid[k].tolist()
            seq.append(target[k])
            neg_items = []
            seq = [eee.index(j) for j in seq]
            if i==10:
                left=0
                right=79706
            else:
                left=l[i]
                right=l[i+1]-1
            for j in range(1):
                sam_neg = random.randint(left,right)
                while sam_neg in seq:
                    sam_neg = random.randint(left,right)
                neg_items.append(eee[sam_neg])
            neg_items=torch.LongTensor(np.array(neg_items))

            term_1 = content_emb[users_history_for_valid[k]]
            term_2 = content_emb[target[k]].reshape(1,-1)
            dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            distance_llm = np.concatenate((distance_llm,dis),0)
            #a=pairwise_distances(term_1,metric="cosine")
            #uptri_idx = np.triu_indices_from(a, k=1)
            #dis=a[(uptri_idx)]
            #distance_llm = np.concatenate((distance_llm,dis),0)
            
            term_3 = content_emb[neg_items].reshape(1,-1)
            dis = pairwise_distances(term_1, term_3, metric='cosine').reshape(term_1.shape[0])
            distance_llm = np.concatenate((distance_llm,dis),0)

            term_1 = id_emb[users_history_for_valid[k]]
            term_2 = id_emb[target[k]].reshape(1,-1)
            dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            #distance_rs= np.concatenate((distance_rs,dis),0)
            #a=pairwise_distances(term_1,metric="cosine")
            #dis=a[(uptri_idx)]
            distance_rs_cos= np.concatenate((distance_rs_cos,dis),0)
            
            term_3 = id_emb[neg_items].reshape(1,-1)
            dis = pairwise_distances(term_1, term_3, metric='cosine').reshape(term_1.shape[0])
            distance_rs_cos = np.concatenate((distance_rs_cos,dis),0)
            '''
            term_1 = id_emb[users_history_for_valid[k]]
            term_2 = id_emb[target[k]].reshape(1,-1)
            #dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            #distance_rs= np.concatenate((distance_rs,dis),0)
            dis = np.dot(term_1,term_2.T).reshape(term_1.shape[0])
            #dis=a[(uptri_idx)]
            distance_rs_dot= np.concatenate((distance_rs_dot,dis),0)
            '''
        distance_llm = np.half(distance_llm)
        distance_rs_cos = np.half(distance_rs_cos)
        #distance_rs_dot = np.half(distance_rs_dot)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_cosine_llm_cos_'+str(num_k)+'.dat'
        distance_llm.tofile(path)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_cosine_rs_cos_'+str(num_k)+'.dat'
        distance_rs_cos.tofile(path)
        #path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_mmd_rs_dot_'+str(num_k)+'.dat'
        #distance_rs_dot.tofile(path)


    #print((distance_llm.shape))
    Kendallta,p_value = kendalltau(distance_rs_cos,distance_llm)
    print(Kendallta,p_value)
    c1 = distance_rs_cos
    d1 = distance_llm
    #e=c1
    #c1=d1
    #d1=e
    length = c1.shape[0]
    left = int(length/num)
    right = int(length/num*(num-1))
    indices = np.argsort(d1)[::-1]
    #plt.hist(d1, bins=60, alpha=0.5, color='grey')  # bins参数定义了直方图的柱子数量
    top_k_indices = indices[:left]
    pairs_1 = d1[top_k_indices]
    #plt.hist(pairs, bins=60, alpha=0.5, color='red')
    btm_k_indices = indices[right:]
    pairs_2 = d1[btm_k_indices]
    #plt.hist(pairs, bins=60, alpha=0.5, color='blue')
    num_bins = 80
    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    n, bins, patches = ax[0].hist(d1, bins=num_bins, edgecolor='skyblue', color='white', hatch='/', label='distance_distribution')
    n_set, bins_set, patches_set = ax[0].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_'+str(int(100/num)))
    n_set_2, bins_set_2, patches_set_2 = ax[0].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_'+str(int(100/num)))
    ax[0].set_title('llm2vec Embedding Distance Distribution', fontsize=16)
    ax[0].set_xlabel('distance', fontsize=14)
    ax[0].set_ylabel('num', fontsize=14)
    ax[0].legend()

    pairs_1 = c1[top_k_indices]
    pairs_2 = c1[btm_k_indices]
    n, bins, patches = ax[1].hist(c1, bins=num_bins, edgecolor='orange', color='white', hatch='/', label='sasrec_distribution')
    n_set, bins_set, patches_set = ax[1].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_'+str(int(100/num)), alpha=1)
    n_set_2, bins_set_2, patches_set_2 = ax[1].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_'+str(int(100/num)), alpha=0.4)
    ax[1].set_title('llm2vec Top/bottom '+str(int(100/num))+'% in SASRec', fontsize=16)
    ax[1].set_xlabel('distance', fontsize=14)
    ax[1].set_ylabel('num', fontsize=14)
    ax[1].legend()
    
    plt.savefig('./figure/target_llm_neg_top_distance-cosine-cos-'+str(i)+'.png', dpi=600)  # 保存图为png格式
    plt.show()
    plt.clf()
    print('done cosine task 3-1')

    return item_num, item_id_to_dic, users_train, users_valid, users_test, \
           users_history_for_valid, users_history_for_test, item_name_to_id

def read_behaviors_fre_target_llm_neg_top_linear(behaviors_path, before_item_id_to_dic, before_item_name_to_id, before_item_id_to_name, max_seq_len, min_seq_len, Log_file,num_k):
    random.seed(12345)
    np.random.seed(12345)
    Log_file.info("##### news number {} {} (before clearing)#####".format(len(before_item_id_to_dic), len(before_item_name_to_id)))
    Log_file.info("##### min seq len {}, max seq len {}#####".format(min_seq_len, max_seq_len))
    before_item_num = len(before_item_name_to_id)
    before_item_counts = [0] * (before_item_num + 1)
    user_seq_dic = {}
    seq_num = 0
    pairs_num = 0
    Log_file.info('rebuild user seqs...')
    with open(behaviors_path, "r") as f:
        for line in f:
            splited = line.strip('\n').split('\t')
            user_name = splited[0]
            history_item_name = splited[1].split(' ')
            if len(history_item_name) < min_seq_len:
                continue
            history_item_name = history_item_name[-(max_seq_len+3):]
            item_ids_sub_seq = [before_item_name_to_id[i] for i in history_item_name]
            user_seq_dic[user_name] = item_ids_sub_seq
            for item_id in item_ids_sub_seq:
                before_item_counts[item_id] += 1
                pairs_num += 1
            seq_num += 1
    Log_file.info("##### pairs_num {}".format(pairs_num))

    item_id = 1
    item_id_to_dic = {}
    item_name_to_id = {}
    item_id_before_to_now = {}
    for before_item_id in range(1, before_item_num + 1):
        if before_item_counts[before_item_id] != 0:
            item_id_before_to_now[before_item_id] = item_id
            item_id_to_dic[item_id] = before_item_id_to_dic[before_item_id]
            item_name_to_id[before_item_id_to_name[before_item_id]] = item_id
            item_id += 1
    item_num = len(item_id_before_to_now)
    Log_file.info("##### items after clearing {}, {}, {} #####".format(item_num, len(item_id_before_to_now), len(item_id_to_dic)))

    users_train = {}
    users_valid = {}
    users_test = {}
    users_history_for_valid = {}
    users_history_for_test = {}
    user_id = 0
    ee=pd.read_csv('/dataset/fre.csv')
    ee.columns = ['id', 'fre']
    ee.drop(0,inplace=True)
    e=ee.sort_values('fre',ascending=False)
    eee=e[['id']]
    eee=list(eee['id'])
    num=10
    #l = [0]
    #for k in range(num):
    #    l.append(int(79707*(1+k)/num))
    l = [0, 41, 127, 276, 517, 889, 1467, 2413, 4070, 7618, 79707, 79707]
    i=num_k###################################
    valid_id=0
    target = []

    for user_name, item_seqs in user_seq_dic.items():
        user_seq = [item_id_before_to_now[i] for i in item_seqs]
        train = user_seq[:-2] ####################################################################
        users_train[user_id] = train

        if (eee.index(user_seq[-2])<l[i+1] and eee.index(user_seq[-2])>=l[i] and i<10) or (i==10):
            valid = user_seq[-(max_seq_len+2):-1]
            users_valid[valid_id] = valid
            users_history_for_valid[valid_id] = torch.LongTensor(np.array(train))
            valid_id+=1
            target.append(user_seq[-2])
        test = user_seq[-(max_seq_len+1):]
        users_test[user_id] = test
        users_history_for_test[user_id] = torch.LongTensor(np.array(user_seq[:-1]))
        user_id += 1
    Log_file.info("##### user seqs after clearing {}, {}, {}, {}, {}#####".
                  format(seq_num, len(user_seq_dic), len(users_train), len(users_valid), len(users_test)))

    id_emb = torch.load('./tensor/order_linear_item_id_128.pt')
    content_emb = torch.load('./tensor/order_linear_content_128.pt')
    
    id_emb = id_emb.detach().numpy()
    content_emb = content_emb.detach().numpy()
    distance_llm = np.array([])
    distance_rs_cos = np.array([])
    distance_rs_dot = np.array([])
    path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_linear_llm_cos_'+str(num_k)+'.dat'
    if os.path.exists(path):
        distance_llm = np.fromfile(path, dtype=np.half)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_linear_rs_cos_'+str(num_k)+'.dat'
        distance_rs_cos = np.fromfile(path, dtype=np.half)
        #path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_mmd_rs_dot_'+str(num_k)+'.dat'
        #distance_rs_dot = np.fromfile(path, dtype=np.half)
    else:
        for k in range(valid_id):
            seq=users_history_for_valid[k].tolist()
            seq.append(target[k])
            neg_items = []
            seq = [eee.index(j) for j in seq]
            if i==10:
                left=0
                right=79706
            else:
                left=l[i]
                right=l[i+1]-1
            for j in range(1):
                sam_neg = random.randint(left,right)
                while sam_neg in seq:
                    sam_neg = random.randint(left,right)
                neg_items.append(eee[sam_neg])
            neg_items=torch.LongTensor(np.array(neg_items))

            term_1 = content_emb[users_history_for_valid[k]]
            term_2 = content_emb[target[k]].reshape(1,-1)
            dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            distance_llm = np.concatenate((distance_llm,dis),0)
            #a=pairwise_distances(term_1,metric="cosine")
            #uptri_idx = np.triu_indices_from(a, k=1)
            #dis=a[(uptri_idx)]
            #distance_llm = np.concatenate((distance_llm,dis),0)
            
            term_3 = content_emb[neg_items].reshape(1,-1)
            dis = pairwise_distances(term_1, term_3, metric='cosine').reshape(term_1.shape[0])
            distance_llm = np.concatenate((distance_llm,dis),0)

            term_1 = id_emb[users_history_for_valid[k]]
            term_2 = id_emb[target[k]].reshape(1,-1)
            dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            #distance_rs= np.concatenate((distance_rs,dis),0)
            #a=pairwise_distances(term_1,metric="cosine")
            #dis=a[(uptri_idx)]
            distance_rs_cos= np.concatenate((distance_rs_cos,dis),0)
            
            term_3 = id_emb[neg_items].reshape(1,-1)
            dis = pairwise_distances(term_1, term_3, metric='cosine').reshape(term_1.shape[0])
            distance_rs_cos = np.concatenate((distance_rs_cos,dis),0)
            '''
            term_1 = id_emb[users_history_for_valid[k]]
            term_2 = id_emb[target[k]].reshape(1,-1)
            #dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            #distance_rs= np.concatenate((distance_rs,dis),0)
            dis = np.dot(term_1,term_2.T).reshape(term_1.shape[0])
            #dis=a[(uptri_idx)]
            distance_rs_dot= np.concatenate((distance_rs_dot,dis),0)
            '''
        distance_llm = np.half(distance_llm)
        distance_rs_cos = np.half(distance_rs_cos)
        #distance_rs_dot = np.half(distance_rs_dot)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_linear_llm_cos_'+str(num_k)+'.dat'
        distance_llm.tofile(path)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_linear_rs_cos_'+str(num_k)+'.dat'
        distance_rs_cos.tofile(path)
        #path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_mmd_rs_dot_'+str(num_k)+'.dat'
        #distance_rs_dot.tofile(path)


    #print((distance_llm.shape))
    Kendallta,p_value = kendalltau(distance_rs_cos,distance_llm)
    print(Kendallta,p_value)
    c1 = distance_rs_cos
    d1 = distance_llm
    #e=c1
    #c1=d1
    #d1=e
    length = c1.shape[0]
    left = int(length/num)
    right = int(length/num*(num-1))
    indices = np.argsort(d1)[::-1]
    #plt.hist(d1, bins=60, alpha=0.5, color='grey')  # bins参数定义了直方图的柱子数量
    top_k_indices = indices[:left]
    pairs_1 = d1[top_k_indices]
    #plt.hist(pairs, bins=60, alpha=0.5, color='red')
    btm_k_indices = indices[right:]
    pairs_2 = d1[btm_k_indices]
    #plt.hist(pairs, bins=60, alpha=0.5, color='blue')
    num_bins = 80
    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    n, bins, patches = ax[0].hist(d1, bins=num_bins, edgecolor='skyblue', color='white', hatch='/', label='distance_distribution')
    n_set, bins_set, patches_set = ax[0].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_'+str(int(100/num)))
    n_set_2, bins_set_2, patches_set_2 = ax[0].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_'+str(int(100/num)))
    ax[0].set_title('llm2vec Embedding Distance Distribution', fontsize=16)
    ax[0].set_xlabel('distance', fontsize=14)
    ax[0].set_ylabel('num', fontsize=14)
    ax[0].legend()

    pairs_1 = c1[top_k_indices]
    pairs_2 = c1[btm_k_indices]
    n, bins, patches = ax[1].hist(c1, bins=num_bins, edgecolor='orange', color='white', hatch='/', label='sasrec_distribution')
    n_set, bins_set, patches_set = ax[1].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_'+str(int(100/num)), alpha=1)
    n_set_2, bins_set_2, patches_set_2 = ax[1].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_'+str(int(100/num)), alpha=0.4)
    ax[1].set_title('llm2vec Top/bottom '+str(int(100/num))+'% in SASRec', fontsize=16)
    ax[1].set_xlabel('distance', fontsize=14)
    ax[1].set_ylabel('num', fontsize=14)
    ax[1].legend()
    
    plt.savefig('./figure/target_llm_neg_top_distance-linear-cos-'+str(i)+'.png', dpi=600)  # 保存图为png格式
    plt.show()
    plt.clf()
    print('done linear task 3-1')

    return item_num, item_id_to_dic, users_train, users_valid, users_test, \
           users_history_for_valid, users_history_for_test, item_name_to_id

def read_behaviors_fre_target_llm_neg_top_infonce(behaviors_path, before_item_id_to_dic, before_item_name_to_id, before_item_id_to_name, max_seq_len, min_seq_len, Log_file,num_k):
    random.seed(12345)
    np.random.seed(12345)
    Log_file.info("##### news number {} {} (before clearing)#####".format(len(before_item_id_to_dic), len(before_item_name_to_id)))
    Log_file.info("##### min seq len {}, max seq len {}#####".format(min_seq_len, max_seq_len))
    before_item_num = len(before_item_name_to_id)
    before_item_counts = [0] * (before_item_num + 1)
    user_seq_dic = {}
    seq_num = 0
    pairs_num = 0
    Log_file.info('rebuild user seqs...')
    with open(behaviors_path, "r") as f:
        for line in f:
            splited = line.strip('\n').split('\t')
            user_name = splited[0]
            history_item_name = splited[1].split(' ')
            if len(history_item_name) < min_seq_len:
                continue
            history_item_name = history_item_name[-(max_seq_len+3):]
            item_ids_sub_seq = [before_item_name_to_id[i] for i in history_item_name]
            user_seq_dic[user_name] = item_ids_sub_seq
            for item_id in item_ids_sub_seq:
                before_item_counts[item_id] += 1
                pairs_num += 1
            seq_num += 1
    Log_file.info("##### pairs_num {}".format(pairs_num))

    item_id = 1
    item_id_to_dic = {}
    item_name_to_id = {}
    item_id_before_to_now = {}
    for before_item_id in range(1, before_item_num + 1):
        if before_item_counts[before_item_id] != 0:
            item_id_before_to_now[before_item_id] = item_id
            item_id_to_dic[item_id] = before_item_id_to_dic[before_item_id]
            item_name_to_id[before_item_id_to_name[before_item_id]] = item_id
            item_id += 1
    item_num = len(item_id_before_to_now)
    Log_file.info("##### items after clearing {}, {}, {} #####".format(item_num, len(item_id_before_to_now), len(item_id_to_dic)))

    users_train = {}
    users_valid = {}
    users_test = {}
    users_history_for_valid = {}
    users_history_for_test = {}
    user_id = 0
    ee=pd.read_csv('/dataset/fre.csv')
    ee.columns = ['id', 'fre']
    ee.drop(0,inplace=True)
    e=ee.sort_values('fre',ascending=False)
    eee=e[['id']]
    eee=list(eee['id'])
    num=10
    #l = [0]
    #for k in range(num):
    #    l.append(int(79707*(1+k)/num))
    l = [0, 41, 127, 276, 517, 889, 1467, 2413, 4070, 7618, 79707, 79707]
    i=num_k###################################
    valid_id=0
    target = []

    for user_name, item_seqs in user_seq_dic.items():
        user_seq = [item_id_before_to_now[i] for i in item_seqs]
        train = user_seq[:-2] ####################################################################
        users_train[user_id] = train

        if (eee.index(user_seq[-2])<l[i+1] and eee.index(user_seq[-2])>=l[i] and i<10) or (i==10):
            valid = user_seq[-(max_seq_len+2):-1]
            users_valid[valid_id] = valid
            users_history_for_valid[valid_id] = torch.LongTensor(np.array(train))
            valid_id+=1
            target.append(user_seq[-2])
        test = user_seq[-(max_seq_len+1):]
        users_test[user_id] = test
        users_history_for_test[user_id] = torch.LongTensor(np.array(user_seq[:-1]))
        user_id += 1
    Log_file.info("##### user seqs after clearing {}, {}, {}, {}, {}#####".
                  format(seq_num, len(user_seq_dic), len(users_train), len(users_valid), len(users_test)))

    id_emb = torch.load('./tensor/order_infonce_item_id_128.pt')
    content_emb = torch.load('./tensor/order_infonce_content_128.pt')
    
    id_emb = id_emb.detach().numpy()
    content_emb = content_emb.detach().numpy()
    distance_llm = np.array([])
    distance_rs_cos = np.array([])
    distance_rs_dot = np.array([])
    path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_infonce_llm_cos_'+str(num_k)+'.dat'
    if os.path.exists(path):
        distance_llm = np.fromfile(path, dtype=np.half)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_infonce_rs_cos_'+str(num_k)+'.dat'
        distance_rs_cos = np.fromfile(path, dtype=np.half)
        #path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_mmd_rs_dot_'+str(num_k)+'.dat'
        #distance_rs_dot = np.fromfile(path, dtype=np.half)
    else:
        for k in range(valid_id):
            seq=users_history_for_valid[k].tolist()
            seq.append(target[k])
            neg_items = []
            seq = [eee.index(j) for j in seq]
            if i==10:
                left=0
                right=79706
            else:
                left=l[i]
                right=l[i+1]-1
            for j in range(1):
                sam_neg = random.randint(left,right)
                while sam_neg in seq:
                    sam_neg = random.randint(left,right)
                neg_items.append(eee[sam_neg])
            neg_items=torch.LongTensor(np.array(neg_items))

            term_1 = content_emb[users_history_for_valid[k]]
            term_2 = content_emb[target[k]].reshape(1,-1)
            dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            distance_llm = np.concatenate((distance_llm,dis),0)
            #a=pairwise_distances(term_1,metric="cosine")
            #uptri_idx = np.triu_indices_from(a, k=1)
            #dis=a[(uptri_idx)]
            #distance_llm = np.concatenate((distance_llm,dis),0)
            
            term_3 = content_emb[neg_items].reshape(1,-1)
            dis = pairwise_distances(term_1, term_3, metric='cosine').reshape(term_1.shape[0])
            distance_llm = np.concatenate((distance_llm,dis),0)

            term_1 = id_emb[users_history_for_valid[k]]
            term_2 = id_emb[target[k]].reshape(1,-1)
            dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            #distance_rs= np.concatenate((distance_rs,dis),0)
            #a=pairwise_distances(term_1,metric="cosine")
            #dis=a[(uptri_idx)]
            distance_rs_cos= np.concatenate((distance_rs_cos,dis),0)
            
            term_3 = id_emb[neg_items].reshape(1,-1)
            dis = pairwise_distances(term_1, term_3, metric='cosine').reshape(term_1.shape[0])
            distance_rs_cos = np.concatenate((distance_rs_cos,dis),0)
            '''
            term_1 = id_emb[users_history_for_valid[k]]
            term_2 = id_emb[target[k]].reshape(1,-1)
            #dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            #distance_rs= np.concatenate((distance_rs,dis),0)
            dis = np.dot(term_1,term_2.T).reshape(term_1.shape[0])
            #dis=a[(uptri_idx)]
            distance_rs_dot= np.concatenate((distance_rs_dot,dis),0)
            '''
        distance_llm = np.half(distance_llm)
        distance_rs_cos = np.half(distance_rs_cos)
        #distance_rs_dot = np.half(distance_rs_dot)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_infonce_llm_cos_'+str(num_k)+'.dat'
        distance_llm.tofile(path)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_infonce_rs_cos_'+str(num_k)+'.dat'
        distance_rs_cos.tofile(path)
        #path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_mmd_rs_dot_'+str(num_k)+'.dat'
        #distance_rs_dot.tofile(path)


    #print((distance_llm.shape))
    Kendallta,p_value = kendalltau(distance_rs_cos,distance_llm)
    print(Kendallta,p_value)
    c1 = distance_rs_cos
    d1 = distance_llm
    #e=c1
    #c1=d1
    #d1=e
    length = c1.shape[0]
    left = int(length/num)
    right = int(length/num*(num-1))
    indices = np.argsort(d1)[::-1]
    #plt.hist(d1, bins=60, alpha=0.5, color='grey')  # bins参数定义了直方图的柱子数量
    top_k_indices = indices[:left]
    pairs_1 = d1[top_k_indices]
    #plt.hist(pairs, bins=60, alpha=0.5, color='red')
    btm_k_indices = indices[right:]
    pairs_2 = d1[btm_k_indices]
    #plt.hist(pairs, bins=60, alpha=0.5, color='blue')
    num_bins = 80
    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    n, bins, patches = ax[0].hist(d1, bins=num_bins, edgecolor='skyblue', color='white', hatch='/', label='distance_distribution')
    n_set, bins_set, patches_set = ax[0].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_'+str(int(100/num)))
    n_set_2, bins_set_2, patches_set_2 = ax[0].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_'+str(int(100/num)))
    ax[0].set_title('llm2vec Embedding Distance Distribution', fontsize=16)
    ax[0].set_xlabel('distance', fontsize=14)
    ax[0].set_ylabel('num', fontsize=14)
    ax[0].legend()

    pairs_1 = c1[top_k_indices]
    pairs_2 = c1[btm_k_indices]
    n, bins, patches = ax[1].hist(c1, bins=num_bins, edgecolor='orange', color='white', hatch='/', label='sasrec_distribution')
    n_set, bins_set, patches_set = ax[1].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_'+str(int(100/num)), alpha=1)
    n_set_2, bins_set_2, patches_set_2 = ax[1].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_'+str(int(100/num)), alpha=0.4)
    ax[1].set_title('llm2vec Top/bottom '+str(int(100/num))+'% in SASRec', fontsize=16)
    ax[1].set_xlabel('distance', fontsize=14)
    ax[1].set_ylabel('num', fontsize=14)
    ax[1].legend()
    
    plt.savefig('./figure/target_llm_neg_top_distance-infonce-cos-'+str(i)+'.png', dpi=600)  # 保存图为png格式
    plt.show()
    plt.clf()
    print('done infonce task 3-1')

    return item_num, item_id_to_dic, users_train, users_valid, users_test, \
           users_history_for_valid, users_history_for_test, item_name_to_id



def read_behaviors_fre_target_llm_neg_top(behaviors_path, before_item_id_to_dic, before_item_name_to_id, before_item_id_to_name, max_seq_len, min_seq_len, Log_file,num_k):
    random.seed(12345)
    np.random.seed(12345)
    Log_file.info("##### news number {} {} (before clearing)#####".format(len(before_item_id_to_dic), len(before_item_name_to_id)))
    Log_file.info("##### min seq len {}, max seq len {}#####".format(min_seq_len, max_seq_len))
    before_item_num = len(before_item_name_to_id)
    before_item_counts = [0] * (before_item_num + 1)
    user_seq_dic = {}
    seq_num = 0
    pairs_num = 0
    Log_file.info('rebuild user seqs...')
    with open(behaviors_path, "r") as f:
        for line in f:
            splited = line.strip('\n').split('\t')
            user_name = splited[0]
            history_item_name = splited[1].split(' ')
            if len(history_item_name) < min_seq_len:
                continue
            history_item_name = history_item_name[-(max_seq_len+3):]
            item_ids_sub_seq = [before_item_name_to_id[i] for i in history_item_name]
            user_seq_dic[user_name] = item_ids_sub_seq
            for item_id in item_ids_sub_seq:
                before_item_counts[item_id] += 1
                pairs_num += 1
            seq_num += 1
    Log_file.info("##### pairs_num {}".format(pairs_num))

    item_id = 1
    item_id_to_dic = {}
    item_name_to_id = {}
    item_id_before_to_now = {}
    for before_item_id in range(1, before_item_num + 1):
        if before_item_counts[before_item_id] != 0:
            item_id_before_to_now[before_item_id] = item_id
            item_id_to_dic[item_id] = before_item_id_to_dic[before_item_id]
            item_name_to_id[before_item_id_to_name[before_item_id]] = item_id
            item_id += 1
    item_num = len(item_id_before_to_now)
    Log_file.info("##### items after clearing {}, {}, {} #####".format(item_num, len(item_id_before_to_now), len(item_id_to_dic)))

    users_train = {}
    users_valid = {}
    users_test = {}
    users_history_for_valid = {}
    users_history_for_test = {}
    user_id = 0
    ee=pd.read_csv('/dataset/fre.csv')
    ee.columns = ['id', 'fre']
    ee.drop(0,inplace=True)
    e=ee.sort_values('fre',ascending=False)
    eee=e[['id']]
    eee=list(eee['id'])
    num=10
    #l = [0]
    #for k in range(num):
    #    l.append(int(79707*(1+k)/num))
    l = [0, 41, 127, 276, 517, 889, 1467, 2413, 4070, 7618, 79707, 79707]
    i=num_k###################################
    valid_id=0

    target = []
    distance_llm = np.array([])
    distance_rs_cos = np.array([])
    distance_rs_dot = np.array([])
    id_emb = torch.load('./tensor/order_pure_id_128.pt')
    #content_emb = torch.load('./tensor/order_pure_content_128.pt')
    a = torch.load('/dataset/bce_text/id_plus_mo-2stage/tensor/llm2vec_0.pt')
    b = torch.load('/dataset/bce_text/id_plus_mo-2stage/tensor/llm2vec_1.pt')
    c = torch.load('/dataset/bce_text/id_plus_mo-2stage/tensor/llm2vec_2.pt')
    d = torch.load('/dataset/bce_text/id_plus_mo-2stage/tensor/llm2vec_3.pt')
    content_emb = torch.cat((a,b,c,d),0)
    
    ind=[eee[j] for j in range(l[i],l[i+1])]
    v=torch.linalg.svdvals(torch.tensor(content_emb[ind,:]))
    print(torch.sum(v)/torch.max(v))
    print(v)

    id_emb = id_emb.detach().numpy()
    content_emb = content_emb.detach().numpy()

    #id_emb = id_emb.detach().numpy()
    #content_emb = content_emb.detach().numpy()

    for user_name, item_seqs in user_seq_dic.items():
        user_seq = [item_id_before_to_now[i] for i in item_seqs]
        train = user_seq[:-2] ####################################################################
        users_train[user_id] = train

        if (eee.index(user_seq[-2])<l[i+1] and eee.index(user_seq[-2])>=l[i] and i<10) or (i==10):
            valid = user_seq[-(max_seq_len+2):-1]
            users_valid[valid_id] = valid
            users_history_for_valid[valid_id] = torch.LongTensor(np.array(train))
            valid_id+=1
            target.append(user_seq[-2])
        test = user_seq[-(max_seq_len+1):]
        users_test[user_id] = test
        users_history_for_test[user_id] = torch.LongTensor(np.array(user_seq[:-1]))
        user_id += 1
    Log_file.info("##### user seqs after clearing {}, {}, {}, {}, {}#####".
                  format(seq_num, len(user_seq_dic), len(users_train), len(users_valid), len(users_test)))

    path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_pure_llm_cos_'+str(num_k)+'.dat'
    if os.path.exists(path):
        distance_llm = np.fromfile(path, dtype=np.half)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_pure_rs_cos_'+str(num_k)+'.dat'
        distance_rs_cos = np.fromfile(path, dtype=np.half)
        #path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_pure_rs_dot_'+str(num_k)+'.dat'
        #distance_rs_dot = np.fromfile(path, dtype=np.half)
    else:
        for k in range(valid_id):
            seq=users_history_for_valid[k].tolist()
            seq.append(target[k])
            neg_items = []
            seq = [eee.index(j) for j in seq]
            if i==10:
                left=0
                right=79706
            else:
                left=l[i]
                right=l[i+1]-1
            for j in range(1):
                sam_neg = random.randint(left,right)
                while sam_neg in seq:
                    sam_neg = random.randint(left,right)
                neg_items.append(eee[sam_neg])
            neg_items=torch.LongTensor(np.array(neg_items))
            
            term_1 = content_emb[users_history_for_valid[k]]
            term_2 = content_emb[target[k]].reshape(1,-1)
            dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            distance_llm = np.concatenate((distance_llm,dis),0)
            #a=pairwise_distances(term_1,metric="cosine")
            #uptri_idx = np.triu_indices_from(a, k=1)
            #dis=a[(uptri_idx)]
            #distance_llm = np.concatenate((distance_llm,dis),0)
            #print(distance_llm.shape)#############################
            term_3 = content_emb[neg_items].reshape(1,-1)
            dis = pairwise_distances(term_1, term_3, metric='cosine').reshape(term_1.shape[0])
            distance_llm = np.concatenate((distance_llm,dis),0)
            #print(term_1.shape)#######################################
            #print(term_2.shape)
            #print(term_3.shape)
            #print()
            term_1 = id_emb[users_history_for_valid[k]]
            term_2 = id_emb[target[k]].reshape(1,-1)
            dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            #distance_rs= np.concatenate((distance_rs,dis),0)
            #a=pairwise_distances(term_1,metric="cosine")
            #dis=a[(uptri_idx)]
            distance_rs_cos= np.concatenate((distance_rs_cos,dis),0)
            
            term_3 = id_emb[neg_items].reshape(1,-1)
            dis = pairwise_distances(term_1, term_3, metric='cosine').reshape(term_1.shape[0])
            distance_rs_cos = np.concatenate((distance_rs_cos,dis),0)

            '''
            term_1 = id_emb[users_history_for_valid[k]]
            term_2 = id_emb[target[k]].reshape(1,-1)
            #dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            #distance_rs= np.concatenate((distance_rs,dis),0)
            dis = np.dot(term_1,term_2.T).reshape(term_1.shape[0])
            #dis=a[(uptri_idx)]
            distance_rs_dot= np.concatenate((distance_rs_dot,dis),0)
            '''
        distance_llm = np.half(distance_llm)
        distance_rs_cos = np.half(distance_rs_cos)
        #distance_rs_dot = np.half(distance_rs_dot)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_pure_llm_cos_'+str(num_k)+'.dat'
        distance_llm.tofile(path)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_pure_rs_cos_'+str(num_k)+'.dat'
        distance_rs_cos.tofile(path)
        #path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_pure_rs_dot_'+str(num_k)+'.dat'
        #distance_rs_dot.tofile(path)


    #print((distance_llm.shape))
    Kendallta,p_value = kendalltau(distance_rs_cos,distance_llm)
    print(Kendallta,p_value)
    c1 = distance_rs_cos
    d1 = distance_llm
    #e=c1
    #c1=d1
    #d1=e
    length = c1.shape[0]
    left = int(length/num)
    right = int(length/num*(num-1))
    indices = np.argsort(d1)[::-1]
    #plt.hist(d1, bins=60, alpha=0.5, color='grey')  # bins参数定义了直方图的柱子数量
    top_k_indices_1 = indices[:left]
    pairs_1 = d1[top_k_indices_1]
    #plt.hist(pairs, bins=60, alpha=0.5, color='red')
    top_k_indices_2 = indices[left:2*left]
    pairs_2 = d1[top_k_indices_2]
    top_k_indices_3 = indices[2*left:3*left]
    pairs_3 = d1[top_k_indices_3]
    #plt.hist(pairs, bins=60, alpha=0.5, color='blue')
    num_bins = 80
    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    n, bins, patches = ax[0].hist(d1, bins=num_bins, edgecolor='skyblue', color='white', hatch='/', label='distance_distribution')
    n_set, bins_set, patches_set = ax[0].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_'+str(int(100/num)))
    n_set_2, bins_set_2, patches_set_2 = ax[0].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='top_'+str(1*int(100/num))+'-'+str(2*int(100/num)))
    n_set_3, bins_set_3, patches_set_3 = ax[0].hist(pairs_3, bins=bins, edgecolor='black', color='lightblue', label='top_'+str(2*int(100/num))+'-'+str(3*int(100/num)))
    ax[0].set_title('llm2vec Embedding Distance Distribution', fontsize=16)
    ax[0].set_xlabel('distance', fontsize=14)
    ax[0].set_ylabel('num', fontsize=14)
    ax[0].legend()

    pairs_1 = c1[top_k_indices_1]
    pairs_2 = c1[top_k_indices_2]
    pairs_3 = c1[top_k_indices_3]
    n, bins, patches = ax[1].hist(c1, bins=num_bins, edgecolor='orange', color='white', hatch='/', label='sasrec_distribution')
    n_set, bins_set, patches_set = ax[1].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_'+str(int(100/num)), alpha=1)
    n_set_2, bins_set_2, patches_set_2 = ax[1].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='top_'+str(1*int(100/num))+'-'+str(2*int(100/num)), alpha=0.4)
    n_set_3, bins_set_3, patches_set_3 = ax[1].hist(pairs_3, bins=bins, edgecolor='black', color='lightblue', label='top_'+str(2*int(100/num))+'-'+str(3*int(100/num)), alpha=0.4)
    ax[1].set_title('llm2vec Top '+str(3*int(100/num))+'% in SASRec', fontsize=16)
    ax[1].set_xlabel('distance', fontsize=14)
    ax[1].set_ylabel('num', fontsize=14)
    ax[1].legend()
    plt.savefig('./figure/target_llm_neg_top_distance-pure-cos-'+str(i)+'.png', dpi=600)  # 保存图为png格式
    plt.show()
    plt.clf()
    print('done task 1-1')
    
    '''
    c1 = distance_rs_dot
    d1 = distance_llm
    pairs_1 = d1[top_k_indices]
    pairs_2 = d1[btm_k_indices]
    #plt.hist(pairs, bins=60, alpha=0.5, color='blue')
    num_bins = 80
    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    n, bins, patches = ax[0].hist(d1, bins=num_bins, edgecolor='skyblue', color='white', hatch='/', label='distance_distribution')
    n_set, bins_set, patches_set = ax[0].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top'+str(int(100/num)))
    n_set_2, bins_set_2, patches_set_2 = ax[0].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_'+str(int(100/num)))
    ax[0].set_title('llm2vec Embedding Distance Distribution', fontsize=16)
    ax[0].set_xlabel('distance', fontsize=14)
    ax[0].set_ylabel('num', fontsize=14)
    ax[0].legend()

    pairs_1 = c1[top_k_indices]
    pairs_2 = c1[btm_k_indices]
    n, bins, patches = ax[1].hist(c1, bins=num_bins, edgecolor='orange', color='white', hatch='/', label='sasrec_distribution')
    n_set, bins_set, patches_set = ax[1].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_'+str(int(100/num)), alpha=1)
    n_set_2, bins_set_2, patches_set_2 = ax[1].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_'+str(int(100/num)), alpha=0.4)
    ax[1].set_title('llm2vec Top/bottom '+str(int(100/num))+'% in SASRec', fontsize=16)
    ax[1].set_xlabel('distance', fontsize=14)
    ax[1].set_ylabel('num', fontsize=14)
    ax[1].legend()
    plt.savefig('./figure/target_llm_neg_distance-pure-dot-'+str(i)+'.png', dpi=600)  # 保存图为png格式
    plt.show()
    plt.clf()
    print('done task 1-2')
    '''










    id_emb = torch.load('./tensor/order_mmd_item_id_128.pt')
    content_emb = torch.load('./tensor/order_mmd_content_128.pt')

    ind=[eee[j] for j in range(l[i],l[i+1])]
    v=torch.linalg.svdvals(torch.tensor(content_emb[ind,:]))
    print(torch.sum(v)/torch.max(v))
    print(v)
    
    id_emb = id_emb.detach().numpy()
    content_emb = content_emb.detach().numpy()
    distance_llm = np.array([])
    distance_rs_cos = np.array([])
    distance_rs_dot = np.array([])
    path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_mmd_llm_cos_'+str(num_k)+'.dat'
    if os.path.exists(path):
        distance_llm = np.fromfile(path, dtype=np.half)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_mmd_rs_cos_'+str(num_k)+'.dat'
        distance_rs_cos = np.fromfile(path, dtype=np.half)
        #path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_mmd_rs_dot_'+str(num_k)+'.dat'
        #distance_rs_dot = np.fromfile(path, dtype=np.half)
    else:
        for k in range(valid_id):
            seq=users_history_for_valid[k].tolist()
            seq.append(target[k])
            neg_items = []
            seq = [eee.index(j) for j in seq]
            if i==10:
                left=0
                right=79706
            else:
                left=l[i]
                right=l[i+1]-1
            for j in range(1):
                sam_neg = random.randint(left,right)
                while sam_neg in seq:
                    sam_neg = random.randint(left,right)
                neg_items.append(eee[sam_neg])
            neg_items=torch.LongTensor(np.array(neg_items))

            term_1 = content_emb[users_history_for_valid[k]]
            term_2 = content_emb[target[k]].reshape(1,-1)
            dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            distance_llm = np.concatenate((distance_llm,dis),0)
            #a=pairwise_distances(term_1,metric="cosine")
            #uptri_idx = np.triu_indices_from(a, k=1)
            #dis=a[(uptri_idx)]
            #distance_llm = np.concatenate((distance_llm,dis),0)
            
            term_3 = content_emb[neg_items].reshape(1,-1)
            dis = pairwise_distances(term_1, term_3, metric='cosine').reshape(term_1.shape[0])
            distance_llm = np.concatenate((distance_llm,dis),0)

            term_1 = id_emb[users_history_for_valid[k]]
            term_2 = id_emb[target[k]].reshape(1,-1)
            dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            #distance_rs= np.concatenate((distance_rs,dis),0)
            #a=pairwise_distances(term_1,metric="cosine")
            #dis=a[(uptri_idx)]
            distance_rs_cos= np.concatenate((distance_rs_cos,dis),0)
            
            term_3 = id_emb[neg_items].reshape(1,-1)
            dis = pairwise_distances(term_1, term_3, metric='cosine').reshape(term_1.shape[0])
            distance_rs_cos = np.concatenate((distance_rs_cos,dis),0)
            '''
            term_1 = id_emb[users_history_for_valid[k]]
            term_2 = id_emb[target[k]].reshape(1,-1)
            #dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            #distance_rs= np.concatenate((distance_rs,dis),0)
            dis = np.dot(term_1,term_2.T).reshape(term_1.shape[0])
            #dis=a[(uptri_idx)]
            distance_rs_dot= np.concatenate((distance_rs_dot,dis),0)
            '''
        distance_llm = np.half(distance_llm)
        distance_rs_cos = np.half(distance_rs_cos)
        #distance_rs_dot = np.half(distance_rs_dot)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_mmd_llm_cos_'+str(num_k)+'.dat'
        distance_llm.tofile(path)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_mmd_rs_cos_'+str(num_k)+'.dat'
        distance_rs_cos.tofile(path)
        #path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_mmd_rs_dot_'+str(num_k)+'.dat'
        #distance_rs_dot.tofile(path)


    #print((distance_llm.shape))
    Kendallta,p_value = kendalltau(distance_rs_cos,distance_llm)
    print(Kendallta,p_value)
    c1 = distance_rs_cos
    d1 = distance_llm
    #e=c1
    #c1=d1
    #d1=e
    length = c1.shape[0]
    left = int(length/num)
    right = int(length/num*(num-1))
    indices = np.argsort(d1)[::-1]
    #plt.hist(d1, bins=60, alpha=0.5, color='grey')  # bins参数定义了直方图的柱子数量
    top_k_indices_1 = indices[:left]
    pairs_1 = d1[top_k_indices_1]
    #plt.hist(pairs, bins=60, alpha=0.5, color='red')
    top_k_indices_2 = indices[left:2*left]
    pairs_2 = d1[top_k_indices_2]
    top_k_indices_3 = indices[2*left:3*left]
    pairs_3 = d1[top_k_indices_3]
    #plt.hist(pairs, bins=60, alpha=0.5, color='blue')
    num_bins = 80
    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    n, bins, patches = ax[0].hist(d1, bins=num_bins, edgecolor='skyblue', color='white', hatch='/', label='distance_distribution')
    n_set, bins_set, patches_set = ax[0].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_'+str(int(100/num)))
    n_set_2, bins_set_2, patches_set_2 = ax[0].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='top_'+str(1*int(100/num))+'-'+str(2*int(100/num)))
    n_set_3, bins_set_3, patches_set_3 = ax[0].hist(pairs_3, bins=bins, edgecolor='black', color='lightblue', label='top_'+str(2*int(100/num))+'-'+str(3*int(100/num)))
    ax[0].set_title('llm2vec Embedding Distance Distribution', fontsize=16)
    ax[0].set_xlabel('distance', fontsize=14)
    ax[0].set_ylabel('num', fontsize=14)
    ax[0].legend()

    pairs_1 = c1[top_k_indices_1]
    pairs_2 = c1[top_k_indices_2]
    pairs_3 = c1[top_k_indices_3]
    n, bins, patches = ax[1].hist(c1, bins=num_bins, edgecolor='orange', color='white', hatch='/', label='sasrec_distribution')
    n_set, bins_set, patches_set = ax[1].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_'+str(int(100/num)), alpha=1)
    n_set_2, bins_set_2, patches_set_2 = ax[1].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='top_'+str(1*int(100/num))+'-'+str(2*int(100/num)), alpha=0.4)
    n_set_3, bins_set_3, patches_set_3 = ax[1].hist(pairs_3, bins=bins, edgecolor='black', color='lightblue', label='top_'+str(2*int(100/num))+'-'+str(3*int(100/num)), alpha=0.4)
    ax[1].set_title('llm2vec Top '+str(3*int(100/num))+'% in SASRec', fontsize=16)
    ax[1].set_xlabel('distance', fontsize=14)
    ax[1].set_ylabel('num', fontsize=14)
    ax[1].legend()
    
    plt.savefig('./figure/target_llm_neg_top_distance-mmd-cos-'+str(i)+'.png', dpi=600)  # 保存图为png格式
    plt.show()
    plt.clf()
    print('done task 3-1')

    '''
    c1 = distance_rs_dot
    d1 = distance_llm
    pairs_1 = d1[top_k_indices]
    pairs_2 = d1[btm_k_indices]
    #plt.hist(pairs, bins=60, alpha=0.5, color='blue')
    num_bins = 80
    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    n, bins, patches = ax[0].hist(d1, bins=num_bins, edgecolor='skyblue', color='white', hatch='/', label='distance_distribution')
    n_set, bins_set, patches_set = ax[0].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_'+str(int(100/num)))
    n_set_2, bins_set_2, patches_set_2 = ax[0].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_'+str(int(100/num)))
    ax[0].set_title('llm2vec Embedding Distance Distribution', fontsize=16)
    ax[0].set_xlabel('distance', fontsize=14)
    ax[0].set_ylabel('num', fontsize=14)
    ax[0].legend()

    pairs_1 = c1[top_k_indices]
    pairs_2 = c1[btm_k_indices]
    n, bins, patches = ax[1].hist(c1, bins=num_bins, edgecolor='orange', color='white', hatch='/', label='sasrec_distribution')
    n_set, bins_set, patches_set = ax[1].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_'+str(int(100/num)), alpha=1)
    n_set_2, bins_set_2, patches_set_2 = ax[1].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_'+str(int(100/num)), alpha=0.4)
    ax[1].set_title('llm2vec Top/bottom '+str(int(100/num))+'% in SASRec', fontsize=16)
    ax[1].set_xlabel('distance', fontsize=14)
    ax[1].set_ylabel('num', fontsize=14)
    ax[1].legend()
    plt.savefig('./figure/target_llm_neg_distance-mmd-dot-'+str(i)+'.png', dpi=600)  # 保存图为png格式
    plt.show()
    plt.clf()
    print('done task 3-2')
    '''
    return item_num, item_id_to_dic, users_train, users_valid, users_test, \
           users_history_for_valid, users_history_for_test, item_name_to_id


def read_behaviors_fre_target_llm_neg(behaviors_path, before_item_id_to_dic, before_item_name_to_id, before_item_id_to_name, max_seq_len, min_seq_len, Log_file,num_k):
    Log_file.info("##### news number {} {} (before clearing)#####".format(len(before_item_id_to_dic), len(before_item_name_to_id)))
    Log_file.info("##### min seq len {}, max seq len {}#####".format(min_seq_len, max_seq_len))
    before_item_num = len(before_item_name_to_id)
    before_item_counts = [0] * (before_item_num + 1)
    user_seq_dic = {}
    seq_num = 0
    pairs_num = 0
    Log_file.info('rebuild user seqs...')
    with open(behaviors_path, "r") as f:
        for line in f:
            splited = line.strip('\n').split('\t')
            user_name = splited[0]
            history_item_name = splited[1].split(' ')
            if len(history_item_name) < min_seq_len:
                continue
            history_item_name = history_item_name[-(max_seq_len+3):]
            item_ids_sub_seq = [before_item_name_to_id[i] for i in history_item_name]
            user_seq_dic[user_name] = item_ids_sub_seq
            for item_id in item_ids_sub_seq:
                before_item_counts[item_id] += 1
                pairs_num += 1
            seq_num += 1
    Log_file.info("##### pairs_num {}".format(pairs_num))

    item_id = 1
    item_id_to_dic = {}
    item_name_to_id = {}
    item_id_before_to_now = {}
    for before_item_id in range(1, before_item_num + 1):
        if before_item_counts[before_item_id] != 0:
            item_id_before_to_now[before_item_id] = item_id
            item_id_to_dic[item_id] = before_item_id_to_dic[before_item_id]
            item_name_to_id[before_item_id_to_name[before_item_id]] = item_id
            item_id += 1
    item_num = len(item_id_before_to_now)
    Log_file.info("##### items after clearing {}, {}, {} #####".format(item_num, len(item_id_before_to_now), len(item_id_to_dic)))

    users_train = {}
    users_valid = {}
    users_test = {}
    users_history_for_valid = {}
    users_history_for_test = {}
    user_id = 0
    ee=pd.read_csv('/dataset/fre.csv')
    ee.columns = ['id', 'fre']
    ee.drop(0,inplace=True)
    e=ee.sort_values('fre',ascending=False)
    eee=e[['id']]
    eee=list(eee['id'])
    num=10
    #l = [0]
    #for k in range(num):
    #    l.append(int(79707*(1+k)/num))
    l = [0, 41, 127, 276, 517, 889, 1467, 2413, 4070, 7618, 79707, 79707]
    i=num_k###################################
    valid_id=0

    target = []
    distance_llm = np.array([])
    distance_rs_cos = np.array([])
    distance_rs_dot = np.array([])
    id_emb = torch.load('./tensor/order_pure_id_128.pt')
    #content_emb = torch.load('./tensor/order_pure_content_128.pt')
    a = torch.load('/dataset/bce_text/id_plus_mo-2stage/tensor/llm2vec_0.pt')
    b = torch.load('/dataset/bce_text/id_plus_mo-2stage/tensor/llm2vec_1.pt')
    c = torch.load('/dataset/bce_text/id_plus_mo-2stage/tensor/llm2vec_2.pt')
    d = torch.load('/dataset/bce_text/id_plus_mo-2stage/tensor/llm2vec_3.pt')
    content_emb = torch.cat((a,b,c,d),0)

    id_emb = id_emb.detach().numpy()
    content_emb = content_emb.detach().numpy()

    #id_emb = id_emb.detach().numpy()
    #content_emb = content_emb.detach().numpy()

    for user_name, item_seqs in user_seq_dic.items():
        user_seq = [item_id_before_to_now[i] for i in item_seqs]
        train = user_seq[:-2] ####################################################################
        users_train[user_id] = train

        if (eee.index(user_seq[-2])<l[i+1] and eee.index(user_seq[-2])>=l[i] and i<10) or (i==10):
            valid = user_seq[-(max_seq_len+2):-1]
            users_valid[valid_id] = valid
            users_history_for_valid[valid_id] = torch.LongTensor(np.array(train))
            valid_id+=1
            target.append(user_seq[-2])
        test = user_seq[-(max_seq_len+1):]
        users_test[user_id] = test
        users_history_for_test[user_id] = torch.LongTensor(np.array(user_seq[:-1]))
        user_id += 1
    Log_file.info("##### user seqs after clearing {}, {}, {}, {}, {}#####".
                  format(seq_num, len(user_seq_dic), len(users_train), len(users_valid), len(users_test)))

    path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_pure_llm_cos_'+str(num_k)+'.dat'
    if os.path.exists(path):
        distance_llm = np.fromfile(path, dtype=np.half)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_pure_rs_cos_'+str(num_k)+'.dat'
        distance_rs_cos = np.fromfile(path, dtype=np.half)
        #path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_pure_rs_dot_'+str(num_k)+'.dat'
        #distance_rs_dot = np.fromfile(path, dtype=np.half)


    #print((distance_llm.shape))
    #Kendallta,p_value = kendalltau(distance_rs_cos,distance_llm)
    #print(Kendallta,p_value)
    c1 = distance_rs_cos
    d1 = distance_llm
    length = c1.shape[0]
    left = int(length/num)
    right = int(length/num*(num-1))
    indices = np.argsort(d1)[::-1]
    #plt.hist(d1, bins=60, alpha=0.5, color='grey')  # bins参数定义了直方图的柱子数量
    top_k_indices = indices[:left]
    pairs_1 = d1[top_k_indices]
    #plt.hist(pairs, bins=60, alpha=0.5, color='red')
    btm_k_indices = indices[right:]
    pairs_2 = d1[btm_k_indices]
    #plt.hist(pairs, bins=60, alpha=0.5, color='blue')
    num_bins = 80
    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    n, bins, patches = ax[0].hist(d1, bins=num_bins, edgecolor='skyblue', color='white', hatch='/', label='All item pairs')
    n_set, bins_set, patches_set = ax[0].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='Top '+str(int(100/num))+'%')
    n_set_2, bins_set_2, patches_set_2 = ax[0].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='Bottom '+str(int(100/num))+'%')
    ax[0].set_title('(a) LLM2Vec Embedding Distance Distribution', font='Times New Roman', fontsize=28)
    ax[0].set_xlabel('Distance', font='Times New Roman', fontsize=24)
    ax[0].set_ylabel('Count', font='Times New Roman', fontsize=24)
    ax[0].legend(prop={'family': 'Times New Roman', 'size':22})

    pairs_1 = c1[top_k_indices]
    pairs_2 = c1[btm_k_indices]
    n, bins, patches = ax[1].hist(c1, bins=num_bins, edgecolor='orange', color='white', hatch='/', label='All item pairs')
    n_set, bins_set, patches_set = ax[1].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='Top '+str(int(100/num))+'%', alpha=0.4)
    n_set_2, bins_set_2, patches_set_2 = ax[1].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='Bottom '+str(int(100/num))+'%', alpha=0.4)
    ax[1].set_title('(b) LLM2Vec Top/bottom '+str(int(100/num))+'% in SASRec', font='Times New Roman', fontsize=28)
    ax[1].set_xlabel('Distance', font='Times New Roman', fontsize=24)
    ax[1].set_ylabel('Count', font='Times New Roman', fontsize=24)
    ax[1].legend(prop={'family': 'Times New Roman', 'size':22})
    #plt.savefig('./figure/llm2vec/target_llm_neg_distance-pure-cos-'+str(i)+'.png', dpi=600)  # 保存图为png格式
    plt.savefig('./figure/llm2vec/target_llm_neg_distance-pure-cos-'+str(i)+'.pdf', dpi=600, bbox_inches='tight')  # 保存图为png格式
    plt.show()
    #plt.clf()
    print('done task 1-1')
    '''
    c1 = distance_rs_dot
    d1 = distance_llm
    pairs_1 = d1[top_k_indices]
    pairs_2 = d1[btm_k_indices]
    #plt.hist(pairs, bins=60, alpha=0.5, color='blue')
    num_bins = 80
    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    n, bins, patches = ax[0].hist(d1, bins=num_bins, edgecolor='skyblue', color='white', hatch='/', label='distance_distribution')
    n_set, bins_set, patches_set = ax[0].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top'+str(int(100/num)))
    n_set_2, bins_set_2, patches_set_2 = ax[0].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_'+str(int(100/num)))
    ax[0].set_title('llm2vec Embedding Distance Distribution', fontsize=16)
    ax[0].set_xlabel('distance', fontsize=14)
    ax[0].set_ylabel('num', fontsize=14)
    ax[0].legend()

    pairs_1 = c1[top_k_indices]
    pairs_2 = c1[btm_k_indices]
    n, bins, patches = ax[1].hist(c1, bins=num_bins, edgecolor='orange', color='white', hatch='/', label='sasrec_distribution')
    n_set, bins_set, patches_set = ax[1].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_'+str(int(100/num)), alpha=1)
    n_set_2, bins_set_2, patches_set_2 = ax[1].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_'+str(int(100/num)), alpha=0.4)
    ax[1].set_title('llm2vec Top/bottom '+str(int(100/num))+'% in SASRec', fontsize=16)
    ax[1].set_xlabel('distance', fontsize=14)
    ax[1].set_ylabel('num', fontsize=14)
    ax[1].legend()
    plt.savefig('./figure/llm2vec/target_llm_neg_distance-pure-dot-'+str(i)+'.png', dpi=600)  # 保存图为png格式
    plt.show()
    plt.clf()
    print('done task 1-2')
    '''










    id_emb = torch.load('./tensor/order_mmd_item_id_128.pt')
    content_emb = torch.load('./tensor/order_mmd_content_128.pt')
    id_emb = id_emb.detach().numpy()
    content_emb = content_emb.detach().numpy()
    distance_llm = np.array([])
    distance_rs_cos = np.array([])
    distance_rs_dot = np.array([])
    path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_mmd_llm_cos_'+str(num_k)+'.dat'
    if os.path.exists(path):
        distance_llm = np.fromfile(path, dtype=np.half)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_mmd_rs_cos_'+str(num_k)+'.dat'
        distance_rs_cos = np.fromfile(path, dtype=np.half)
        #path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_neg_mmd_rs_dot_'+str(num_k)+'.dat'
        #distance_rs_dot = np.fromfile(path, dtype=np.half)
    

    #print((distance_llm.shape))
    #Kendallta,p_value = kendalltau(distance_rs_cos,distance_llm)
    #print(Kendallta,p_value)
    c1 = distance_rs_cos
    d1 = distance_llm
    length = c1.shape[0]
    left = int(length/num)
    right = int(length/num*(num-1))
    indices = np.argsort(d1)[::-1]
    #plt.hist(d1, bins=60, alpha=0.5, color='grey')  # bins参数定义了直方图的柱子数量
    top_k_indices = indices[:left]
    pairs_1 = d1[top_k_indices]
    #plt.hist(pairs, bins=60, alpha=0.5, color='red')
    btm_k_indices = indices[right:]
    pairs_2 = d1[btm_k_indices]
    #plt.hist(pairs, bins=60, alpha=0.5, color='blue')
    num_bins = 80
    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    n, bins, patches = ax[0].hist(d1, bins=num_bins, edgecolor='skyblue', color='white', hatch='/', label='distance_distribution')
    n_set, bins_set, patches_set = ax[0].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_'+str(int(100/num)))
    n_set_2, bins_set_2, patches_set_2 = ax[0].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_'+str(int(100/num)))
    ax[0].set_title('llm2vec Embedding Distance Distribution', fontsize=16)
    ax[0].set_xlabel('distance', fontsize=14)
    ax[0].set_ylabel('num', fontsize=14)
    ax[0].legend()

    pairs_1 = c1[top_k_indices]
    pairs_2 = c1[btm_k_indices]
    n, bins, patches = ax[1].hist(c1, bins=num_bins, edgecolor='orange', color='white', hatch='/', label='sasrec_distribution')
    n_set, bins_set, patches_set = ax[1].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_'+str(int(100/num)), alpha=1)
    n_set_2, bins_set_2, patches_set_2 = ax[1].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_'+str(int(100/num)), alpha=0.4)
    ax[1].set_title('llm2vec Top/bottom '+str(int(100/num))+'% in SASRec', fontsize=16)
    ax[1].set_xlabel('distance', fontsize=14)
    ax[1].set_ylabel('num', fontsize=14)
    ax[1].legend()
    plt.savefig('./figure/llm2vec/target_llm_neg_distance-mmd-cos-'+str(i)+'.png', dpi=600)  # 保存图为png格式
    plt.show()
    plt.clf()
    print('done task 3-1')
    '''
    c1 = distance_rs_dot
    d1 = distance_llm
    pairs_1 = d1[top_k_indices]
    pairs_2 = d1[btm_k_indices]
    #plt.hist(pairs, bins=60, alpha=0.5, color='blue')
    num_bins = 80
    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    n, bins, patches = ax[0].hist(d1, bins=num_bins, edgecolor='skyblue', color='white', hatch='/', label='distance_distribution')
    n_set, bins_set, patches_set = ax[0].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_'+str(int(100/num)))
    n_set_2, bins_set_2, patches_set_2 = ax[0].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_'+str(int(100/num)))
    ax[0].set_title('llm2vec Embedding Distance Distribution', fontsize=16)
    ax[0].set_xlabel('distance', fontsize=14)
    ax[0].set_ylabel('num', fontsize=14)
    ax[0].legend()

    pairs_1 = c1[top_k_indices]
    pairs_2 = c1[btm_k_indices]
    n, bins, patches = ax[1].hist(c1, bins=num_bins, edgecolor='orange', color='white', hatch='/', label='sasrec_distribution')
    n_set, bins_set, patches_set = ax[1].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_'+str(int(100/num)), alpha=1)
    n_set_2, bins_set_2, patches_set_2 = ax[1].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_'+str(int(100/num)), alpha=0.4)
    ax[1].set_title('llm2vec Top/bottom '+str(int(100/num))+'% in SASRec', fontsize=16)
    ax[1].set_xlabel('distance', fontsize=14)
    ax[1].set_ylabel('num', fontsize=14)
    ax[1].legend()
    plt.savefig('./figure/llm2vec/target_llm_neg_distance-mmd-dot-'+str(i)+'.png', dpi=600)  # 保存图为png格式
    plt.show()
    plt.clf()
    print('done task 3-2')
    '''
    return item_num, item_id_to_dic, users_train, users_valid, users_test, \
           users_history_for_valid, users_history_for_test, item_name_to_id

def read_behaviors_fre_target_llm_top(behaviors_path, before_item_id_to_dic, before_item_name_to_id, before_item_id_to_name, max_seq_len, min_seq_len, Log_file,num_k):
    Log_file.info("##### news number {} {} (before clearing)#####".format(len(before_item_id_to_dic), len(before_item_name_to_id)))
    Log_file.info("##### min seq len {}, max seq len {}#####".format(min_seq_len, max_seq_len))
    before_item_num = len(before_item_name_to_id)
    before_item_counts = [0] * (before_item_num + 1)
    user_seq_dic = {}
    seq_num = 0
    pairs_num = 0
    Log_file.info('rebuild user seqs...')
    with open(behaviors_path, "r") as f:
        for line in f:
            splited = line.strip('\n').split('\t')
            user_name = splited[0]
            history_item_name = splited[1].split(' ')
            if len(history_item_name) < min_seq_len:
                continue
            history_item_name = history_item_name[-(max_seq_len+3):]
            item_ids_sub_seq = [before_item_name_to_id[i] for i in history_item_name]
            user_seq_dic[user_name] = item_ids_sub_seq
            for item_id in item_ids_sub_seq:
                before_item_counts[item_id] += 1
                pairs_num += 1
            seq_num += 1
    Log_file.info("##### pairs_num {}".format(pairs_num))

    item_id = 1
    item_id_to_dic = {}
    item_name_to_id = {}
    item_id_before_to_now = {}
    for before_item_id in range(1, before_item_num + 1):
        if before_item_counts[before_item_id] != 0:
            item_id_before_to_now[before_item_id] = item_id
            item_id_to_dic[item_id] = before_item_id_to_dic[before_item_id]
            item_name_to_id[before_item_id_to_name[before_item_id]] = item_id
            item_id += 1
    item_num = len(item_id_before_to_now)
    Log_file.info("##### items after clearing {}, {}, {} #####".format(item_num, len(item_id_before_to_now), len(item_id_to_dic)))

    users_train = {}
    users_valid = {}
    users_test = {}
    users_history_for_valid = {}
    users_history_for_test = {}
    user_id = 0
    ee=pd.read_csv('/dataset/fre.csv')
    ee.columns = ['id', 'fre']
    ee.drop(0,inplace=True)
    e=ee.sort_values('fre',ascending=False)
    eee=e[['id']]
    eee=list(eee['id'])
    num=20
    #l = [0]
    #for k in range(num):
    #    l.append(int(79707*(1+k)/num))
    l = [0, 41, 127, 276, 517, 889, 1467, 2413, 4070, 7618, 79707, 79707]
    i=num_k###################################
    valid_id=0

    target = []
    distance_llm = np.array([])
    distance_rs_cos = np.array([])
    distance_rs_dot = np.array([])
    id_emb = torch.load('./tensor/order_pure_id_128.pt')
    #content_emb = torch.load('./tensor/order_pure_content_128.pt')
    a = torch.load('/dataset/bce_text/id_plus_mo-2stage/tensor/llm2vec_0.pt')
    b = torch.load('/dataset/bce_text/id_plus_mo-2stage/tensor/llm2vec_1.pt')
    c = torch.load('/dataset/bce_text/id_plus_mo-2stage/tensor/llm2vec_2.pt')
    d = torch.load('/dataset/bce_text/id_plus_mo-2stage/tensor/llm2vec_3.pt')
    content_emb = torch.cat((a,b,c,d),0)

    id_emb = id_emb.detach().numpy()
    content_emb = content_emb.detach().numpy()

    #id_emb = id_emb.detach().numpy()
    #content_emb = content_emb.detach().numpy()

    for user_name, item_seqs in user_seq_dic.items():
        user_seq = [item_id_before_to_now[i] for i in item_seqs]
        train = user_seq[:-2] ####################################################################
        users_train[user_id] = train

        if (eee.index(user_seq[-2])<l[i+1] and eee.index(user_seq[-2])>=l[i] and i<10) or (i==10):
            valid = user_seq[-(max_seq_len+2):-1]
            users_valid[valid_id] = valid
            users_history_for_valid[valid_id] = torch.LongTensor(np.array(train))
            valid_id+=1
            target.append(user_seq[-2])
        test = user_seq[-(max_seq_len+1):]
        users_test[user_id] = test
        users_history_for_test[user_id] = torch.LongTensor(np.array(user_seq[:-1]))
        user_id += 1
    Log_file.info("##### user seqs after clearing {}, {}, {}, {}, {}#####".
                  format(seq_num, len(user_seq_dic), len(users_train), len(users_valid), len(users_test)))

    path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_pure_llm_cos_'+str(num_k)+'.dat'
    if os.path.exists(path):
        distance_llm = np.fromfile(path, dtype=np.half)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_pure_rs_cos_'+str(num_k)+'.dat'
        distance_rs_cos = np.fromfile(path, dtype=np.half)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_pure_rs_dot_'+str(num_k)+'.dat'
        distance_rs_dot = np.fromfile(path, dtype=np.half)
    else:
        for k in range(valid_id):
            term_1 = content_emb[users_history_for_valid[k]]
            term_2 = content_emb[target[k]].reshape(1,-1)
            dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            distance_llm = np.concatenate((distance_llm,dis),0)
            #a=pairwise_distances(term_1,metric="cosine")
            #uptri_idx = np.triu_indices_from(a, k=1)
            #dis=a[(uptri_idx)]
            #distance_llm = np.concatenate((distance_llm,dis),0)

            term_1 = id_emb[users_history_for_valid[k]]
            term_2 = id_emb[target[k]].reshape(1,-1)
            dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            #distance_rs= np.concatenate((distance_rs,dis),0)
            #a=pairwise_distances(term_1,metric="cosine")
            #dis=a[(uptri_idx)]
            distance_rs_cos= np.concatenate((distance_rs_cos,dis),0)

            term_1 = id_emb[users_history_for_valid[k]]
            term_2 = id_emb[target[k]].reshape(1,-1)
            #dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            #distance_rs= np.concatenate((distance_rs,dis),0)
            dis = np.dot(term_1,term_2.T).reshape(term_1.shape[0])
            #dis=a[(uptri_idx)]
            distance_rs_dot= np.concatenate((distance_rs_dot,dis),0)
        distance_llm = np.half(distance_llm)
        distance_rs_cos = np.half(distance_rs_cos)
        distance_rs_dot = np.half(distance_rs_dot)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_pure_llm_cos_'+str(num_k)+'.dat'
        distance_llm.tofile(path)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_pure_rs_cos_'+str(num_k)+'.dat'
        distance_rs_cos.tofile(path)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_pure_rs_dot_'+str(num_k)+'.dat'
        distance_rs_dot.tofile(path)


    #print((distance_llm.shape))
    #Kendallta,p_value = kendalltau(distance_rs_cos,distance_llm)
    #print(Kendallta,p_value)
    c1 = distance_rs_cos
    d1 = distance_llm
    #e=c1
    #c1=d1
    #d1=e
    length = c1.shape[0]
    left = int(length/num)
    right = int(length/num*(num-1))
    indices = np.argsort(d1)[::-1]
    #plt.hist(d1, bins=60, alpha=0.5, color='grey')  # bins参数定义了直方图的柱子数量
    top_k_indices_1 = indices[:left]
    pairs_1 = d1[top_k_indices_1]
    #plt.hist(pairs, bins=60, alpha=0.5, color='red')
    top_k_indices_2 = indices[left:2*left]
    pairs_2 = d1[top_k_indices_2]
    top_k_indices_3 = indices[2*left:3*left]
    pairs_3 = d1[top_k_indices_3]
    #plt.hist(pairs, bins=60, alpha=0.5, color='blue')
    num_bins = 80
    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    n, bins, patches = ax[0].hist(d1, bins=num_bins, edgecolor='skyblue', color='white', hatch='/', label='distance_distribution')
    n_set, bins_set, patches_set = ax[0].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_'+str(int(100/num)))
    n_set_2, bins_set_2, patches_set_2 = ax[0].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='top_'+str(2*int(100/num))+'-'+str(3*int(100/num)))
    n_set_3, bins_set_3, patches_set_3 = ax[0].hist(pairs_3, bins=bins, edgecolor='black', color='lightblue', label='top_'+str(4*int(100/num))+'-'+str(5*int(100/num)))
    ax[0].set_title('llm2vec Embedding Distance Distribution', fontsize=16)
    ax[0].set_xlabel('distance', fontsize=14)
    ax[0].set_ylabel('num', fontsize=14)
    ax[0].legend()

    pairs_1 = c1[top_k_indices_1]
    pairs_2 = c1[top_k_indices_2]
    pairs_3 = c1[top_k_indices_3]
    n, bins, patches = ax[1].hist(c1, bins=num_bins, edgecolor='orange', color='white', hatch='/', label='sasrec_distribution')
    n_set, bins_set, patches_set = ax[1].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_'+str(int(100/num)), alpha=1)
    n_set_2, bins_set_2, patches_set_2 = ax[1].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='top_'+str(2*int(100/num))+'-'+str(3*int(100/num)), alpha=0.4)
    n_set_3, bins_set_3, patches_set_3 = ax[1].hist(pairs_3, bins=bins, edgecolor='black', color='lightblue', label='top_'+str(4*int(100/num))+'-'+str(5*int(100/num)), alpha=0.4)
    ax[1].set_title('llm2vec Top '+str(5*int(100/num))+'% in SASRec', fontsize=16)
    ax[1].set_xlabel('distance', fontsize=14)
    ax[1].set_ylabel('num', fontsize=14)
    ax[1].legend()
    plt.savefig('./figure/target_llm_top_distance-pure-cos-'+str(i)+'.png', dpi=600)  # 保存图为png格式
    plt.show()
    plt.clf()
    print('done task 1-1')
    
    '''
    c1 = distance_rs_dot
    d1 = distance_llm
    pairs_1 = d1[top_k_indices]
    pairs_2 = d1[btm_k_indices]
    #plt.hist(pairs, bins=60, alpha=0.5, color='blue')
    num_bins = 80
    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    n, bins, patches = ax[0].hist(d1, bins=num_bins, edgecolor='skyblue', color='white', hatch='/', label='distance_distribution')
    n_set, bins_set, patches_set = ax[0].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top'+str(int(100/num)))
    n_set_2, bins_set_2, patches_set_2 = ax[0].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_'+str(int(100/num)))
    ax[0].set_title('llm2vec Embedding Distance Distribution', fontsize=16)
    ax[0].set_xlabel('distance', fontsize=14)
    ax[0].set_ylabel('num', fontsize=14)
    ax[0].legend()

    pairs_1 = c1[top_k_indices]
    pairs_2 = c1[btm_k_indices]
    n, bins, patches = ax[1].hist(c1, bins=num_bins, edgecolor='orange', color='white', hatch='/', label='sasrec_distribution')
    n_set, bins_set, patches_set = ax[1].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_'+str(int(100/num)), alpha=1)
    n_set_2, bins_set_2, patches_set_2 = ax[1].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_'+str(int(100/num)), alpha=0.4)
    ax[1].set_title('llm2vec Top/bottom '+str(int(100/num))+'% in SASRec', fontsize=16)
    ax[1].set_xlabel('distance', fontsize=14)
    ax[1].set_ylabel('num', fontsize=14)
    ax[1].legend()
    plt.savefig('./figure/target_llm_distance-pure-dot-'+str(i)+'.png', dpi=600)  # 保存图为png格式
    plt.show()
    plt.clf()
    print('done task 1-2')
    '''










    id_emb = torch.load('./tensor/order_mmd_item_id_128.pt')
    content_emb = torch.load('./tensor/order_mmd_content_128.pt')
    id_emb = id_emb.detach().numpy()
    content_emb = content_emb.detach().numpy()
    distance_llm = np.array([])
    distance_rs_cos = np.array([])
    distance_rs_dot = np.array([])
    path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_mmd_llm_cos_'+str(num_k)+'.dat'
    if os.path.exists(path):
        distance_llm = np.fromfile(path, dtype=np.half)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_mmd_rs_cos_'+str(num_k)+'.dat'
        distance_rs_cos = np.fromfile(path, dtype=np.half)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_mmd_rs_dot_'+str(num_k)+'.dat'
        distance_rs_dot = np.fromfile(path, dtype=np.half)
    else:
        for k in range(valid_id):
            term_1 = content_emb[users_history_for_valid[k]]
            term_2 = content_emb[target[k]].reshape(1,-1)
            dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            distance_llm = np.concatenate((distance_llm,dis),0)
            #a=pairwise_distances(term_1,metric="cosine")
            #uptri_idx = np.triu_indices_from(a, k=1)
            #dis=a[(uptri_idx)]
            #distance_llm = np.concatenate((distance_llm,dis),0)

            term_1 = id_emb[users_history_for_valid[k]]
            term_2 = id_emb[target[k]].reshape(1,-1)
            dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            #distance_rs= np.concatenate((distance_rs,dis),0)
            #a=pairwise_distances(term_1,metric="cosine")
            #dis=a[(uptri_idx)]
            distance_rs_cos= np.concatenate((distance_rs_cos,dis),0)

            term_1 = id_emb[users_history_for_valid[k]]
            term_2 = id_emb[target[k]].reshape(1,-1)
            #dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            #distance_rs= np.concatenate((distance_rs,dis),0)
            dis = np.dot(term_1,term_2.T).reshape(term_1.shape[0])
            #dis=a[(uptri_idx)]
            distance_rs_dot= np.concatenate((distance_rs_dot,dis),0)
        distance_llm = np.half(distance_llm)
        distance_rs_cos = np.half(distance_rs_cos)
        distance_rs_dot = np.half(distance_rs_dot)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_mmd_llm_cos_'+str(num_k)+'.dat'
        distance_llm.tofile(path)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_mmd_rs_cos_'+str(num_k)+'.dat'
        distance_rs_cos.tofile(path)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_mmd_rs_dot_'+str(num_k)+'.dat'
        distance_rs_dot.tofile(path)


    #print((distance_llm.shape))
    #Kendallta,p_value = kendalltau(distance_rs_cos,distance_llm)
    #print(Kendallta,p_value)
    c1 = distance_rs_cos
    d1 = distance_llm
    #e=c1
    #c1=d1
    #d1=e
    length = c1.shape[0]
    left = int(length/num)
    right = int(length/num*(num-1))
    indices = np.argsort(d1)[::-1]
    #plt.hist(d1, bins=60, alpha=0.5, color='grey')  # bins参数定义了直方图的柱子数量
    top_k_indices_1 = indices[:left]
    pairs_1 = d1[top_k_indices_1]
    #plt.hist(pairs, bins=60, alpha=0.5, color='red')
    top_k_indices_2 = indices[left:2*left]
    pairs_2 = d1[top_k_indices_2]
    top_k_indices_3 = indices[2*left:3*left]
    pairs_3 = d1[top_k_indices_3]
    #plt.hist(pairs, bins=60, alpha=0.5, color='blue')
    num_bins = 80
    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    n, bins, patches = ax[0].hist(d1, bins=num_bins, edgecolor='skyblue', color='white', hatch='/', label='distance_distribution')
    n_set, bins_set, patches_set = ax[0].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_'+str(int(100/num)))
    n_set_2, bins_set_2, patches_set_2 = ax[0].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='top_'+str(2*int(100/num))+'-'+str(3*int(100/num)))
    n_set_3, bins_set_3, patches_set_3 = ax[0].hist(pairs_3, bins=bins, edgecolor='black', color='lightblue', label='top_'+str(4*int(100/num))+'-'+str(5*int(100/num)))
    ax[0].set_title('llm2vec Embedding Distance Distribution', fontsize=16)
    ax[0].set_xlabel('distance', fontsize=14)
    ax[0].set_ylabel('num', fontsize=14)
    ax[0].legend()

    pairs_1 = c1[top_k_indices_1]
    pairs_2 = c1[top_k_indices_2]
    pairs_3 = c1[top_k_indices_3]
    n, bins, patches = ax[1].hist(c1, bins=num_bins, edgecolor='orange', color='white', hatch='/', label='sasrec_distribution')
    n_set, bins_set, patches_set = ax[1].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_'+str(int(100/num)), alpha=1)
    n_set_2, bins_set_2, patches_set_2 = ax[1].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='top_'+str(2*int(100/num))+'-'+str(3*int(100/num)), alpha=0.4)
    n_set_3, bins_set_3, patches_set_3 = ax[1].hist(pairs_3, bins=bins, edgecolor='black', color='lightblue', label='top_'+str(4*int(100/num))+'-'+str(5*int(100/num)), alpha=0.4)
    ax[1].set_title('llm2vec Top '+str(5*int(100/num))+'% in SASRec', fontsize=16)
    ax[1].set_xlabel('distance', fontsize=14)
    ax[1].set_ylabel('num', fontsize=14)
    ax[1].legend()
    
    plt.savefig('./figure/target_llm_top_distance-mmd-cos-'+str(i)+'.png', dpi=600)  # 保存图为png格式
    plt.show()
    plt.clf()
    print('done task 3-1')

    '''
    c1 = distance_rs_dot
    d1 = distance_llm
    pairs_1 = d1[top_k_indices]
    pairs_2 = d1[btm_k_indices]
    #plt.hist(pairs, bins=60, alpha=0.5, color='blue')
    num_bins = 80
    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    n, bins, patches = ax[0].hist(d1, bins=num_bins, edgecolor='skyblue', color='white', hatch='/', label='distance_distribution')
    n_set, bins_set, patches_set = ax[0].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_'+str(int(100/num)))
    n_set_2, bins_set_2, patches_set_2 = ax[0].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_'+str(int(100/num)))
    ax[0].set_title('llm2vec Embedding Distance Distribution', fontsize=16)
    ax[0].set_xlabel('distance', fontsize=14)
    ax[0].set_ylabel('num', fontsize=14)
    ax[0].legend()

    pairs_1 = c1[top_k_indices]
    pairs_2 = c1[btm_k_indices]
    n, bins, patches = ax[1].hist(c1, bins=num_bins, edgecolor='orange', color='white', hatch='/', label='sasrec_distribution')
    n_set, bins_set, patches_set = ax[1].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_'+str(int(100/num)), alpha=1)
    n_set_2, bins_set_2, patches_set_2 = ax[1].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_'+str(int(100/num)), alpha=0.4)
    ax[1].set_title('llm2vec Top/bottom '+str(int(100/num))+'% in SASRec', fontsize=16)
    ax[1].set_xlabel('distance', fontsize=14)
    ax[1].set_ylabel('num', fontsize=14)
    ax[1].legend()
    plt.savefig('./figure/target_llm_distance-mmd-dot-'+str(i)+'.png', dpi=600)  # 保存图为png格式
    plt.show()
    plt.clf()
    print('done task 3-2')
    '''
    return item_num, item_id_to_dic, users_train, users_valid, users_test, \
           users_history_for_valid, users_history_for_test, item_name_to_id


def read_behaviors_fre_target_llm(behaviors_path, before_item_id_to_dic, before_item_name_to_id, before_item_id_to_name, max_seq_len, min_seq_len, Log_file,num_k):
    Log_file.info("##### news number {} {} (before clearing)#####".format(len(before_item_id_to_dic), len(before_item_name_to_id)))
    Log_file.info("##### min seq len {}, max seq len {}#####".format(min_seq_len, max_seq_len))
    before_item_num = len(before_item_name_to_id)
    before_item_counts = [0] * (before_item_num + 1)
    user_seq_dic = {}
    seq_num = 0
    pairs_num = 0
    Log_file.info('rebuild user seqs...')
    with open(behaviors_path, "r") as f:
        for line in f:
            splited = line.strip('\n').split('\t')
            user_name = splited[0]
            history_item_name = splited[1].split(' ')
            if len(history_item_name) < min_seq_len:
                continue
            history_item_name = history_item_name[-(max_seq_len+3):]
            item_ids_sub_seq = [before_item_name_to_id[i] for i in history_item_name]
            user_seq_dic[user_name] = item_ids_sub_seq
            for item_id in item_ids_sub_seq:
                before_item_counts[item_id] += 1
                pairs_num += 1
            seq_num += 1
    Log_file.info("##### pairs_num {}".format(pairs_num))

    item_id = 1
    item_id_to_dic = {}
    item_name_to_id = {}
    item_id_before_to_now = {}
    for before_item_id in range(1, before_item_num + 1):
        if before_item_counts[before_item_id] != 0:
            item_id_before_to_now[before_item_id] = item_id
            item_id_to_dic[item_id] = before_item_id_to_dic[before_item_id]
            item_name_to_id[before_item_id_to_name[before_item_id]] = item_id
            item_id += 1
    item_num = len(item_id_before_to_now)
    Log_file.info("##### items after clearing {}, {}, {} #####".format(item_num, len(item_id_before_to_now), len(item_id_to_dic)))

    users_train = {}
    users_valid = {}
    users_test = {}
    users_history_for_valid = {}
    users_history_for_test = {}
    user_id = 0
    ee=pd.read_csv('/dataset/fre.csv')
    ee.columns = ['id', 'fre']
    ee.drop(0,inplace=True)
    e=ee.sort_values('fre',ascending=False)
    eee=e[['id']]
    eee=list(eee['id'])
    num=5
    #l = [0]
    #for k in range(num):
    #    l.append(int(79707*(1+k)/num))
    l = [0, 41, 127, 276, 517, 889, 1467, 2413, 4070, 7618, 79707, 79707]
    i=num_k###################################
    valid_id=0

    target = []
    distance_llm = np.array([])
    distance_rs_cos = np.array([])
    distance_rs_dot = np.array([])
    id_emb = torch.load('./tensor/order_pure_id_128.pt')
    #content_emb = torch.load('./tensor/order_pure_content_128.pt')
    a = torch.load('/dataset/bce_text/id_plus_mo-2stage/tensor/llm2vec_0.pt')
    b = torch.load('/dataset/bce_text/id_plus_mo-2stage/tensor/llm2vec_1.pt')
    c = torch.load('/dataset/bce_text/id_plus_mo-2stage/tensor/llm2vec_2.pt')
    d = torch.load('/dataset/bce_text/id_plus_mo-2stage/tensor/llm2vec_3.pt')
    content_emb = torch.cat((a,b,c,d),0)

    id_emb = id_emb.detach().numpy()
    content_emb = content_emb.detach().numpy()

    #id_emb = id_emb.detach().numpy()
    #content_emb = content_emb.detach().numpy()

    for user_name, item_seqs in user_seq_dic.items():
        user_seq = [item_id_before_to_now[i] for i in item_seqs]
        train = user_seq[:-2] ####################################################################
        users_train[user_id] = train

        if (eee.index(user_seq[-2])<l[i+1] and eee.index(user_seq[-2])>=l[i] and i<10) or (i==10):
            valid = user_seq[-(max_seq_len+2):-1]
            users_valid[valid_id] = valid
            users_history_for_valid[valid_id] = torch.LongTensor(np.array(train))
            valid_id+=1
            target.append(user_seq[-2])
        test = user_seq[-(max_seq_len+1):]
        users_test[user_id] = test
        users_history_for_test[user_id] = torch.LongTensor(np.array(user_seq[:-1]))
        user_id += 1
    Log_file.info("##### user seqs after clearing {}, {}, {}, {}, {}#####".
                  format(seq_num, len(user_seq_dic), len(users_train), len(users_valid), len(users_test)))

    path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_pure_llm_cos_'+str(num_k)+'.dat'
    if os.path.exists(path):
        distance_llm = np.fromfile(path, dtype=np.half)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_pure_rs_cos_'+str(num_k)+'.dat'
        distance_rs_cos = np.fromfile(path, dtype=np.half)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_pure_rs_dot_'+str(num_k)+'.dat'
        distance_rs_dot = np.fromfile(path, dtype=np.half)
    else:
        for k in range(valid_id):
            term_1 = content_emb[users_history_for_valid[k]]
            term_2 = content_emb[target[k]].reshape(1,-1)
            dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            distance_llm = np.concatenate((distance_llm,dis),0)
            #a=pairwise_distances(term_1,metric="cosine")
            #uptri_idx = np.triu_indices_from(a, k=1)
            #dis=a[(uptri_idx)]
            #distance_llm = np.concatenate((distance_llm,dis),0)

            term_1 = id_emb[users_history_for_valid[k]]
            term_2 = id_emb[target[k]].reshape(1,-1)
            dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            #distance_rs= np.concatenate((distance_rs,dis),0)
            #a=pairwise_distances(term_1,metric="cosine")
            #dis=a[(uptri_idx)]
            distance_rs_cos= np.concatenate((distance_rs_cos,dis),0)

            term_1 = id_emb[users_history_for_valid[k]]
            term_2 = id_emb[target[k]].reshape(1,-1)
            #dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            #distance_rs= np.concatenate((distance_rs,dis),0)
            dis = np.dot(term_1,term_2.T).reshape(term_1.shape[0])
            #dis=a[(uptri_idx)]
            distance_rs_dot= np.concatenate((distance_rs_dot,dis),0)
        distance_llm = np.half(distance_llm)
        distance_rs_cos = np.half(distance_rs_cos)
        distance_rs_dot = np.half(distance_rs_dot)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_pure_llm_cos_'+str(num_k)+'.dat'
        distance_llm.tofile(path)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_pure_rs_cos_'+str(num_k)+'.dat'
        distance_rs_cos.tofile(path)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_pure_rs_dot_'+str(num_k)+'.dat'
        distance_rs_dot.tofile(path)


    print((distance_llm.shape))
    Kendallta,p_value = kendalltau(distance_rs_cos,distance_llm)
    print(Kendallta,p_value)
    c1 = distance_rs_cos
    d1 = distance_llm
    length = c1.shape[0]
    left = int(length/num)
    right = int(length/num*(num-1))
    indices = np.argsort(d1)[::-1]
    #plt.hist(d1, bins=60, alpha=0.5, color='grey')  # bins参数定义了直方图的柱子数量
    top_k_indices = indices[:left]
    pairs_1 = d1[top_k_indices]
    #plt.hist(pairs, bins=60, alpha=0.5, color='red')
    btm_k_indices = indices[right:]
    pairs_2 = d1[btm_k_indices]
    #plt.hist(pairs, bins=60, alpha=0.5, color='blue')
    num_bins = 80
    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    n, bins, patches = ax[0].hist(d1, bins=num_bins, edgecolor='skyblue', color='white', hatch='/', label='distance_distribution')
    n_set, bins_set, patches_set = ax[0].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_'+str(int(100/num)))
    n_set_2, bins_set_2, patches_set_2 = ax[0].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_'+str(int(100/num)))
    ax[0].set_title('llm2vec Embedding Distance Distribution', fontsize=16)
    ax[0].set_xlabel('distance', fontsize=14)
    ax[0].set_ylabel('num', fontsize=14)
    ax[0].legend()

    pairs_1 = c1[top_k_indices]
    pairs_2 = c1[btm_k_indices]
    n, bins, patches = ax[1].hist(c1, bins=num_bins, edgecolor='orange', color='white', hatch='/', label='sasrec_distribution')
    n_set, bins_set, patches_set = ax[1].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_'+str(int(100/num)), alpha=1)
    n_set_2, bins_set_2, patches_set_2 = ax[1].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_'+str(int(100/num)), alpha=0.4)
    ax[1].set_title('llm2vec Top/bottom '+str(int(100/num))+'% in SASRec', fontsize=16)
    ax[1].set_xlabel('distance', fontsize=14)
    ax[1].set_ylabel('num', fontsize=14)
    ax[1].legend()
    plt.savefig('./figure/llm2vec/target_llm_distance-pure-cos-'+str(i)+'.png', dpi=600)  # 保存图为png格式
    plt.show()
    plt.clf()
    print('done task 1-1')

    c1 = distance_rs_dot
    d1 = distance_llm
    pairs_1 = d1[top_k_indices]
    pairs_2 = d1[btm_k_indices]
    #plt.hist(pairs, bins=60, alpha=0.5, color='blue')
    num_bins = 80
    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    n, bins, patches = ax[0].hist(d1, bins=num_bins, edgecolor='skyblue', color='white', hatch='/', label='distance_distribution')
    n_set, bins_set, patches_set = ax[0].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top'+str(int(100/num)))
    n_set_2, bins_set_2, patches_set_2 = ax[0].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_'+str(int(100/num)))
    ax[0].set_title('llm2vec Embedding Distance Distribution', fontsize=16)
    ax[0].set_xlabel('distance', fontsize=14)
    ax[0].set_ylabel('num', fontsize=14)
    ax[0].legend()

    pairs_1 = c1[top_k_indices]
    pairs_2 = c1[btm_k_indices]
    n, bins, patches = ax[1].hist(c1, bins=num_bins, edgecolor='orange', color='white', hatch='/', label='sasrec_distribution')
    n_set, bins_set, patches_set = ax[1].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_'+str(int(100/num)), alpha=1)
    n_set_2, bins_set_2, patches_set_2 = ax[1].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_'+str(int(100/num)), alpha=0.4)
    ax[1].set_title('llm2vec Top/bottom '+str(int(100/num))+'% in SASRec', fontsize=16)
    ax[1].set_xlabel('distance', fontsize=14)
    ax[1].set_ylabel('num', fontsize=14)
    ax[1].legend()
    plt.savefig('./figure/llm2vec/target_llm_distance-pure-dot-'+str(i)+'.png', dpi=600)  # 保存图为png格式
    plt.show()
    plt.clf()
    print('done task 1-2')











    id_emb = torch.load('./tensor/order_mmd_item_id_128.pt')
    content_emb = torch.load('./tensor/order_mmd_content_128.pt')
    id_emb = id_emb.detach().numpy()
    content_emb = content_emb.detach().numpy()
    distance_llm = np.array([])
    distance_rs_cos = np.array([])
    distance_rs_dot = np.array([])
    path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_mmd_llm_cos_'+str(num_k)+'.dat'
    if os.path.exists(path):
        distance_llm = np.fromfile(path, dtype=np.half)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_mmd_rs_cos_'+str(num_k)+'.dat'
        distance_rs_cos = np.fromfile(path, dtype=np.half)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_mmd_rs_dot_'+str(num_k)+'.dat'
        distance_rs_dot = np.fromfile(path, dtype=np.half)
    else:
        for k in range(valid_id):
            term_1 = content_emb[users_history_for_valid[k]]
            term_2 = content_emb[target[k]].reshape(1,-1)
            dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            distance_llm = np.concatenate((distance_llm,dis),0)
            #a=pairwise_distances(term_1,metric="cosine")
            #uptri_idx = np.triu_indices_from(a, k=1)
            #dis=a[(uptri_idx)]
            #distance_llm = np.concatenate((distance_llm,dis),0)

            term_1 = id_emb[users_history_for_valid[k]]
            term_2 = id_emb[target[k]].reshape(1,-1)
            dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            #distance_rs= np.concatenate((distance_rs,dis),0)
            #a=pairwise_distances(term_1,metric="cosine")
            #dis=a[(uptri_idx)]
            distance_rs_cos= np.concatenate((distance_rs_cos,dis),0)

            term_1 = id_emb[users_history_for_valid[k]]
            term_2 = id_emb[target[k]].reshape(1,-1)
            #dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            #distance_rs= np.concatenate((distance_rs,dis),0)
            dis = np.dot(term_1,term_2.T).reshape(term_1.shape[0])
            #dis=a[(uptri_idx)]
            distance_rs_dot= np.concatenate((distance_rs_dot,dis),0)
        distance_llm = np.half(distance_llm)
        distance_rs_cos = np.half(distance_rs_cos)
        distance_rs_dot = np.half(distance_rs_dot)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_mmd_llm_cos_'+str(num_k)+'.dat'
        distance_llm.tofile(path)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_mmd_rs_cos_'+str(num_k)+'.dat'
        distance_rs_cos.tofile(path)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_llm_mmd_rs_dot_'+str(num_k)+'.dat'
        distance_rs_dot.tofile(path)


    print((distance_llm.shape))
    Kendallta,p_value = kendalltau(distance_rs_cos,distance_llm)
    print(Kendallta,p_value)
    c1 = distance_rs_cos
    d1 = distance_llm
    length = c1.shape[0]
    left = int(length/num)
    right = int(length/num*(num-1))
    indices = np.argsort(d1)[::-1]
    #plt.hist(d1, bins=60, alpha=0.5, color='grey')  # bins参数定义了直方图的柱子数量
    top_k_indices = indices[:left]
    pairs_1 = d1[top_k_indices]
    #plt.hist(pairs, bins=60, alpha=0.5, color='red')
    btm_k_indices = indices[right:]
    pairs_2 = d1[btm_k_indices]
    #plt.hist(pairs, bins=60, alpha=0.5, color='blue')
    num_bins = 80
    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    n, bins, patches = ax[0].hist(d1, bins=num_bins, edgecolor='skyblue', color='white', hatch='/', label='distance_distribution')
    n_set, bins_set, patches_set = ax[0].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_'+str(int(100/num)))
    n_set_2, bins_set_2, patches_set_2 = ax[0].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_'+str(int(100/num)))
    ax[0].set_title('llm2vec Embedding Distance Distribution', fontsize=16)
    ax[0].set_xlabel('distance', fontsize=14)
    ax[0].set_ylabel('num', fontsize=14)
    ax[0].legend()

    pairs_1 = c1[top_k_indices]
    pairs_2 = c1[btm_k_indices]
    n, bins, patches = ax[1].hist(c1, bins=num_bins, edgecolor='orange', color='white', hatch='/', label='sasrec_distribution')
    n_set, bins_set, patches_set = ax[1].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_'+str(int(100/num)), alpha=1)
    n_set_2, bins_set_2, patches_set_2 = ax[1].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_'+str(int(100/num)), alpha=0.4)
    ax[1].set_title('llm2vec Top/bottom '+str(int(100/num))+'% in SASRec', fontsize=16)
    ax[1].set_xlabel('distance', fontsize=14)
    ax[1].set_ylabel('num', fontsize=14)
    ax[1].legend()
    plt.savefig('./figure/llm2vec/target_llm_distance-mmd-cos-'+str(i)+'.png', dpi=600)  # 保存图为png格式
    plt.show()
    plt.clf()
    print('done task 3-1')

    c1 = distance_rs_dot
    d1 = distance_llm
    pairs_1 = d1[top_k_indices]
    pairs_2 = d1[btm_k_indices]
    #plt.hist(pairs, bins=60, alpha=0.5, color='blue')
    num_bins = 80
    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    n, bins, patches = ax[0].hist(d1, bins=num_bins, edgecolor='skyblue', color='white', hatch='/', label='distance_distribution')
    n_set, bins_set, patches_set = ax[0].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_'+str(int(100/num)))
    n_set_2, bins_set_2, patches_set_2 = ax[0].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_'+str(int(100/num)))
    ax[0].set_title('llm2vec Embedding Distance Distribution', fontsize=16)
    ax[0].set_xlabel('distance', fontsize=14)
    ax[0].set_ylabel('num', fontsize=14)
    ax[0].legend()

    pairs_1 = c1[top_k_indices]
    pairs_2 = c1[btm_k_indices]
    n, bins, patches = ax[1].hist(c1, bins=num_bins, edgecolor='orange', color='white', hatch='/', label='sasrec_distribution')
    n_set, bins_set, patches_set = ax[1].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_'+str(int(100/num)), alpha=1)
    n_set_2, bins_set_2, patches_set_2 = ax[1].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_'+str(int(100/num)), alpha=0.4)
    ax[1].set_title('llm2vec Top/bottom '+str(int(100/num))+'% in SASRec', fontsize=16)
    ax[1].set_xlabel('distance', fontsize=14)
    ax[1].set_ylabel('num', fontsize=14)
    ax[1].legend()
    plt.savefig('./figure/llm2vec/target_llm_distance-mmd-dot-'+str(i)+'.png', dpi=600)  # 保存图为png格式
    plt.show()
    plt.clf()
    print('done task 3-2')

    return item_num, item_id_to_dic, users_train, users_valid, users_test, \
           users_history_for_valid, users_history_for_test, item_name_to_id


def read_behaviors_fre_target(behaviors_path, before_item_id_to_dic, before_item_name_to_id, before_item_id_to_name, max_seq_len, min_seq_len, Log_file,num_k):
    Log_file.info("##### news number {} {} (before clearing)#####".format(len(before_item_id_to_dic), len(before_item_name_to_id)))
    Log_file.info("##### min seq len {}, max seq len {}#####".format(min_seq_len, max_seq_len))
    before_item_num = len(before_item_name_to_id)
    before_item_counts = [0] * (before_item_num + 1)
    user_seq_dic = {}
    seq_num = 0
    pairs_num = 0
    Log_file.info('rebuild user seqs...')
    with open(behaviors_path, "r") as f:
        for line in f:
            splited = line.strip('\n').split('\t')
            user_name = splited[0]
            history_item_name = splited[1].split(' ')
            if len(history_item_name) < min_seq_len:
                continue
            history_item_name = history_item_name[-(max_seq_len+3):]
            item_ids_sub_seq = [before_item_name_to_id[i] for i in history_item_name]
            user_seq_dic[user_name] = item_ids_sub_seq
            for item_id in item_ids_sub_seq:
                before_item_counts[item_id] += 1
                pairs_num += 1
            seq_num += 1
    Log_file.info("##### pairs_num {}".format(pairs_num))

    item_id = 1
    item_id_to_dic = {}
    item_name_to_id = {}
    item_id_before_to_now = {}
    for before_item_id in range(1, before_item_num + 1):
        if before_item_counts[before_item_id] != 0:
            item_id_before_to_now[before_item_id] = item_id
            item_id_to_dic[item_id] = before_item_id_to_dic[before_item_id]
            item_name_to_id[before_item_id_to_name[before_item_id]] = item_id
            item_id += 1
    item_num = len(item_id_before_to_now)
    Log_file.info("##### items after clearing {}, {}, {} #####".format(item_num, len(item_id_before_to_now), len(item_id_to_dic)))

    users_train = {}
    users_valid = {}
    users_test = {}
    users_history_for_valid = {}
    users_history_for_test = {}
    user_id = 0
    ee=pd.read_csv('/dataset/fre.csv')
    ee.columns = ['id', 'fre']
    ee.drop(0,inplace=True)
    e=ee.sort_values('fre',ascending=False)
    eee=e[['id']]
    eee=list(eee['id'])
    num=20
    #l = [0]
    #for k in range(num):
    #    l.append(int(79707*(1+k)/num))
    l = [0, 41, 127, 276, 517, 889, 1467, 2413, 4070, 7618, 79707, 79707]
    i=num_k###################################
    valid_id=0
    
    target = []
    distance_llm = np.array([])
    distance_rs_cos = np.array([])
    distance_rs_dot = np.array([])
    id_emb = torch.load('./tensor/order_pure_id_128.pt')
    content_emb = torch.load('./tensor/order_pure_content_128.pt')
    id_emb = id_emb.detach().numpy()
    content_emb = content_emb.detach().numpy()

    #id_emb = id_emb.detach().numpy()
    #content_emb = content_emb.detach().numpy()
    
    for user_name, item_seqs in user_seq_dic.items():
        user_seq = [item_id_before_to_now[i] for i in item_seqs]
        train = user_seq[:-2] ####################################################################
        users_train[user_id] = train
        
        if (eee.index(user_seq[-2])<l[i+1] and eee.index(user_seq[-2])>=l[i] and i<10) or (i==10):
            valid = user_seq[-(max_seq_len+2):-1]
            users_valid[valid_id] = valid
            users_history_for_valid[valid_id] = torch.LongTensor(np.array(train))
            valid_id+=1
            target.append(user_seq[-2])
        test = user_seq[-(max_seq_len+1):]
        users_test[user_id] = test
        users_history_for_test[user_id] = torch.LongTensor(np.array(user_seq[:-1]))
        user_id += 1
    Log_file.info("##### user seqs after clearing {}, {}, {}, {}, {}#####".
                  format(seq_num, len(user_seq_dic), len(users_train), len(users_valid), len(users_test)))
    
    path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_pure_llm_cos_'+str(num_k)+'.dat'
    if os.path.exists(path):
        distance_llm = np.fromfile(path, dtype=np.half)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_pure_rs_cos_'+str(num_k)+'.dat'
        distance_rs_cos = np.fromfile(path, dtype=np.half)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_pure_rs_dot_'+str(num_k)+'.dat'
        distance_rs_dot = np.fromfile(path, dtype=np.half)
    else:
        for k in range(valid_id):
            term_1 = content_emb[users_history_for_valid[k]]
            term_2 = content_emb[target[k]].reshape(1,-1)
            dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            distance_llm = np.concatenate((distance_llm,dis),0)
            #a=pairwise_distances(term_1,metric="cosine")
            #uptri_idx = np.triu_indices_from(a, k=1)
            #dis=a[(uptri_idx)]
            #distance_llm = np.concatenate((distance_llm,dis),0)
            
            term_1 = id_emb[users_history_for_valid[k]]
            term_2 = id_emb[target[k]].reshape(1,-1)
            dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            #distance_rs= np.concatenate((distance_rs,dis),0)
            #a=pairwise_distances(term_1,metric="cosine")
            #dis=a[(uptri_idx)]
            distance_rs_cos= np.concatenate((distance_rs_cos,dis),0)
            
            term_1 = id_emb[users_history_for_valid[k]]
            term_2 = id_emb[target[k]].reshape(1,-1)
            #dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            #distance_rs= np.concatenate((distance_rs,dis),0)
            dis = np.dot(term_1,term_2.T).reshape(term_1.shape[0])
            #dis=a[(uptri_idx)]
            distance_rs_dot= np.concatenate((distance_rs_dot,dis),0)
        distance_llm = np.half(distance_llm)
        distance_rs_cos = np.half(distance_rs_cos)
        distance_rs_dot = np.half(distance_rs_dot)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_pure_llm_cos_'+str(num_k)+'.dat'
        distance_llm.tofile(path)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_pure_rs_cos_'+str(num_k)+'.dat'
        distance_rs_cos.tofile(path)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_pure_rs_dot_'+str(num_k)+'.dat'
        distance_rs_dot.tofile(path)
        
    
    print((distance_llm.shape))
    #Kendallta,p_value = kendalltau(distance_rs,distance_llm)
    #print(Kendallta,p_value)
    c1 = distance_rs_cos
    d1 = distance_llm
    length = c1.shape[0]
    left = int(length/num)
    right = int(length/num*(num-1))
    indices = np.argsort(d1)[::-1]
    #plt.hist(d1, bins=60, alpha=0.5, color='grey')  # bins参数定义了直方图的柱子数量
    top_k_indices = indices[:left]
    pairs_1 = d1[top_k_indices]
    #plt.hist(pairs, bins=60, alpha=0.5, color='red')
    btm_k_indices = indices[right:]
    pairs_2 = d1[btm_k_indices]
    #plt.hist(pairs, bins=60, alpha=0.5, color='blue')
    num_bins = 40
    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    n, bins, patches = ax[0].hist(d1, bins=num_bins, edgecolor='skyblue', color='white', hatch='/', label='distance_distribution')
    n_set, bins_set, patches_set = ax[0].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_'+str(int(100/num)))
    n_set_2, bins_set_2, patches_set_2 = ax[0].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_'+str(int(100/num)))
    ax[0].set_title('BERT Embedding Distance Distribution', fontsize=16)
    ax[0].set_xlabel('distance', fontsize=14)
    ax[0].set_ylabel('num', fontsize=14)
    ax[0].legend()
    
    pairs_1 = c1[top_k_indices]
    pairs_2 = c1[btm_k_indices]
    n, bins, patches = ax[1].hist(c1, bins=num_bins, edgecolor='orange', color='white', hatch='/', label='sasrec_distribution')
    n_set, bins_set, patches_set = ax[1].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_'+str(int(100/num)), alpha=1)
    n_set_2, bins_set_2, patches_set_2 = ax[1].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_'+str(int(100/num)), alpha=0.4)
    ax[1].set_title('BERT Top/bottom '+str(int(100/num))+'% in SASRec', fontsize=16)
    ax[1].set_xlabel('distance', fontsize=14)
    ax[1].set_ylabel('num', fontsize=14)
    ax[1].legend()
    plt.savefig('./figure/target_distance-pure-cos-'+str(i)+'.png', dpi=600)  # 保存图为png格式
    plt.show()
    plt.clf()
    print('done task 1-1')
    
    c1 = distance_rs_dot
    d1 = distance_llm
    pairs_1 = d1[top_k_indices]
    pairs_2 = d1[btm_k_indices]
    #plt.hist(pairs, bins=60, alpha=0.5, color='blue')
    num_bins = 40
    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    n, bins, patches = ax[0].hist(d1, bins=num_bins, edgecolor='skyblue', color='white', hatch='/', label='distance_distribution')
    n_set, bins_set, patches_set = ax[0].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top'+str(int(100/num)))
    n_set_2, bins_set_2, patches_set_2 = ax[0].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_'+str(int(100/num)))
    ax[0].set_title('BERT Embedding Distance Distribution', fontsize=16)
    ax[0].set_xlabel('distance', fontsize=14)
    ax[0].set_ylabel('num', fontsize=14)
    ax[0].legend()
    
    pairs_1 = c1[top_k_indices]
    pairs_2 = c1[btm_k_indices]
    n, bins, patches = ax[1].hist(c1, bins=num_bins, edgecolor='orange', color='white', hatch='/', label='sasrec_distribution')
    n_set, bins_set, patches_set = ax[1].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_'+str(int(100/num)), alpha=1)
    n_set_2, bins_set_2, patches_set_2 = ax[1].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_'+str(int(100/num)), alpha=0.4)
    ax[1].set_title('BERT Top/bottom '+str(int(100/num))+'% in SASRec', fontsize=16)
    ax[1].set_xlabel('distance', fontsize=14)
    ax[1].set_ylabel('num', fontsize=14)
    ax[1].legend()
    plt.savefig('./figure/target_distance-pure-dot-'+str(i)+'.png', dpi=600)  # 保存图为png格式
    plt.show()
    plt.clf()
    print('done task 1-2')
    
    
    
    
    
    
    
    id_emb = torch.load('./tensor/order_item_id_128.pt')
    content_emb = torch.load('./tensor/order_content_128.pt')
    id_emb = id_emb.detach().numpy()
    content_emb = content_emb.detach().numpy()
    distance_llm = np.array([])
    distance_rs_cos = np.array([])
    distance_rs_dot = np.array([])
    path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_cotrain_llm_cos_'+str(num_k)+'.dat'
    if os.path.exists(path):
        distance_llm = np.fromfile(path, dtype=np.half)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_cotrain_rs_cos_'+str(num_k)+'.dat'
        distance_rs_cos = np.fromfile(path, dtype=np.half)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_cotrain_rs_dot_'+str(num_k)+'.dat'
        distance_rs_dot = np.fromfile(path, dtype=np.half)
    else:
        for k in range(valid_id):
            term_1 = content_emb[users_history_for_valid[k]]
            term_2 = content_emb[target[k]].reshape(1,-1)
            dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            distance_llm = np.concatenate((distance_llm,dis),0)
            #a=pairwise_distances(term_1,metric="cosine")
            #uptri_idx = np.triu_indices_from(a, k=1)
            #dis=a[(uptri_idx)]
            #distance_llm = np.concatenate((distance_llm,dis),0)
            
            term_1 = id_emb[users_history_for_valid[k]]
            term_2 = id_emb[target[k]].reshape(1,-1)
            dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            #distance_rs= np.concatenate((distance_rs,dis),0)
            #a=pairwise_distances(term_1,metric="cosine")
            #dis=a[(uptri_idx)]
            distance_rs_cos= np.concatenate((distance_rs_cos,dis),0)
            
            term_1 = id_emb[users_history_for_valid[k]]
            term_2 = id_emb[target[k]].reshape(1,-1)
            #dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            #distance_rs= np.concatenate((distance_rs,dis),0)
            dis = np.dot(term_1,term_2.T).reshape(term_1.shape[0])
            #dis=a[(uptri_idx)]
            distance_rs_dot= np.concatenate((distance_rs_dot,dis),0)
        distance_llm = np.half(distance_llm)
        distance_rs_cos = np.half(distance_rs_cos)
        distance_rs_dot = np.half(distance_rs_dot)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_cotrain_llm_cos_'+str(num_k)+'.dat'
        distance_llm.tofile(path)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_cotrain_rs_cos_'+str(num_k)+'.dat'
        distance_rs_cos.tofile(path)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_cotrain_rs_dot_'+str(num_k)+'.dat'
        distance_rs_dot.tofile(path)
        
    
    print((distance_llm.shape))
    #Kendallta,p_value = kendalltau(distance_rs,distance_llm)
    #print(Kendallta,p_value)
    c1 = distance_rs_cos
    d1 = distance_llm
    length = c1.shape[0]
    left = int(length/num)
    right = int(length/num*(num-1))
    indices = np.argsort(d1)[::-1]
    #plt.hist(d1, bins=60, alpha=0.5, color='grey')  # bins参数定义了直方图的柱子数量
    top_k_indices = indices[:left]
    pairs_1 = d1[top_k_indices]
    #plt.hist(pairs, bins=60, alpha=0.5, color='red')
    btm_k_indices = indices[right:]
    pairs_2 = d1[btm_k_indices]
    #plt.hist(pairs, bins=60, alpha=0.5, color='blue')
    num_bins = 40
    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    n, bins, patches = ax[0].hist(d1, bins=num_bins, edgecolor='skyblue', color='white', hatch='/', label='distance_distribution')
    n_set, bins_set, patches_set = ax[0].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_'+str(int(100/num)))
    n_set_2, bins_set_2, patches_set_2 = ax[0].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_'+str(int(100/num)))
    ax[0].set_title('BERT Embedding Distance Distribution', fontsize=16)
    ax[0].set_xlabel('distance', fontsize=14)
    ax[0].set_ylabel('num', fontsize=14)
    ax[0].legend()
    
    pairs_1 = c1[top_k_indices]
    pairs_2 = c1[btm_k_indices]
    n, bins, patches = ax[1].hist(c1, bins=num_bins, edgecolor='orange', color='white', hatch='/', label='sasrec_distribution')
    n_set, bins_set, patches_set = ax[1].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_'+str(int(100/num)), alpha=1)
    n_set_2, bins_set_2, patches_set_2 = ax[1].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_'+str(int(100/num)), alpha=0.4)
    ax[1].set_title('BERT Top/bottom '+str(int(100/num))+'% in SASRec', fontsize=16)
    ax[1].set_xlabel('distance', fontsize=14)
    ax[1].set_ylabel('num', fontsize=14)
    ax[1].legend()
    plt.savefig('./figure/target_distance-cotrain-cos-'+str(i)+'.png', dpi=600)  # 保存图为png格式
    plt.show()
    plt.clf()
    print('done task 2-1')
    
    c1 = distance_rs_dot
    d1 = distance_llm
    pairs_1 = d1[top_k_indices]
    pairs_2 = d1[btm_k_indices]
    #plt.hist(pairs, bins=60, alpha=0.5, color='blue')
    num_bins = 40
    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    n, bins, patches = ax[0].hist(d1, bins=num_bins, edgecolor='skyblue', color='white', hatch='/', label='distance_distribution')
    n_set, bins_set, patches_set = ax[0].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_'+str(int(100/num)))
    n_set_2, bins_set_2, patches_set_2 = ax[0].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_'+str(int(100/num)))
    ax[0].set_title('BERT Embedding Distance Distribution', fontsize=16)
    ax[0].set_xlabel('distance', fontsize=14)
    ax[0].set_ylabel('num', fontsize=14)
    ax[0].legend()
    
    pairs_1 = c1[top_k_indices]
    pairs_2 = c1[btm_k_indices]
    n, bins, patches = ax[1].hist(c1, bins=num_bins, edgecolor='orange', color='white', hatch='/', label='sasrec_distribution')
    n_set, bins_set, patches_set = ax[1].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_'+str(int(100/num)), alpha=1)
    n_set_2, bins_set_2, patches_set_2 = ax[1].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_'+str(int(100/num)), alpha=0.4)
    ax[1].set_title('BERT Top/bottom '+str(int(100/num))+'% in SASRec', fontsize=16)
    ax[1].set_xlabel('distance', fontsize=14)
    ax[1].set_ylabel('num', fontsize=14)
    ax[1].legend()
    plt.savefig('./figure/target_distance-cotrain-dot-'+str(i)+'.png', dpi=600)  # 保存图为png格式
    plt.show()
    plt.clf()
    print('done task 2-2')
    
    
    
    
    
    
     
    id_emb = torch.load('./tensor/order_mmd_item_id_128.pt')
    content_emb = torch.load('./tensor/order_mmd_content_128.pt')
    id_emb = id_emb.detach().numpy()
    content_emb = content_emb.detach().numpy()
    distance_llm = np.array([])
    distance_rs_cos = np.array([])
    distance_rs_dot = np.array([])
    path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_mmd_llm_cos_'+str(num_k)+'.dat'
    if os.path.exists(path):
        distance_llm = np.fromfile(path, dtype=np.half)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_mmd_rs_cos_'+str(num_k)+'.dat'
        distance_rs_cos = np.fromfile(path, dtype=np.half)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_mmd_rs_dot_'+str(num_k)+'.dat'
        distance_rs_dot = np.fromfile(path, dtype=np.half)
    else:
        for k in range(valid_id):
            term_1 = content_emb[users_history_for_valid[k]]
            term_2 = content_emb[target[k]].reshape(1,-1)
            dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            distance_llm = np.concatenate((distance_llm,dis),0)
            #a=pairwise_distances(term_1,metric="cosine")
            #uptri_idx = np.triu_indices_from(a, k=1)
            #dis=a[(uptri_idx)]
            #distance_llm = np.concatenate((distance_llm,dis),0)
            
            term_1 = id_emb[users_history_for_valid[k]]
            term_2 = id_emb[target[k]].reshape(1,-1)
            dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            #distance_rs= np.concatenate((distance_rs,dis),0)
            #a=pairwise_distances(term_1,metric="cosine")
            #dis=a[(uptri_idx)]
            distance_rs_cos= np.concatenate((distance_rs_cos,dis),0)
            
            term_1 = id_emb[users_history_for_valid[k]]
            term_2 = id_emb[target[k]].reshape(1,-1)
            #dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            #distance_rs= np.concatenate((distance_rs,dis),0)
            dis = np.dot(term_1,term_2.T).reshape(term_1.shape[0])
            #dis=a[(uptri_idx)]
            distance_rs_dot= np.concatenate((distance_rs_dot,dis),0)
        distance_llm = np.half(distance_llm)
        distance_rs_cos = np.half(distance_rs_cos)
        distance_rs_dot = np.half(distance_rs_dot)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_mmd_llm_cos_'+str(num_k)+'.dat'
        distance_llm.tofile(path)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_mmd_rs_cos_'+str(num_k)+'.dat'
        distance_rs_cos.tofile(path)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/target_mmd_rs_dot_'+str(num_k)+'.dat'
        distance_rs_dot.tofile(path)
        
    
    print((distance_llm.shape))
    #Kendallta,p_value = kendalltau(distance_rs,distance_llm)
    #print(Kendallta,p_value)
    c1 = distance_rs_cos
    d1 = distance_llm
    length = c1.shape[0]
    left = int(length/num)
    right = int(length/num*(num-1))
    indices = np.argsort(d1)[::-1]
    #plt.hist(d1, bins=60, alpha=0.5, color='grey')  # bins参数定义了直方图的柱子数量
    top_k_indices = indices[:left]
    pairs_1 = d1[top_k_indices]
    #plt.hist(pairs, bins=60, alpha=0.5, color='red')
    btm_k_indices = indices[right:]
    pairs_2 = d1[btm_k_indices]
    #plt.hist(pairs, bins=60, alpha=0.5, color='blue')
    num_bins = 40
    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    n, bins, patches = ax[0].hist(d1, bins=num_bins, edgecolor='skyblue', color='white', hatch='/', label='distance_distribution')
    n_set, bins_set, patches_set = ax[0].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_'+str(int(100/num)))
    n_set_2, bins_set_2, patches_set_2 = ax[0].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_'+str(int(100/num)))
    ax[0].set_title('BERT Embedding Distance Distribution', fontsize=16)
    ax[0].set_xlabel('distance', fontsize=14)
    ax[0].set_ylabel('num', fontsize=14)
    ax[0].legend()
    
    pairs_1 = c1[top_k_indices]
    pairs_2 = c1[btm_k_indices]
    n, bins, patches = ax[1].hist(c1, bins=num_bins, edgecolor='orange', color='white', hatch='/', label='sasrec_distribution')
    n_set, bins_set, patches_set = ax[1].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_'+str(int(100/num)), alpha=1)
    n_set_2, bins_set_2, patches_set_2 = ax[1].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_'+str(int(100/num)), alpha=0.4)
    ax[1].set_title('BERT Top/bottom '+str(int(100/num))+'% in SASRec', fontsize=16)
    ax[1].set_xlabel('distance', fontsize=14)
    ax[1].set_ylabel('num', fontsize=14)
    ax[1].legend()
    plt.savefig('./figure/target_distance-mmd-cos-'+str(i)+'.png', dpi=600)  # 保存图为png格式
    plt.show()
    plt.clf()
    print('done task 3-1')
    
    c1 = distance_rs_dot
    d1 = distance_llm
    pairs_1 = d1[top_k_indices]
    pairs_2 = d1[btm_k_indices]
    #plt.hist(pairs, bins=60, alpha=0.5, color='blue')
    num_bins = 40
    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    n, bins, patches = ax[0].hist(d1, bins=num_bins, edgecolor='skyblue', color='white', hatch='/', label='distance_distribution')
    n_set, bins_set, patches_set = ax[0].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_'+str(int(100/num)))
    n_set_2, bins_set_2, patches_set_2 = ax[0].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_'+str(int(100/num)))
    ax[0].set_title('BERT Embedding Distance Distribution', fontsize=16)
    ax[0].set_xlabel('distance', fontsize=14)
    ax[0].set_ylabel('num', fontsize=14)
    ax[0].legend()
    
    pairs_1 = c1[top_k_indices]
    pairs_2 = c1[btm_k_indices]
    n, bins, patches = ax[1].hist(c1, bins=num_bins, edgecolor='orange', color='white', hatch='/', label='sasrec_distribution')
    n_set, bins_set, patches_set = ax[1].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_'+str(int(100/num)), alpha=1)
    n_set_2, bins_set_2, patches_set_2 = ax[1].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_'+str(int(100/num)), alpha=0.4)
    ax[1].set_title('BERT Top/bottom '+str(int(100/num))+'% in SASRec', fontsize=16)
    ax[1].set_xlabel('distance', fontsize=14)
    ax[1].set_ylabel('num', fontsize=14)
    ax[1].legend()
    plt.savefig('./figure/target_distance-mmd-dot-'+str(i)+'.png', dpi=600)  # 保存图为png格式
    plt.show()
    plt.clf()
    print('done task 3-2')

    return item_num, item_id_to_dic, users_train, users_valid, users_test, \
           users_history_for_valid, users_history_for_test, item_name_to_id

def read_behaviors_fre_full(behaviors_path, before_item_id_to_dic, before_item_name_to_id, before_item_id_to_name, max_seq_len, min_seq_len, Log_file,num_k):
    Log_file.info("##### news number {} {} (before clearing)#####".format(len(before_item_id_to_dic), len(before_item_name_to_id)))
    Log_file.info("##### min seq len {}, max seq len {}#####".format(min_seq_len, max_seq_len))
    before_item_num = len(before_item_name_to_id)
    before_item_counts = [0] * (before_item_num + 1)
    user_seq_dic = {}
    seq_num = 0
    pairs_num = 0
    Log_file.info('rebuild user seqs...')
    with open(behaviors_path, "r") as f:
        for line in f:
            splited = line.strip('\n').split('\t')
            user_name = splited[0]
            history_item_name = splited[1].split(' ')
            if len(history_item_name) < min_seq_len:
                continue
            history_item_name = history_item_name[-(max_seq_len+3):]
            item_ids_sub_seq = [before_item_name_to_id[i] for i in history_item_name]
            user_seq_dic[user_name] = item_ids_sub_seq
            for item_id in item_ids_sub_seq:
                before_item_counts[item_id] += 1
                pairs_num += 1
            seq_num += 1
    Log_file.info("##### pairs_num {}".format(pairs_num))

    item_id = 1
    item_id_to_dic = {}
    item_name_to_id = {}
    item_id_before_to_now = {}
    for before_item_id in range(1, before_item_num + 1):
        if before_item_counts[before_item_id] != 0:
            item_id_before_to_now[before_item_id] = item_id
            item_id_to_dic[item_id] = before_item_id_to_dic[before_item_id]
            item_name_to_id[before_item_id_to_name[before_item_id]] = item_id
            item_id += 1
    item_num = len(item_id_before_to_now)
    Log_file.info("##### items after clearing {}, {}, {} #####".format(item_num, len(item_id_before_to_now), len(item_id_to_dic)))

    users_train = {}
    users_valid = {}
    users_test = {}
    users_history_for_valid = {}
    users_history_for_test = {}
    user_id = 0
    ee=pd.read_csv('/dataset/fre.csv')
    ee.columns = ['id', 'fre']
    ee.drop(0,inplace=True)
    e=ee.sort_values('fre',ascending=False)
    eee=e[['id']]
    eee=list(eee['id'])
    num=20
    #l = [0]
    #for k in range(num):
    #    l.append(int(79707*(1+k)/num))
    l = [0, 41, 127, 276, 517, 889, 1467, 2413, 4070, 7618, 79707, 79707]
    i=num_k###################################
    valid_id=0
    
    target = []
    distance_llm = np.array([])
    distance_rs_cos = np.array([])
    distance_rs_dot = np.array([])
    id_emb = torch.load('./tensor/order_pure_id_128.pt')
    content_emb = torch.load('./tensor/order_pure_content_128.pt')
    id_emb = id_emb.detach().numpy()
    content_emb = content_emb.detach().numpy()

    #id_emb = id_emb.detach().numpy()
    #content_emb = content_emb.detach().numpy()
    
    for user_name, item_seqs in user_seq_dic.items():
        user_seq = [item_id_before_to_now[i] for i in item_seqs]
        train = user_seq[:-1] ####################################################################
        users_train[user_id] = train
        
        if (eee.index(user_seq[-2])<l[i+1] and eee.index(user_seq[-2])>=l[i] and i<10) or (i==10):
            valid = user_seq[-(max_seq_len+2):-1]
            users_valid[valid_id] = valid
            users_history_for_valid[valid_id] = torch.LongTensor(np.array(train))
            valid_id+=1
            target.append(user_seq[-2])
        test = user_seq[-(max_seq_len+1):]
        users_test[user_id] = test
        users_history_for_test[user_id] = torch.LongTensor(np.array(user_seq[:-1]))
        user_id += 1
    Log_file.info("##### user seqs after clearing {}, {}, {}, {}, {}#####".
                  format(seq_num, len(user_seq_dic), len(users_train), len(users_valid), len(users_test)))
    
    path = '/dataset/bce_text/id_plus_mo-2stage/tensor/full_pure_llm_cos_'+str(num_k)+'.dat'
    if os.path.exists(path):
        distance_llm = np.fromfile(path, dtype=np.half)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/full_pure_rs_cos_'+str(num_k)+'.dat'
        distance_rs_cos = np.fromfile(path, dtype=np.half)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/full_pure_rs_dot_'+str(num_k)+'.dat'
        distance_rs_dot = np.fromfile(path, dtype=np.half)
    else:
        for k in range(valid_id):
            term_1 = content_emb[users_history_for_valid[k]]
            #term_2 = content_emb[target[k]].reshape(1,-1)
            #dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            #distance_llm = np.concatenate((distance_llm,dis),0)
            a=pairwise_distances(term_1,metric="cosine")
            uptri_idx = np.triu_indices_from(a, k=1)
            dis=a[(uptri_idx)]
            distance_llm = np.concatenate((distance_llm,dis),0)
            
            term_1 = id_emb[users_history_for_valid[k]]
            #term_2 = id_emb[target[k]].reshape(1,-1)
            #dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            #distance_rs= np.concatenate((distance_rs,dis),0)
            a=pairwise_distances(term_1,metric="cosine")
            dis=a[(uptri_idx)]
            distance_rs_cos= np.concatenate((distance_rs_cos,dis),0)
            
            term_1 = id_emb[users_history_for_valid[k]]
            #term_2 = id_emb[target[k]].reshape(1,-1)
            #dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            #distance_rs= np.concatenate((distance_rs,dis),0)
            a = np.dot(term_1,term_1.T)
            dis=a[(uptri_idx)]
            distance_rs_dot= np.concatenate((distance_rs_dot,dis),0)
        distance_llm = np.half(distance_llm)
        distance_rs_cos = np.half(distance_rs_cos)
        distance_rs_dot = np.half(distance_rs_dot)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/full_pure_llm_cos_'+str(num_k)+'.dat'
        distance_llm.tofile(path)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/full_pure_rs_cos_'+str(num_k)+'.dat'
        distance_rs_cos.tofile(path)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/full_pure_rs_dot_'+str(num_k)+'.dat'
        distance_rs_dot.tofile(path)
        
    
    print((distance_llm.shape))
    #Kendallta,p_value = kendalltau(distance_rs,distance_llm)
    #print(Kendallta,p_value)
    c1 = distance_rs_cos
    d1 = distance_llm
    length = c1.shape[0]
    left = int(length/num)
    right = int(length/num*(num-1))
    indices = np.argsort(d1)[::-1]
    #plt.hist(d1, bins=60, alpha=0.5, color='grey')  # bins参数定义了直方图的柱子数量
    top_k_indices = indices[:left]
    pairs_1 = d1[top_k_indices]
    #plt.hist(pairs, bins=60, alpha=0.5, color='red')
    btm_k_indices = indices[right:]
    pairs_2 = d1[btm_k_indices]
    #plt.hist(pairs, bins=60, alpha=0.5, color='blue')
    num_bins = 80
    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    n, bins, patches = ax[0].hist(d1, bins=num_bins, edgecolor='skyblue', color='white', hatch='/', label='distance_distribution')
    n_set, bins_set, patches_set = ax[0].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_10')
    n_set_2, bins_set_2, patches_set_2 = ax[0].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_10')
    ax[0].set_title('BERT Embedding Distance Distribution', fontsize=16)
    ax[0].set_xlabel('distance', fontsize=14)
    ax[0].set_ylabel('num', fontsize=14)
    ax[0].legend()
    
    pairs_1 = c1[top_k_indices]
    pairs_2 = c1[btm_k_indices]
    n, bins, patches = ax[1].hist(c1, bins=num_bins, edgecolor='orange', color='white', hatch='/', label='sasrec_distribution')
    n_set, bins_set, patches_set = ax[1].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_10', alpha=1)
    n_set_2, bins_set_2, patches_set_2 = ax[1].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_10', alpha=0.4)
    ax[1].set_title('BERT Top/bottom 10% in SASRec', fontsize=16)
    ax[1].set_xlabel('distance', fontsize=14)
    ax[1].set_ylabel('num', fontsize=14)
    ax[1].legend()
    plt.savefig('./figure/full_distance-pure-cos-'+str(i)+'.png', dpi=600)  # 保存图为png格式
    plt.show()
    plt.clf()
    print('done task 1-1')
    
    c1 = distance_rs_dot
    d1 = distance_llm
    pairs_1 = d1[top_k_indices]
    pairs_2 = d1[btm_k_indices]
    #plt.hist(pairs, bins=60, alpha=0.5, color='blue')
    num_bins = 80
    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    n, bins, patches = ax[0].hist(d1, bins=num_bins, edgecolor='skyblue', color='white', hatch='/', label='distance_distribution')
    n_set, bins_set, patches_set = ax[0].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_10')
    n_set_2, bins_set_2, patches_set_2 = ax[0].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_10')
    ax[0].set_title('BERT Embedding Distance Distribution', fontsize=16)
    ax[0].set_xlabel('distance', fontsize=14)
    ax[0].set_ylabel('num', fontsize=14)
    ax[0].legend()
    
    pairs_1 = c1[top_k_indices]
    pairs_2 = c1[btm_k_indices]
    n, bins, patches = ax[1].hist(c1, bins=num_bins, edgecolor='orange', color='white', hatch='/', label='sasrec_distribution')
    n_set, bins_set, patches_set = ax[1].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_10', alpha=1)
    n_set_2, bins_set_2, patches_set_2 = ax[1].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_10', alpha=0.4)
    ax[1].set_title('BERT Top/bottom 10% in SASRec', fontsize=16)
    ax[1].set_xlabel('distance', fontsize=14)
    ax[1].set_ylabel('num', fontsize=14)
    ax[1].legend()
    plt.savefig('./figure/full_distance-pure-dot-'+str(i)+'.png', dpi=600)  # 保存图为png格式
    plt.show()
    plt.clf()
    print('done task 1-2')
    
    
    
    
    
    
    
    id_emb = torch.load('./tensor/order_item_id_128.pt')
    content_emb = torch.load('./tensor/order_content_128.pt')
    id_emb = id_emb.detach().numpy()
    content_emb = content_emb.detach().numpy()
    distance_llm = np.array([])
    distance_rs_cos = np.array([])
    distance_rs_dot = np.array([])
    path = '/dataset/bce_text/id_plus_mo-2stage/tensor/full_cotrain_llm_cos_'+str(num_k)+'.dat'
    if os.path.exists(path):
        distance_llm = np.fromfile(path, dtype=np.half)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/full_cotrain_rs_cos_'+str(num_k)+'.dat'
        distance_rs_cos = np.fromfile(path, dtype=np.half)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/full_cotrain_rs_dot_'+str(num_k)+'.dat'
        distance_rs_dot = np.fromfile(path, dtype=np.half)
    else:
        for k in range(valid_id):
            term_1 = content_emb[users_history_for_valid[k]]
            #term_2 = content_emb[target[k]].reshape(1,-1)
            #dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            #distance_llm = np.concatenate((distance_llm,dis),0)
            a=pairwise_distances(term_1,metric="cosine")
            uptri_idx = np.triu_indices_from(a, k=1)
            dis=a[(uptri_idx)]
            distance_llm = np.concatenate((distance_llm,dis),0)
            
            term_1 = id_emb[users_history_for_valid[k]]
            #term_2 = id_emb[target[k]].reshape(1,-1)
            #dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            #distance_rs= np.concatenate((distance_rs,dis),0)
            a=pairwise_distances(term_1,metric="cosine")
            dis=a[(uptri_idx)]
            distance_rs_cos= np.concatenate((distance_rs_cos,dis),0)
            
            term_1 = id_emb[users_history_for_valid[k]]
            #term_2 = id_emb[target[k]].reshape(1,-1)
            #dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            #distance_rs= np.concatenate((distance_rs,dis),0)
            a = np.dot(term_1,term_1.T)
            dis=a[(uptri_idx)]
            distance_rs_dot= np.concatenate((distance_rs_dot,dis),0)
        distance_llm = np.half(distance_llm)
        distance_rs_cos = np.half(distance_rs_cos)
        distance_rs_dot = np.half(distance_rs_dot)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/full_cotrain_llm_cos_'+str(num_k)+'.dat'
        distance_llm.tofile(path)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/full_cotrain_rs_cos_'+str(num_k)+'.dat'
        distance_rs_cos.tofile(path)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/full_cotrain_rs_dot_'+str(num_k)+'.dat'
        distance_rs_dot.tofile(path)
        
    
    print((distance_llm.shape))
    #Kendallta,p_value = kendalltau(distance_rs,distance_llm)
    #print(Kendallta,p_value)
    c1 = distance_rs_cos
    d1 = distance_llm
    length = c1.shape[0]
    left = int(length/num)
    right = int(length/num*(num-1))
    indices = np.argsort(d1)[::-1]
    #plt.hist(d1, bins=60, alpha=0.5, color='grey')  # bins参数定义了直方图的柱子数量
    top_k_indices = indices[:left]
    pairs_1 = d1[top_k_indices]
    #plt.hist(pairs, bins=60, alpha=0.5, color='red')
    btm_k_indices = indices[right:]
    pairs_2 = d1[btm_k_indices]
    #plt.hist(pairs, bins=60, alpha=0.5, color='blue')
    num_bins = 80
    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    n, bins, patches = ax[0].hist(d1, bins=num_bins, edgecolor='skyblue', color='white', hatch='/', label='distance_distribution')
    n_set, bins_set, patches_set = ax[0].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_10')
    n_set_2, bins_set_2, patches_set_2 = ax[0].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_10')
    ax[0].set_title('BERT Embedding Distance Distribution', fontsize=16)
    ax[0].set_xlabel('distance', fontsize=14)
    ax[0].set_ylabel('num', fontsize=14)
    ax[0].legend()
    
    pairs_1 = c1[top_k_indices]
    pairs_2 = c1[btm_k_indices]
    n, bins, patches = ax[1].hist(c1, bins=num_bins, edgecolor='orange', color='white', hatch='/', label='sasrec_distribution')
    n_set, bins_set, patches_set = ax[1].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_10', alpha=1)
    n_set_2, bins_set_2, patches_set_2 = ax[1].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_10', alpha=0.4)
    ax[1].set_title('BERT Top/bottom 10% in SASRec', fontsize=16)
    ax[1].set_xlabel('distance', fontsize=14)
    ax[1].set_ylabel('num', fontsize=14)
    ax[1].legend()
    plt.savefig('./figure/full_distance-cotrain-cos-'+str(i)+'.png', dpi=600)  # 保存图为png格式
    plt.show()
    plt.clf()
    print('done task 2-1')
    
    c1 = distance_rs_dot
    d1 = distance_llm
    pairs_1 = d1[top_k_indices]
    pairs_2 = d1[btm_k_indices]
    #plt.hist(pairs, bins=60, alpha=0.5, color='blue')
    num_bins = 80
    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    n, bins, patches = ax[0].hist(d1, bins=num_bins, edgecolor='skyblue', color='white', hatch='/', label='distance_distribution')
    n_set, bins_set, patches_set = ax[0].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_10')
    n_set_2, bins_set_2, patches_set_2 = ax[0].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_10')
    ax[0].set_title('BERT Embedding Distance Distribution', fontsize=16)
    ax[0].set_xlabel('distance', fontsize=14)
    ax[0].set_ylabel('num', fontsize=14)
    ax[0].legend()
    
    pairs_1 = c1[top_k_indices]
    pairs_2 = c1[btm_k_indices]
    n, bins, patches = ax[1].hist(c1, bins=num_bins, edgecolor='orange', color='white', hatch='/', label='sasrec_distribution')
    n_set, bins_set, patches_set = ax[1].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_10', alpha=1)
    n_set_2, bins_set_2, patches_set_2 = ax[1].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_10', alpha=0.4)
    ax[1].set_title('BERT Top/bottom 10% in SASRec', fontsize=16)
    ax[1].set_xlabel('distance', fontsize=14)
    ax[1].set_ylabel('num', fontsize=14)
    ax[1].legend()
    plt.savefig('./figure/full_distance-cotrain-dot-'+str(i)+'.png', dpi=600)  # 保存图为png格式
    plt.show()
    plt.clf()
    print('done task 2-2')
    
    
    
    
    
    
     
    id_emb = torch.load('./tensor/order_mmd_item_id_128.pt')
    content_emb = torch.load('./tensor/order_mmd_content_128.pt')
    id_emb = id_emb.detach().numpy()
    content_emb = content_emb.detach().numpy()
    distance_llm = np.array([])
    distance_rs_cos = np.array([])
    distance_rs_dot = np.array([])
    path = '/dataset/bce_text/id_plus_mo-2stage/tensor/full_mmd_llm_cos_'+str(num_k)+'.dat'
    if os.path.exists(path):
        distance_llm = np.fromfile(path, dtype=np.half)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/full_mmd_rs_cos_'+str(num_k)+'.dat'
        distance_rs_cos = np.fromfile(path, dtype=np.half)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/full_mmd_rs_dot_'+str(num_k)+'.dat'
        distance_rs_dot = np.fromfile(path, dtype=np.half)
    else:
        for k in range(valid_id):
            term_1 = content_emb[users_history_for_valid[k]]
            #term_2 = content_emb[target[k]].reshape(1,-1)
            #dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            #distance_llm = np.concatenate((distance_llm,dis),0)
            a=pairwise_distances(term_1,metric="cosine")
            uptri_idx = np.triu_indices_from(a, k=1)
            dis=a[(uptri_idx)]
            distance_llm = np.concatenate((distance_llm,dis),0)
            
            term_1 = id_emb[users_history_for_valid[k]]
            #term_2 = id_emb[target[k]].reshape(1,-1)
            #dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            #distance_rs= np.concatenate((distance_rs,dis),0)
            a=pairwise_distances(term_1,metric="cosine")
            dis=a[(uptri_idx)]
            distance_rs_cos= np.concatenate((distance_rs_cos,dis),0)
            
            term_1 = id_emb[users_history_for_valid[k]]
            #term_2 = id_emb[target[k]].reshape(1,-1)
            #dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
            #distance_rs= np.concatenate((distance_rs,dis),0)
            a = np.dot(term_1,term_1.T)
            dis=a[(uptri_idx)]
            distance_rs_dot= np.concatenate((distance_rs_dot,dis),0)
        distance_llm = np.half(distance_llm)
        distance_rs_cos = np.half(distance_rs_cos)
        distance_rs_dot = np.half(distance_rs_dot)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/full_mmd_llm_cos_'+str(num_k)+'.dat'
        distance_llm.tofile(path)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/full_mmd_rs_cos_'+str(num_k)+'.dat'
        distance_rs_cos.tofile(path)
        path = '/dataset/bce_text/id_plus_mo-2stage/tensor/full_mmd_rs_dot_'+str(num_k)+'.dat'
        distance_rs_dot.tofile(path)
        
    
    print((distance_llm.shape))
    #Kendallta,p_value = kendalltau(distance_rs,distance_llm)
    #print(Kendallta,p_value)
    c1 = distance_rs_cos
    d1 = distance_llm
    length = c1.shape[0]
    left = int(length/num)
    right = int(length/num*(num-1))
    indices = np.argsort(d1)[::-1]
    #plt.hist(d1, bins=60, alpha=0.5, color='grey')  # bins参数定义了直方图的柱子数量
    top_k_indices = indices[:left]
    pairs_1 = d1[top_k_indices]
    #plt.hist(pairs, bins=60, alpha=0.5, color='red')
    btm_k_indices = indices[right:]
    pairs_2 = d1[btm_k_indices]
    #plt.hist(pairs, bins=60, alpha=0.5, color='blue')
    num_bins = 80
    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    n, bins, patches = ax[0].hist(d1, bins=num_bins, edgecolor='skyblue', color='white', hatch='/', label='distance_distribution')
    n_set, bins_set, patches_set = ax[0].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_10')
    n_set_2, bins_set_2, patches_set_2 = ax[0].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_10')
    ax[0].set_title('BERT Embedding Distance Distribution', fontsize=16)
    ax[0].set_xlabel('distance', fontsize=14)
    ax[0].set_ylabel('num', fontsize=14)
    ax[0].legend()
    
    pairs_1 = c1[top_k_indices]
    pairs_2 = c1[btm_k_indices]
    n, bins, patches = ax[1].hist(c1, bins=num_bins, edgecolor='orange', color='white', hatch='/', label='sasrec_distribution')
    n_set, bins_set, patches_set = ax[1].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_10', alpha=1)
    n_set_2, bins_set_2, patches_set_2 = ax[1].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_10', alpha=0.4)
    ax[1].set_title('BERT Top/bottom 10% in SASRec', fontsize=16)
    ax[1].set_xlabel('distance', fontsize=14)
    ax[1].set_ylabel('num', fontsize=14)
    ax[1].legend()
    plt.savefig('./figure/full_distance-mmd-cos-'+str(i)+'.png', dpi=600)  # 保存图为png格式
    plt.show()
    plt.clf()
    print('done task 3-1')
    
    c1 = distance_rs_dot
    d1 = distance_llm
    pairs_1 = d1[top_k_indices]
    pairs_2 = d1[btm_k_indices]
    #plt.hist(pairs, bins=60, alpha=0.5, color='blue')
    num_bins = 80
    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    n, bins, patches = ax[0].hist(d1, bins=num_bins, edgecolor='skyblue', color='white', hatch='/', label='distance_distribution')
    n_set, bins_set, patches_set = ax[0].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_10')
    n_set_2, bins_set_2, patches_set_2 = ax[0].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_10')
    ax[0].set_title('BERT Embedding Distance Distribution', fontsize=16)
    ax[0].set_xlabel('distance', fontsize=14)
    ax[0].set_ylabel('num', fontsize=14)
    ax[0].legend()
    
    pairs_1 = c1[top_k_indices]
    pairs_2 = c1[btm_k_indices]
    n, bins, patches = ax[1].hist(c1, bins=num_bins, edgecolor='orange', color='white', hatch='/', label='sasrec_distribution')
    n_set, bins_set, patches_set = ax[1].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_10', alpha=1)
    n_set_2, bins_set_2, patches_set_2 = ax[1].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_10', alpha=0.4)
    ax[1].set_title('BERT Top/bottom 10% in SASRec', fontsize=16)
    ax[1].set_xlabel('distance', fontsize=14)
    ax[1].set_ylabel('num', fontsize=14)
    ax[1].legend()
    plt.savefig('./figure/full_distance-mmd-dot-'+str(i)+'.png', dpi=600)  # 保存图为png格式
    plt.show()
    plt.clf()
    print('done task 3-2')

    return item_num, item_id_to_dic, users_train, users_valid, users_test, \
           users_history_for_valid, users_history_for_test, item_name_to_id

def read_behaviors_fre_2(behaviors_path, before_item_id_to_dic, before_item_name_to_id, before_item_id_to_name, max_seq_len, min_seq_len, Log_file,num_k):
    Log_file.info("##### news number {} {} (before clearing)#####".format(len(before_item_id_to_dic), len(before_item_name_to_id)))
    Log_file.info("##### min seq len {}, max seq len {}#####".format(min_seq_len, max_seq_len))
    before_item_num = len(before_item_name_to_id)
    before_item_counts = [0] * (before_item_num + 1)
    user_seq_dic = {}
    seq_num = 0
    pairs_num = 0
    Log_file.info('rebuild user seqs...')
    with open(behaviors_path, "r") as f:
        for line in f:
            splited = line.strip('\n').split('\t')
            user_name = splited[0]
            history_item_name = splited[1].split(' ')
            if len(history_item_name) < min_seq_len:
                continue
            history_item_name = history_item_name[-(max_seq_len+3):]
            item_ids_sub_seq = [before_item_name_to_id[i] for i in history_item_name]
            user_seq_dic[user_name] = item_ids_sub_seq
            for item_id in item_ids_sub_seq:
                before_item_counts[item_id] += 1
                pairs_num += 1
            seq_num += 1
    Log_file.info("##### pairs_num {}".format(pairs_num))

    item_id = 1
    item_id_to_dic = {}
    item_name_to_id = {}
    item_id_before_to_now = {}
    for before_item_id in range(1, before_item_num + 1):
        if before_item_counts[before_item_id] != 0:
            item_id_before_to_now[before_item_id] = item_id
            item_id_to_dic[item_id] = before_item_id_to_dic[before_item_id]
            item_name_to_id[before_item_id_to_name[before_item_id]] = item_id
            item_id += 1
    item_num = len(item_id_before_to_now)
    Log_file.info("##### items after clearing {}, {}, {} #####".format(item_num, len(item_id_before_to_now), len(item_id_to_dic)))

    users_train = {}
    users_valid = {}
    users_test = {}
    users_history_for_valid = {}
    users_history_for_test = {}
    user_id = 0
    ee=pd.read_csv('/dataset/fre.csv')
    ee.columns = ['id', 'fre']
    ee.drop(0,inplace=True)
    e=ee.sort_values('fre',ascending=False)
    eee=e[['id']]
    eee=list(eee['id'])
    num=10
    #l = [0]
    #for k in range(num):
    #    l.append(int(79707*(1+k)/num))
    l = [0, 41, 127, 276, 517, 889, 1467, 2413, 4070, 7618, 79707, 79707]
    i=num_k###################################
    valid_id=0
    
    target = []
    distance_llm = np.array([])
    distance_rs = np.array([])
    id_emb = torch.load('./tensor/order_pure_id_128.pt')
    content_emb = torch.load('./tensor/order_pure_content_128.pt')
    id_emb = id_emb.detach().numpy()
    content_emb = content_emb.detach().numpy()

    #id_emb = id_emb.detach().numpy()
    #content_emb = content_emb.detach().numpy()
    
    for user_name, item_seqs in user_seq_dic.items():
        user_seq = [item_id_before_to_now[i] for i in item_seqs]
        train = user_seq[:-1] ####################################################################
        users_train[user_id] = train
        
        if (eee.index(user_seq[-2])<l[i+1] and eee.index(user_seq[-2])>=l[i] and i<10) or (i==10):
            valid = user_seq[-(max_seq_len+2):-1]
            users_valid[valid_id] = valid
            users_history_for_valid[valid_id] = torch.LongTensor(np.array(train))
            valid_id+=1
            target.append(user_seq[-2])
        test = user_seq[-(max_seq_len+1):]
        users_test[user_id] = test
        users_history_for_test[user_id] = torch.LongTensor(np.array(user_seq[:-1]))
        user_id += 1
    Log_file.info("##### user seqs after clearing {}, {}, {}, {}, {}#####".
                  format(seq_num, len(user_seq_dic), len(users_train), len(users_valid), len(users_test)))
    
    for k in range(valid_id):
        term_1 = content_emb[users_history_for_valid[k]]
        #term_2 = content_emb[target[k]].reshape(1,-1)
        #print(term_1.shape)
        #print(term_2.shape)
        #dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
        #distance_llm = np.concatenate((distance_llm,dis),0)
        a=pairwise_distances(term_1,metric="cosine")
        uptri_idx = np.triu_indices_from(a, k=1)
        dis=a[(uptri_idx)]
        distance_llm = np.concatenate((distance_llm,dis),0)

        term_1 = id_emb[users_history_for_valid[k]]
        #term_2 = id_emb[target[k]].reshape(1,-1)
        #dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
        #distance_rs= np.concatenate((distance_rs,dis),0)
        #a=pairwise_distances(term_1,metric="cosine")
        a = np.dot(term_1,term_1.T)
        dis=a[(uptri_idx)]
        distance_rs= np.concatenate((distance_rs,dis),0)
        
    
    print((distance_llm.shape))
    #Kendallta,p_value = kendalltau(distance_rs,distance_llm)
    #print(Kendallta,p_value)
    c1 = distance_rs
    d1 = distance_llm
    length = c1.shape[0]
    left = int(length/num)
    right = int(length/num*(num-1))
    indices = np.argsort(d1)[::-1]
    #plt.hist(d1, bins=60, alpha=0.5, color='grey')  # bins参数定义了直方图的柱子数量
    top_k_indices = indices[:left]
    pairs_1 = d1[top_k_indices]
    #plt.hist(pairs, bins=60, alpha=0.5, color='red')
    btm_k_indices = indices[right:]
    pairs_2 = d1[btm_k_indices]
    #plt.hist(pairs, bins=60, alpha=0.5, color='blue')
    num_bins = 80
    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    n, bins, patches = ax[0].hist(d1, bins=num_bins, edgecolor='skyblue', color='white', hatch='/', label='distance_distribution')
    n_set, bins_set, patches_set = ax[0].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_10')
    n_set_2, bins_set_2, patches_set_2 = ax[0].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_10')
    ax[0].set_title('BERT Embedding Distance Distribution', fontsize=16)
    ax[0].set_xlabel('distance', fontsize=14)
    ax[0].set_ylabel('num', fontsize=14)
    ax[0].legend()
    
    pairs_1 = c1[top_k_indices]
    pairs_2 = c1[btm_k_indices]
    n, bins, patches = ax[1].hist(c1, bins=num_bins, edgecolor='orange', color='white', hatch='/', label='sasrec_distribution')
    n_set, bins_set, patches_set = ax[1].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_10', alpha=1)
    n_set_2, bins_set_2, patches_set_2 = ax[1].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_10', alpha=0.4)
    ax[1].set_title('BERT Top/bottom 10% in SASRec', fontsize=16)
    ax[1].set_xlabel('distance', fontsize=14)
    ax[1].set_ylabel('num', fontsize=14)
    ax[1].legend()
    plt.savefig('./figure/distance-pure-'+str(i)+'.png', dpi=600)  # 保存图为png格式
    plt.show()
    plt.clf()
    print('done task 1')
    
    id_emb = torch.load('./tensor/order_item_id_128.pt')
    content_emb = torch.load('./tensor/order_content_128.pt')
    id_emb = id_emb.detach().numpy()
    content_emb = content_emb.detach().numpy()
    distance_llm = np.array([])
    distance_rs = np.array([])
    for k in range(valid_id):
        term_1 = content_emb[users_history_for_valid[k]]
        #term_2 = content_emb[target[k]].reshape(1,-1)
        #print(term_1.shape)
        #print(term_2.shape)
        '''
        dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
        distance_llm = np.concatenate((distance_llm,dis),0)
        term_1 = id_emb[users_history_for_valid[k]]
        term_2 = id_emb[target[k]].reshape(1,-1)
        dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
        distance_rs= np.concatenate((distance_rs,dis),0)
        '''
        #dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
        #distance_llm = np.concatenate((distance_llm,dis),0)
        a=pairwise_distances(term_1,metric="cosine")
        uptri_idx = np.triu_indices_from(a, k=1)
        dis=a[(uptri_idx)]
        distance_llm = np.concatenate((distance_llm,dis),0)

        term_1 = id_emb[users_history_for_valid[k]]
        #term_2 = id_emb[target[k]].reshape(1,-1)
        #dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
        #distance_rs= np.concatenate((distance_rs,dis),0)
        #a=pairwise_distances(term_1,metric="cosine")
        a = np.dot(term_1,term_1.T)
        dis=a[(uptri_idx)]
        distance_rs= np.concatenate((distance_rs,dis),0)
    #print((distance_llm))
    #Kendallta,p_value = kendalltau(distance_rs,distance_llm)
    #print(Kendallta,p_value)
    c1 = distance_rs
    d1 = distance_llm
    length = c1.shape[0]
    left = int(length/num)
    right = int(length/num*(num-1))
    indices = np.argsort(d1)[::-1]
    #plt.hist(d1, bins=60, alpha=0.5, color='grey')  # bins参数定义了直方图的柱子数量
    top_k_indices = indices[:left]
    pairs_1 = d1[top_k_indices]
    #plt.hist(pairs, bins=60, alpha=0.5, color='red')
    btm_k_indices = indices[right:]
    pairs_2 = d1[btm_k_indices]
    #plt.hist(pairs, bins=60, alpha=0.5, color='blue')
    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    n, bins, patches = ax[0].hist(d1, bins=num_bins, edgecolor='skyblue', color='white', hatch='/', label='distance_distribution')
    n_set, bins_set, patches_set = ax[0].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_10')
    n_set_2, bins_set_2, patches_set_2 = ax[0].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_10')
    ax[0].set_title('BERT Embedding Distance Distribution', fontsize=16)
    ax[0].set_xlabel('distance', fontsize=14)
    ax[0].set_ylabel('num', fontsize=14)
    ax[0].legend()

    pairs_1 = c1[top_k_indices]
    pairs_2 = c1[btm_k_indices]
    n, bins, patches = ax[1].hist(c1, bins=num_bins, edgecolor='orange', color='white', hatch='/', label='sasrec_distribution')
    n_set, bins_set, patches_set = ax[1].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_10', alpha=1)
    n_set_2, bins_set_2, patches_set_2 = ax[1].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_10', alpha=0.4)
    ax[1].set_title('BERT Top/bottom 10% in SASRec', fontsize=16)
    ax[1].set_xlabel('distance', fontsize=14)
    ax[1].set_ylabel('num', fontsize=14)
    ax[1].legend()
    plt.savefig('./figure/distance-cotrain-'+str(i)+'.png', dpi=600)  # 保存图为png格式
    plt.show()
    plt.clf()
    print('done task 2')
     
    id_emb = torch.load('./tensor/order_mmd_item_id_128.pt')
    content_emb = torch.load('./tensor/order_mmd_content_128.pt')
    id_emb = id_emb.detach().numpy()
    content_emb = content_emb.detach().numpy()
    distance_llm = np.array([])
    distance_rs = np.array([])
    for k in range(valid_id):
        term_1 = content_emb[users_history_for_valid[k]]
        #term_2 = content_emb[target[k]].reshape(1,-1)
        #print(term_1.shape)
        #print(term_2.shape)
        '''
        dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
        distance_llm = np.concatenate((distance_llm,dis),0)
        term_1 = id_emb[users_history_for_valid[k]]
        term_2 = id_emb[target[k]].reshape(1,-1)
        dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
        distance_rs= np.concatenate((distance_rs,dis),0)
        '''
        #dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
        #distance_llm = np.concatenate((distance_llm,dis),0)
        a=pairwise_distances(term_1,metric="cosine")
        uptri_idx = np.triu_indices_from(a, k=1)
        dis=a[(uptri_idx)]
        distance_llm = np.concatenate((distance_llm,dis),0)

        term_1 = id_emb[users_history_for_valid[k]]
        #term_2 = id_emb[target[k]].reshape(1,-1)
        #dis = pairwise_distances(term_1, term_2, metric='cosine').reshape(term_1.shape[0])
        #distance_rs= np.concatenate((distance_rs,dis),0)
        #a=pairwise_distances(term_1,metric="cosine")
        a = np.dot(term_1,term_1.T)
        dis=a[(uptri_idx)]
        distance_rs= np.concatenate((distance_rs,dis),0)

    #print((distance_llm))
    #Kendallta,p_value = kendalltau(distance_rs,distance_llm)
    #print(Kendallta,p_value)
    c1 = distance_rs
    d1 = distance_llm
    length = c1.shape[0]
    left = int(length/num)
    right = int(length/num*(num-1))
    indices = np.argsort(d1)[::-1]
    #plt.hist(d1, bins=60, alpha=0.5, color='grey')  # bins参数定义了直方图的柱子数量
    top_k_indices = indices[:left]
    pairs_1 = d1[top_k_indices]
    #plt.hist(pairs, bins=60, alpha=0.5, color='red')
    btm_k_indices = indices[right:]
    pairs_2 = d1[btm_k_indices]
    #plt.hist(pairs, bins=60, alpha=0.5, color='blue')
    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    n, bins, patches = ax[0].hist(d1, bins=num_bins, edgecolor='skyblue', color='white', hatch='/', label='distance_distribution')
    n_set, bins_set, patches_set = ax[0].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_10')
    n_set_2, bins_set_2, patches_set_2 = ax[0].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_10')
    ax[0].set_title('BERT Embedding Distance Distribution', fontsize=16)
    ax[0].set_xlabel('distance', fontsize=14)
    ax[0].set_ylabel('num', fontsize=14)
    ax[0].legend()

    pairs_1 = c1[top_k_indices]
    pairs_2 = c1[btm_k_indices]
    n, bins, patches = ax[1].hist(c1, bins=num_bins, edgecolor='orange', color='white', hatch='/', label='sasrec_distribution')
    n_set, bins_set, patches_set = ax[1].hist(pairs_1, bins=bins, edgecolor='black', color='pink', label='top_10', alpha=1)
    n_set_2, bins_set_2, patches_set_2 = ax[1].hist(pairs_2, bins=bins, edgecolor='black', color='lightgreen', label='bottom_10', alpha=0.4)
    ax[1].set_title('BERT Top/bottom 10% in SASRec', fontsize=16)
    ax[1].set_xlabel('distance', fontsize=14)
    ax[1].set_ylabel('num', fontsize=14)
    ax[1].legend()
    plt.savefig('./figure/distance-mmd-'+str(i)+'.png', dpi=600)  # 保存图为png格式
    plt.show()
    plt.clf()
    print('done task 3')

    return item_num, item_id_to_dic, users_train, users_valid, users_test, \
           users_history_for_valid, users_history_for_test, item_name_to_id

def read_behaviors_fre(behaviors_path, before_item_id_to_dic, before_item_name_to_id, before_item_id_to_name, max_seq_len, min_seq_len, Log_file,num_k):
    Log_file.info("##### news number {} {} (before clearing)#####".format(len(before_item_id_to_dic), len(before_item_name_to_id)))
    Log_file.info("##### min seq len {}, max seq len {}#####".format(min_seq_len, max_seq_len))
    before_item_num = len(before_item_name_to_id)
    before_item_counts = [0] * (before_item_num + 1)
    user_seq_dic = {}
    seq_num = 0
    pairs_num = 0
    Log_file.info('rebuild user seqs...')
    with open(behaviors_path, "r") as f:
        for line in f:
            splited = line.strip('\n').split('\t')
            user_name = splited[0]
            history_item_name = splited[1].split(' ')
            if len(history_item_name) < min_seq_len:
                continue
            history_item_name = history_item_name[-(max_seq_len+3):]
            item_ids_sub_seq = [before_item_name_to_id[i] for i in history_item_name]
            user_seq_dic[user_name] = item_ids_sub_seq
            for item_id in item_ids_sub_seq:
                before_item_counts[item_id] += 1
                pairs_num += 1
            seq_num += 1
    Log_file.info("##### pairs_num {}".format(pairs_num))

    item_id = 1
    item_id_to_dic = {}
    item_name_to_id = {}
    item_id_before_to_now = {}
    for before_item_id in range(1, before_item_num + 1):
        if before_item_counts[before_item_id] != 0:
            item_id_before_to_now[before_item_id] = item_id
            item_id_to_dic[item_id] = before_item_id_to_dic[before_item_id]
            item_name_to_id[before_item_id_to_name[before_item_id]] = item_id
            item_id += 1
    item_num = len(item_id_before_to_now)
    Log_file.info("##### items after clearing {}, {}, {} #####".format(item_num, len(item_id_before_to_now), len(item_id_to_dic)))

    users_train = {}
    users_valid = {}
    users_test = {}
    users_history_for_valid = {}
    users_history_for_test = {}
    user_id = 0
    ee=pd.read_csv('/dataset/fre.csv')
    ee.columns = ['id', 'fre']
    ee.drop(0,inplace=True)
    e=ee.sort_values('fre',ascending=False)
    eee=e[['id']]
    eee=list(eee['id'])
    num=10
    #l = [0]
    #for k in range(num):
    #    l.append(int(79707*(1+k)/num))
    l = [0, 41, 127, 276, 517, 889, 1467, 2413, 4070, 7618, 79707, 79707]
    i=num_k###################################
    valid_id=0
    for user_name, item_seqs in user_seq_dic.items():
        user_seq = [item_id_before_to_now[i] for i in item_seqs]
        train = user_seq[:-2]
        users_train[user_id] = train
        
        if eee.index(user_seq[-2])<l[i+1] and eee.index(user_seq[-2])>=l[i]:
            valid = user_seq[-(max_seq_len+2):-1]
            users_valid[valid_id] = valid
            users_history_for_valid[valid_id] = torch.LongTensor(np.array(train))
            valid_id+=1
        test = user_seq[-(max_seq_len+1):]
        users_test[user_id] = test
        users_history_for_test[user_id] = torch.LongTensor(np.array(user_seq[:-1]))
        user_id += 1
    Log_file.info("##### user seqs after clearing {}, {}, {}, {}, {}#####".
                  format(seq_num, len(user_seq_dic), len(users_train), len(users_valid), len(users_test)))
    return item_num, item_id_to_dic, users_train, users_valid, users_test, \
           users_history_for_valid, users_history_for_test, item_name_to_id

def read_behaviors(behaviors_path, before_item_id_to_dic, before_item_name_to_id, before_item_id_to_name, max_seq_len, min_seq_len, Log_file):
    Log_file.info("##### news number {} {} (before clearing)#####".format(len(before_item_id_to_dic), len(before_item_name_to_id)))
    Log_file.info("##### min seq len {}, max seq len {}#####".format(min_seq_len, max_seq_len))
    before_item_num = len(before_item_name_to_id)
    before_item_counts = [0] * (before_item_num + 1)
    user_seq_dic = {}
    seq_num = 0
    pairs_num = 0
    Log_file.info('rebuild user seqs...')
    with open(behaviors_path, "r") as f:
        for line in f:
            splited = line.strip('\n').split('\t')
            user_name = splited[0]
            history_item_name = splited[1].split(' ')
            if len(history_item_name) < min_seq_len:
                continue
            history_item_name = history_item_name[-(max_seq_len+3):]
            item_ids_sub_seq = [before_item_name_to_id[i] for i in history_item_name]
            user_seq_dic[user_name] = item_ids_sub_seq
            for item_id in item_ids_sub_seq:
                before_item_counts[item_id] += 1
                pairs_num += 1
            seq_num += 1
    Log_file.info("##### pairs_num {}".format(pairs_num))

    item_id = 1
    item_id_to_dic = {}
    item_name_to_id = {}
    item_id_before_to_now = {}
    for before_item_id in range(1, before_item_num + 1):
        if before_item_counts[before_item_id] != 0:
            item_id_before_to_now[before_item_id] = item_id
            item_id_to_dic[item_id] = before_item_id_to_dic[before_item_id]
            item_name_to_id[before_item_id_to_name[before_item_id]] = item_id
            item_id += 1
    item_num = len(item_id_before_to_now)
    Log_file.info("##### items after clearing {}, {}, {} #####".format(item_num, len(item_id_before_to_now), len(item_id_to_dic)))

    users_train = {}
    users_valid = {}
    users_test = {}
    users_history_for_valid = {}
    users_history_for_test = {}
    user_id = 0
    for user_name, item_seqs in user_seq_dic.items():
        user_seq = [item_id_before_to_now[i] for i in item_seqs]
        train = user_seq[:-2]
        valid = user_seq[-(max_seq_len+2):-1]
        test = user_seq[-(max_seq_len+1):]
        users_train[user_id] = train
        users_valid[user_id] = valid
        users_test[user_id] = test

        users_history_for_valid[user_id] = torch.LongTensor(np.array(train))
        users_history_for_test[user_id] = torch.LongTensor(np.array(user_seq[:-1]))
        user_id += 1
    Log_file.info("##### user seqs after clearing {}, {}, {}, {}, {}#####".
                  format(seq_num, len(user_seq_dic), len(users_train), len(users_valid), len(users_test)))
    return item_num, item_id_to_dic, users_train, users_valid, users_test, \
           users_history_for_valid, users_history_for_test, item_name_to_id


def read_news(news_path):
    item_id_to_dic = {}
    item_id_to_name = {}
    item_name_to_id = {}
    item_id = 1
    with open(news_path, "r") as f:
        for line in f:
            splited = line.strip('\n').split('\t')
            doc_name, _, _ = splited
            item_name_to_id[doc_name] = item_id
            item_id_to_dic[item_id] = doc_name
            item_id_to_name[item_id] = doc_name
            item_id += 1
    return item_id_to_dic, item_name_to_id, item_id_to_name

def read_news_bert_new(news_path, args, tokenizer):
    item_id_to_dic = {}
    item_id_to_name = {}
    item_name_to_id = {}
    #item_id_to_text = {}
    item_id = 1
    #item_id_to_text[0]='empty'
    
    import torch
    from llm2vec import LLM2Vec

    import os
    #os.environ["TOKENIZERS_PARALLELISM"] = "false"
    l2v = LLM2Vec.from_pretrained(
        "/llama3-8B",
        peft_model_name_or_path="/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
        device_map="cpu" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.bfloat16,
    )

    #item_dataset = ItemsDataset(data=item_content)
    #item_dataloader = DataLoader(item_dataset, batch_size=test_batch_size, num_workers=args.num_workers,
    #                             pin_memory=True, collate_fn=item_collate_fn)
    item_word_embs = []
    
    #item_feature = l2v.encode(input_ids)
    #item_word_embs.extend(item_feature)
    #return torch.stack(tensors=item_word_embs, dim=0)
    with open(news_path, "r") as f:
        for line in f:
            splited = line.strip('\n').split('\t')
            doc_name, title1, abstract = splited
            '''
            #print(doc_name, title, abstract)
            if 'title' in args.news_attributes:
                title = tokenizer(title1.lower(), max_length=args.num_words_title, padding='max_length', truncation=True)
            else:
                title = []

            if 'abstract' in args.news_attributes:
                abstract = tokenizer(abstract.lower(), max_length=args.num_words_abstract, padding='max_length', truncation=True)
            else:
                abstract = []

            if 'body' in args.news_attributes:
                body = tokenizer(body.lower()[:2000], max_length=args.num_words_body, padding='max_length', truncation=True)
            else:
                body = []
            item_name_to_id[doc_name] = item_id
            item_id_to_name[item_id] = doc_name
            item_id_to_dic[item_id] = [title, abstract, body]
            '''
            item_feature = l2v.encode(title1)
            item_word_embs.extend(item_feature)
            item_id += 1
            if item_id>5:
               break
    a = torch.stack(tensors=item_word_embs, dim=0)
    print(a.shape)
    return item_id_to_dic, item_name_to_id, item_id_to_name, a

def read_news_bert(news_path, args, tokenizer):
    item_id_to_dic = {}
    item_id_to_name = {}
    item_name_to_id = {}
    item_id = 1
    with open(news_path, "r") as f:
        for line in f:
            splited = line.strip('\n').split('\t')
            doc_name, title, abstract = splited
            #print(doc_name, title, abstract)
            if 'title' in args.news_attributes:
                title = tokenizer(title.lower(), max_length=args.num_words_title, padding='max_length', truncation=True)
            else:
                title = []

            if 'abstract' in args.news_attributes:
                abstract = tokenizer(abstract.lower(), max_length=args.num_words_abstract, padding='max_length', truncation=True)
            else:
                abstract = []

            if 'body' in args.news_attributes:
                body = tokenizer(body.lower()[:2000], max_length=args.num_words_body, padding='max_length', truncation=True)
            else:
                body = []
            item_name_to_id[doc_name] = item_id
            item_id_to_name[item_id] = doc_name
            item_id_to_dic[item_id] = [title, abstract, body]
            item_id += 1
    return item_id_to_dic, item_name_to_id, item_id_to_name


def get_doc_input_bert(item_id_to_content, args):
    item_num = len(item_id_to_content) + 1

    if 'title' in args.news_attributes:
        news_title = np.zeros((item_num, args.num_words_title), dtype='int32')
        news_title_attmask = np.zeros((item_num, args.num_words_title), dtype='int32')
    else:
        news_title = None
        news_title_attmask = None

    if 'abstract' in args.news_attributes:
        news_abstract = np.zeros((item_num, args.num_words_abstract), dtype='int32')
        news_abstract_attmask = np.zeros((item_num, args.num_words_abstract), dtype='int32')
    else:
        news_abstract = None
        news_abstract_attmask = None

    if 'body' in args.news_attributes:
        news_body = np.zeros((item_num, args.num_words_body), dtype='int32')
        news_body_attmask = np.zeros((item_num, args.num_words_body), dtype='int32')
    else:
        news_body = None
        news_body_attmask = None

    for item_id in range(1, item_num):
        title, abstract, body = item_id_to_content[item_id]

        if 'title' in args.news_attributes:
            news_title[item_id] = title['input_ids']
            news_title_attmask[item_id] = title['attention_mask']

        if 'abstract' in args.news_attributes:
            news_abstract[item_id] = abstract['input_ids']
            news_abstract_attmask[item_id] = abstract['attention_mask']

        if 'body' in args.news_attributes:
            news_body[item_id] = body['input_ids']
            news_body_attmask[item_id] = body['attention_mask']

    return news_title, news_title_attmask, \
        news_abstract, news_abstract_attmask, \
        news_body, news_body_attmask


