import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import math
import random

class BuildTrainDataset_kl(Dataset): 
    def __init__(self, u2seq, item_content,
                 item_num, max_seq_len, use_modal):
        self.u2seq = u2seq
        self.item_content = item_content
        self.item_num = item_num
        self.max_seq_len = max_seq_len + 1
        self.use_modal = use_modal
        ee=pd.read_csv('/dataset/fre.csv')
        ee.columns = ['id', 'fre']
        e=ee.sort_values('fre',ascending=False)
        l = [0, 41, 127, 276, 517, 889, 1467, 2413, 4070, 7618, 79707, 79708]
        e['bin']=0
        for k in range(10):
            e.iloc[l[k]:l[k+1],2]=k+1
        self.bins=e['bin'].to_numpy()

    def __len__(self):
        return len(self.u2seq)

    def __getitem__(self, user_id):
        seq = self.u2seq[user_id]
        seq_Len = len(seq)
        tokens_Len = seq_Len - 1
        mask_len_head = self.max_seq_len - seq_Len
        log_mask = [0] * mask_len_head + [1] * tokens_Len

        sample_items = []
        padding_seq = [0] * mask_len_head + seq
        sample_items.append(padding_seq)
        #print(sample_items)
        #bin_pos=self.bins[padding_seq[1:]]
        neg_items = []
        for i in range(tokens_Len):
            sam_neg = random.randint(1, self.item_num)
            while sam_neg in seq:
                sam_neg = random.randint(1, self.item_num)
            neg_items.append(sam_neg)
        neg_items = [0] * mask_len_head + neg_items + [0]
        #bin_neg=self.bins[neg_items[:-1]]
        #print(len(sample_items[1:]))
        #print(neg_items)
        #print(bin_neg.shape)
        #print('')
        sample_items.append(neg_items)
        bins=self.bins[sample_items]
        sample_items_id = torch.LongTensor(np.array(sample_items)).transpose(0, 1)
 
        if self.use_modal:
            sample_items_content = self.item_content[sample_items_id]
            return torch.LongTensor(sample_items_id), sample_items_content, \
                   torch.FloatTensor(log_mask), torch.LongTensor(bins)
        else:
            return torch.LongTensor(sample_items_id), sample_items_id, \
                   torch.FloatTensor(log_mask)
 

class BuildTrainDataset_ablation(Dataset): 
    def __init__(self, u2seq, item_content,
                 item_num, max_seq_len, use_modal):
        self.u2seq = u2seq
        self.item_content = item_content
        self.item_num = item_num
        self.max_seq_len = max_seq_len + 1
        self.use_modal = use_modal
        ee=pd.read_csv('/dataset/fre.csv')
        ee.columns = ['id', 'fre']
        e=ee.sort_values('fre',ascending=False)
        l = [0, 41, 127, 276, 517, 889, 1467, 2413, 4070, 7618, 79707, 79708]
        e['bin']=0
        #for k in range(10):
        #    e.iloc[l[k]:l[k+1],2]=k+1
        #e=e.sort_values('id',ascending=True) ##################
        self.bins=e['bin'].to_numpy()

    def __len__(self):
        return len(self.u2seq)

    def __getitem__(self, user_id):
        seq = self.u2seq[user_id]
        seq_Len = len(seq)
        tokens_Len = seq_Len - 1
        mask_len_head = self.max_seq_len - seq_Len
        log_mask = [0] * mask_len_head + [1] * tokens_Len

        sample_items = []
        padding_seq = [0] * mask_len_head + seq
        sample_items.append(padding_seq)
        #print(sample_items)
        bin_pos=self.bins[padding_seq[1:]]
        neg_items = []
        for i in range(tokens_Len):
            sam_neg = random.randint(1, self.item_num)
            while sam_neg in seq:
                sam_neg = random.randint(1, self.item_num)
            neg_items.append(sam_neg)
        neg_items = [0] * mask_len_head + neg_items + [0]
        bin_neg=self.bins[neg_items[:-1]]
        #print(len(sample_items[1:]))
        #print(neg_items)
        #print(bin_neg.shape)
        #print('')
        sample_items.append(neg_items)
        sample_items_id = torch.LongTensor(np.array(sample_items)).transpose(0, 1)

        if self.use_modal:
            sample_items_content = self.item_content[sample_items_id]
            return torch.LongTensor(sample_items_id), sample_items_content, \
                   torch.FloatTensor(log_mask), torch.LongTensor(bin_pos), torch.LongTensor(bin_neg)
        else:
            return torch.LongTensor(sample_items_id), sample_items_id, \
                   torch.FloatTensor(log_mask)
 

class BuildTrainDataset_new(Dataset): 
    def __init__(self, u2seq, item_content,
                 item_num, max_seq_len, use_modal):
        self.u2seq = u2seq
        self.item_content = item_content
        self.item_num = item_num
        self.max_seq_len = max_seq_len + 1
        self.use_modal = use_modal
        ee=pd.read_csv('/dataset/fre.csv')
        ee.columns = ['id', 'fre']
        e=ee.sort_values('fre',ascending=False)
        l = [0, 41, 127, 276, 517, 889, 1467, 2413, 4070, 7618, 79707, 79708]
        e['bin']=0
        for k in range(10):
            e.iloc[l[k]:l[k+1],2]=k+1
        e=e.sort_values('id',ascending=True) ##################
        self.bins=e['bin'].to_numpy()

    def __len__(self):
        return len(self.u2seq)

    def __getitem__(self, user_id):
        seq = self.u2seq[user_id]
        seq_Len = len(seq)
        tokens_Len = seq_Len - 1
        mask_len_head = self.max_seq_len - seq_Len
        log_mask = [0] * mask_len_head + [1] * tokens_Len

        sample_items = []
        padding_seq = [0] * mask_len_head + seq
        sample_items.append(padding_seq)
        #print(sample_items)
        bin_pos=self.bins[padding_seq[1:]]
        neg_items = []
        for i in range(tokens_Len):
            sam_neg = random.randint(1, self.item_num)
            while sam_neg in seq:
                sam_neg = random.randint(1, self.item_num)
            neg_items.append(sam_neg)
        neg_items = [0] * mask_len_head + neg_items + [0]
        bin_neg=self.bins[neg_items[:-1]]
        #print(len(sample_items[1:]))
        #print(neg_items)
        #print(bin_neg.shape)
        #print('')
        sample_items.append(neg_items)
        sample_items_id = torch.LongTensor(np.array(sample_items)).transpose(0, 1)

        if self.use_modal:
            sample_items_content = self.item_content[sample_items_id]
            return torch.LongTensor(sample_items_id), sample_items_content, \
                   torch.FloatTensor(log_mask), torch.LongTensor(bin_pos), torch.LongTensor(bin_neg)
        else:
            return torch.LongTensor(sample_items_id), sample_items_id, \
                   torch.FloatTensor(log_mask)
 
class BuildTrainDataset_new_amazon_ele(Dataset): 
    def __init__(self, u2seq, item_content,
                 item_num, max_seq_len, use_modal):
        self.u2seq = u2seq
        self.item_content = item_content
        self.item_num = item_num
        self.max_seq_len = max_seq_len + 1
        self.use_modal = use_modal
        ee=pd.read_csv('/dataset/Electronics_fre.csv')
        #ee.columns = ['id', 'fre']
        e=ee.sort_values('fre',ascending=False)
        #l = [0, 41, 127, 276, 517, 889, 1467, 2413, 4070, 7618, 79707, 79708]
        l=[0, 233, 856, 2090, 4327, 8214, 14950, 27012, 50742, 109145, 423192]
        e['bin']=0
        for k in range(10):
            e.iloc[l[k]:l[k+1],2]=k+1
        e=e.sort_values('id',ascending=True) ##################
        self.bins=e['bin'].to_numpy()
        self.bins[0]=0

    def __len__(self):
        return len(self.u2seq)

    def __getitem__(self, user_id):
        seq = self.u2seq[user_id]
        seq_Len = len(seq)
        tokens_Len = seq_Len - 1
        mask_len_head = self.max_seq_len - seq_Len
        log_mask = [0] * mask_len_head + [1] * tokens_Len

        sample_items = []
        padding_seq = [0] * mask_len_head + seq
        sample_items.append(padding_seq)
        #print(sample_items)
        bin_pos=self.bins[padding_seq[1:]]
        neg_items = []
        for i in range(tokens_Len):
            sam_neg = random.randint(1, self.item_num)
            while sam_neg in seq:
                sam_neg = random.randint(1, self.item_num)
            neg_items.append(sam_neg)
        neg_items = [0] * mask_len_head + neg_items + [0]
        bin_neg=self.bins[neg_items[:-1]]
        #print(len(sample_items[1:]))
        #print(neg_items)
        #print(bin_neg.shape)
        #print('')
        sample_items.append(neg_items)
        sample_items_id = torch.LongTensor(np.array(sample_items)).transpose(0, 1)

        if self.use_modal:
            sample_items_content = self.item_content[sample_items_id]
            return torch.LongTensor(sample_items_id), sample_items_content, \
                   torch.FloatTensor(log_mask), torch.LongTensor(bin_pos), torch.LongTensor(bin_neg)
        else:
            return torch.LongTensor(sample_items_id), sample_items_id, \
                   torch.FloatTensor(log_mask)
 
class BuildTrainDataset_new_amazon_pantry(Dataset): 
    def __init__(self, u2seq, item_content,
                 item_num, max_seq_len, use_modal):
        self.u2seq = u2seq
        self.item_content = item_content
        self.item_num = item_num
        self.max_seq_len = max_seq_len + 1
        self.use_modal = use_modal
        ee=pd.read_csv('/dataset/Prime_Pantry_fre.csv')
        #ee.columns = ['id', 'fre']
        e=ee.sort_values('fre',ascending=False)
        #l = [0, 41, 127, 276, 517, 889, 1467, 2413, 4070, 7618, 79707, 79708]
        #l=[0, 233, 856, 2090, 4327, 8214, 14950, 27012, 50742, 109145, 423192]
        l=[0, 35, 100, 205, 361, 585, 898, 1357, 2074, 3348, 8348]
        e['bin']=0
        for k in range(10):
            e.iloc[l[k]:l[k+1],2]=k+1
        e=e.sort_values('id',ascending=True) ##################
        self.bins=e['bin'].to_numpy()
        self.bins[0]=0
    
    def __len__(self):
        return len(self.u2seq)

    def __getitem__(self, user_id):
        seq = self.u2seq[user_id]
        seq_Len = len(seq)
        tokens_Len = seq_Len - 1
        mask_len_head = self.max_seq_len - seq_Len
        log_mask = [0] * mask_len_head + [1] * tokens_Len

        sample_items = []
        padding_seq = [0] * mask_len_head + seq
        sample_items.append(padding_seq)
        #print(sample_items)
        bin_pos=self.bins[padding_seq[1:]]
        neg_items = []
        for i in range(tokens_Len):
            sam_neg = random.randint(1, self.item_num)
            while sam_neg in seq:
                sam_neg = random.randint(1, self.item_num)
            neg_items.append(sam_neg)
        neg_items = [0] * mask_len_head + neg_items + [0]
        bin_neg=self.bins[neg_items[:-1]]
        #print(len(sample_items[1:]))
        #print(neg_items)
        #print(bin_neg.shape)
        #print('')
        sample_items.append(neg_items)
        sample_items_id = torch.LongTensor(np.array(sample_items)).transpose(0, 1)

        if self.use_modal:
            sample_items_content = self.item_content[sample_items_id]
            return torch.LongTensor(sample_items_id), sample_items_content, \
                   torch.FloatTensor(log_mask), torch.LongTensor(bin_pos), torch.LongTensor(bin_neg)
        else:
            return torch.LongTensor(sample_items_id), sample_items_id, \
                   torch.FloatTensor(log_mask)
 

class BuildTrainDataset_modified(Dataset):
    def __init__(self, u2seq, item_content,
                 item_num, max_seq_len, use_modal):
        self.u2seq = u2seq
        self.item_content = item_content
        self.item_num = item_num
        self.max_seq_len = max_seq_len + 1
        self.use_modal = use_modal

    def __len__(self):
        return len(self.u2seq)

    def __getitem__(self, user_id):
        seq = self.u2seq[user_id]
        seq_Len = len(seq)
        tokens_Len = seq_Len - 1
        mask_len_head = self.max_seq_len - seq_Len
        log_mask = [0] * mask_len_head + [1] * tokens_Len

        sample_items = []
        modified = []
        padding_seq = [0] * mask_len_head + seq
        sample_items.append(padding_seq)
        modified.append(seq)
        neg_items = []
        for i in range(tokens_Len):
            sam_neg = random.randint(1, self.item_num)
            while sam_neg in seq:
                sam_neg = random.randint(1, self.item_num)
            neg_items.append(sam_neg)
        neg_items = neg_items + [0]
        modified.append(neg_items)
        neg_items = [0] * mask_len_head + neg_items
        sample_items.append(neg_items)
        
        sample_items_id = torch.LongTensor(np.array(sample_items)).transpose(0, 1)
        modified= torch.LongTensor(np.array(modified)).transpose(0, 1)
        if self.use_modal:
            sample_items_content = self.item_content[sample_items_id]
            content_modified = self.item_content[modified]
            return torch.LongTensor(sample_items_id), sample_items_content, \
                   torch.FloatTensor(log_mask), torch.LongTensor(modified), content_modified
        else:
            return torch.LongTensor(sample_items_id), sample_items_id, \
                   torch.FloatTensor(log_mask)

class BuildTrainDataset(Dataset):
    def __init__(self, u2seq, item_content,
                 item_num, max_seq_len, use_modal):
        self.u2seq = u2seq
        self.item_content = item_content
        self.item_num = item_num
        self.max_seq_len = max_seq_len + 1
        self.use_modal = use_modal

    def __len__(self):
        return len(self.u2seq)

    def __getitem__(self, user_id):
        seq = self.u2seq[user_id]
        seq_Len = len(seq)
        tokens_Len = seq_Len - 1
        mask_len_head = self.max_seq_len - seq_Len
        log_mask = [0] * mask_len_head + [1] * tokens_Len

        sample_items = []
        padding_seq = [0] * mask_len_head + seq
        sample_items.append(padding_seq)
        neg_items = []
        for i in range(tokens_Len):
            sam_neg = random.randint(1, self.item_num)
            while sam_neg in seq:
                sam_neg = random.randint(1, self.item_num)
            neg_items.append(sam_neg)
        neg_items = [0] * mask_len_head + neg_items + [0]
        sample_items.append(neg_items)
        sample_items_id = torch.LongTensor(np.array(sample_items)).transpose(0, 1)

        if self.use_modal:
            sample_items_content = self.item_content[sample_items_id]
            return torch.LongTensor(sample_items_id), sample_items_content, \
                   torch.FloatTensor(log_mask)
        else:
            return torch.LongTensor(sample_items_id), sample_items_id, \
                   torch.FloatTensor(log_mask)
        
class BuildTrainDataset2(Dataset):
    def __init__(self, u2seq, item_content,
                 item_num, max_seq_len, use_modal):
        self.u2seq = u2seq
        self.item_content = item_content
        self.item_num = item_num
        self.max_seq_len = max_seq_len + 1
        self.use_modal = use_modal
        
        self.ee=pd.read_csv('/dataset/fre.csv')
        #self.ee.columns = ['id', 'fre']
        e=self.ee['fre']
        self.q1=torch.tensor(((e<=6)&(e>0))).detach().numpy()
        self.q2=torch.tensor(((e>6)&(e<=50))).detach().numpy()
        self.q3=torch.tensor((e>50)).detach().numpy()
        

    def __len__(self):
        return len(self.u2seq)

    def __getitem__(self, user_id):
        seq = self.u2seq[user_id]
        seq_Len = len(seq)
        tokens_Len = seq_Len - 1
        mask_len_head = self.max_seq_len - seq_Len
        log_mask = [0] * mask_len_head + [1] * tokens_Len

        sample_items = []
        padding_seq = [0] * mask_len_head + seq
        sample_items.append(padding_seq)
        neg_items = []
        for i in range(tokens_Len):
            sam_neg = random.randint(1, self.item_num)
            while sam_neg in seq:
                sam_neg = random.randint(1, self.item_num)
            neg_items.append(sam_neg)
        neg_items = [0] * mask_len_head + neg_items + [0]
        sample_items.append(neg_items)
        
        q1=self.q1[sample_items]
        q2=self.q2[sample_items]
        q3=self.q3[sample_items]
        
        q1=list(map(list, zip(*q1)))
        q2=list(map(list, zip(*q2)))
        q3=list(map(list, zip(*q3)))       
        sample_items_id = torch.LongTensor(np.array(sample_items)).transpose(0, 1)
        #print(sample_items_id.shape)
        #print(torch.LongTensor(sample_items_id).shape)
        if self.use_modal:
            sample_items_content = self.item_content[sample_items_id]
            return torch.LongTensor(sample_items_id), sample_items_content, \
                   torch.FloatTensor(log_mask),torch.LongTensor(q1),torch.LongTensor(q2),torch.LongTensor(q3)
        else:
            return torch.LongTensor(sample_items_id), sample_items_id, \
                   torch.FloatTensor(log_mask)


class BuildEvalDataset(Dataset):
    def __init__(self, u2seq, item_content, max_seq_len, item_num):
        self.u2seq = u2seq
        self.item_content = item_content
        self.max_seq_len = max_seq_len + 1
        self.item_num = item_num

    def __len__(self):
        return len(self.u2seq)

    def __getitem__(self, user_id):
        seq = self.u2seq[user_id]
        tokens = seq[:-1]
        target = seq[-1]
        mask_len = self.max_seq_len - len(seq)
        pad_tokens = [0] * mask_len + tokens
        log_mask = [0] * mask_len + [1] * len(tokens)
        input_embs = self.item_content[pad_tokens]
        labels = np.zeros(self.item_num)
        labels[target - 1] = 1.0
        return torch.LongTensor([user_id]), \
            input_embs, \
            torch.FloatTensor(log_mask), \
            labels

class BuildEvalDataset_3(Dataset):
    def __init__(self, u2seq, item_content,item_content3,id_embs, max_seq_len, item_num):
        self.u2seq = u2seq
        self.item_content = item_content
        self.item_content3 = item_content3
        self.id_embs = id_embs
        self.max_seq_len = max_seq_len + 1
        self.item_num = item_num
        #print(str(self.item_content))
        #print(str(self.id_embs))

    def __len__(self):
        return len(self.u2seq)

    def __getitem__(self, user_id):
        seq = self.u2seq[user_id]
        tokens = seq[:-1]
        target = seq[-1]
        mask_len = self.max_seq_len - len(seq)
        pad_tokens = [0] * mask_len + tokens
        log_mask = [0] * mask_len + [1] * len(tokens)
        #print(pad_tokens)
        input_embs = self.item_content[pad_tokens]
        input_embs3 = self.item_content3[pad_tokens]
        embs = self.id_embs[pad_tokens]
        #print(input_embs)
        #print(embs)
        #print()
        labels = np.zeros(self.item_num)
        labels[target - 1] = 1.0
        return torch.LongTensor([user_id]), \
            input_embs, \
            input_embs3,\
            torch.FloatTensor(log_mask), \
            labels, embs

class BuildEvalDataset_2(Dataset):
    def __init__(self, u2seq, item_content,id_embs, max_seq_len, item_num):
        self.u2seq = u2seq
        self.item_content = item_content
        self.id_embs = id_embs
        self.max_seq_len = max_seq_len + 1
        self.item_num = item_num
        #print(str(self.item_content))
        #print(str(self.id_embs))

    def __len__(self):
        return len(self.u2seq)

    def __getitem__(self, user_id):
        seq = self.u2seq[user_id]
        tokens = seq[:-1]
        target = seq[-1]
        mask_len = self.max_seq_len - len(seq)
        pad_tokens = [0] * mask_len + tokens
        log_mask = [0] * mask_len + [1] * len(tokens)
        #print(pad_tokens)
        input_embs = self.item_content[pad_tokens]
        embs = self.id_embs[pad_tokens]
        #print(input_embs)
        #print(embs)
        #print()
        labels = np.zeros(self.item_num)
        labels[target - 1] = 1.0
        return torch.LongTensor([user_id]), \
            input_embs, \
            torch.FloatTensor(log_mask), \
            labels, embs

class BuildTrainDataset_caser(Dataset):
    def __init__(self, u2seq, item_content,
                 item_num, max_seq_len, use_modal):
        self.u2seq = u2seq
        self.item_content = item_content
        self.item_num = item_num
        self.max_seq_len = max_seq_len + 20
        self.use_modal = use_modal

    def __len__(self):
        return len(self.u2seq)

    def __getitem__(self, user_id):
        seq = self.u2seq[user_id]
        seq_Len = len(seq)
        tokens_Len = seq_Len - 1
        mask_len_head = self.max_seq_len - seq_Len
        log_mask = [0] * mask_len_head + [1] * tokens_Len

        sample_items = []
        padding_seq = [0] * mask_len_head + seq
        sample_items.append(padding_seq)
        neg_items = []
        for i in range(tokens_Len):
            sam_neg = random.randint(1, self.item_num)
            while sam_neg in seq:
                sam_neg = random.randint(1, self.item_num)
            neg_items.append(sam_neg)
        neg_items = [0] * mask_len_head + neg_items + [0]
        sample_items.append(neg_items)
        sample_items_id = torch.LongTensor(np.array(sample_items)).transpose(0, 1)

        if self.use_modal:
            sample_items_content = self.item_content[sample_items_id]
            return torch.LongTensor(sample_items_id), sample_items_content, \
                   torch.FloatTensor(log_mask)
        else:
            return torch.LongTensor(sample_items_id), sample_items_id, \
                   torch.FloatTensor(log_mask)
 
class SequentialDistributedSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, batch_size, rank=None, num_replicas=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.batch_size / self.num_replicas)) * self.batch_size
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += [indices[-1]] * (self.total_size - len(indices))
        # subsample
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        return iter(indices)

    def __len__(self):
        return self.num_samples
