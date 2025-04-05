import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .dataset import BuildEvalDataset, BuildEvalDataset_2,BuildEvalDataset_3, SequentialDistributedSampler
import torch.distributed as dist
import math
import pandas as pd
import torch.nn.functional as F

class ItemsDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.data.shape[0]


def item_collate_fn(arr):
    arr = torch.LongTensor(arr)
    return arr


class BuildItemEmbeddingDataset(Dataset):
    def __init__(self, item_word_embs):
        self.item_word_embs = item_word_embs

    def __len__(self):
        return len(self.item_word_embs)

    def __getitem__(self, item_id):
        return torch.LongTensor([item_id]), torch.FloatTensor(self.item_word_embs[item_id])


def print_metrics(x, Log_file, v_or_t):
    Log_file.info(v_or_t+"_results   {}".format('\t'.join(["{:0.5f}".format(i * 100) for i in x])))


def get_mean(arr):
    return [i.mean() for i in arr]


def distributed_concat(tensor, num_total_examples):
    output_tensors = [tensor.clone() for _ in range(dist.get_world_size())]
    dist.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    return concat[:num_total_examples]


def eval_concat(eval_list, test_sampler):
    eval_result = []
    for eval_m in eval_list:
        eval_m_cpu = distributed_concat(eval_m, len(test_sampler.dataset))\
            .to(torch.device("cpu")).numpy()
        eval_result.append(eval_m_cpu.mean())
    return eval_result


def metrics_topK(y_score, y_true, item_rank, topK, local_rank):
    order = torch.argsort(y_score, descending=True)
    y_true = torch.take(y_true, order)
    rank = torch.sum(y_true * item_rank)
    eval_ra = torch.zeros(2).to(local_rank)
    if rank <= topK:
        eval_ra[0] = 1
        eval_ra[1] = 1 / math.log2(rank + 1)
    return eval_ra


def get_item_word_embs(bert_encoder, item_content, test_batch_size, args, local_rank):
    bert_encoder.eval()
    item_dataset = ItemsDataset(data=item_content)
    item_dataloader = DataLoader(item_dataset, batch_size=test_batch_size, num_workers=args.num_workers,
                                 pin_memory=True, collate_fn=item_collate_fn)
    #print(item_content)
    item_word_embs = []
    with torch.no_grad():
        for input_ids in item_dataloader:
            input_ids = input_ids.to(local_rank)
            item_feature = bert_encoder(input_ids).to(torch.device("cpu")).detach()
            item_word_embs.extend(item_feature)
    return torch.stack(tensors=item_word_embs, dim=0)

def get_item_word_embs_llm(item_content,  args):
    import torch
    from llm2vec import LLM2Vec
    
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    l2v = LLM2Vec.from_pretrained(
        "/llama3-8B",
        peft_model_name_or_path="/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.bfloat16,
    )
    
    #item_dataset = ItemsDataset(data=item_content)
    #item_dataloader = DataLoader(item_dataset, batch_size=test_batch_size, num_workers=args.num_workers,
    #                             pin_memory=True, collate_fn=item_collate_fn)
    item_word_embs = []
    for i in range(79708):
        input_ids = item_content[i]
        if i==1:
            print(input_ids)
        item_feature = l2v.encode(input_ids)
        item_word_embs.extend(item_feature)
    return torch.stack(tensors=item_word_embs, dim=0)

def get_item_embeddings_llm(model, item_word_embs, test_batch_size, args, use_modal, local_rank):
    model.eval()
    item_dataset = BuildItemEmbeddingDataset(item_word_embs)
    item_dataloader = DataLoader(item_dataset, batch_size=test_batch_size, num_workers=args.num_workers,
                                 pin_memory=True)
    item_embeddings = []
    with torch.no_grad():
        for input_id, input_embs_content in item_dataloader:
            input_id, input_embs_content = input_id.to(local_rank).squeeze(), input_embs_content.to(local_rank)
            input_embs_id = model.module.id_embedding(input_id)

            if use_modal:
                input_embs_all = \
                    model.module.mlp_layers(
                        model.module.fc(input_embs_id,
                                        model.module.turn_dim1(input_embs_content)))\
                    .to(torch.device("cpu")).detach()
            else:
                input_embs_all = input_embs_id.to(torch.device("cpu")).detach()
            item_embeddings.extend(input_embs_all)
    return torch.stack(tensors=item_embeddings, dim=0)

def get_item_embeddings_llm_morec(model, item_word_embs, test_batch_size, args, use_modal, local_rank):
    model.eval()
    item_dataset = BuildItemEmbeddingDataset(item_word_embs)
    item_dataloader = DataLoader(item_dataset, batch_size=test_batch_size, num_workers=args.num_workers,
                                 pin_memory=True)
    item_embeddings = []
    with torch.no_grad():
        for input_id, input_embs_content in item_dataloader:
            input_id, input_embs_content = input_id.to(local_rank).squeeze(), input_embs_content.to(local_rank)
            #input_embs_id = model.module.id_embedding(input_id)

            if use_modal:
                input_embs_all = \
                    model.module.mlp_layers(model.module.turn_dim1(input_embs_content))\
                    .to(torch.device("cpu")).detach()
            else:
                input_embs_all = input_embs_id.to(torch.device("cpu")).detach()
            item_embeddings.extend(input_embs_all)
    return torch.stack(tensors=item_embeddings, dim=0)


def get_item_embeddings_llm_disco(model, item_word_embs, test_batch_size, args, use_modal, local_rank):
    model.eval()
    item_dataset = BuildItemEmbeddingDataset(item_word_embs)
    item_dataloader = DataLoader(item_dataset, batch_size=test_batch_size, num_workers=args.num_workers,
                                 pin_memory=True)
    item_embeddings = []
    with torch.no_grad():
        for input_id, input_embs_content in item_dataloader:
            input_id, input_embs_content = input_id.to(local_rank).squeeze(), input_embs_content.to(local_rank)
            input_embs_id = model.module.other_embedding(input_id)
            original_id = model.module.id_embedding(input_id)
            if use_modal:
                input_embs_all = \
                    model.module.mlp_layers(
                        model.module.fc1(model.module.mlp1(original_id), model.module.mlp2(input_embs_id),
                                        model.module.mlp3(model.module.turn_dim1(input_embs_content)), model.module.mlp4(model.module.turn_dim2(input_embs_content))))\
                    .to(torch.device("cpu")).detach()
            else:
                input_embs_all = input_embs_id.to(torch.device("cpu")).detach()
            item_embeddings.extend(input_embs_all)
    return torch.stack(tensors=item_embeddings, dim=0)


def get_item_embeddings_llm_4(model, item_word_embs, test_batch_size, args, use_modal, local_rank):
    model.eval()
    item_dataset = BuildItemEmbeddingDataset(item_word_embs)
    item_dataloader = DataLoader(item_dataset, batch_size=test_batch_size, num_workers=args.num_workers,
                                 pin_memory=True)
    item_embeddings = []
    with torch.no_grad():
        for input_id, input_embs_content in item_dataloader:
            input_id, input_embs_content = input_id.to(local_rank).squeeze(), input_embs_content.to(local_rank)
            input_embs_id = model.module.other_embedding(input_id)

            if use_modal:
                input_embs_all = \
                    model.module.mlp_layers(
                        model.module.fc(input_embs_id,
                                        model.module.turn_dim1(input_embs_content)))\
                    .to(torch.device("cpu")).detach()
            else:
                input_embs_all = input_embs_id.to(torch.device("cpu")).detach()
            item_embeddings.extend(input_embs_all)
    return torch.stack(tensors=item_embeddings, dim=0)


def get_item_embeddings_llm_2(model, item_word_embs, test_batch_size, args, use_modal, local_rank):
    model.eval()
    item_dataset = BuildItemEmbeddingDataset(item_word_embs)
    item_dataloader = DataLoader(item_dataset, batch_size=test_batch_size, num_workers=args.num_workers,
                                 pin_memory=True)
    item_embeddings = []
    with torch.no_grad():
        for input_id, input_embs_content in item_dataloader:
            input_id, input_embs_content = input_id.to(local_rank).squeeze(), input_embs_content.to(local_rank)
            input_embs_id = model.module.id_embedding(input_id)
            input_other = model.module.other_embedding(input_id)
            if use_modal:
                input_embs_all = \
                    model.module.mlp_layers(
                        model.module.fc1(input_other,input_embs_id,
                                        model.module.turn_dim1(input_embs_content)))\
                    .to(torch.device("cpu")).detach()
            else:
                input_embs_all = input_embs_id.to(torch.device("cpu")).detach()
            item_embeddings.extend(input_embs_all)
    return torch.stack(tensors=item_embeddings, dim=0)

def get_item_embeddings(model, item_word_embs, test_batch_size, args, use_modal, local_rank):
    model.eval()
    item_dataset = BuildItemEmbeddingDataset(item_word_embs)
    item_dataloader = DataLoader(item_dataset, batch_size=test_batch_size, num_workers=args.num_workers,
                                 pin_memory=True)
    item_embeddings = []
    with torch.no_grad():
        for input_id, input_embs_content in item_dataloader:
            input_id, input_embs_content = input_id.to(local_rank).squeeze(), input_embs_content.to(local_rank)
            input_embs_id = model.module.id_embedding(input_id)

            if use_modal:
                input_embs_all = \
                    model.module.mlp_layers(
                        model.module.fc(input_embs_id,
                                        model.module.turn_dim(input_embs_content)))\
                    .to(torch.device("cpu")).detach()
            else:
                input_embs_all = input_embs_id.to(torch.device("cpu")).detach()
            item_embeddings.extend(input_embs_all)
    return torch.stack(tensors=item_embeddings, dim=0)

def get_item_embeddings_all(model, item_word_embs, test_batch_size, args, use_modal, local_rank):
    model.eval()
    item_dataset = BuildItemEmbeddingDataset(item_word_embs)
    item_dataloader = DataLoader(item_dataset, batch_size=test_batch_size, num_workers=args.num_workers,
                                 pin_memory=True)
    item_embeddings = []
    with torch.no_grad():
        for input_id, input_embs_content in item_dataloader:
            input_id, input_embs_content = input_id.to(local_rank).squeeze(), input_embs_content.to(local_rank)
            input_embs_id = model.module.id_embedding(input_id)

            if use_modal:
                input_embs_all = \
                    model.module.mlp_layers(
                        model.module.fc1(input_embs_id,
                                        model.module.turn_dim(input_embs_content),input_embs_id*model.module.turn_dim(input_embs_content)))\
                    .to(torch.device("cpu")).detach()
            else:
                input_embs_all = input_embs_id.to(torch.device("cpu")).detach()
            item_embeddings.extend(input_embs_all)
    return torch.stack(tensors=item_embeddings, dim=0)

def get_id_embeddings_amazon(model, item_word_embs, test_batch_size, args, use_modal, local_rank):
    model.eval()
    item_dataset = BuildItemEmbeddingDataset(item_word_embs)
    item_dataloader = DataLoader(item_dataset, batch_size=test_batch_size, num_workers=args.num_workers,
                                 pin_memory=False)
    item_embeddings = []
    with torch.no_grad():
        for input_id, input_embs_content in item_dataloader:
            input_id, input_embs_content = input_id.to(local_rank).squeeze(), input_embs_content.to(local_rank)
            input_embs_id = model.module.id_embedding(input_id)


            input_embs_all = input_embs_id.to(torch.device("cpu")).detach()
            item_embeddings.extend(input_embs_all)
    return torch.stack(tensors=item_embeddings, dim=0)

def get_id_embeddings(model, item_word_embs, test_batch_size, args, use_modal, local_rank):
    model.eval()
    item_dataset = BuildItemEmbeddingDataset(item_word_embs)
    item_dataloader = DataLoader(item_dataset, batch_size=test_batch_size, num_workers=args.num_workers,
                                 pin_memory=True)
    item_embeddings = []
    with torch.no_grad():
        for input_id, input_embs_content in item_dataloader:
            input_id, input_embs_content = input_id.to(local_rank).squeeze(), input_embs_content.to(local_rank)
            input_embs_id = model.module.id_embedding(input_id)

            
            input_embs_all = input_embs_id.to(torch.device("cpu")).detach()
            item_embeddings.extend(input_embs_all)
    return torch.stack(tensors=item_embeddings, dim=0)

def get_id_embeddings_other(model, item_word_embs, test_batch_size, args, use_modal, local_rank):
    model.eval()
    item_dataset = BuildItemEmbeddingDataset(item_word_embs)
    item_dataloader = DataLoader(item_dataset, batch_size=test_batch_size, num_workers=args.num_workers,
                                 pin_memory=True)
    item_embeddings = []
    with torch.no_grad():
        for input_id, input_embs_content in item_dataloader:
            input_id, input_embs_content = input_id.to(local_rank).squeeze(), input_embs_content.to(local_rank)
            input_embs_id = model.module.other_embedding(input_id)


            input_embs_all = input_embs_id.to(torch.device("cpu")).detach()
            item_embeddings.extend(input_embs_all)
    return torch.stack(tensors=item_embeddings, dim=0)

def get_item_embeddings_llm_3(model, item_word_embs, test_batch_size, args, use_modal, local_rank):
    model.eval()
    item_dataset = BuildItemEmbeddingDataset(item_word_embs)
    item_dataloader = DataLoader(item_dataset, batch_size=test_batch_size, num_workers=args.num_workers,
                                 pin_memory=True)
    item_embeddings = []
    id_embs = []
    with torch.no_grad():
        for input_id, input_embs_content in item_dataloader:
            input_id, input_embs_content = input_id.to(local_rank).squeeze(), input_embs_content.to(local_rank)
            input_embs_id = model.module.other_embedding(input_id)
            emb = model.module.id_embedding(input_id).to(torch.device("cpu")).detach()
            if use_modal:
                input_embs_all = \
                    model.module.mlp_layers(
                        model.module.fc(input_embs_id,
                                        model.module.turn_dim1(input_embs_content)))\
                    .to(torch.device("cpu")).detach()
            else:
                input_embs_all = input_embs_id.to(torch.device("cpu")).detach()
            item_embeddings.extend(input_embs_all)
            id_embs.extend(emb)
    return torch.stack(tensors=item_embeddings, dim=0), torch.stack(tensors=id_embs, dim=0)

def get_item_embeddings_llm_junguang(model, item_word_embs, test_batch_size, args, use_modal, local_rank):
    model.eval()
    item_dataset = BuildItemEmbeddingDataset(item_word_embs)
    item_dataloader = DataLoader(item_dataset, batch_size=test_batch_size, num_workers=args.num_workers,
                                 pin_memory=True)
    item_embeddings = []
    id_embs = []
    with torch.no_grad():
        for input_id, input_embs_content in item_dataloader:
            input_id, input_embs_content = input_id.to(local_rank).squeeze(), input_embs_content.to(local_rank)
            #input_embs_id = model.module.other_embedding(input_id)
            emb = model.module.id_embedding(input_id).to(torch.device("cpu")).detach()
            if use_modal:
                input_embs_all = \
                    model.module.mlp_layers(model.module.turn_dim1(input_embs_content))\
                    .to(torch.device("cpu")).detach()
            else:
                input_embs_all = input_embs_id.to(torch.device("cpu")).detach()
            item_embeddings.extend(input_embs_all)
            id_embs.extend(emb)
    return torch.stack(tensors=item_embeddings, dim=0), torch.stack(tensors=id_embs, dim=0)

def get_item_embeddings_llm_noid(model, item_word_embs, test_batch_size, args, use_modal, local_rank):
    model.eval()
    item_dataset = BuildItemEmbeddingDataset(item_word_embs)
    item_dataloader = DataLoader(item_dataset, batch_size=test_batch_size, num_workers=args.num_workers,
                                 pin_memory=True)
    item_embeddings = []
    id_embs = []
    with torch.no_grad():
        for input_id, input_embs_content in item_dataloader:
            input_id, input_embs_content = input_id.to(local_rank).squeeze(), input_embs_content.to(local_rank)
            #input_embs_id = model.module.other_embedding(input_id)
            #emb = model.module.id_embedding(input_id).to(torch.device("cpu")).detach()
            other_emb = model.module.other_embedding(input_id)
            if use_modal:
                input_embs_cont = \
                    model.module.mlp_layers2(model.module.turn_dim2(input_embs_content))\
                    .to(torch.device("cpu")).detach()
                input_embs_all = \
                    model.module.mlp_layers(
                        model.module.fc(other_emb,
                                        model.module.turn_dim1(input_embs_content)))\
                    .to(torch.device("cpu")).detach()
            else:
                input_embs_all = input_embs_id.to(torch.device("cpu")).detach()
            item_embeddings.extend(input_embs_all)
            id_embs.extend(input_embs_cont)
    return torch.stack(tensors=item_embeddings, dim=0), torch.stack(tensors=id_embs, dim=0)

def get_item_embeddings_llm_3tower(model, item_word_embs, test_batch_size, args, use_modal, local_rank):
    model.eval()
    item_dataset = BuildItemEmbeddingDataset(item_word_embs)
    item_dataloader = DataLoader(item_dataset, batch_size=test_batch_size, num_workers=args.num_workers,
                                 pin_memory=True)
    item_embeddings = []
    item_embeddings3 = []
    id_embs = []
    with torch.no_grad():
        for input_id, input_embs_content in item_dataloader:
            input_id, input_embs_content = input_id.to(local_rank).squeeze(), input_embs_content.to(local_rank)
            other_emb = model.module.other_embedding(input_id)
            emb = model.module.id_embedding(input_id).to(torch.device("cpu")).detach()
            if use_modal:
                input_embs_all3 = \
                    model.module.mlp_layer_3(model.module.turn_dim3(input_embs_content))\
                    .to(torch.device("cpu")).detach()
                input_embs_all = \
                    model.module.mlp_layers(
                        model.module.fc(other_emb,
                                        model.module.turn_dim1(input_embs_content)))\
                    .to(torch.device("cpu")).detach()
            item_embeddings.extend(input_embs_all)
            item_embeddings3.extend(input_embs_all3)
            id_embs.extend(emb)
    return torch.stack(tensors=item_embeddings3, dim=0), torch.stack(tensors=item_embeddings, dim=0), torch.stack(tensors=id_embs, dim=0)

def eval_model(model, user_history, eval_seq, item_embeddings, test_batch_size, args, item_num, Log_file, v_or_t, local_rank):
    eval_dataset = BuildEvalDataset(u2seq=eval_seq, item_content=item_embeddings,
                                    max_seq_len=args.max_seq_len, item_num=item_num)
    test_sampler = SequentialDistributedSampler(eval_dataset, batch_size=test_batch_size)
    eval_dl = DataLoader(eval_dataset, batch_size=test_batch_size, \
                         num_workers=args.num_workers, pin_memory=False, sampler=test_sampler)
    #                     num_workers=args.num_workers, pin_memory=True, sampler=test_sampler)
    model.eval()
    topK = 10
    Log_file.info(v_or_t + "_methods   {}".format('\t'.join(['Hit{}'.format(topK), 'nDCG{}'.format(topK)])))
    item_embeddings = item_embeddings.to(local_rank)
    with torch.no_grad():
        eval_all_user = []
        item_rank = torch.Tensor(np.arange(item_num) + 1).to(local_rank)
        for data in eval_dl:
            user_ids, input_embs, log_mask, labels = data
            user_ids, input_embs, log_mask, labels = \
                user_ids.to(local_rank), input_embs.to(local_rank),\
                log_mask.to(local_rank), labels.to(local_rank).detach()
            prec_emb = model.module.user_encoder(input_embs, log_mask, local_rank)[:, -1].detach()
            scores = torch.matmul(prec_emb, item_embeddings.t()).squeeze(dim=-1).detach()
            #print(scores.shape)
            for user_id, label, score in zip(user_ids, labels, scores):
                user_id = user_id[0].item()
                history = user_history[user_id].to(local_rank)
                score[history] = -np.inf
                score = score[1:]
                eval_all_user.append(metrics_topK(score, label, item_rank, topK, local_rank))
        eval_all_user = torch.stack(tensors=eval_all_user, dim=0).t().contiguous()
        Hit10, nDCG10 = eval_all_user
        mean_eval = eval_concat([Hit10, nDCG10], test_sampler)
        print_metrics(mean_eval, Log_file, v_or_t)
    return mean_eval[0]

def eval_model_step2(model, user_history, eval_seq, item_embeddings, test_batch_size, args, item_num, Log_file, v_or_t, local_rank):
    eval_dataset = BuildEvalDataset(u2seq=eval_seq, item_content=item_embeddings,
                                    max_seq_len=args.max_seq_len, item_num=item_num)
    test_sampler = SequentialDistributedSampler(eval_dataset, batch_size=test_batch_size)
    eval_dl = DataLoader(eval_dataset, batch_size=test_batch_size,
                         num_workers=args.num_workers, pin_memory=True, sampler=test_sampler)
    model.eval()
    topK = 10
    Log_file.info(v_or_t + "_methods   {}".format('\t'.join(['Hit{}'.format(topK), 'nDCG{}'.format(topK)])))
    item_embeddings = item_embeddings.to(local_rank)
    with torch.no_grad():
        eval_all_user = []
        item_rank = torch.Tensor(np.arange(item_num) + 1).to(local_rank)
        for data in eval_dl:
            user_ids, input_embs, log_mask, labels = data
            user_ids, input_embs, log_mask, labels = \
                user_ids.to(local_rank), input_embs.to(local_rank),\
                log_mask.to(local_rank), labels.to(local_rank).detach()
            prec_emb = model.module.user_enc_2(input_embs, log_mask, local_rank)[:, -1].detach()
            scores = torch.matmul(prec_emb, item_embeddings.t()).squeeze(dim=-1).detach()
            #print(scores.shape)
            for user_id, label, score in zip(user_ids, labels, scores):
                user_id = user_id[0].item()
                history = user_history[user_id].to(local_rank)
                score[history] = -np.inf
                score = score[1:]
                eval_all_user.append(metrics_topK(score, label, item_rank, topK, local_rank))
        eval_all_user = torch.stack(tensors=eval_all_user, dim=0).t().contiguous()
        Hit10, nDCG10 = eval_all_user
        mean_eval = eval_concat([Hit10, nDCG10], test_sampler)
        print_metrics(mean_eval, Log_file, v_or_t)
    return mean_eval[0]

def eval_model_2(model, user_history, eval_seq, item_embeddings,id_embs, test_batch_size, args, item_num, Log_file, v_or_t, local_rank):
    eval_dataset = BuildEvalDataset_2(u2seq=eval_seq, item_content=item_embeddings, id_embs = id_embs, \
                                    max_seq_len=args.max_seq_len, item_num=item_num)
    #eval_dataset = BuildEvalDataset(u2seq=eval_seq, item_content=item_embeddings,
    #                                max_seq_len=args.max_seq_len, item_num=item_num)
    test_sampler = SequentialDistributedSampler(eval_dataset, batch_size=test_batch_size)
    eval_dl = DataLoader(eval_dataset, batch_size=test_batch_size,
                         num_workers=args.num_workers, pin_memory=True, sampler=test_sampler)
    model.eval()
    topK = 10
    ee=pd.read_csv('/dataset/fre.csv')
    ee.columns = ['id', 'fre']
    e=ee.sort_values('fre',ascending=False)
    l = [0, 41, 127, 276, 517, 889, 1467, 2413, 4070, 7618, 79707, 79708]
    e['bin']=0
    for k in range(10):
        e.iloc[l[k]:l[k+1],2]=k+1
    e=e.sort_values('id',ascending=True)
    bins=e['bin'].to_numpy()
    alpha = model.module.alpha[bins]
    Log_file.info(v_or_t + "_methods   {}".format('\t'.join(['Hit{}'.format(topK), 'nDCG{}'.format(topK)])))
    item_embeddings = item_embeddings.to(local_rank)
    embs = id_embs.to(local_rank)
    with torch.no_grad():
        eval_all_user = []
        item_rank = torch.Tensor(np.arange(item_num) + 1).to(local_rank)
        for data in eval_dl:
            user_ids, input_embs, log_mask, labels, id_embs = data
            user_ids, input_embs, log_mask, labels, id_embs = \
                user_ids.to(local_rank), input_embs.to(local_rank),\
                log_mask.to(local_rank), labels.to(local_rank).detach(), id_embs.to(local_rank)
            prec_emb_2 = model.module.user_enc_2(input_embs, log_mask, local_rank)[:, -1].detach()
            prec_emb_1 = model.module.user_encoder(id_embs, log_mask, local_rank)[:, -1].detach()
            #print(item_embeddings.shape)
            #print(id_embs.shape)
            a = torch.matmul(prec_emb_2, item_embeddings.t()).squeeze(dim=-1).detach()
            #alpha = model.module.alpha[bin_pos].detach()
            #print(a.shape)
            scores_2 = (1-alpha)*a
            #print(a.shape)
            #print(alpha.shape)
            #print(scores_2.shape)
            b = torch.matmul(prec_emb_1, embs.t()).squeeze(dim=-1).detach()
            #print(b.shape)
            #print()
            scores_1 = alpha*b
            scores=scores_2+scores_1
            for user_id, label, score in zip(user_ids, labels, scores):
                user_id = user_id[0].item()
                history = user_history[user_id].to(local_rank)
                score[history] = -np.inf
                score = score[1:]
                eval_all_user.append(metrics_topK(score, label, item_rank, topK, local_rank))
        eval_all_user = torch.stack(tensors=eval_all_user, dim=0).t().contiguous()
        Hit10, nDCG10 = eval_all_user
        mean_eval = eval_concat([Hit10, nDCG10], test_sampler)
        print_metrics(mean_eval, Log_file, v_or_t)
    return mean_eval[0]


def eval_model_2_2(model, user_history, eval_seq, item_embeddings,id_embs, test_batch_size, args, item_num, Log_file, v_or_t, local_rank):
    eval_dataset = BuildEvalDataset_2(u2seq=eval_seq, item_content=item_embeddings, id_embs = id_embs, \
                                    max_seq_len=args.max_seq_len, item_num=item_num)
    #eval_dataset = BuildEvalDataset(u2seq=eval_seq, item_content=item_embeddings,
    #                                max_seq_len=args.max_seq_len, item_num=item_num)
    test_sampler = SequentialDistributedSampler(eval_dataset, batch_size=test_batch_size)
    eval_dl = DataLoader(eval_dataset, batch_size=test_batch_size,
                         num_workers=args.num_workers, pin_memory=True, sampler=test_sampler)
    model.eval()
    topK = 10
    ee=pd.read_csv('/dataset/fre.csv')
    ee.columns = ['id', 'fre']
    e=ee.sort_values('fre',ascending=False)
    l = [0, 41, 127, 276, 517, 889, 1467, 2413, 4070, 7618, 79707, 79708]
    e['bin']=0
    for k in range(10):
        e.iloc[l[k]:l[k+1],2]=k+1
    e=e.sort_values('id',ascending=True)
    bins=e['bin'].to_numpy()
    act=torch.nn.Sigmoid()
    alpha = act(model.module.alpha[bins])
    beta = act(model.module.beta[bins])
    s=alpha+beta
    Log_file.info(v_or_t + "_methods   {}".format('\t'.join(['Hit{}'.format(topK), 'nDCG{}'.format(topK)])))
    item_embeddings = item_embeddings.to(local_rank)
    embs = id_embs.to(local_rank)
    with torch.no_grad():
        eval_all_user = []
        item_rank = torch.Tensor(np.arange(item_num) + 1).to(local_rank)
        for data in eval_dl:
            user_ids, input_embs, log_mask, labels, id_embs = data
            user_ids, input_embs, log_mask, labels, id_embs = \
                user_ids.to(local_rank), input_embs.to(local_rank),\
                log_mask.to(local_rank), labels.to(local_rank).detach(), id_embs.to(local_rank)
            prec_emb_2 = model.module.user_enc_2(input_embs, log_mask, local_rank)[:, -1].detach()
            prec_emb_1 = model.module.user_encoder(id_embs, log_mask, local_rank)[:, -1].detach()
            #print(item_embeddings.shape)
            #print(id_embs.shape)
            a = torch.matmul(prec_emb_2, item_embeddings.t()).squeeze(dim=-1).detach()
            #alpha = model.module.alpha[bin_pos].detach()
            #print(a.shape)
            scores_2 = beta/s*a
            #print(a.shape)
            #print(alpha.shape)
            #print(scores_2.shape)
            b = torch.matmul(prec_emb_1, embs.t()).squeeze(dim=-1).detach()
            #print(b.shape)
            #print()
            scores_1 = alpha/s*b
            scores=scores_2+scores_1
            for user_id, label, score in zip(user_ids, labels, scores):
                user_id = user_id[0].item()
                history = user_history[user_id].to(local_rank)
                score[history] = -np.inf
                score = score[1:]
                eval_all_user.append(metrics_topK(score, label, item_rank, topK, local_rank))
        eval_all_user = torch.stack(tensors=eval_all_user, dim=0).t().contiguous()
        Hit10, nDCG10 = eval_all_user
        mean_eval = eval_concat([Hit10, nDCG10], test_sampler)
        print_metrics(mean_eval, Log_file, v_or_t)
    return mean_eval[0]


def eval_model_noid(model, user_history, eval_seq, item_embeddings,id_embs, test_batch_size, args, item_num, Log_file, v_or_t, local_rank):
    eval_dataset = BuildEvalDataset_2(u2seq=eval_seq, item_content=item_embeddings, id_embs = id_embs, \
                                    max_seq_len=args.max_seq_len, item_num=item_num)
    #eval_dataset = BuildEvalDataset(u2seq=eval_seq, item_content=item_embeddings,
    #                                max_seq_len=args.max_seq_len, item_num=item_num)
    test_sampler = SequentialDistributedSampler(eval_dataset, batch_size=test_batch_size)
    eval_dl = DataLoader(eval_dataset, batch_size=test_batch_size,
                         num_workers=args.num_workers, pin_memory=True, sampler=test_sampler)
    model.eval()
    topK = 10
    ee=pd.read_csv('/dataset/fre.csv')
    ee.columns = ['id', 'fre']
    e=ee.sort_values('fre',ascending=False)
    l = [0, 41, 127, 276, 517, 889, 1467, 2413, 4070, 7618, 79707, 79708]
    e['bin']=0
    for k in range(10):
        e.iloc[l[k]:l[k+1],2]=k+1
    e=e.sort_values('id',ascending=True)
    bins=e['bin'].to_numpy()
    act=torch.nn.Sigmoid()
    alpha = act(model.module.alpha[bins])
    beta = act(model.module.beta[bins])
    s=alpha+beta
    Log_file.info(v_or_t + "_methods   {}".format('\t'.join(['Hit{}'.format(topK), 'nDCG{}'.format(topK)])))
    item_embeddings = item_embeddings.to(local_rank)
    embs = id_embs.to(local_rank)
    with torch.no_grad():
        eval_all_user = []
        item_rank = torch.Tensor(np.arange(item_num) + 1).to(local_rank)
        for data in eval_dl:
            user_ids, input_embs, log_mask, labels, id_embs = data
            user_ids, input_embs, log_mask, labels, id_embs = \
                user_ids.to(local_rank), input_embs.to(local_rank),\
                log_mask.to(local_rank), labels.to(local_rank).detach(), id_embs.to(local_rank)
            prec_emb_2 = model.module.user_enc_2(input_embs, log_mask, local_rank)[:, -1].detach()
            prec_emb_1 = model.module.user_enc_3(id_embs, log_mask, local_rank)[:, -1].detach()
            #print(item_embeddings.shape)
            #print(id_embs.shape)
            a = torch.matmul(prec_emb_2, item_embeddings.t()).squeeze(dim=-1).detach()
            #alpha = model.module.alpha[bin_pos].detach()
            #print(a.shape)
            scores_2 = beta/s*a
            #print(a.shape)
            #print(alpha.shape)
            #print(scores_2.shape)
            b = torch.matmul(prec_emb_1, embs.t()).squeeze(dim=-1).detach()
            #print(b.shape)
            #print()
            scores_1 = alpha/s*b
            scores=scores_2+scores_1
            for user_id, label, score in zip(user_ids, labels, scores):
                user_id = user_id[0].item()
                history = user_history[user_id].to(local_rank)
                score[history] = -np.inf
                score = score[1:]
                eval_all_user.append(metrics_topK(score, label, item_rank, topK, local_rank))
        eval_all_user = torch.stack(tensors=eval_all_user, dim=0).t().contiguous()
        Hit10, nDCG10 = eval_all_user
        mean_eval = eval_concat([Hit10, nDCG10], test_sampler)
        print_metrics(mean_eval, Log_file, v_or_t)
    return mean_eval[0]


def eval_model_2_3(model, user_history, eval_seq, item_embeddings,id_embs, test_batch_size, args, item_num, Log_file, v_or_t, local_rank):
    eval_dataset = BuildEvalDataset_2(u2seq=eval_seq, item_content=item_embeddings, id_embs = id_embs, \
                                    max_seq_len=args.max_seq_len, item_num=item_num)
    #eval_dataset = BuildEvalDataset(u2seq=eval_seq, item_content=item_embeddings,
    #                                max_seq_len=args.max_seq_len, item_num=item_num)
    test_sampler = SequentialDistributedSampler(eval_dataset, batch_size=test_batch_size)
    eval_dl = DataLoader(eval_dataset, batch_size=test_batch_size,
                         num_workers=args.num_workers, pin_memory=True, sampler=test_sampler)
    model.eval()
    topK = 10
    ee=pd.read_csv('/dataset/fre.csv')
    ee.columns = ['id', 'fre']
    e=ee.sort_values('fre',ascending=False)
    l = [0, 41, 127, 276, 517, 889, 1467, 2413, 4070, 7618, 79707, 79708]
    e['bin']=0
    for k in range(10):
        e.iloc[l[k]:l[k+1],2]=k+1
    e=e.sort_values('id',ascending=True)
    bins=e['bin'].to_numpy()
    act=torch.nn.Sigmoid()
    alpha = 2*act(model.module.alpha[bins])
    Log_file.info(v_or_t + "_methods   {}".format('\t'.join(['Hit{}'.format(topK), 'nDCG{}'.format(topK)])))
    item_embeddings = item_embeddings.to(local_rank)
    embs = id_embs.to(local_rank)
    with torch.no_grad():
        eval_all_user = []
        item_rank = torch.Tensor(np.arange(item_num) + 1).to(local_rank)
        for data in eval_dl:
            user_ids, input_embs, log_mask, labels, id_embs = data
            user_ids, input_embs, log_mask, labels, id_embs = \
                user_ids.to(local_rank), input_embs.to(local_rank),\
                log_mask.to(local_rank), labels.to(local_rank).detach(), id_embs.to(local_rank)
            prec_emb_2 = model.module.user_enc_2(input_embs, log_mask, local_rank)[:, -1].detach()
            prec_emb_1 = model.module.user_encoder(id_embs, log_mask, local_rank)[:, -1].detach()
            #print(item_embeddings.shape)
            #print(id_embs.shape)
            a = torch.matmul(prec_emb_2, item_embeddings.t()).squeeze(dim=-1).detach()
            #alpha = model.module.alpha[bin_pos].detach()
            #print(a.shape)
            scores_2 = (2-alpha)*a
            #print(a.shape)
            #print(alpha.shape)
            #print(scores_2.shape)
            b = torch.matmul(prec_emb_1, embs.t()).squeeze(dim=-1).detach()
            #print(b.shape)
            #print()
            scores_1 = alpha*b
            scores=scores_2+scores_1
            for user_id, label, score in zip(user_ids, labels, scores):
                user_id = user_id[0].item()
                history = user_history[user_id].to(local_rank)
                score[history] = -np.inf
                score = score[1:]
                eval_all_user.append(metrics_topK(score, label, item_rank, topK, local_rank))
        eval_all_user = torch.stack(tensors=eval_all_user, dim=0).t().contiguous()
        Hit10, nDCG10 = eval_all_user
        mean_eval = eval_concat([Hit10, nDCG10], test_sampler)
        print_metrics(mean_eval, Log_file, v_or_t)
    return mean_eval[0]

def eval_model_2_3tower_ablation(model, user_history, eval_seq,item_embeddings3, item_embeddings,id_embs, test_batch_size, args, item_num, Log_file, v_or_t, local_rank):
    eval_dataset = BuildEvalDataset_3(u2seq=eval_seq, item_content=item_embeddings, item_content3=item_embeddings3,id_embs = id_embs, \
                                    max_seq_len=args.max_seq_len, item_num=item_num)
    #eval_dataset = BuildEvalDataset(u2seq=eval_seq, item_content=item_embeddings,
    #                                max_seq_len=args.max_seq_len, item_num=item_num)
    test_sampler = SequentialDistributedSampler(eval_dataset, batch_size=test_batch_size)
    eval_dl = DataLoader(eval_dataset, batch_size=test_batch_size,
                         num_workers=args.num_workers, pin_memory=True, sampler=test_sampler)
    model.eval()
    topK = 10
    ee=pd.read_csv('/dataset/fre.csv')
    ee.columns = ['id', 'fre']
    e=ee.sort_values('fre',ascending=False)
    l = [0, 41, 127, 276, 517, 889, 1467, 2413, 4070, 7618, 79707, 79708]
    e['bin']=0
    #for k in range(10):
    #    e.iloc[l[k]:l[k+1],2]=k+1
    #e=e.sort_values('id',ascending=True)
    bins=e['bin'].to_numpy()
    act=torch.nn.Sigmoid()
    alpha = act(model.module.alpha[bins])
    alpha2 = act(model.module.alpha2[bins])
    alpha3 = act(model.module.alpha3[bins])
    s=alpha+alpha2+alpha3
    Log_file.info(v_or_t + "_methods   {}".format('\t'.join(['Hit{}'.format(topK), 'nDCG{}'.format(topK)])))
    item_embeddings = item_embeddings.to(local_rank)
    item_embeddings3 = item_embeddings3.to(local_rank)
    embs = id_embs.to(local_rank)
    with torch.no_grad():
        eval_all_user = []
        item_rank = torch.Tensor(np.arange(item_num) + 1).to(local_rank)
        for data in eval_dl:
            user_ids, input_embs, emb3, log_mask, labels, id_embs = data
            user_ids, input_embs, emb3, log_mask, labels, id_embs = \
                user_ids.to(local_rank), input_embs.to(local_rank), emb3.to(local_rank),\
                log_mask.to(local_rank), labels.to(local_rank).detach(), id_embs.to(local_rank)
            prec_emb_2 = model.module.user_enc_2(input_embs, log_mask, local_rank)[:, -1].detach()
            prec_emb_3 = model.module.user_enc_3(emb3, log_mask, local_rank)[:, -1].detach()
            prec_emb_1 = model.module.user_encoder(id_embs, log_mask, local_rank)[:, -1].detach()
            #print(item_embeddings.shape)
            #print(id_embs.shape)
            a = torch.matmul(prec_emb_2, item_embeddings.t()).squeeze(dim=-1).detach()
            #alpha = model.module.alpha[bin_pos].detach()
            #print(a.shape)
            scores_2 = alpha2/s*a
            #print(a.shape)
            #print(alpha.shape)
            #print(scores_2.shape)
            b = torch.matmul(prec_emb_1, embs.t()).squeeze(dim=-1).detach()
            #print(b.shape)
            #print()
            scores_1 = alpha/s*b
            c = torch.matmul(prec_emb_3, item_embeddings3.t()).squeeze(dim=-1).detach()
            scores_3 = alpha3/s*c
            scores=scores_2+scores_1+scores_3
            #print(scores.shape)
            for user_id, label, score in zip(user_ids, labels, scores):
                user_id = user_id[0].item()
                history = user_history[user_id].to(local_rank)
                score[history] = -np.inf
                score = score[1:]
                eval_all_user.append(metrics_topK(score, label, item_rank, topK, local_rank))
        eval_all_user = torch.stack(tensors=eval_all_user, dim=0).t().contiguous()
        Hit10, nDCG10 = eval_all_user
        mean_eval = eval_concat([Hit10, nDCG10], test_sampler)
        print_metrics(mean_eval, Log_file, v_or_t)
    return mean_eval[0]


def eval_model_2_3tower(model, user_history, eval_seq,item_embeddings3, item_embeddings,id_embs, test_batch_size, args, item_num, Log_file, v_or_t, local_rank):
    eval_dataset = BuildEvalDataset_3(u2seq=eval_seq, item_content=item_embeddings, item_content3=item_embeddings3,id_embs = id_embs, \
                                    max_seq_len=args.max_seq_len, item_num=item_num)
    #eval_dataset = BuildEvalDataset(u2seq=eval_seq, item_content=item_embeddings,
    #                                max_seq_len=args.max_seq_len, item_num=item_num)
    test_sampler = SequentialDistributedSampler(eval_dataset, batch_size=test_batch_size)
    eval_dl = DataLoader(eval_dataset, batch_size=test_batch_size,
                         num_workers=args.num_workers, pin_memory=True, sampler=test_sampler)
    model.eval()
    topK = 10
    ee=pd.read_csv('/dataset/fre.csv')
    ee.columns = ['id', 'fre']
    e=ee.sort_values('fre',ascending=False)
    l = [0, 41, 127, 276, 517, 889, 1467, 2413, 4070, 7618, 79707, 79708]
    e['bin']=0
    for k in range(10):
        e.iloc[l[k]:l[k+1],2]=k+1
    e=e.sort_values('id',ascending=True)
    bins=e['bin'].to_numpy()
    act=torch.nn.Sigmoid()
    alpha = act(model.module.alpha[bins])
    alpha2 = act(model.module.alpha2[bins])
    alpha3 = act(model.module.alpha3[bins])
    s=alpha+alpha2+alpha3
    Log_file.info(v_or_t + "_methods   {}".format('\t'.join(['Hit{}'.format(topK), 'nDCG{}'.format(topK)])))
    item_embeddings = item_embeddings.to(local_rank)
    item_embeddings3 = item_embeddings3.to(local_rank)
    embs = id_embs.to(local_rank)
    with torch.no_grad():
        eval_all_user = []
        item_rank = torch.Tensor(np.arange(item_num) + 1).to(local_rank)
        for data in eval_dl:
            user_ids, input_embs, emb3, log_mask, labels, id_embs = data
            user_ids, input_embs, emb3, log_mask, labels, id_embs = \
                user_ids.to(local_rank), input_embs.to(local_rank), emb3.to(local_rank),\
                log_mask.to(local_rank), labels.to(local_rank).detach(), id_embs.to(local_rank)
            prec_emb_2 = model.module.user_enc_2(input_embs, log_mask, local_rank)[:, -1].detach()
            prec_emb_3 = model.module.user_enc_3(emb3, log_mask, local_rank)[:, -1].detach()
            prec_emb_1 = model.module.user_encoder(id_embs, log_mask, local_rank)[:, -1].detach()
            #print(item_embeddings.shape)
            #print(id_embs.shape)
            a = torch.matmul(prec_emb_2, item_embeddings.t()).squeeze(dim=-1).detach()
            #alpha = model.module.alpha[bin_pos].detach()
            #print(a.shape)
            scores_2 = alpha2/s*a
            #print(a.shape)
            #print(alpha.shape)
            #print(scores_2.shape)
            b = torch.matmul(prec_emb_1, embs.t()).squeeze(dim=-1).detach()
            #print(b.shape)
            #print()
            scores_1 = alpha/s*b
            c = torch.matmul(prec_emb_3, item_embeddings3.t()).squeeze(dim=-1).detach()
            scores_3 = alpha3/s*c
            scores=scores_2+scores_1+scores_3
            #print(scores.shape)
            for user_id, label, score in zip(user_ids, labels, scores):
                user_id = user_id[0].item()
                history = user_history[user_id].to(local_rank)
                score[history] = -np.inf
                score = score[1:]
                eval_all_user.append(metrics_topK(score, label, item_rank, topK, local_rank))
        eval_all_user = torch.stack(tensors=eval_all_user, dim=0).t().contiguous()
        Hit10, nDCG10 = eval_all_user
        mean_eval = eval_concat([Hit10, nDCG10], test_sampler)
        print_metrics(mean_eval, Log_file, v_or_t)
    return mean_eval[0]

def eval_model_gru_amazon(topk, model, user_history, eval_seq, item_embeddings, test_batch_size, args, item_num, Log_file, v_or_t, local_rank):
    eval_dataset = BuildEvalDataset(u2seq=eval_seq, item_content=item_embeddings,
                                    max_seq_len=args.max_seq_len, item_num=item_num)
    test_sampler = SequentialDistributedSampler(eval_dataset, batch_size=test_batch_size)
    eval_dl = DataLoader(eval_dataset, batch_size=test_batch_size, \
    #                     num_workers=args.num_workers, pin_memory=False, sampler=test_sampler)
                         num_workers=args.num_workers, pin_memory=True, sampler=test_sampler)
    model.eval()
    topK = topk
    Log_file.info(v_or_t + "_methods   {}".format('\t'.join(['Hit{}'.format(topK), 'nDCG{}'.format(topK)])))
    item_embeddings = item_embeddings.to(local_rank)
    #print(item_embeddings.shape)
    with torch.no_grad():
        eval_all_user = []
        item_rank = torch.Tensor(np.arange(item_num) + 1).to(local_rank)
        for data in eval_dl:
            user_ids, input_embs, log_mask, labels = data
            user_ids, input_embs, log_mask, labels = \
                user_ids.to(local_rank), input_embs.to(local_rank),\
                log_mask.to(local_rank), labels.to(local_rank).detach()
            gru_output, _=model.module.gru_layers(input_embs)
            prec_emb = model.module.dense(gru_output)[:, -1].detach()
            scores = torch.matmul(prec_emb, item_embeddings.t()).squeeze(dim=-1).detach()
            #print(scores.shape)
            for user_id, label, score in zip(user_ids, labels, scores):
                user_id = user_id[0].item()
                history = user_history[user_id].to(local_rank)
                score[history] = -np.inf
                score = score[1:]
                eval_all_user.append(metrics_topK(score, label, item_rank, topK, local_rank))
        eval_all_user = torch.stack(tensors=eval_all_user, dim=0).t().contiguous()
        Hit10, nDCG10 = eval_all_user
        mean_eval = eval_concat([Hit10, nDCG10], test_sampler)
        print_metrics(mean_eval, Log_file, v_or_t)
    return mean_eval[0]


def eval_model_amazon(topk, model, user_history, eval_seq, item_embeddings, test_batch_size, args, item_num, Log_file, v_or_t, local_rank):
    eval_dataset = BuildEvalDataset(u2seq=eval_seq, item_content=item_embeddings,
                                    max_seq_len=args.max_seq_len, item_num=item_num)
    test_sampler = SequentialDistributedSampler(eval_dataset, batch_size=test_batch_size)
    eval_dl = DataLoader(eval_dataset, batch_size=test_batch_size, \
    #                     num_workers=args.num_workers, pin_memory=False, sampler=test_sampler)
                         num_workers=args.num_workers, pin_memory=True, sampler=test_sampler)
    model.eval()
    topK = topk
    Log_file.info(v_or_t + "_methods   {}".format('\t'.join(['Hit{}'.format(topK), 'nDCG{}'.format(topK)])))
    item_embeddings = item_embeddings.to(local_rank)
    #print(item_embeddings.shape)
    with torch.no_grad():
        eval_all_user = []
        item_rank = torch.Tensor(np.arange(item_num) + 1).to(local_rank)
        for data in eval_dl:
            user_ids, input_embs, log_mask, labels = data
            user_ids, input_embs, log_mask, labels = \
                user_ids.to(local_rank), input_embs.to(local_rank),\
                log_mask.to(local_rank), labels.to(local_rank).detach()
            prec_emb = model.module.user_encoder(input_embs, log_mask, local_rank)[:, -1].detach()
            scores = torch.matmul(prec_emb, item_embeddings.t()).squeeze(dim=-1).detach()
            #print(scores.shape)
            for user_id, label, score in zip(user_ids, labels, scores):
                user_id = user_id[0].item()
                history = user_history[user_id].to(local_rank)
                score[history] = -np.inf
                score = score[1:]
                eval_all_user.append(metrics_topK(score, label, item_rank, topK, local_rank))
        eval_all_user = torch.stack(tensors=eval_all_user, dim=0).t().contiguous()
        Hit10, nDCG10 = eval_all_user
        mean_eval = eval_concat([Hit10, nDCG10], test_sampler)
        print_metrics(mean_eval, Log_file, v_or_t)
    return mean_eval[0]

def eval_model_2_2_amazon(topk, model, user_history, eval_seq, item_embeddings,id_embs, test_batch_size, args, item_num, Log_file, v_or_t, local_rank):
    eval_dataset = BuildEvalDataset_2(u2seq=eval_seq, item_content=item_embeddings, id_embs = id_embs, \
                                    max_seq_len=args.max_seq_len, item_num=item_num)
    #eval_dataset = BuildEvalDataset(u2seq=eval_seq, item_content=item_embeddings,
    #                                max_seq_len=args.max_seq_len, item_num=item_num)
    test_sampler = SequentialDistributedSampler(eval_dataset, batch_size=test_batch_size)
    eval_dl = DataLoader(eval_dataset, batch_size=test_batch_size,
                         num_workers=args.num_workers, pin_memory=True, sampler=test_sampler)
    model.eval()
    topK = topk
    ee=pd.read_csv('/dataset/dataset/amazon/Electronics_fre.csv')
    #ee.columns = ['id', 'fre']
    e=ee.sort_values('fre',ascending=False)
    #l = [0, 41, 127, 276, 517, 889, 1467, 2413, 4070, 7618, 79707, 79708]
    l=[0, 233, 856, 2090, 4327, 8214, 14950, 27012, 50742, 109145, 423192]
    e['bin']=0
    for k in range(10):
        e.iloc[l[k]:l[k+1],2]=k+1
    e=e.sort_values('id',ascending=True)
    bins=e['bin'].to_numpy()
    bins[0]=0
    act=torch.nn.Sigmoid()
    alpha = act(model.module.alpha[bins])
    beta = act(model.module.beta[bins])
    s=alpha+beta
    Log_file.info(v_or_t + "_methods   {}".format('\t'.join(['Hit{}'.format(topK), 'nDCG{}'.format(topK)])))
    item_embeddings = item_embeddings.to(local_rank)
    embs = id_embs.to(local_rank)
    with torch.no_grad():
        eval_all_user = []
        item_rank = torch.Tensor(np.arange(item_num) + 1).to(local_rank)
        for data in eval_dl:
            user_ids, input_embs, log_mask, labels, id_embs = data
            user_ids, input_embs, log_mask, labels, id_embs = \
                user_ids.to(local_rank), input_embs.to(local_rank),\
                log_mask.to(local_rank), labels.to(local_rank).detach(), id_embs.to(local_rank)
            prec_emb_2 = model.module.user_enc_2(input_embs, log_mask, local_rank)[:, -1].detach()
            prec_emb_1 = model.module.user_encoder(id_embs, log_mask, local_rank)[:, -1].detach()
            #print(item_embeddings.shape)
            #print(id_embs.shape)
            a = torch.matmul(prec_emb_2, item_embeddings.t()).squeeze(dim=-1).detach()
            #alpha = model.module.alpha[bin_pos].detach()
            #print(a.shape)
            scores_2 = beta/s*a
            #print(a.shape)
            #print(alpha.shape)
            #print(scores_2.shape)
            b = torch.matmul(prec_emb_1, embs.t()).squeeze(dim=-1).detach()
            #print(b.shape)
            #print()
            scores_1 = alpha/s*b
            scores=scores_2+scores_1
            for user_id, label, score in zip(user_ids, labels, scores):
                user_id = user_id[0].item()
                history = user_history[user_id].to(local_rank)
                score[history] = -np.inf
                score = score[1:]
                eval_all_user.append(metrics_topK(score, label, item_rank, topK, local_rank))
        eval_all_user = torch.stack(tensors=eval_all_user, dim=0).t().contiguous()
        Hit10, nDCG10 = eval_all_user
        mean_eval = eval_concat([Hit10, nDCG10], test_sampler)
        print_metrics(mean_eval, Log_file, v_or_t)
    return mean_eval[0]


def eval_model_2_3tower_amazon(topk, model, user_history, eval_seq,item_embeddings3, item_embeddings,id_embs, test_batch_size, args, item_num, Log_file, v_or_t, local_rank):
    eval_dataset = BuildEvalDataset_3(u2seq=eval_seq, item_content=item_embeddings, item_content3=item_embeddings3,id_embs = id_embs, \
                                    max_seq_len=args.max_seq_len, item_num=item_num)
    #eval_dataset = BuildEvalDataset(u2seq=eval_seq, item_content=item_embeddings,
    #                                max_seq_len=args.max_seq_len, item_num=item_num)
    test_sampler = SequentialDistributedSampler(eval_dataset, batch_size=test_batch_size)
    eval_dl = DataLoader(eval_dataset, batch_size=test_batch_size,
                         num_workers=args.num_workers, pin_memory=True, sampler=test_sampler)
    model.eval()
    topK = topk
    ee=pd.read_csv('/dataset/dataset/amazon/Electronics_fre.csv')
    #ee.columns = ['id', 'fre']
    e=ee.sort_values('fre',ascending=False)
    #l = [0, 41, 127, 276, 517, 889, 1467, 2413, 4070, 7618, 79707, 79708]
    l=[0, 233, 856, 2090, 4327, 8214, 14950, 27012, 50742, 109145, 423192]
    e['bin']=0
    for k in range(10):
        e.iloc[l[k]:l[k+1],2]=k+1
    e=e.sort_values('id',ascending=True)
    bins=e['bin'].to_numpy()
    bins[0]=0
    act=torch.nn.Sigmoid()
    alpha = act(model.module.alpha[bins])
    alpha2 = act(model.module.alpha2[bins])
    alpha3 = act(model.module.alpha3[bins])
    s=alpha+alpha2+alpha3
    Log_file.info(v_or_t + "_methods   {}".format('\t'.join(['Hit{}'.format(topK), 'nDCG{}'.format(topK)])))
    item_embeddings = item_embeddings.to(local_rank)
    item_embeddings3 = item_embeddings3.to(local_rank)
    embs = id_embs.to(local_rank)
    with torch.no_grad():
        eval_all_user = []
        item_rank = torch.Tensor(np.arange(item_num) + 1).to(local_rank)
        for data in eval_dl:
            user_ids, input_embs, emb3, log_mask, labels, id_embs = data
            user_ids, input_embs, emb3, log_mask, labels, id_embs = \
                user_ids.to(local_rank), input_embs.to(local_rank), emb3.to(local_rank),\
                log_mask.to(local_rank), labels.to(local_rank).detach(), id_embs.to(local_rank)
            prec_emb_2 = model.module.user_enc_2(input_embs, log_mask, local_rank)[:, -1].detach()
            prec_emb_3 = model.module.user_enc_3(emb3, log_mask, local_rank)[:, -1].detach()
            prec_emb_1 = model.module.user_encoder(id_embs, log_mask, local_rank)[:, -1].detach()
            #print(item_embeddings.shape)
            #print(id_embs.shape)
            a = torch.matmul(prec_emb_2, item_embeddings.t()).squeeze(dim=-1).detach()
            #alpha = model.module.alpha[bin_pos].detach()
            #print(a.shape)
            scores_2 = alpha2/s*a
            #print(a.shape)
            #print(alpha.shape)
            #print(scores_2.shape)
            b = torch.matmul(prec_emb_1, embs.t()).squeeze(dim=-1).detach()
            #print(b.shape)
            #print()
            scores_1 = alpha/s*b
            c = torch.matmul(prec_emb_3, item_embeddings3.t()).squeeze(dim=-1).detach()
            scores_3 = alpha3/s*c
            scores=scores_2+scores_1+scores_3
            for user_id, label, score in zip(user_ids, labels, scores):
                user_id = user_id[0].item()
                history = user_history[user_id].to(local_rank)
                score[history] = -np.inf
                score = score[1:]
                eval_all_user.append(metrics_topK(score, label, item_rank, topK, local_rank))
        eval_all_user = torch.stack(tensors=eval_all_user, dim=0).t().contiguous()
        Hit10, nDCG10 = eval_all_user
        mean_eval = eval_concat([Hit10, nDCG10], test_sampler)
        print_metrics(mean_eval, Log_file, v_or_t)
    return mean_eval[0]


def eval_model_2_2_amazon_pantry(topk, model, user_history, eval_seq, item_embeddings,id_embs, test_batch_size, args, item_num, Log_file, v_or_t, local_rank):
    eval_dataset = BuildEvalDataset_2(u2seq=eval_seq, item_content=item_embeddings, id_embs = id_embs, \
                                    max_seq_len=args.max_seq_len, item_num=item_num)
    #eval_dataset = BuildEvalDataset(u2seq=eval_seq, item_content=item_embeddings,
    #                                max_seq_len=args.max_seq_len, item_num=item_num)
    test_sampler = SequentialDistributedSampler(eval_dataset, batch_size=test_batch_size)
    eval_dl = DataLoader(eval_dataset, batch_size=test_batch_size,
                         num_workers=args.num_workers, pin_memory=True, sampler=test_sampler)
    model.eval()
    topK = topk
    ee=pd.read_csv('/dataset/dataset/amazon/Prime_Pantry_fre.csv')
    #ee.columns = ['id', 'fre']
    e=ee.sort_values('fre',ascending=False)
    #l = [0, 41, 127, 276, 517, 889, 1467, 2413, 4070, 7618, 79707, 79708]
    l=[0, 35, 100, 205, 361, 585, 898, 1357, 2074, 3348, 8348]
    e['bin']=0
    for k in range(10):
        e.iloc[l[k]:l[k+1],2]=k+1
    e=e.sort_values('id',ascending=True)
    bins=e['bin'].to_numpy()
    bins[0]=0
    act=torch.nn.Sigmoid()
    alpha = act(model.module.alpha[bins])
    beta = act(model.module.beta[bins])
    s=alpha+beta
    Log_file.info(v_or_t + "_methods   {}".format('\t'.join(['Hit{}'.format(topK), 'nDCG{}'.format(topK)])))
    item_embeddings = item_embeddings.to(local_rank)
    embs = id_embs.to(local_rank)
    with torch.no_grad():
        eval_all_user = []
        item_rank = torch.Tensor(np.arange(item_num) + 1).to(local_rank)
        for data in eval_dl:
            user_ids, input_embs, log_mask, labels, id_embs = data
            user_ids, input_embs, log_mask, labels, id_embs = \
                user_ids.to(local_rank), input_embs.to(local_rank),\
                log_mask.to(local_rank), labels.to(local_rank).detach(), id_embs.to(local_rank)
            prec_emb_2 = model.module.user_enc_2(input_embs, log_mask, local_rank)[:, -1].detach()
            prec_emb_1 = model.module.user_encoder(id_embs, log_mask, local_rank)[:, -1].detach()
            #print(item_embeddings.shape)
            #print(id_embs.shape)
            a = torch.matmul(prec_emb_2, item_embeddings.t()).squeeze(dim=-1).detach()
            #alpha = model.module.alpha[bin_pos].detach()
            #print(a.shape)
            scores_2 = beta/s*a
            #print(a.shape)
            #print(alpha.shape)
            #print(scores_2.shape)
            b = torch.matmul(prec_emb_1, embs.t()).squeeze(dim=-1).detach()
            #print(b.shape)
            #print()
            scores_1 = alpha/s*b
            scores=scores_2+scores_1
            for user_id, label, score in zip(user_ids, labels, scores):
                user_id = user_id[0].item()
                history = user_history[user_id].to(local_rank)
                score[history] = -np.inf
                score = score[1:]
                eval_all_user.append(metrics_topK(score, label, item_rank, topK, local_rank))
        eval_all_user = torch.stack(tensors=eval_all_user, dim=0).t().contiguous()
        Hit10, nDCG10 = eval_all_user
        mean_eval = eval_concat([Hit10, nDCG10], test_sampler)
        print_metrics(mean_eval, Log_file, v_or_t)
    return mean_eval[0]


def eval_model_2_3tower_amazon_pantry(topk, model, user_history, eval_seq,item_embeddings3, item_embeddings,id_embs, test_batch_size, args, item_num, Log_file, v_or_t, local_rank):
    eval_dataset = BuildEvalDataset_3(u2seq=eval_seq, item_content=item_embeddings, item_content3=item_embeddings3,id_embs = id_embs, \
                                    max_seq_len=args.max_seq_len, item_num=item_num)
    #eval_dataset = BuildEvalDataset(u2seq=eval_seq, item_content=item_embeddings,
    #                                max_seq_len=args.max_seq_len, item_num=item_num)
    test_sampler = SequentialDistributedSampler(eval_dataset, batch_size=test_batch_size)
    eval_dl = DataLoader(eval_dataset, batch_size=test_batch_size,
                         num_workers=args.num_workers, pin_memory=True, sampler=test_sampler)
    model.eval()
    topK = topk
    ee=pd.read_csv('/dataset/dataset/amazon/Prime_Pantry_fre.csv')
    #ee.columns = ['id', 'fre']
    e=ee.sort_values('fre',ascending=False)
    #l = [0, 41, 127, 276, 517, 889, 1467, 2413, 4070, 7618, 79707, 79708]
    l=[0, 35, 100, 205, 361, 585, 898, 1357, 2074, 3348, 8348]
    e['bin']=0
    for k in range(10):
        e.iloc[l[k]:l[k+1],2]=k+1
    e=e.sort_values('id',ascending=True)
    bins=e['bin'].to_numpy()
    bins[0]=0
    act=torch.nn.Sigmoid()
    alpha = act(model.module.alpha[bins])
    alpha2 = act(model.module.alpha2[bins])
    alpha3 = act(model.module.alpha3[bins])
    s=alpha+alpha2+alpha3
    Log_file.info(v_or_t + "_methods   {}".format('\t'.join(['Hit{}'.format(topK), 'nDCG{}'.format(topK)])))
    item_embeddings = item_embeddings.to(local_rank)
    item_embeddings3 = item_embeddings3.to(local_rank)
    embs = id_embs.to(local_rank)
    with torch.no_grad():
        eval_all_user = []
        item_rank = torch.Tensor(np.arange(item_num) + 1).to(local_rank)
        for data in eval_dl:
            user_ids, input_embs, emb3, log_mask, labels, id_embs = data
            user_ids, input_embs, emb3, log_mask, labels, id_embs = \
                user_ids.to(local_rank), input_embs.to(local_rank), emb3.to(local_rank),\
                log_mask.to(local_rank), labels.to(local_rank).detach(), id_embs.to(local_rank)
            prec_emb_2 = model.module.user_enc_2(input_embs, log_mask, local_rank)[:, -1].detach()
            prec_emb_3 = model.module.user_enc_3(emb3, log_mask, local_rank)[:, -1].detach()
            prec_emb_1 = model.module.user_encoder(id_embs, log_mask, local_rank)[:, -1].detach()
            #print(item_embeddings.shape)
            #print(id_embs.shape)
            a = torch.matmul(prec_emb_2, item_embeddings.t()).squeeze(dim=-1).detach()
            #alpha = model.module.alpha[bin_pos].detach()
            #print(a.shape)
            scores_2 = alpha2/s*a
            #print(a.shape)
            #print(alpha.shape)
            #print(scores_2.shape)
            b = torch.matmul(prec_emb_1, embs.t()).squeeze(dim=-1).detach()
            #print(b.shape)
            #print()
            scores_1 = alpha/s*b
            c = torch.matmul(prec_emb_3, item_embeddings3.t()).squeeze(dim=-1).detach()
            scores_3 = alpha3/s*c
            scores=scores_2+scores_1+scores_3
            for user_id, label, score in zip(user_ids, labels, scores):
                user_id = user_id[0].item()
                history = user_history[user_id].to(local_rank)
                score[history] = -np.inf
                score = score[1:]
                eval_all_user.append(metrics_topK(score, label, item_rank, topK, local_rank))
        eval_all_user = torch.stack(tensors=eval_all_user, dim=0).t().contiguous()
        Hit10, nDCG10 = eval_all_user
        mean_eval = eval_concat([Hit10, nDCG10], test_sampler)
        print_metrics(mean_eval, Log_file, v_or_t)
    return mean_eval[0]


def eval_model_amazon_step2(topk, model, user_history, eval_seq, item_embeddings, test_batch_size, args, item_num, Log_file, v_or_t, local_rank):
    eval_dataset = BuildEvalDataset(u2seq=eval_seq, item_content=item_embeddings,
                                    max_seq_len=args.max_seq_len, item_num=item_num)
    test_sampler = SequentialDistributedSampler(eval_dataset, batch_size=test_batch_size)
    eval_dl = DataLoader(eval_dataset, batch_size=test_batch_size, \
    #                     num_workers=args.num_workers, pin_memory=False, sampler=test_sampler)
                         num_workers=args.num_workers, pin_memory=True, sampler=test_sampler)
    model.eval()
    topK = topk
    Log_file.info(v_or_t + "_methods   {}".format('\t'.join(['Hit{}'.format(topK), 'nDCG{}'.format(topK)])))
    item_embeddings = item_embeddings.to(local_rank)
    with torch.no_grad():
        eval_all_user = []
        item_rank = torch.Tensor(np.arange(item_num) + 1).to(local_rank)
        for data in eval_dl:
            user_ids, input_embs, log_mask, labels = data
            user_ids, input_embs, log_mask, labels = \
                user_ids.to(local_rank), input_embs.to(local_rank),\
                log_mask.to(local_rank), labels.to(local_rank).detach()
            prec_emb = model.module.user_enc_2(input_embs, log_mask, local_rank)[:, -1].detach()
            scores = torch.matmul(prec_emb, item_embeddings.t()).squeeze(dim=-1).detach()
            #print(scores.shape)
            for user_id, label, score in zip(user_ids, labels, scores):
                user_id = user_id[0].item()
                history = user_history[user_id].to(local_rank)
                score[history] = -np.inf
                score = score[1:]
                eval_all_user.append(metrics_topK(score, label, item_rank, topK, local_rank))
        eval_all_user = torch.stack(tensors=eval_all_user, dim=0).t().contiguous()
        Hit10, nDCG10 = eval_all_user
        mean_eval = eval_concat([Hit10, nDCG10], test_sampler)
        print_metrics(mean_eval, Log_file, v_or_t)
    return mean_eval[0]


'''
def eval_model_amazon(model, user_history, item_embeddings, test_batch_size, args, item_num, Log_file, v_or_t, local_rank,eval_dl,test_sampler):
    #eval_dataset = BuildEvalDataset(u2seq=eval_seq, item_content=item_embeddings,
    #                                max_seq_len=args.max_seq_len, item_num=item_num)
    #test_sampler = SequentialDistributedSampler(eval_dataset, batch_size=test_batch_size)
    #eval_dl = DataLoader(eval_dataset, batch_size=test_batch_size,
    #                     num_workers=args.num_workers, pin_memory=False, sampler=test_sampler)
    model.eval()
    topK = 10
    Log_file.info(v_or_t + "_methods   {}".format('\t'.join(['Hit{}'.format(topK), 'nDCG{}'.format(topK)])))
    item_embeddings = item_embeddings.to(local_rank)
    with torch.no_grad():
        eval_all_user = []
        item_rank = torch.Tensor(np.arange(item_num) + 1).to(local_rank)
        for data in eval_dl:
            user_ids, input_embs, log_mask, labels = data
            user_ids, input_embs, log_mask, labels = \
                user_ids.to(local_rank), input_embs.to(local_rank),\
                log_mask.to(local_rank), labels.to(local_rank).detach()
            prec_emb = model.module.user_encoder(input_embs, log_mask, local_rank)[:, -1].detach()
            scores = torch.matmul(prec_emb, item_embeddings.t()).squeeze(dim=-1).detach()
            #print(scores.shape)
            for user_id, label, score in zip(user_ids, labels, scores):
                user_id = user_id[0].item()
                history = user_history[user_id].to(local_rank)
                score[history] = -np.inf
                score = score[1:]
                eval_all_user.append(metrics_topK(score, label, item_rank, topK, local_rank))
'''

def eval_model_caser_amazon(topk, model, user_history, eval_seq, item_embeddings, test_batch_size, args, item_num, Log_file, v_or_t, local_rank):
    eval_dataset = BuildEvalDataset(u2seq=eval_seq, item_content=item_embeddings,
                                    max_seq_len=args.max_seq_len, item_num=item_num)
    test_sampler = SequentialDistributedSampler(eval_dataset, batch_size=test_batch_size)
    eval_dl = DataLoader(eval_dataset, batch_size=test_batch_size, \
    #                     num_workers=args.num_workers, pin_memory=False, sampler=test_sampler)
                         num_workers=args.num_workers, pin_memory=True, sampler=test_sampler)
    model.eval()
    topK = topk
    Log_file.info(v_or_t + "_methods   {}".format('\t'.join(['Hit{}'.format(topK), 'nDCG{}'.format(topK)])))
    item_embeddings = item_embeddings.to(local_rank)
    #print(item_embeddings.shape)
    with torch.no_grad():
        eval_all_user = []
        item_rank = torch.Tensor(np.arange(item_num) + 1).to(local_rank)
        for data in eval_dl:
            user_ids, input_embs, log_mask, labels = data
            user_ids, input_embs, log_mask, labels = \
                user_ids.to(local_rank), input_embs.to(local_rank),\
                log_mask.to(local_rank), labels.to(local_rank).detach()
            #print(input_embs.shape)
            item_seq_emb = input_embs.unsqueeze(1)
            out_v = model.module.conv_v(item_seq_emb)
            out_v = out_v.view(-1, model.module.fc1_dim_v)  # prepare for fully connect
            out_hs = list()
            for conv in model.module.conv_h:
                conv_out = model.module.ac_conv(conv(item_seq_emb).squeeze(3))
                pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
                out_hs.append(pool_out)
            out_h = torch.cat(out_hs, 1)  # prepare for fully connect
            # Fully-connected Layers
            out = torch.cat([out_v, out_h], 1)
            # apply dropout
            #out = model.module.dropout(out)
            # fully-connected layer
            z = model.module.ac_fc(model.module.fc1(out))
            #z = model.module.fc1(out)
            #prec_emb = z.unsqueeze(1)
            #print(prec_emb.shape)
            #print('')
            #print(item_embeddings.shape)
            scores = torch.matmul(z, item_embeddings.t()).detach()
            #print(scores.shape)
            for user_id, label, score in zip(user_ids, labels, scores):
                user_id = user_id[0].item()
                history = user_history[user_id].to(local_rank)
                score[history] = -np.inf
                score = score[1:]
                eval_all_user.append(metrics_topK(score, label, item_rank, topK, local_rank))
        eval_all_user = torch.stack(tensors=eval_all_user, dim=0).t().contiguous()
        Hit10, nDCG10 = eval_all_user
        mean_eval = eval_concat([Hit10, nDCG10], test_sampler)
        print_metrics(mean_eval, Log_file, v_or_t)
    return mean_eval[0]

def eval_model_step2_caser(model, user_history, eval_seq, item_embeddings, test_batch_size, args, item_num, Log_file, v_or_t, local_rank):
    eval_dataset = BuildEvalDataset(u2seq=eval_seq, item_content=item_embeddings,
                                    max_seq_len=args.max_seq_len, item_num=item_num)
    test_sampler = SequentialDistributedSampler(eval_dataset, batch_size=test_batch_size)
    eval_dl = DataLoader(eval_dataset, batch_size=test_batch_size,
                         num_workers=args.num_workers, pin_memory=True, sampler=test_sampler)
    model.eval()
    topK = 10
    Log_file.info(v_or_t + "_methods   {}".format('\t'.join(['Hit{}'.format(topK), 'nDCG{}'.format(topK)])))
    item_embeddings = item_embeddings.to(local_rank)
    with torch.no_grad():
        eval_all_user = []
        item_rank = torch.Tensor(np.arange(item_num) + 1).to(local_rank)
        for data in eval_dl:
            user_ids, input_embs, log_mask, labels = data
            user_ids, input_embs, log_mask, labels = \
                user_ids.to(local_rank), input_embs.to(local_rank),\
                log_mask.to(local_rank), labels.to(local_rank).detach()
            
            item_seq_emb = input_embs.unsqueeze(1) #16*1*20*128
            # Convolutional Layers
            out, out_h, out_v = None, None, None
            # vertical conv layer
            out_v = model.module.conv_v2(item_seq_emb)
            out_v = out_v.view(-1, model.module.fc1_dim_v)  # prepare for fully connect

            # horizontal conv layer
            out_hs = list()
            for conv in model.module.conv_h2:
                conv_out = model.module.ac_conv(conv(item_seq_emb).squeeze(3))
                pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
                out_hs.append(pool_out)
            out_h = torch.cat(out_hs, 1)  # prepare for fully connect
    
            # Fully-connected Layers
            out = torch.cat([out_v, out_h], 1)
            prec_emb = model.module.ac_fc(model.module.fc12(out)).detach()
        
            #prec_emb = model.module.user_enc_2(input_embs, log_mask, local_rank)[:, -1].detach()
            scores = torch.matmul(prec_emb, item_embeddings.t()).detach()
            #print(scores.shape)
            for user_id, label, score in zip(user_ids, labels, scores):
                user_id = user_id[0].item()
                history = user_history[user_id].to(local_rank)
                score[history] = -np.inf
                score = score[1:]
                eval_all_user.append(metrics_topK(score, label, item_rank, topK, local_rank))
        eval_all_user = torch.stack(tensors=eval_all_user, dim=0).t().contiguous()
        Hit10, nDCG10 = eval_all_user
        mean_eval = eval_concat([Hit10, nDCG10], test_sampler)
        print_metrics(mean_eval, Log_file, v_or_t)
    return mean_eval[0]


def eval_model_2_3tower_amazon_pantry_caser(topk, model, user_history, eval_seq,item_embeddings3, item_embeddings,id_embs, test_batch_size, args, item_num, Log_file, v_or_t, local_rank):
    eval_dataset = BuildEvalDataset_3(u2seq=eval_seq, item_content=item_embeddings, item_content3=item_embeddings3,id_embs = id_embs, \
                                    max_seq_len=args.max_seq_len, item_num=item_num)
    #eval_dataset = BuildEvalDataset(u2seq=eval_seq, item_content=item_embeddings,
    #                                max_seq_len=args.max_seq_len, item_num=item_num)
    test_sampler = SequentialDistributedSampler(eval_dataset, batch_size=test_batch_size)
    eval_dl = DataLoader(eval_dataset, batch_size=test_batch_size,
                         num_workers=args.num_workers, pin_memory=True, sampler=test_sampler)
    model.eval()
    topK = topk
    ee=pd.read_csv('/dataset/dataset/amazon/Prime_Pantry_fre.csv')
    #ee.columns = ['id', 'fre']
    e=ee.sort_values('fre',ascending=False)
    #l = [0, 41, 127, 276, 517, 889, 1467, 2413, 4070, 7618, 79707, 79708]
    l=[0, 35, 100, 205, 361, 585, 898, 1357, 2074, 3348, 8348]
    e['bin']=0
    for k in range(10):
        e.iloc[l[k]:l[k+1],2]=k+1
    e=e.sort_values('id',ascending=True)
    bins=e['bin'].to_numpy()
    bins[0]=0
    act=torch.nn.Sigmoid()
    alpha = act(model.module.alpha[bins])
    alpha2 = act(model.module.alpha2[bins])
    alpha3 = act(model.module.alpha3[bins])
    s=alpha+alpha2+alpha3
    Log_file.info(v_or_t + "_methods   {}".format('\t'.join(['Hit{}'.format(topK), 'nDCG{}'.format(topK)])))
    item_embeddings = item_embeddings.to(local_rank)
    item_embeddings3 = item_embeddings3.to(local_rank)
    embs = id_embs.to(local_rank)
    with torch.no_grad():
        eval_all_user = []
        item_rank = torch.Tensor(np.arange(item_num) + 1).to(local_rank)
        for data in eval_dl:
            user_ids, input_embs, emb3, log_mask, labels, id_embs = data
            user_ids, input_embs, emb3, log_mask, labels, id_embs = \
                user_ids.to(local_rank), input_embs.to(local_rank), emb3.to(local_rank),\
                log_mask.to(local_rank), labels.to(local_rank).detach(), id_embs.to(local_rank)
            
            
            #part 2
            item_seq_emb = input_embs.unsqueeze(1) #16*1*20*128
            # Convolutional Layers
            out, out_h, out_v = None, None, None
            # vertical conv layer
            out_v = model.module.conv_v2(item_seq_emb)
            out_v = out_v.view(-1, model.module.fc1_dim_v)  # prepare for fully connect

            # horizontal conv layer
            out_hs = list()
            for conv in model.module.conv_h2:
                conv_out = model.module.ac_conv(conv(item_seq_emb).squeeze(3))
                pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
                out_hs.append(pool_out)
            out_h = torch.cat(out_hs, 1)  # prepare for fully connect
    
            # Fully-connected Layers
            out = torch.cat([out_v, out_h], 1)
            prec_emb_2 = (model.module.fc12(out)).detach()
            
            #part 3
            item_seq_emb = emb3.unsqueeze(1) #16*1*20*128
            # Convolutional Layers
            out, out_h, out_v = None, None, None
            # vertical conv layer
            out_v = model.module.conv_v3(item_seq_emb)
            out_v = out_v.view(-1, model.module.fc1_dim_v)  # prepare for fully connect

            # horizontal conv layer
            out_hs = list()
            for conv in model.module.conv_h3:
                conv_out = model.module.ac_conv(conv(item_seq_emb).squeeze(3))
                pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
                out_hs.append(pool_out)
            out_h = torch.cat(out_hs, 1)  # prepare for fully connect
    
            # Fully-connected Layers
            out = torch.cat([out_v, out_h], 1)
            prec_emb_3 = (model.module.fc13(out)).detach()
            
            #part 1
            item_seq_emb = id_embs.unsqueeze(1) #16*1*20*128
            # Convolutional Layers
            out, out_h, out_v = None, None, None
            # vertical conv layer
            out_v = model.module.conv_v(item_seq_emb)
            out_v = out_v.view(-1, model.module.fc1_dim_v)  # prepare for fully connect

            # horizontal conv layer
            out_hs = list()
            for conv in model.module.conv_h:
                conv_out = model.module.ac_conv(conv(item_seq_emb).squeeze(3))
                pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
                out_hs.append(pool_out)
            out_h = torch.cat(out_hs, 1)  # prepare for fully connect
    
            # Fully-connected Layers
            out = torch.cat([out_v, out_h], 1)
            prec_emb_1 = (model.module.fc1(out)).detach()
            
            #print(item_embeddings.shape)
            #print(id_embs.shape)
            a = torch.matmul(prec_emb_2, item_embeddings.t()).detach()
            #alpha = model.module.alpha[bin_pos].detach()
            #print(a.shape)
            scores_2 = alpha2/s*a
            #print(a.shape)
            #print(alpha.shape)
            #print(scores_2.shape)
            b = torch.matmul(prec_emb_1, embs.t()).detach()
            #print(b.shape)
            #print()
            scores_1 = alpha/s*b
            c = torch.matmul(prec_emb_3, item_embeddings3.t()).detach()
            scores_3 = alpha3/s*c
            scores=scores_2+scores_1+scores_3
            
            for user_id, label, score in zip(user_ids, labels, scores):
                user_id = user_id[0].item()
                history = user_history[user_id].to(local_rank)
                score[history] = -np.inf
                score = score[1:]
                eval_all_user.append(metrics_topK(score, label, item_rank, topK, local_rank))
        eval_all_user = torch.stack(tensors=eval_all_user, dim=0).t().contiguous()
        Hit10, nDCG10 = eval_all_user
        mean_eval = eval_concat([Hit10, nDCG10], test_sampler)
        print_metrics(mean_eval, Log_file, v_or_t)
    return mean_eval[0]


def eval_model_2_junguang_amazon_pantry_caser(topk, model, user_history, eval_seq,item_embeddings3, item_embeddings,id_embs, test_batch_size, args, item_num, Log_file, v_or_t, local_rank):
    eval_dataset = BuildEvalDataset_3(u2seq=eval_seq, item_content=item_embeddings, item_content3=item_embeddings3,id_embs = id_embs, \
                                    max_seq_len=args.max_seq_len, item_num=item_num)
    #eval_dataset = BuildEvalDataset(u2seq=eval_seq, item_content=item_embeddings,
    #                                max_seq_len=args.max_seq_len, item_num=item_num)
    test_sampler = SequentialDistributedSampler(eval_dataset, batch_size=test_batch_size)
    eval_dl = DataLoader(eval_dataset, batch_size=test_batch_size,
                         num_workers=args.num_workers, pin_memory=True, sampler=test_sampler)
    model.eval()
    topK = topk
    ee=pd.read_csv('/dataset/dataset/amazon/Prime_Pantry_fre.csv')
    #ee.columns = ['id', 'fre']
    e=ee.sort_values('fre',ascending=False)
    #l = [0, 41, 127, 276, 517, 889, 1467, 2413, 4070, 7618, 79707, 79708]
    l=[0, 35, 100, 205, 361, 585, 898, 1357, 2074, 3348, 8348]
    e['bin']=0
    for k in range(10):
        e.iloc[l[k]:l[k+1],2]=k+1
    e=e.sort_values('id',ascending=True)
    bins=e['bin'].to_numpy()
    bins[0]=0
    act=torch.nn.Sigmoid()
    alpha = act(model.module.alpha[bins])
    alpha2 = act(model.module.alpha2[bins])
    alpha3 = act(model.module.alpha3[bins])
    s=alpha+alpha3
    Log_file.info(v_or_t + "_methods   {}".format('\t'.join(['Hit{}'.format(topK), 'nDCG{}'.format(topK)])))
    item_embeddings = item_embeddings.to(local_rank)
    item_embeddings3 = item_embeddings3.to(local_rank)
    embs = id_embs.to(local_rank)
    with torch.no_grad():
        eval_all_user = []
        item_rank = torch.Tensor(np.arange(item_num) + 1).to(local_rank)
        for data in eval_dl:
            user_ids, input_embs, emb3, log_mask, labels, id_embs = data
            user_ids, input_embs, emb3, log_mask, labels, id_embs = \
                user_ids.to(local_rank), input_embs.to(local_rank), emb3.to(local_rank),\
                log_mask.to(local_rank), labels.to(local_rank).detach(), id_embs.to(local_rank)
            
            
            #part 2
            item_seq_emb = input_embs.unsqueeze(1) #16*1*20*128
            # Convolutional Layers
            out, out_h, out_v = None, None, None
            # vertical conv layer
            out_v = model.module.conv_v2(item_seq_emb)
            out_v = out_v.view(-1, model.module.fc1_dim_v)  # prepare for fully connect

            # horizontal conv layer
            out_hs = list()
            for conv in model.module.conv_h2:
                conv_out = model.module.ac_conv(conv(item_seq_emb).squeeze(3))
                pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
                out_hs.append(pool_out)
            out_h = torch.cat(out_hs, 1)  # prepare for fully connect
    
            # Fully-connected Layers
            out = torch.cat([out_v, out_h], 1)
            prec_emb_2 = model.module.ac_fc(model.module.fc12(out)).detach()
            
            #part 3
            item_seq_emb = emb3.unsqueeze(1) #16*1*20*128
            # Convolutional Layers
            out, out_h, out_v = None, None, None
            # vertical conv layer
            out_v = model.module.conv_v3(item_seq_emb)
            out_v = out_v.view(-1, model.module.fc1_dim_v)  # prepare for fully connect

            # horizontal conv layer
            out_hs = list()
            for conv in model.module.conv_h3:
                conv_out = model.module.ac_conv(conv(item_seq_emb).squeeze(3))
                pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
                out_hs.append(pool_out)
            out_h = torch.cat(out_hs, 1)  # prepare for fully connect
    
            # Fully-connected Layers
            out = torch.cat([out_v, out_h], 1)
            prec_emb_3 = model.module.ac_fc(model.module.fc13(out)).detach()
            
            #part 1
            item_seq_emb = id_embs.unsqueeze(1) #16*1*20*128
            # Convolutional Layers
            out, out_h, out_v = None, None, None
            # vertical conv layer
            out_v = model.module.conv_v(item_seq_emb)
            out_v = out_v.view(-1, model.module.fc1_dim_v)  # prepare for fully connect

            # horizontal conv layer
            out_hs = list()
            for conv in model.module.conv_h:
                conv_out = model.module.ac_conv(conv(item_seq_emb).squeeze(3))
                pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
                out_hs.append(pool_out)
            out_h = torch.cat(out_hs, 1)  # prepare for fully connect
    
            # Fully-connected Layers
            out = torch.cat([out_v, out_h], 1)
            prec_emb_1 = model.module.ac_fc(model.module.fc1(out)).detach()
            
            #print(item_embeddings.shape)
            #print(id_embs.shape)
            a = torch.matmul(prec_emb_2, item_embeddings.t()).detach()
            #alpha = model.module.alpha[bin_pos].detach()
            #print(a.shape)
            scores_2 = alpha2/s*a
            #print(a.shape)
            #print(alpha.shape)
            #print(scores_2.shape)
            b = torch.matmul(prec_emb_1, embs.t()).detach()
            #print(b.shape)
            #print()
            scores_1 = alpha/s*b
            c = torch.matmul(prec_emb_3, item_embeddings3.t()).detach()
            scores_3 = alpha3/s*c
            scores=scores_1+scores_3
            
            for user_id, label, score in zip(user_ids, labels, scores):
                #print(score.shape)
                user_id = user_id[0].item()
                history = user_history[user_id].to(local_rank)
                score[history] = -np.inf
                score = score[1:]
                eval_all_user.append(metrics_topK(score, label, item_rank, topK, local_rank))
        eval_all_user = torch.stack(tensors=eval_all_user, dim=0).t().contiguous()
        Hit10, nDCG10 = eval_all_user
        mean_eval = eval_concat([Hit10, nDCG10], test_sampler)
        print_metrics(mean_eval, Log_file, v_or_t)
    return mean_eval[0]

def eval_model_step2_gru(model, user_history, eval_seq, item_embeddings, test_batch_size, args, item_num, Log_file, v_or_t, local_rank):
    eval_dataset = BuildEvalDataset(u2seq=eval_seq, item_content=item_embeddings,
                                    max_seq_len=args.max_seq_len, item_num=item_num)
    test_sampler = SequentialDistributedSampler(eval_dataset, batch_size=test_batch_size)
    eval_dl = DataLoader(eval_dataset, batch_size=test_batch_size,
                         num_workers=args.num_workers, pin_memory=True, sampler=test_sampler)
    model.eval()
    topK = 10
    Log_file.info(v_or_t + "_methods   {}".format('\t'.join(['Hit{}'.format(topK), 'nDCG{}'.format(topK)])))
    item_embeddings = item_embeddings.to(local_rank)
    with torch.no_grad():
        eval_all_user = []
        item_rank = torch.Tensor(np.arange(item_num) + 1).to(local_rank)
        for data in eval_dl:
            user_ids, input_embs, log_mask, labels = data
            user_ids, input_embs, log_mask, labels = \
                user_ids.to(local_rank), input_embs.to(local_rank),\
                log_mask.to(local_rank), labels.to(local_rank).detach()
            gru_output, _=model.module.gru_layers_2(input_embs)
            prec_emb = model.module.dense_2(gru_output)[:, -1].detach()
            '''
            item_seq_emb = input_embs.unsqueeze(1) #16*1*20*128
            # Convolutional Layers
            out, out_h, out_v = None, None, None
            # vertical conv layer
            out_v = model.module.conv_v2(item_seq_emb)
            out_v = out_v.view(-1, model.module.fc1_dim_v)  # prepare for fully connect

            # horizontal conv layer
            out_hs = list()
            for conv in model.module.conv_h2:
                conv_out = model.module.ac_conv(conv(item_seq_emb).squeeze(3))
                pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
                out_hs.append(pool_out)
            out_h = torch.cat(out_hs, 1)  # prepare for fully connect
    
            # Fully-connected Layers
            out = torch.cat([out_v, out_h], 1)
            prec_emb = model.module.ac_fc(model.module.fc12(out)).detach()
            '''
            #prec_emb = model.module.user_enc_2(input_embs, log_mask, local_rank)[:, -1].detach()
            scores = torch.matmul(prec_emb, item_embeddings.t()).detach()
            #print(scores.shape)
            for user_id, label, score in zip(user_ids, labels, scores):
                user_id = user_id[0].item()
                history = user_history[user_id].to(local_rank)
                score[history] = -np.inf
                score = score[1:]
                eval_all_user.append(metrics_topK(score, label, item_rank, topK, local_rank))
        eval_all_user = torch.stack(tensors=eval_all_user, dim=0).t().contiguous()
        Hit10, nDCG10 = eval_all_user
        mean_eval = eval_concat([Hit10, nDCG10], test_sampler)
        print_metrics(mean_eval, Log_file, v_or_t)
    return mean_eval[0]


def eval_model_2_3tower_amazon_pantry_gru(topk, model, user_history, eval_seq,item_embeddings3, item_embeddings,id_embs, test_batch_size, args, item_num, Log_file, v_or_t, local_rank):
    eval_dataset = BuildEvalDataset_3(u2seq=eval_seq, item_content=item_embeddings, item_content3=item_embeddings3,id_embs = id_embs, \
                                    max_seq_len=args.max_seq_len, item_num=item_num)
    #eval_dataset = BuildEvalDataset(u2seq=eval_seq, item_content=item_embeddings,
    #                                max_seq_len=args.max_seq_len, item_num=item_num)
    test_sampler = SequentialDistributedSampler(eval_dataset, batch_size=test_batch_size)
    eval_dl = DataLoader(eval_dataset, batch_size=test_batch_size,
                         num_workers=args.num_workers, pin_memory=True, sampler=test_sampler)
    model.eval()
    topK = topk
    ee=pd.read_csv('/dataset/dataset/amazon/Prime_Pantry_fre.csv')
    #ee.columns = ['id', 'fre']
    e=ee.sort_values('fre',ascending=False)
    #l = [0, 41, 127, 276, 517, 889, 1467, 2413, 4070, 7618, 79707, 79708]
    l=[0, 35, 100, 205, 361, 585, 898, 1357, 2074, 3348, 8348]
    e['bin']=0
    for k in range(10):
        e.iloc[l[k]:l[k+1],2]=k+1
    e=e.sort_values('id',ascending=True)
    bins=e['bin'].to_numpy()
    bins[0]=0
    act=torch.nn.Sigmoid()
    alpha = act(model.module.alpha[bins])
    alpha2 = act(model.module.alpha2[bins])
    alpha3 = act(model.module.alpha3[bins])
    s=alpha+alpha2+alpha3
    Log_file.info(v_or_t + "_methods   {}".format('\t'.join(['Hit{}'.format(topK), 'nDCG{}'.format(topK)])))
    item_embeddings = item_embeddings.to(local_rank)
    item_embeddings3 = item_embeddings3.to(local_rank)
    embs = id_embs.to(local_rank)
    with torch.no_grad():
        eval_all_user = []
        item_rank = torch.Tensor(np.arange(item_num) + 1).to(local_rank)
        for data in eval_dl:
            user_ids, input_embs, emb3, log_mask, labels, id_embs = data
            user_ids, input_embs, emb3, log_mask, labels, id_embs = \
                user_ids.to(local_rank), input_embs.to(local_rank), emb3.to(local_rank),\
                log_mask.to(local_rank), labels.to(local_rank).detach(), id_embs.to(local_rank)
            #prec_emb_2 = model.module.user_enc_2(input_embs, log_mask, local_rank)[:, -1].detach()
            #prec_emb_3 = model.module.user_enc_3(emb3, log_mask, local_rank)[:, -1].detach()
            #prec_emb_1 = model.module.user_encoder(id_embs, log_mask, local_rank)[:, -1].detach()
            gru_output, _=model.module.gru_layers_2(input_embs)
            prec_emb_2 = model.module.dense_2(gru_output)[:, -1].detach()
            gru_output, _=model.module.gru_layers_3(input_embs)
            prec_emb_3 = model.module.dense_3(gru_output)[:, -1].detach()
            gru_output, _=model.module.gru_layers(input_embs)
            prec_emb_1 = model.module.dense(gru_output)[:, -1].detach()
            
            a = torch.matmul(prec_emb_2, item_embeddings.t()).squeeze(dim=-1).detach()
            #alpha = model.module.alpha[bin_pos].detach()
            #print(a.shape)
            scores_2 = alpha2/s*a
            #print(a.shape)
            #print(alpha.shape)
            #print(scores_2.shape)
            b = torch.matmul(prec_emb_1, embs.t()).squeeze(dim=-1).detach()
            #print(b.shape)
            #print()
            scores_1 = alpha/s*b
            c = torch.matmul(prec_emb_3, item_embeddings3.t()).squeeze(dim=-1).detach()
            scores_3 = alpha3/s*c
            scores=scores_2+scores_1+scores_3
            for user_id, label, score in zip(user_ids, labels, scores):
                user_id = user_id[0].item()
                history = user_history[user_id].to(local_rank)
                score[history] = -np.inf
                score = score[1:]
                eval_all_user.append(metrics_topK(score, label, item_rank, topK, local_rank))
        eval_all_user = torch.stack(tensors=eval_all_user, dim=0).t().contiguous()
        Hit10, nDCG10 = eval_all_user
        mean_eval = eval_concat([Hit10, nDCG10], test_sampler)
        print_metrics(mean_eval, Log_file, v_or_t)
    return mean_eval[0]


def eval_model_2_3tower_amazon_gru(topk, model, user_history, eval_seq,item_embeddings3, item_embeddings,id_embs, test_batch_size, args, item_num, Log_file, v_or_t, local_rank):
    eval_dataset = BuildEvalDataset_3(u2seq=eval_seq, item_content=item_embeddings, item_content3=item_embeddings3,id_embs = id_embs, \
                                    max_seq_len=args.max_seq_len, item_num=item_num)
    #eval_dataset = BuildEvalDataset(u2seq=eval_seq, item_content=item_embeddings,
    #                                max_seq_len=args.max_seq_len, item_num=item_num)
    test_sampler = SequentialDistributedSampler(eval_dataset, batch_size=test_batch_size)
    eval_dl = DataLoader(eval_dataset, batch_size=test_batch_size,
                         num_workers=args.num_workers, pin_memory=True, sampler=test_sampler)
    model.eval()
    topK = topk
    ee=pd.read_csv('/dataset/dataset/amazon/Electronics_fre.csv')
    #ee.columns = ['id', 'fre']
    e=ee.sort_values('fre',ascending=False)
    #l = [0, 41, 127, 276, 517, 889, 1467, 2413, 4070, 7618, 79707, 79708]
    l=[0, 233, 856, 2090, 4327, 8214, 14950, 27012, 50742, 109145, 423192]
    e['bin']=0
    for k in range(10):
        e.iloc[l[k]:l[k+1],2]=k+1
    e=e.sort_values('id',ascending=True)
    bins=e['bin'].to_numpy()
    bins[0]=0
    act=torch.nn.Sigmoid()
    alpha = act(model.module.alpha[bins])
    alpha2 = act(model.module.alpha2[bins])
    alpha3 = act(model.module.alpha3[bins])
    s=alpha+alpha2+alpha3
    Log_file.info(v_or_t + "_methods   {}".format('\t'.join(['Hit{}'.format(topK), 'nDCG{}'.format(topK)])))
    item_embeddings = item_embeddings.to(local_rank)
    item_embeddings3 = item_embeddings3.to(local_rank)
    embs = id_embs.to(local_rank)
    with torch.no_grad():
        eval_all_user = []
        item_rank = torch.Tensor(np.arange(item_num) + 1).to(local_rank)
        for data in eval_dl:
            user_ids, input_embs, emb3, log_mask, labels, id_embs = data
            user_ids, input_embs, emb3, log_mask, labels, id_embs = \
                user_ids.to(local_rank), input_embs.to(local_rank), emb3.to(local_rank),\
                log_mask.to(local_rank), labels.to(local_rank).detach(), id_embs.to(local_rank)
            #prec_emb_2 = model.module.user_enc_2(input_embs, log_mask, local_rank)[:, -1].detach()
            #prec_emb_3 = model.module.user_enc_3(emb3, log_mask, local_rank)[:, -1].detach()
            #prec_emb_1 = model.module.user_encoder(id_embs, log_mask, local_rank)[:, -1].detach()
            gru_output, _=model.module.gru_layers_2(input_embs)
            prec_emb_2 = model.module.dense_2(gru_output)[:, -1].detach()
            gru_output, _=model.module.gru_layers_3(input_embs)
            prec_emb_3 = model.module.dense_3(gru_output)[:, -1].detach()
            gru_output, _=model.module.gru_layers(input_embs)
            prec_emb_1 = model.module.dense(gru_output)[:, -1].detach()
            
            a = torch.matmul(prec_emb_2, item_embeddings.t()).squeeze(dim=-1).detach()
            #alpha = model.module.alpha[bin_pos].detach()
            #print(a.shape)
            scores_2 = alpha2/s*a
            #print(a.shape)
            #print(alpha.shape)
            #print(scores_2.shape)
            b = torch.matmul(prec_emb_1, embs.t()).squeeze(dim=-1).detach()
            #print(b.shape)
            #print()
            scores_1 = alpha/s*b
            c = torch.matmul(prec_emb_3, item_embeddings3.t()).squeeze(dim=-1).detach()
            scores_3 = alpha3/s*c
            scores=scores_2+scores_1+scores_3
            for user_id, label, score in zip(user_ids, labels, scores):
                user_id = user_id[0].item()
                history = user_history[user_id].to(local_rank)
                score[history] = -np.inf
                score = score[1:]
                eval_all_user.append(metrics_topK(score, label, item_rank, topK, local_rank))
        eval_all_user = torch.stack(tensors=eval_all_user, dim=0).t().contiguous()
        Hit10, nDCG10 = eval_all_user
        mean_eval = eval_concat([Hit10, nDCG10], test_sampler)
        print_metrics(mean_eval, Log_file, v_or_t)
    return mean_eval[0]


def eval_model_2_3tower_mind_gru(topk, model, user_history, eval_seq,item_embeddings3, item_embeddings,id_embs, test_batch_size, args, item_num, Log_file, v_or_t, local_rank):
    eval_dataset = BuildEvalDataset_3(u2seq=eval_seq, item_content=item_embeddings, item_content3=item_embeddings3,id_embs = id_embs, \
                                    max_seq_len=args.max_seq_len, item_num=item_num)
    #eval_dataset = BuildEvalDataset(u2seq=eval_seq, item_content=item_embeddings,
    #                                max_seq_len=args.max_seq_len, item_num=item_num)
    test_sampler = SequentialDistributedSampler(eval_dataset, batch_size=test_batch_size)
    eval_dl = DataLoader(eval_dataset, batch_size=test_batch_size,
                         num_workers=args.num_workers, pin_memory=True, sampler=test_sampler)
    model.eval()
    topK = topk
    ee=pd.read_csv('/dataset/fre.csv')
    ee.columns = ['id', 'fre']
    e=ee.sort_values('fre',ascending=False)
    l = [0, 41, 127, 276, 517, 889, 1467, 2413, 4070, 7618, 79707, 79708]
    e['bin']=0
    for k in range(10):
        e.iloc[l[k]:l[k+1],2]=k+1
    e=e.sort_values('id',ascending=True)
    bins=e['bin'].to_numpy()
    act=torch.nn.Sigmoid()
    alpha = act(model.module.alpha[bins])
    alpha2 = act(model.module.alpha2[bins])
    alpha3 = act(model.module.alpha3[bins])
    s=alpha+alpha2+alpha3
    Log_file.info(v_or_t + "_methods   {}".format('\t'.join(['Hit{}'.format(topK), 'nDCG{}'.format(topK)])))
    item_embeddings = item_embeddings.to(local_rank)
    item_embeddings3 = item_embeddings3.to(local_rank)
    embs = id_embs.to(local_rank)
    with torch.no_grad():
        eval_all_user = []
        item_rank = torch.Tensor(np.arange(item_num) + 1).to(local_rank)
        for data in eval_dl:
            user_ids, input_embs, emb3, log_mask, labels, id_embs = data
            user_ids, input_embs, emb3, log_mask, labels, id_embs = \
                user_ids.to(local_rank), input_embs.to(local_rank), emb3.to(local_rank),\
                log_mask.to(local_rank), labels.to(local_rank).detach(), id_embs.to(local_rank)
            #prec_emb_2 = model.module.user_enc_2(input_embs, log_mask, local_rank)[:, -1].detach()
            #prec_emb_3 = model.module.user_enc_3(emb3, log_mask, local_rank)[:, -1].detach()
            #prec_emb_1 = model.module.user_encoder(id_embs, log_mask, local_rank)[:, -1].detach()
            gru_output, _=model.module.gru_layers_2(input_embs)
            prec_emb_2 = model.module.dense_2(gru_output)[:, -1].detach()
            gru_output, _=model.module.gru_layers_3(input_embs)
            prec_emb_3 = model.module.dense_3(gru_output)[:, -1].detach()
            gru_output, _=model.module.gru_layers(input_embs)
            prec_emb_1 = model.module.dense(gru_output)[:, -1].detach()
            
            a = torch.matmul(prec_emb_2, item_embeddings.t()).squeeze(dim=-1).detach()
            #alpha = model.module.alpha[bin_pos].detach()
            #print(a.shape)
            scores_2 = alpha2/s*a
            #print(a.shape)
            #print(alpha.shape)
            #print(scores_2.shape)
            b = torch.matmul(prec_emb_1, embs.t()).squeeze(dim=-1).detach()
            #print(b.shape)
            #print()
            scores_1 = alpha/s*b
            c = torch.matmul(prec_emb_3, item_embeddings3.t()).squeeze(dim=-1).detach()
            scores_3 = alpha3/s*c
            scores=scores_2+scores_1+scores_3
            for user_id, label, score in zip(user_ids, labels, scores):
                user_id = user_id[0].item()
                history = user_history[user_id].to(local_rank)
                score[history] = -np.inf
                score = score[1:]
                eval_all_user.append(metrics_topK(score, label, item_rank, topK, local_rank))
        eval_all_user = torch.stack(tensors=eval_all_user, dim=0).t().contiguous()
        Hit10, nDCG10 = eval_all_user
        mean_eval = eval_concat([Hit10, nDCG10], test_sampler)
        print_metrics(mean_eval, Log_file, v_or_t)
    return mean_eval[0]


