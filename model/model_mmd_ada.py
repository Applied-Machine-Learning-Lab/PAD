

import torch
import numpy as np
from torch import nn
from .encoders import Bert_Encoder, FC_Layers, User_Encoder, ADD, CAT, MLP_Layers
from torch.nn.init import xavier_normal_
from tllib.modules.kernels import GaussianKernel
from tllib.alignment.dan import MultipleKernelMaximumMeanDiscrepancy
import torch.nn.functional as F
import copy
from info_nce import InfoNCE, info_nce
from typing import Optional

class LinearKernel(nn.Module):
    def __init__(self, sigma: Optional[float] = None, track_running_stats: Optional[bool] = True,
                 alpha: Optional[float] = 1.):
        super(LinearKernel, self).__init__()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X.mm(X.t())

class CosKernel(nn.Module):
    def __init__(self, sigma: Optional[float] = None, track_running_stats: Optional[bool] = True,
                 eps: Optional[float] = 1e-8):
        super(CosKernel, self).__init__()
        self.eps = eps

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        d=X.norm(dim=1)[:, None]
        #e=X/d.unsqueeze(1)
        #f=1-e.mm(e.t())
        #print('tensor X')
        #print(X)
        #print('distance')
        #print(f)
        e = X/torch.clamp(d, min=self.eps)
        return 1-e.mm(e.t())

class LapKernel(nn.Module):
    def __init__(self, sigma: Optional[float] = None, track_running_stats: Optional[bool] = True,
                 alpha: Optional[float] = 1.):
        super(LapKernel, self).__init__()
        assert track_running_stats or sigma is not None
        self.sigma_square = torch.tensor(sigma * sigma) if sigma is not None else None
        self.track_running_stats = track_running_stats
        self.alpha = alpha

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        man_distance_square = (torch.abs(X.unsqueeze(0) - X.unsqueeze(1))).sum(2)

        if self.track_running_stats:
            self.sigma_square = self.alpha * torch.mean(man_distance_square.detach())

        return torch.exp(-man_distance_square / (self.sigma_square))


class Model2(torch.nn.Module):

    def __init__(self, args, item_num, use_modal):
        super(Model2, self).__init__()
        self.args = args
        self.use_modal = use_modal
        self.max_seq_len = args.max_seq_len + 1
        self.dnn_layers = args.dnn_layers
        self.mo_dnn_layers = args.mo_dnn_layers
        self.gamma = args.gamma
        self.user_encoder = User_Encoder(
            item_num=item_num,
            max_seq_len=args.max_seq_len,
            item_dim=args.embedding_dim,
            num_attention_heads=args.num_attention_heads,
            dropout=args.drop_rate,
            n_layers=args.transformer_block)
        #print(args.word_embedding_dim)
        self.turn_dim1 = FC_Layers(word_embedding_dim=args.word_embedding_dim,
                                  item_embedding_dim=args.embedding_dim,
                                  dnn_layers=self.mo_dnn_layers, drop_rate=args.drop_rate)

        if 'add' in args.item_tower:
            self.fc = ADD()
        elif 'cat' in args.item_tower:
            self.fc = CAT(input_dim=args.embedding_dim*2,
                          output_dim=args.embedding_dim,
                          drop_rate=args.drop_rate)
        else:
            self.fc = None

        self.mlp_layers = MLP_Layers(layers=[args.embedding_dim] * (self.dnn_layers + 1),
                              dnn_layers=self.dnn_layers,
                              drop_rate=args.drop_rate)

        self.id_embedding = nn.Embedding(item_num + 1, args.embedding_dim, padding_idx=0)
        xavier_normal_(self.id_embedding.weight.data)
        self.criterion = nn.BCEWithLogitsLoss()
        
        self.mkmmd_loss = MultipleKernelMaximumMeanDiscrepancy(
        #kernels=[GaussianKernel(alpha=0.5), GaussianKernel(alpha=1.), GaussianKernel(alpha=2.)])
        kernels=[GaussianKernel(alpha=2 ** k) for k in range(-3, 2)])

    def forward(self, sample_items_id, input_embs_content, log_mask, local_rank):

        #input_embs_id = self.id_embedding(sample_items_id)
        if self.use_modal:
            input_embs_content=self.turn_dim1(input_embs_content)
            #input_embs_all = self.mlp_layers(self.fc(input_embs_id, input_embs_content)) ##########################
            input_embs_all = self.mlp_layers(input_embs_content)
            #input_embs_all = input_embs_id 
            #mkmmd_loss = self.gamma*self.mkmmd_loss(input_embs_content, input_embs_id)
            #mkmmd_loss = self.gamma*(self.mkmmd_loss(input_embs_id, input_embs_content)+self.mkmmd_loss(input_embs_content, input_embs_id))
            #print('mmd loss')
            #print(mkmmd_loss)
        else:
            input_embs_all = input_embs_id
        input_embs = input_embs_all.view(-1, self.max_seq_len, 2, self.args.embedding_dim)
        pos_items_embs = input_embs[:, :, 0]
        neg_items_embs = input_embs[:, :, 1]

        input_logs_embs = pos_items_embs[:, :-1, :]
        target_pos_embs = pos_items_embs[:, 1:, :]
        target_neg_embs = neg_items_embs[:, :-1, :]

        prec_vec = self.user_encoder(input_logs_embs, log_mask, local_rank)
        pos_score = (prec_vec * target_pos_embs).sum(-1)
        neg_score = (prec_vec * target_neg_embs).sum(-1)
        pos_labels, neg_labels = torch.ones(pos_score.shape).to(local_rank), torch.zeros(neg_score.shape).to(local_rank)

        indices = torch.where(log_mask != 0)
        loss = self.criterion(pos_score[indices], pos_labels[indices]) + \
               self.criterion(neg_score[indices], neg_labels[indices])
        #if self.use_modal:
        #    loss+=mkmmd_loss
        return loss

class Model2_align(torch.nn.Module):

    def __init__(self, args, item_num, use_modal):
        super(Model2_align, self).__init__()
        self.args = args
        self.use_modal = use_modal
        self.max_seq_len = args.max_seq_len + 1
        self.dnn_layers = args.dnn_layers
        self.mo_dnn_layers = args.mo_dnn_layers
        self.gamma = args.gamma
        self.user_encoder = User_Encoder(
            item_num=item_num,
            max_seq_len=args.max_seq_len,
            item_dim=args.embedding_dim,
            num_attention_heads=args.num_attention_heads,
            dropout=args.drop_rate,
            n_layers=args.transformer_block)
        #print(args.word_embedding_dim)
        self.turn_dim1 = FC_Layers(word_embedding_dim=args.word_embedding_dim,
                                  item_embedding_dim=args.embedding_dim,
                                  dnn_layers=self.mo_dnn_layers, drop_rate=args.drop_rate)

        if 'add' in args.item_tower:
            self.fc = ADD()
        elif 'cat' in args.item_tower:
            self.fc = CAT(input_dim=args.embedding_dim*2,
                          output_dim=args.embedding_dim,
                          drop_rate=args.drop_rate)
        else:
            self.fc = None

        self.mlp_layers = MLP_Layers(layers=[args.embedding_dim] * (self.dnn_layers + 1),
                              dnn_layers=self.dnn_layers,
                              drop_rate=args.drop_rate)

        self.id_embedding = nn.Embedding(item_num + 1, args.embedding_dim, padding_idx=0)
        xavier_normal_(self.id_embedding.weight.data)
        self.criterion = nn.BCEWithLogitsLoss()
        
        self.mkmmd_loss = MultipleKernelMaximumMeanDiscrepancy(
        #kernels=[GaussianKernel(alpha=0.5), GaussianKernel(alpha=1.), GaussianKernel(alpha=2.)])
        kernels=[GaussianKernel(alpha=2 ** k) for k in range(-3, 2)])
        self.loss=InfoNCE()
    def forward(self, sample_items_id, input_embs_content, log_mask, local_rank):

        input_embs_id = self.id_embedding(sample_items_id)
        if self.use_modal:
            input_embs_content=self.turn_dim1(input_embs_content)
            input_embs_all = self.mlp_layers(self.fc(input_embs_id, input_embs_content)) ##########################
            #input_embs_all = self.mlp_layers(input_embs_content)
            #input_embs_all = input_embs_id 
            #mkmmd_loss = self.mkmmd_loss(input_embs_content, input_embs_id)
            #mkmmd_loss = self.gamma*self.mkmmd_loss(input_embs_content, input_embs_id)
            mkmmd_loss = 0.01*self.loss(input_embs_content, input_embs_id)
            #mkmmd_loss = self.gamma*(self.mkmmd_loss(input_embs_id, input_embs_content)+self.mkmmd_loss(input_embs_content, input_embs_id))
            #print('mmd loss')
            #print(mkmmd_loss)
        else:
            input_embs_all = input_embs_id
        
        input_embs = input_embs_all.view(-1, self.max_seq_len, 2, self.args.embedding_dim)
        pos_items_embs = input_embs[:, :, 0]
        neg_items_embs = input_embs[:, :, 1]

        input_logs_embs = pos_items_embs[:, :-1, :]
        target_pos_embs = pos_items_embs[:, 1:, :]
        target_neg_embs = neg_items_embs[:, :-1, :]

        prec_vec = self.user_encoder(input_logs_embs, log_mask, local_rank)
        pos_score = (prec_vec * target_pos_embs).sum(-1)
        neg_score = (prec_vec * target_neg_embs).sum(-1)
        pos_labels, neg_labels = torch.ones(pos_score.shape).to(local_rank), torch.zeros(neg_score.shape).to(local_rank)

        indices = torch.where(log_mask != 0)
        loss = self.criterion(pos_score[indices], pos_labels[indices]) + \
               self.criterion(neg_score[indices], neg_labels[indices])
        #if self.use_modal:
        loss+=mkmmd_loss
        
        return loss

class Model2_id(torch.nn.Module):

    def __init__(self, args, item_num, use_modal):
        super(Model2_id, self).__init__()
        self.args = args
        self.use_modal = use_modal
        self.max_seq_len = args.max_seq_len + 1
        self.dnn_layers = args.dnn_layers
        self.mo_dnn_layers = args.mo_dnn_layers
        self.gamma = args.gamma
        self.user_encoder = User_Encoder(
            item_num=item_num,
            max_seq_len=args.max_seq_len,
            item_dim=args.embedding_dim,
            num_attention_heads=args.num_attention_heads,
            dropout=args.drop_rate,
            n_layers=args.transformer_block)
        self.user_enc_2 = User_Encoder(
            item_num=item_num,
            max_seq_len=args.max_seq_len,
            item_dim=args.embedding_dim,
            num_attention_heads=args.num_attention_heads,
            dropout=args.drop_rate,
            n_layers=args.transformer_block)
        #print(args.word_embedding_dim)
        self.turn_dim1 = FC_Layers(word_embedding_dim=args.word_embedding_dim,
                                  item_embedding_dim=args.embedding_dim,
                                  dnn_layers=self.mo_dnn_layers, drop_rate=args.drop_rate)

        if 'add' in args.item_tower:
            self.fc = ADD()
        elif 'cat' in args.item_tower:
            self.fc = CAT(input_dim=args.embedding_dim*2,
                          output_dim=args.embedding_dim,
                          drop_rate=args.drop_rate)
        else:
            self.fc = None

        self.mlp_layers = MLP_Layers(layers=[args.embedding_dim] * (self.dnn_layers + 1),
                              dnn_layers=self.dnn_layers,
                              drop_rate=args.drop_rate)

        self.id_embedding = nn.Embedding(item_num + 1, args.embedding_dim, padding_idx=0)
        self.other_embedding = nn.Embedding(item_num + 1, args.embedding_dim, padding_idx=0)
        xavier_normal_(self.id_embedding.weight.data)
        self.criterion = nn.BCEWithLogitsLoss()
        
        self.mkmmd_loss = MultipleKernelMaximumMeanDiscrepancy(
        #kernels=[GaussianKernel(alpha=0.5), GaussianKernel(alpha=1.), GaussianKernel(alpha=2.)])
        kernels=[GaussianKernel(alpha=2 ** k) for k in range(-3, 2)])
    def freeze(self):
        self.user_encoder=copy.deepcopy(self.user_enc_2)
        self.id_embedding.weight.data.copy_(self.other_embedding.weight.data)
    def forward(self, sample_items_id, input_embs_content, log_mask, local_rank):

        input_embs_id = self.id_embedding(sample_items_id)
        if self.use_modal:
            #input_embs_content=self.turn_dim1(input_embs_content)
            #input_embs_all = self.mlp_layers(self.fc(input_embs_id, input_embs_content)) ##########################
            #input_embs_all = self.mlp_layers(input_embs_content)
            input_embs_all = input_embs_id 
            #mkmmd_loss = self.gamma*self.mkmmd_loss(input_embs_content, input_embs_id)
            #mkmmd_loss = self.gamma*(self.mkmmd_loss(input_embs_id, input_embs_content)+self.mkmmd_loss(input_embs_content, input_embs_id))
            #print('mmd loss')
            #print(mkmmd_loss)
        else:
            input_embs_all = input_embs_id
        input_embs = input_embs_all.view(-1, self.max_seq_len, 2, self.args.embedding_dim)
        pos_items_embs = input_embs[:, :, 0]
        neg_items_embs = input_embs[:, :, 1]

        input_logs_embs = pos_items_embs[:, :-1, :]
        target_pos_embs = pos_items_embs[:, 1:, :]
        target_neg_embs = neg_items_embs[:, :-1, :]
        #print(input_logs_embs.shape)
        prec_vec = self.user_encoder(input_logs_embs, log_mask, local_rank)
        pos_score = (prec_vec * target_pos_embs).sum(-1)
        neg_score = (prec_vec * target_neg_embs).sum(-1)
        pos_labels, neg_labels = torch.ones(pos_score.shape).to(local_rank), torch.zeros(neg_score.shape).to(local_rank)
        #print(prec_vec.shape)
        #print(target_pos_embs.shape)
        
        indices = torch.where(log_mask != 0)
        loss = self.criterion(pos_score[indices], pos_labels[indices]) + \
               self.criterion(neg_score[indices], neg_labels[indices])
        #if self.use_modal:
        #    loss+=mkmmd_loss
        return loss


class Model2_transfer(torch.nn.Module):

    def __init__(self, args, item_num, use_modal):
        super(Model2_transfer, self).__init__()
        self.args = args
        self.use_modal = use_modal
        self.max_seq_len = args.max_seq_len + 1
        self.dnn_layers = args.dnn_layers
        self.mo_dnn_layers = args.mo_dnn_layers
        self.gamma = args.gamma
        self.user_encoder = User_Encoder(
            item_num=item_num,
            max_seq_len=args.max_seq_len,
            item_dim=args.embedding_dim,
            num_attention_heads=args.num_attention_heads,
            dropout=args.drop_rate,
            n_layers=args.transformer_block)
        self.user_enc_2 = User_Encoder(
            item_num=item_num,
            max_seq_len=args.max_seq_len,
            item_dim=args.embedding_dim,
            num_attention_heads=args.num_attention_heads,
            dropout=args.drop_rate,
            n_layers=args.transformer_block)
        #print(args.word_embedding_dim)
        self.turn_dim1 = FC_Layers(word_embedding_dim=args.word_embedding_dim,
                                  item_embedding_dim=args.embedding_dim,
                                  dnn_layers=self.mo_dnn_layers, drop_rate=args.drop_rate)
        self.turn_dim_random = FC_Layers(word_embedding_dim=args.word_embedding_dim,
                                  item_embedding_dim=args.embedding_dim,
                                  dnn_layers=self.mo_dnn_layers, drop_rate=args.drop_rate)
        if 'add' in args.item_tower:
            self.fc = ADD()
        elif 'cat' in args.item_tower:
            self.fc = CAT(input_dim=args.embedding_dim*2,
                          output_dim=args.embedding_dim,
                          drop_rate=args.drop_rate)
        else:
            self.fc = None

        self.mlp_layers = MLP_Layers(layers=[args.embedding_dim] * (self.dnn_layers + 1),
                              dnn_layers=self.dnn_layers,
                              drop_rate=args.drop_rate)

        self.id_embedding = nn.Embedding(item_num + 1, args.embedding_dim, padding_idx=0)
        self.other_embedding = nn.Embedding(item_num + 1, args.embedding_dim, padding_idx=0)
        xavier_normal_(self.id_embedding.weight.data)
        self.criterion = nn.BCEWithLogitsLoss()
        #l=[-8,-4,-2,-1,0]
        self.mkmmd_loss = MultipleKernelMaximumMeanDiscrepancy(
        #kernels=[GaussianKernel(alpha=0.5), GaussianKernel(alpha=1.), GaussianKernel(alpha=2.)])
        kernels=[GaussianKernel(alpha=2 ** k) for k in range(-3, 2)])
        #l=[-8,-4,-2,-1,0]
        #kernels=[GaussianKernel(alpha=2 ** k) for k in l])
        #kernels=[LapKernel(alpha=2 ** k) for k in range(-3, 2)])
    def freeze(self):
        self.other_embedding.weight.data.copy_(self.id_embedding.weight.data)
        self.user_enc_2=copy.deepcopy(self.user_encoder)
    def forward(self, sample_items_id, input_embs_content, log_mask, local_rank):

        input_embs_id = self.other_embedding(sample_items_id)
        if self.use_modal:
            input_embs_content=self.turn_dim1(input_embs_content)
            input_embs_all = self.mlp_layers(self.fc(input_embs_id, input_embs_content))
            mkmmd_loss = 0.2*self.mkmmd_loss(input_embs_content, input_embs_id)
            #mkmmd_loss = self.gamma*(self.mkmmd_loss(input_embs_id, input_embs_content)+self.mkmmd_loss(input_embs_content, input_embs_id))
            #print('mmd loss')
            #print(mkmmd_loss)
            #input_embs_all = input_embs_id
        else:
            input_embs_all = input_embs_id
        input_embs = input_embs_all.view(-1, self.max_seq_len, 2, self.args.embedding_dim)
        pos_items_embs = input_embs[:, :, 0]
        neg_items_embs = input_embs[:, :, 1]

        input_logs_embs = pos_items_embs[:, :-1, :]
        target_pos_embs = pos_items_embs[:, 1:, :]
        target_neg_embs = neg_items_embs[:, :-1, :]

        prec_vec = self.user_enc_2(input_logs_embs, log_mask, local_rank)
        pos_score = (prec_vec * target_pos_embs).sum(-1)
        neg_score = (prec_vec * target_neg_embs).sum(-1)
        pos_labels, neg_labels = torch.ones(pos_score.shape).to(local_rank), torch.zeros(neg_score.shape).to(local_rank)

        indices = torch.where(log_mask != 0)
        loss = self.criterion(pos_score[indices], pos_labels[indices]) + \
               self.criterion(neg_score[indices], neg_labels[indices])
        #if self.use_modal:
        loss+=mkmmd_loss
        return loss



class Model_new3_3(torch.nn.Module):

    def __init__(self, args, item_num, use_modal):
        super(Model_new3_3, self).__init__()
        self.args = args
        self.use_modal = use_modal
        self.max_seq_len = args.max_seq_len + 1
        self.dnn_layers = args.dnn_layers
        self.mo_dnn_layers = args.mo_dnn_layers
        self.gamma = args.gamma
        self.gamma2 = args.gamma2
        self.gamma3 = args.gamma3
        self.user_encoder = User_Encoder(
            item_num=item_num,
            max_seq_len=args.max_seq_len,
            item_dim=args.embedding_dim,
            num_attention_heads=args.num_attention_heads,
            dropout=args.drop_rate,
            n_layers=args.transformer_block)
        self.user_enc_2 = User_Encoder(
            item_num=item_num,
            max_seq_len=args.max_seq_len,
            item_dim=args.embedding_dim,
            num_attention_heads=args.num_attention_heads,
            dropout=args.drop_rate,
            n_layers=args.transformer_block)
        self.user_enc_3 = User_Encoder(
            item_num=item_num,
            max_seq_len=args.max_seq_len,
            item_dim=args.embedding_dim,
            num_attention_heads=args.num_attention_heads,
            dropout=args.drop_rate,
            n_layers=args.transformer_block)
        #print(args.word_embedding_dim)
        self.turn_dim1 = FC_Layers(word_embedding_dim=args.word_embedding_dim,
                                  item_embedding_dim=args.embedding_dim,
                                  dnn_layers=self.mo_dnn_layers, drop_rate=args.drop_rate)
        self.turn_dim3 = FC_Layers(word_embedding_dim=args.word_embedding_dim,
                                  item_embedding_dim=args.embedding_dim,
                                  dnn_layers=self.mo_dnn_layers, drop_rate=args.drop_rate)
        if 'add' in args.item_tower:
            self.fc = ADD()
        elif 'cat' in args.item_tower:
            self.fc = CAT(input_dim=args.embedding_dim*2,
                          output_dim=args.embedding_dim,
                          drop_rate=args.drop_rate)
        else:
            self.fc = None

        self.mlp_layers = MLP_Layers(layers=[args.embedding_dim] * (self.dnn_layers + 1),
                              dnn_layers=self.dnn_layers,
                              drop_rate=args.drop_rate)
        self.mlp_layer_3 = MLP_Layers(layers=[args.embedding_dim] * (self.dnn_layers + 1),
                              dnn_layers=self.dnn_layers,
                              drop_rate=args.drop_rate)
        
        self.id_embedding = nn.Embedding(item_num + 1, args.embedding_dim, padding_idx=0)
        self.other_embedding = nn.Embedding(item_num + 1, args.embedding_dim, padding_idx=0)
        xavier_normal_(self.id_embedding.weight.data)
        self.criterion = nn.BCEWithLogitsLoss()
        self.act = nn.Sigmoid()
        a=np.linspace(1,0,11)
        #a=torch.tensor([0.7333,0.7515, 0.7375, 0.7263, 0.7165, 0.7066, 0.6978, 0.6913, 0.6865, 0.6870, 0.6977],dtype=float)
        #self.alpha = torch.nn.Parameter(a)
        self.alpha = torch.nn.Parameter(torch.tensor(a))
        b=np.linspace(0,1,11)
        #self.alpha = torch.nn.Parameter(torch.zeros(11)+self.gamma)
        c=np.linspace(self.gamma2,self.gamma2-1,11)
        self.alpha2 = torch.nn.Parameter(torch.tensor(c))
        #self.alpha2 = torch.nn.Parameter(torch.zeros(11)+self.gamma)
        #b=torch.tensor([0.5, 0.4889, 0.5193, 0.5410, 0.5631, 0.5869, 0.6101, 0.6370, 0.6673, 0.7082, 0.7678],dtype=float)
        self.alpha3 = torch.nn.Parameter(torch.tensor(b))
        #self.alpha3 = torch.nn.Parameter(torch.tensor(b))
        #self.alpha3 = torch.nn.Parameter(torch.tensor(b))
        self.mkmmd_loss = MultipleKernelMaximumMeanDiscrepancy(
        #kernels=[GaussianKernel(alpha=0.5), GaussianKernel(alpha=1.), GaussianKernel(alpha=2.)])
        kernels=[GaussianKernel(alpha=2 ** k) for k in range(-3, 2)])
        #kernels=[GaussianKernel(alpha=2 ** -3),  GaussianKernel(alpha=2 ** -2), GaussianKernel(alpha=2 ** -1), GaussianKernel(alpha=2 ** 0), GaussianKernel(alpha=2 ** 1), LapKernel(alpha=2 ** -3),  LapKernel(alpha=2 ** -2), LapKernel(alpha=2 ** -1), LapKernel(alpha=2 ** 0), LapKernel(alpha=2 ** 1)])
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = True
        self.other_embedding.weight.data.copy_(self.id_embedding.weight.data)
        #self.user_enc_2=copy.deepcopy(self.user_encoder)
        #for name, param in self.named_parameters():
        #    if "id_embedding" in name or "user_encoder" in name:
        #        param.requires_grad = False
    def freeze1(self):
        for param in self.parameters():
            param.requires_grad = True
        self.other_embedding.weight.data.copy_(self.id_embedding.weight.data)
        self.user_enc_2=copy.deepcopy(self.user_encoder)
        for name, param in self.named_parameters():
            if "id_embedding" in name  or "user_encoder" in name:
                param.requires_grad = False
    def freeze2(self):
        for param in self.parameters():
            param.requires_grad = True
        self.other_embedding.weight.data.copy_(self.id_embedding.weight.data)
        self.user_enc_2=copy.deepcopy(self.user_encoder)
        for name, param in self.named_parameters():
            if "id_embedding" in name: # or "user_encoder" in name:
                param.requires_grad = False
    def freeze3(self):
        for param in self.parameters():
            param.requires_grad = True
        self.other_embedding.weight.data.copy_(self.id_embedding.weight.data)
        #self.user_enc_2=copy.deepcopy(self.user_encoder)
        for name, param in self.named_parameters():
            if "id_embedding" in name: # or "user_encoder" in name:
                param.requires_grad = False
    def freeze4(self):
        for param in self.parameters():
            param.requires_grad = True
        #self.other_embedding.weight.data.copy_(self.id_embedding.weight.data)
        #self.user_enc_2=copy.deepcopy(self.user_encoder)
        for name, param in self.named_parameters():
            if "user_encoder" in name:
                param.requires_grad = False
    def freeze5(self):
        for param in self.parameters():
            param.requires_grad = True
        for name, param in self.named_parameters():
            if "user_encoder" in name or "enc_2" in name or "turn_dim1" in name or "fc" in name or "mlp_layers" in name:
                param.requires_grad = False
    def freeze6(self):
        for param in self.parameters():
            param.requires_grad = True
    def freeze7(self):
        for param in self.parameters():
            param.requires_grad = True
        for name, param in self.named_parameters():
            if "enc_2" in name or "turn_dim1" in name or "fc" in name or "mlp_layers" in name:
                param.requires_grad = False
    def freeze8(self):
        for param in self.parameters():
            param.requires_grad = True
        for name, param in self.named_parameters():
            if "user_encoder" in name:
                param.requires_grad = False
    def freeze9(self):
        for param in self.parameters():
            param.requires_grad = True
        for name, param in self.named_parameters():
            if "user_enc_2" in name:
                param.requires_grad = False
    def freeze10(self):
        for param in self.parameters():
            param.requires_grad = True
        for name, param in self.named_parameters():
            if "user_encoder" in name or "user_enc_2" in name:
                param.requires_grad = False

    def forward(self, sample_items_id, input_embs_content, log_mask,bin_pos_item,bin_neg_item, local_rank):
        #print(bin_pos_item.shape)
        #print(bin_neg_item.shape)
        #print()
        input_embs_id = self.id_embedding(sample_items_id)
        input_other = self.other_embedding(sample_items_id)
        if self.use_modal:
            input_embs_content1=self.turn_dim1(input_embs_content)
            input_embs_all = self.mlp_layers(self.fc(input_other,input_embs_content1))
            #mkmmd_loss = 0.02*self.mkmmd_loss(input_embs_content1, input_other)
            input_embs_content3=self.turn_dim3(input_embs_content)
            input_embs_all3 = self.mlp_layer_3(input_embs_content3)
            #mkmmd_loss = self.gamma*(self.mkmmd_loss(input_embs_id, input_embs_content)+self.mkmmd_loss(input_embs_content, input_embs_id))
            #print('mmd loss')
            #print(mkmmd_loss)
        else:
            input_embs_all = input_embs_id
        input_embs = input_embs_all.view(-1, self.max_seq_len, 2, self.args.embedding_dim)
        pos_items_embs = input_embs[:, :, 0]
        neg_items_embs = input_embs[:, :, 1]

        input_logs_embs = pos_items_embs[:, :-1, :]
        target_pos_embs = pos_items_embs[:, 1:, :]
        target_neg_embs = neg_items_embs[:, :-1, :]

        prec_vec_2 = self.user_enc_2(input_logs_embs, log_mask, local_rank)
        pos_score_2 = (prec_vec_2 * target_pos_embs).sum(-1)
        neg_score_2 = (prec_vec_2 * target_neg_embs).sum(-1)


        input_embs1 = input_embs_id.view(-1, self.max_seq_len, 2, self.args.embedding_dim)
        pos_items_embs = input_embs1[:, :, 0]
        neg_items_embs = input_embs1[:, :, 1]

        input_logs_embs = pos_items_embs[:, :-1, :]
        target_pos_embs = pos_items_embs[:, 1:, :]
        target_neg_embs = neg_items_embs[:, :-1, :]

        prec_vec_1 = self.user_encoder(input_logs_embs, log_mask, local_rank)
        pos_score_1 = (prec_vec_1 * target_pos_embs).sum(-1)
        neg_score_1 = (prec_vec_1 * target_neg_embs).sum(-1)
   

        input_embs3 = input_embs_all3.view(-1, self.max_seq_len, 2, self.args.embedding_dim)
        pos_items_embs = input_embs3[:, :, 0]
        neg_items_embs = input_embs3[:, :, 1]

        input_logs_embs = pos_items_embs[:, :-1, :]
        target_pos_embs = pos_items_embs[:, 1:, :]
        target_neg_embs = neg_items_embs[:, :-1, :]

        prec_vec_3 = self.user_enc_3(input_logs_embs, log_mask, local_rank)
        pos_score_3 = (prec_vec_3 * target_pos_embs).sum(-1)
        neg_score_3 = (prec_vec_3 * target_neg_embs).sum(-1)

        #pos_score_2 = (prec_vec_2 * target_pos_embs).sum(-1)
        #neg_score_2 = (prec_vec_2 * target_neg_embs).sum(-1)
        
        #pos_item = sample_items_id[:,:self.max_seq_len]
        #neg_item
        #print(bin_pos_item.shape)
        #print(pos_score_1.shape)
        
        alpha = self.act(self.alpha[bin_pos_item])
        alpha2 = self.act(self.alpha2[bin_pos_item])
        alpha3 = self.act(self.alpha3[bin_pos_item])
        beta = self.act(self.alpha[bin_neg_item])
        beta2 = self.act(self.alpha2[bin_neg_item])
        beta3 = self.act(self.alpha3[bin_neg_item])
        alpha_s=alpha+alpha2+alpha3
        beta_s=beta+beta2+beta3
        #alpha=alpha/(alpha+alpha2+alpha3)
        #alpha2=alpha2/(alpha+alpha2+alpha3)
        #alpha3=alpha3/(alpha+alpha2+alpha3)
        #beta=beta/(beta+beta2+beta3)
        #beta2=beta2/(beta+beta2+beta3)
        #beta3=beta3/(beta+beta2+beta3)
        #alpha = self.alpha[bin_pos_item]
        #beta = self.alpha[bin_neg_item]
        #print(alpha.shape)
        #print()
        pos_score = alpha/alpha_s*pos_score_1+alpha2/alpha_s*pos_score_2+alpha3/alpha_s*pos_score_3
        neg_score = beta/beta_s*neg_score_1+beta2/beta_s*neg_score_2+beta3/beta_s*neg_score_3
        pos_labels, neg_labels = torch.ones(pos_score.shape).to(local_rank), torch.zeros(neg_score.shape).to(local_rank)

        indices = torch.where(log_mask != 0)
        loss = self.criterion(pos_score[indices], pos_labels[indices]) + \
               self.criterion(neg_score[indices], neg_labels[indices])
        #if self.use_modal:
        #    loss+=mkmmd_loss
        return loss

