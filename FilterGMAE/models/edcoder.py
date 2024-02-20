from typing import Optional
from itertools import chain
from functools import partial
import torch.nn.functional as F
import numpy as np
import random
import math

import torch
import torch.nn as nn

from .gin import GIN
from .gat import GAT
from .gcn import GCN
from .dot_gat import DotGAT
from .guard   import GATGuard,GCNAdjNorm
from .loss_func import sce_loss
from FilterGMAE.utils import create_norm, drop_edge


def setup_module(m_type, enc_dec, in_dim, num_hidden, out_dim, num_layers, dropout, activation, residual, norm, nhead, nhead_out, attn_drop, negative_slope=0.2, concat_out=True) -> nn.Module:
    if m_type == "gat":
        mod = GAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "dotgat":
        mod = DotGAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "gin":
        mod = GIN(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            residual=residual,
            norm=norm,
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "gcn":
        mod = GCN(
            in_dim=in_dim, 
            num_hidden=num_hidden, 
            out_dim=out_dim, 
            num_layers=num_layers, 
            dropout=dropout, 
            activation=activation, 
            residual=residual, 
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding")
        )
    elif m_type == "mlp":
        # * just for decoder 
        mod = nn.Sequential(
            nn.Linear(in_dim, num_hidden),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(num_hidden, out_dim)
        )
    elif m_type == "linear":
        mod = nn.Linear(in_dim, out_dim)
    else:
        raise NotImplementedError
    
    return mod


class PreModel(nn.Module):
    def __init__(
            self,
            in_dim: int,
            num_hidden: int,
            num_layers: int,
            nhead: int,
            nhead_out: int,
            activation: str,
            feat_drop: float,
            attn_drop: float,
            negative_slope: float,
            residual: bool,
            norm: Optional[str],
            mask_rate: float = 0.3,
            encoder_type: str = "gat",
            decoder_type: str = "gat",
            loss_fn: str = "sce",
            drop_edge_rate: float = 0.0,
            replace_rate: float = 0.1,
            alpha_l: float = 2,
            concat_hidden: bool = False,
            # add by ssh
            mode                  : str   = 'tranductive',
            scheme                : str   = 'sample',
            num_nodes             : int   = 2485,
            mask_strategy         : str   = 'degree',
            keep_scope_centerline : float = 0.1,
            keep_scope_interval   : float = 0.1,
            mask_scope_centerline : float = 0.9,
            mask_scope_interval   : float = 0.1,
            keep_num_ratio        : float = 1.0,
            mask_num_ratio        : float = 1.0
         ):
        super(PreModel, self).__init__()
        self._mask_rate = mask_rate

        self._encoder_type = encoder_type
        self._decoder_type = decoder_type
        self._drop_edge_rate = drop_edge_rate
        self._output_hidden_size = num_hidden
        self._concat_hidden = concat_hidden
        self._replace_rate = replace_rate
        self._mask_token_rate = 1 - self._replace_rate
        # add by ssh
        self._scheme                = scheme
        self._mask_strategy         = mask_strategy
        self._keep_scope_centerline = keep_scope_centerline
        self._keep_scope_interval   = keep_scope_interval
        self._mask_scope_centerline = mask_scope_centerline
        self._mask_scope_interval   = mask_scope_interval
        self._keep_num_ratio        = keep_num_ratio
        self._mask_num_ratio        = mask_num_ratio
        if mode == 'tranductive':
            # add by ssh 全图的时候可以在init里面这样算，但是mini batch的时候这样就不行了,需要根据每个minibatch的g计算一下，这部分要放在forward里面
            self._keep_scope_centerline_index       = int(num_nodes * self._keep_scope_centerline)
            self._keep_scope_len                    = int(num_nodes * self._keep_scope_interval) 
            self._mask_scope_centerline_index       = int(num_nodes * self._mask_scope_centerline)
            self._mask_scope_len                    = int(num_nodes * self._mask_scope_interval) 
            self._keep_low  = self._keep_scope_centerline_index - self._keep_scope_len
            self._keep_high = self._keep_scope_centerline_index + self._keep_scope_len
            self._mask_low  = self._mask_scope_centerline_index - self._mask_scope_len
            self._mask_high = self._mask_scope_centerline_index + self._mask_scope_len
            print(self._keep_low)
            print(self._keep_high)
            print(self._mask_low)
            print(self._mask_high)
            assert  self._keep_low >= 0 
            assert (self._keep_high >= self._keep_low) &  (self._keep_high <= self._mask_low)
            assert (self._mask_high <= num_nodes)      &  (self._mask_high >= self._mask_low)
        # add by ssh

        assert num_hidden % nhead == 0
        assert num_hidden % nhead_out == 0
        if encoder_type in ("gat", "dotgat"):
            enc_num_hidden = num_hidden // nhead
            enc_nhead = nhead
        else:
            enc_num_hidden = num_hidden
            enc_nhead = 1

        dec_in_dim = num_hidden
        dec_num_hidden = num_hidden // nhead_out if decoder_type in ("gat", "dotgat") else num_hidden 

        # add by ssh compare 使用不同的鲁棒性编解码器
        # self.encoder = GATGuard(in_features=in_dim,
        #                         out_features=enc_num_hidden,
        #                         hidden_features=enc_num_hidden,
        #                         n_layers=num_layers,
        #                         n_heads=enc_nhead,
        #                         activation=F.leaky_relu,
        #                         layer_norm=False,
        #                         feat_norm=None,
        #                         adj_norm_func=GCNAdjNorm,
        #                         drop=False,
        #                         attention=True,
        #                         dropout=0.0)
        # self.decoder = GATGuard(in_features=dec_in_dim,
        #                         out_features=in_dim,
        #                         hidden_features=dec_num_hidden,
        #                         n_layers=1,
        #                         n_heads=nhead_out,
        #                         activation=F.leaky_relu,
        #                         layer_norm=False,
        #                         feat_norm=None,
        #                         adj_norm_func=GCNAdjNorm,
        #                         drop=False,
        #                         attention=True,
        #                         dropout=0.0)
        # add by ssh compare



        # build encoder
        self.encoder = setup_module(
            m_type=encoder_type,
            enc_dec="encoding",
            in_dim=in_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=num_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
        )

        # build decoder for attribute prediction
        self.decoder = setup_module(
            m_type=decoder_type,
            enc_dec="decoding",
            in_dim=dec_in_dim,
            num_hidden=dec_num_hidden,
            out_dim=in_dim,
            num_layers=1,
            nhead=nhead,
            nhead_out=nhead_out,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
            concat_out=True,
        )

        ######################################### ema teacher student add by ssh
        # self.encoder_ema = setup_module(
        #     m_type=encoder_type,
        #     enc_dec="encoding",
        #     in_dim=in_dim,
        #     num_hidden=enc_num_hidden,
        #     out_dim=enc_num_hidden,
        #     num_layers=num_layers,
        #     nhead=enc_nhead,
        #     nhead_out=enc_nhead,
        #     concat_out=True,
        #     activation=activation,
        #     dropout=feat_drop,
        #     attn_drop=attn_drop,
        #     negative_slope=negative_slope,
        #     residual=residual,
        #     norm=norm,
        # )
        # self.encoder_ema.load_state_dict(self.encoder.state_dict())
        # for p in self.encoder_ema.parameters():
        #     p.requires_grad = False
        #     p.detach_()
        # # mutihead attention
        # self.attn = nn.MultiheadAttention(dec_in_dim, 4, dropout=0.5)
        # self.dec_mask_token = nn.Parameter(torch.zeros(1, dec_in_dim))
        # self._delayed_ema_epoch = 0
        # self._momentum          = 0.996
        # self.alignment          = self.setup_loss_fn('mse', alpha_l)
        ######################################### ema teacher student add by ssh




        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        if concat_hidden:
            self.encoder_to_decoder = nn.Linear(dec_in_dim * num_layers, dec_in_dim, bias=False)
        else:
            self.encoder_to_decoder = nn.Linear(dec_in_dim, dec_in_dim, bias=False)

        # * setup loss function
        self.criterion = self.setup_loss_fn(loss_fn, alpha_l)

    @property
    def output_hidden_dim(self):
        return self._output_hidden_size

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion
    
    def encoding_mask_noise(self, g, x, mask_rate, mode):
        num_nodes = g.num_nodes()

        # add by ssh # 如果是mini batch，需要根据每个subgraph算一下范围
        if mode == 'mini_batch':
            self._keep_scope_centerline_index       = int(num_nodes * self._keep_scope_centerline)
            self._keep_scope_len                    = int(num_nodes * self._keep_scope_interval) 
            self._mask_scope_centerline_index       = int(num_nodes * self._mask_scope_centerline)
            self._mask_scope_len                    = int(num_nodes * self._mask_scope_interval) 
            self._keep_low  = self._keep_scope_centerline_index - self._keep_scope_len
            self._keep_high = self._keep_scope_centerline_index + self._keep_scope_len
            self._mask_low  = self._mask_scope_centerline_index - self._mask_scope_len
            self._mask_high = self._mask_scope_centerline_index + self._mask_scope_len
            assert  self._keep_low >= 0 
            assert (self._keep_high >= self._keep_low) &  (self._keep_high <= self._mask_low)
            assert (self._mask_high <= num_nodes)      &  (self._mask_high >= self._mask_low)
        # add by ssh


        ######## random masking
        if self._mask_strategy == 'random':
            num_mask_nodes = int(mask_rate * num_nodes)
            perm = torch.randperm(num_nodes, device=x.device)
            mask_nodes = perm[: num_mask_nodes]
            keep_nodes = perm[num_mask_nodes: ]
        else:
            # 在前面已经整理好
            idx_score      = g.ndata['final_score']
            if self._scheme == 'sample':
                # scheme 1 按照权重采样
                sample_weights = 1 / (1 +  torch.exp(idx_score)) # score越大权重越低
                idx_sorted_all = torch.multinomial(sample_weights, num_nodes)
            elif self._scheme == 'fix':
                # scheme 2 固定
                idx_sorted_value, idx_sorted_all = torch.sort(idx_score)
            
            # print(min(g.ndata['csim_score']))
            # print(max(g.ndata['csim_score']))
            # print(g.ndata['csim_score'][idx_sorted_all[-1]])
            # print(g.in_degrees()[idx_sorted_all[-1]])
            # print(min(g.in_degrees()))
            # print(max(g.in_degrees()))
            # # print(g.in_degrees()[idx_sorted_all[-20:]])
            # quit()

            # scheme 3 不同的权重设计
            # idx_sorted_all = torch.multinomial(1.0 - idx_score, num_nodes)# degree不行
            # print(g.ndata['csim_score'][idx_sorted_all[-20:]])
            # print(g.in_degrees()[idx_sorted_all[:20]])
            # quit()

            # def count(a,b):
            #     num = 0
            #     for i in range(len(a)):
            #         if a[i] == b[i]:
            #             num+=1
            #         else:
            #             break
            #     return num
            
            
            # idx_train = g.ndata['train_mask'].nonzero().squeeze()
            # print(max(g.in_degrees()[idx_train]))
            # print(min(g.in_degrees()[idx_train]))
            # quit()
            # print(len(idx_train))
            # print(count(torch.sort(idx_sorted_all[:len(idx_train)])[0], idx_train))
            # quit()



            keep_scope                   = idx_sorted_all[self._keep_low: self._keep_high]
            mask_scope                   = idx_sorted_all[self._mask_low: self._mask_high]
            # keep_scope                   = idx_sorted_all[1500:1900]
            # mask_scope                   = idx_sorted_all[2200:]
            perm_keep  = torch.randperm(len(keep_scope), device=x.device)
            perm_mask  = torch.randperm(len(mask_scope), device=x.device)
            num_dis_keep_nodes = int(len(keep_scope) * self._keep_num_ratio)
            num_dis_mask_nodes = int(len(mask_scope) * self._mask_num_ratio)
            idx_keep   = keep_scope[perm_keep[:num_dis_keep_nodes]]
            idx_mask   = mask_scope[perm_mask[:num_dis_mask_nodes]]  
            keep_nodes = idx_keep.cuda()
            mask_nodes = idx_mask.cuda()
            num_mask_nodes = len(mask_nodes)


        ######## distribution masking 
        # num_mask_nodes = int(mask_rate * num_nodes)
        # idx_val   = g.ndata['val_mask'].nonzero().squeeze()
        # idx_test  = g.ndata['test_mask'].nonzero().squeeze()
        # idx_train = g.ndata['train_mask'].nonzero().squeeze()
   
        # # scheme 1
        # # idx_keep = torch.cat((idx_test, idx_val))
        # # idx_mask = idx_train  # 可以设计实验不断的减小有效信息的可见范围

        # # scheme 2
        # # 不重建被攻击的点，而是重建没有被攻击的同比例的节点
        # # idx_all  = torch.cat((idx_train, torch.cat((idx_test, idx_val))))
        # # idx_mask = idx_all[:num_mask_nodes]
        # # idx_keep = idx_all[num_mask_nodes:]

        # # scheme 3
        # idx_keep_dis = torch.cat((idx_train, idx_val))
        # idx_mask_dis = idx_test
        # perm_keep  = torch.randperm(len(idx_keep_dis), device=x.device)
        # perm_mask  = torch.randperm(len(idx_mask_dis), device=x.device)
        # idx_keep   = idx_keep_dis#[perm_keep[:num_mask_nodes]]
        # idx_mask   = idx_mask_dis  


        # keep_nodes = idx_keep.cuda()
        # mask_nodes = idx_mask.cuda()
        # num_mask_nodes = len(mask_nodes)
        

        ######### node center cosine similarity masking
        # 这里有点意思，sim低的节点都是度比较大的点，看来不能理想话，还是得实验看
        # idx_mask_dis = g.ndata['csim_sorted'][:800] # keep的范围大一些 比例低一些
        # idx_keep_dis = g.ndata['csim_sorted'][2000:]
        # perm_keep  = torch.randperm(len(idx_keep_dis), device=x.device)
        # perm_mask  = torch.randperm(len(idx_mask_dis), device=x.device)

        # num_dis_keep_nodes = int(len(idx_keep_dis) * 0.4)
        # num_dis_mask_nodes = int(len(idx_mask_dis) * 0.8)
        # idx_keep   = idx_keep_dis[perm_keep[:num_dis_keep_nodes]]
        # idx_mask   = idx_mask_dis[perm_mask[:num_dis_mask_nodes]]  

        # keep_nodes = idx_keep.cuda()
        # mask_nodes = idx_mask.cuda()
        # num_mask_nodes = len(mask_nodes)


        ######### degree masking
        # sample_weights = 1 / (1 +  torch.exp(g.in_degrees())) # 度越大权重越低
        # nodes_sample = torch.multinomial(sample_weights, num_nodes)
        # # 用小度的恢复大度的，用鲁棒性差的恢复鲁棒性好的
        # idx_keep_dis = nodes_sample[:1500].cuda()
        # idx_mask_dis = nodes_sample[2200:].cuda()
        # perm_keep  = torch.randperm(len(idx_keep_dis), device=x.device)
        # perm_mask  = torch.randperm(len(idx_mask_dis), device=x.device)
        # num_dis_keep_nodes = int(len(idx_keep_dis) * 0.9)
        # num_dis_mask_nodes = int(len(idx_mask_dis) * 0.9)
        # idx_keep   = idx_keep_dis[perm_keep[:num_dis_keep_nodes]]
        # idx_mask   = idx_mask_dis[perm_mask[:num_dis_mask_nodes]]  
        # keep_nodes = idx_keep.cuda()
        # mask_nodes = idx_mask.cuda()
        # num_mask_nodes = len(mask_nodes)


        # 这个感觉可以放大一下，用train的去重建test的，有效果，但用test的重建train的，越来越差，相当于重建了错误的信息，可以进一步分析, 引发我们讨论重建信息的正确性和鲁棒性问题


        if self._replace_rate > 0:
            num_noise_nodes = int(self._replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self._mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            # keep
            out_x  = torch.zeros_like(x)
            out_x[keep_nodes] = x[keep_nodes]
            # out_x = x.clone()

            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            # keep
            out_x  = torch.zeros_like(x)
            out_x[keep_nodes] = x[keep_nodes]
            # out_x = x.clone()

            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token
        use_g = g.clone()

        return use_g, out_x, (mask_nodes, keep_nodes)

    def forward(self, g, x, weight=None, mode='tranductive'):
        # ---- attribute reconstruction ----
        loss = self.mask_attr_prediction(g, x, weight, mode)
        loss_item = {"loss": loss.item()}
        return loss, loss_item
    
    def mask_attr_prediction(self, g, x, weight, mode):
        pre_use_g, use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(g, x, self._mask_rate, mode)

        if self._drop_edge_rate > 0:
            use_g, masked_edges = drop_edge(pre_use_g, self._drop_edge_rate, return_edges=True)
        else:
            use_g = pre_use_g

        enc_rep, all_hidden = self.encoder(use_g, use_x, weight, return_hidden=True) # add weight

        if self._concat_hidden:
            enc_rep = torch.cat(all_hidden, dim=1)


        ######################## add by ssh
        # 折腾一晚上，无效！！go
        # with torch.no_grad():
        #     embed_masked = self.encoder_ema(use_g, x,return_hidden=False)[mask_nodes]
        # # k和v的来源总是相同的,q 可以不同 query,key,value  q决定输出的维度，
        # embed_masked_token = self.dec_mask_token.expand(embed_masked.shape[0],embed_masked.shape[1])
        # rich_embed_masked_token, token_attn = self.attn(embed_masked_token ,enc_rep[keep_nodes], enc_rep[keep_nodes])
        # rep = torch.zeros_like(enc_rep)
        # rep[mask_nodes] = rich_embed_masked_token
        ######################## add by ssh

        # ---- attribute reconstruction ----
        rep = self.encoder_to_decoder(enc_rep)
        if self._decoder_type not in ("mlp", "linear"):
            # * remask, re-mask
            rep[mask_nodes] = 0

        if self._decoder_type in ("mlp", "liear") :
            recon = self.decoder(rep)
        else:
            recon = self.decoder(pre_use_g, rep, weight) # add weight

        x_init = x[mask_nodes]
        # x_init  = g.ndata['low_signal'][mask_nodes]

        x_rec = recon[mask_nodes]
        loss = self.criterion(x_rec, x_init) 

        ######################## add by ssh
        # loss += 10 * self.alignment(embed_masked, rich_embed_masked_token)
        # if epoch >= self._delayed_ema_epoch:
        #     self.ema_update()
        ######################## add by ssh

        return loss


    # def ema_update(self):
    #     def update(student, teacher):
    #         with torch.no_grad():
    #         # m = momentum_schedule[it]  # momentum parameter
    #             m = self._momentum
    #             for param_q, param_k in zip(student.parameters(), teacher.parameters()):
    #                 param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
    #     update(self.encoder, self.encoder_ema)


    def embed(self, g, x):
        rep = self.encoder(g, x, return_hidden=True)
        return rep

    @property
    def enc_params(self):
        return self.encoder.parameters()
    
    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])
