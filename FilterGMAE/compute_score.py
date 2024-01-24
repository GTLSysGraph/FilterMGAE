import torch
import dgl.function as fn
import numpy as np
import pandas as pd
import scipy.sparse as sp

from datasets_dgl.utils import to_scipy
from datasets_dgl.utils_generate_synthetic import calculate

def compute_score(graph, mask_type, dataset_name, dis_lambda=1.0, mix_type=None, mix_weight=None):
    if mask_type == 'csim':
        ######### part 1 node central sim score       get graph.ndata['csim_score']
        graph.ndata['feat_norm'] = graph.ndata['feat'] / torch.sqrt(torch.sum(graph.ndata['feat'] * graph.ndata['feat'],dim=1)).unsqueeze(-1)
        graph.update_all(fn.u_dot_v('feat_norm', 'feat_norm', 'e'), fn.sum('e', 'csim_score'))
        graph.ndata['csim_score'] = (graph.ndata['csim_score'] / graph.in_degrees().unsqueeze(-1)).squeeze()
        # graph.ndata['csim_score'] = 1 / (1 +  torch.exp(graph.ndata['csim'])) # cim越大权重越小，希望在混合权重的时候可以靠后
        graph.ndata['final_score'] = graph.ndata['csim_score']
    elif mask_type == 'dis':
        ######### part 2 distribution score            get graph.ndata['dis_score']
        # 首先给定一个部分识别标志
        Tag         = torch.zeros(graph.num_nodes())
        lamda       = dis_lambda#控制交叉的缩放区间范围，1.0即完全和数量较少的分布重合，越小重合度越低，默认1.0
        adj = graph.adj().to_dense()

        idx_train = graph.ndata['train_mask'].nonzero().squeeze().numpy()
        idx_val = graph.ndata['val_mask'].nonzero().squeeze().numpy()
        idx_test = graph.ndata['test_mask'].nonzero().squeeze().numpy()
        idx_unlabeled = np.union1d(idx_val, idx_test)
        if len(idx_train) <= len(idx_unlabeled):
            Tag[idx_train] = 1
            psmall_dis_ratio = torch.sum(adj[:, idx_train], dim =1) / graph.in_degrees()
        else:
            Tag[idx_unlabeled] = 1
            psmall_dis_ratio = torch.sum(adj[:, idx_unlabeled], dim =1) / graph.in_degrees()
        graph.ndata['dis_score'] = 1 / torch.exp(lamda * Tag + psmall_dis_ratio)
        graph.ndata['final_score'] = graph.ndata['dis_score']
        #count函数的作用是统计按照分布排序之后排在前面的节点和设定分布之间的重合度，当lamda值为1的时候完全重合
        def count(a,b):
            num = 0
            for i in range(len(a)):
                if a[i] == b[i]:
                    num+=1
                else:
                    break
            return num
        dis_sorted_value, dis_sorted_indices = torch.sort(graph.ndata['dis_score'])
        print('small part Coincidence rate: {}'.format(count(torch.sort(dis_sorted_indices[:len(idx_train)])[0], idx_train)))
        # quit()
    elif mask_type == 'degree':
        ######### part 3 degree score                  get graph.ndata['degree_score']
        graph.ndata['degree_score'] = graph.in_degrees()
        graph.ndata['final_score']  = graph.ndata['degree_score']
        # graph.ndata['degree_score'] = 1 / (1 +  torch.exp(graph.in_degrees())) # 度越大权重越小，希望在混合权重的时候可以靠后
    elif mask_type == 'certifyk':
        ######### part 4 CertifyK score                get graph.ndata['certifiedk_score'] 
        if dataset_name in ['Attack-Cora' , 'Attack-Citeseer']:
            certifyk = torch.zeros(graph.num_nodes())
            file_path = './generate_certify_K/dir_get_K/%s_approx_alpha_0.8' % (dataset_name.split('-',1)[1].lower())
            df = pd.read_csv(file_path, sep="\t")
            certifyk[df['idx']] = torch.tensor(np.array(df['CertifiedK'])).float()
            # graph.ndata['certifyk_score'] = 1 / (1 +  torch.exp(certifyk))
            graph.ndata['certifyk_score'] = certifyk
            graph.ndata['final_score'] = graph.ndata['certifyk_score']
    elif mask_type == 'mix':
        print(mix_type.split('-'))
        ######### part 5 mix score                    get graph.ndata['mix_score'] 
        # 利用normalize_score划分到相同的区间，然后设置每个部分的权重
        if ('certifyk' in mix_type.split('-')) & (dataset_name not in ['Attack-Cora','Attack-Citeseer']):
            raise Exception('Now certifyk only support Cora and Citeseer')
        weight_sum = 0
        mix_score  = 0
        for stg in mix_type.split('-'):
            compute_score(graph, stg, dataset_name)
            print(graph.ndata[stg + '_score'])
            print(normalize_score(graph.ndata[stg + '_score']))
            print(mix_weight['w_' + stg])
            print('*******************************')
            mix_score  += mix_weight['w_' + stg] * normalize_score(graph.ndata[stg + '_score'])
            weight_sum += mix_weight['w_' + stg]
        assert weight_sum == 1.0
        graph.ndata['mix_score'] = mix_score
        graph.ndata['final_score'] = graph.ndata['mix_score']
    elif mask_type == 'random':
        pass
    return


def feat_filter(graph):
    ######### part 5 low signal and high signal    get graph.ndata['low_signal'] & graph.ndata['high_signal']
    low_signal, high_signal    = calculate(to_scipy(graph.adj().to_dense()), graph.ndata['feat'])
    graph.ndata['low_signal']  = torch.tensor(low_signal).float()
    graph.ndata['high_signal'] = torch.tensor(high_signal).float()
    return

def normalize_score(tensor):
    # 计算张量的最小值和最大值
    min_val = tensor.min()
    max_val = tensor.max()
    # 将张量标准化到 [0, 1] 范围
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor