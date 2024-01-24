import logging
import numpy as np
from tqdm import tqdm
import torch
import dgl.function as fn
from  easydict  import EasyDict
import pandas as pd

from FilterGMAE.utils import (
    create_optimizer,
    set_random_seed,
    TBLogger,
    get_current_lr,
    load_best_configs,
)

from datasets_dgl.data_dgl import *
from datasets_dgl.utils import to_scipy
from datasets_dgl.utils_generate_synthetic import *
from build_easydict import build_easydict

from FilterGMAE.evaluation    import node_classification_evaluation
from FilterGMAE.models        import build_model
from FilterGMAE.compute_score import * 

import matplotlib.pyplot as plt

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

# compare jaccard svd
from FilterGMAE.compare_jaccard_svd import drop_dissimilar_edges,truncatedSVD



def pretrain(model, graph, feat, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, logger=None):
    logging.info("start training..")
    graph = graph.to(device)
    x = feat.to(device)

    # # 对比实验，在评估的时候依然使用原始图，训练过程使用处理后的图，公平比较
    # # Jaccard
    # modified_adj = drop_dissimilar_edges(x.cpu().numpy(), to_scipy(graph.adj().to_dense()), binary_feature=True, threshold=0.0)
    # svd
    modified_adj = to_scipy(torch.tensor(truncatedSVD(to_scipy(graph.adj().to_dense()), k=50)))
    weight = torch.tensor(modified_adj.data).to(device)

    graph_pretrain = dgl.from_scipy(modified_adj)
    graph_pretrain = dgl.remove_self_loop(graph_pretrain)
    graph_pretrain = dgl.add_self_loop(graph_pretrain) 
    graph_pretrain = graph_pretrain.to(device)
    graph_pretrain.ndata['train_mask'] = graph.ndata['train_mask']
    graph_pretrain.ndata['val_mask'] = graph.ndata['val_mask']
    graph_pretrain.ndata['test_mask'] = graph.ndata['test_mask']
    graph_pretrain.ndata['label']      = graph.ndata['label']
    graph_pretrain.ndata['feat']      = graph.ndata['feat']
    print(graph_pretrain)
    print(graph)


    epoch_iter = tqdm(range(max_epoch))
    for epoch in epoch_iter:
        model.train()

        # compare
        # loss, loss_dict = model(graph_pretrain, x)           # jaccard
        loss, loss_dict = model(graph_pretrain, x, weight)   # svd
        # loss, loss_dict = model(graph, x)                    # origin

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        epoch_iter.set_description(f"# Epoch {epoch}: train_loss: {loss.item():.4f}")
        if logger is not None:
            loss_dict["lr"] = get_current_lr(optimizer)
            logger.note(loss_dict, step=epoch)

        if (epoch + 1) % 200 == 0:
            node_classification_evaluation(model, graph, graph.ndata['feat'], num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob, mute=True)

    # return best_model

    # pretrain结束之后，保存一下embedding进行t-sne可视化并保存
    # with torch.no_grad():
    #     color = np.array(['Salmon', 'MediumPurple', 'LimeGreen', 'Orange','Cyan', 'RoyalBlue','Yellow'])
    #     pretrain_final_embedding = model.embed(graph.to(device), x.to(device)).cpu()
    #     from sklearn.manifold import TSNE
    #     tsne = TSNE(n_components = 2 ,random_state=42)
    #     result = tsne.fit_transform(pretrain_final_embedding)
    #     plt.figure()
    #     plt.scatter(result[:, 0], result[:, 1], s=40, alpha= 0.5, c=color.take(graph.ndata['label'].cpu().numpy()))
    #     plt.savefig('./t-sne/cora_metattack_0.0.pdf')

    return model


def Train_FilterMGAE_nodecls(margs):
    device = f"cuda" if torch.cuda.is_available() else "cpu"

    print('Preparing dataset')
    dataset_name = margs.dataset
    if dataset_name.split('-',1)[0] == 'Attack':
        print("attack type: " + margs.attack.split('-')[0] + ' | ' + "ptb_rate: " + margs.attack.split('-')[1])
        assert margs.attack != None
        DATASET = EasyDict()
        DATASET.ATTACK = {
            "data"          :dataset_name,
            "attack"        :margs.attack.split('-')[0],
            "ptb_rate"      :margs.attack.split('-')[1],
            # use split
            # "train_size"    :margs.train_size,
            # "val_size"      :margs.val_size,
            # "test_size"     :margs.test_size,
            # "group"         :margs.group, # train_size * 10
            # "use_g1_split"  :margs.use_g1_split
        }
        # now just attack use
        dataset  = load_attack_data(DATASET['ATTACK'])
        graph = dataset.graph    
    else:
        dataset  = load_data(dataset_name) # graph,labels = dataset[0] ogbn-arxiv tuple
        graph = dataset[0] if dataset_name.split('-')[0] != 'syn' else dataset.graph

    num_features = graph.ndata["feat"].shape[1]
    num_classes = dataset.num_classes

    graph = dgl.remove_self_loop(graph)
    graph = dgl.add_self_loop(graph) #在扰动大的情况下不加自环貌似更好


    MDT = build_easydict()
    param         = MDT['MODEL']['PARAM']
    if param.use_cfg:
        param = load_best_configs(param, dataset_name.split('-',1)[1].lower() if dataset_name.split('-',1)[0] == 'Attack' else dataset_name.lower() , "./configs.yml")
    print(param)

    param.num_nodes    = graph.num_nodes()
    param.num_features = num_features
    seeds          = param.seeds
    max_epoch      = param.max_epoch
    max_epoch_f    = param.max_epoch_f
    num_hidden     = param.num_hidden
    num_layers     = param.num_layers
    encoder_type   = param.encoder
    decoder_type   = param.decoder
    replace_rate   = param.replace_rate
    optim_type     = param.optimizer 
    loss_fn        = param.loss_fn
    lr             = param.lr
    weight_decay   = param.weight_decay
    lr_f           = param.lr_f
    weight_decay_f = param.weight_decay_f
    linear_prob    = param.linear_prob
    load_model     = param.load_model
    save_model     = param.save_model
    logs           = param.logging
    use_scheduler  = param.scheduler
    # add by ssh
    use_feat_filter = param.use_feat_filter
    use_low         = param.use_low
    mask_strategy   = param.mask_strategy
    dis_lambda      = param.dis_lambda 
    mix_type        = param.mix_type
    mix_weights     = {}
    mix_weights['w_degree']      = param.w_degree
    mix_weights['w_dis']         = param.w_dis
    mix_weights['w_csim']        = param.w_csim
    mix_weights['w_certifyk']    = param.w_certifyk


    compute_score(graph, mask_strategy, dataset_name, dis_lambda, mix_type, mix_weights)
    if use_feat_filter:
        feat_filter(graph)





    # mask design
    # 本文关注一个非常有意思的点，即什么样的掩蔽策略可以提升预训练的鲁棒性
    # if param.mask_strategy == 'csim':
    #     csim_sorted_value, csim_sorted_indices = torch.sort(graph.ndata['csim_score'])
    #     graph.ndata['csim_sorted']             = csim_sorted_indices
    #     graph.ndata['final_sorted']            = graph.ndata['csim_sorted']
    #     ## 额外的分析评估，sorted score分数
    #     # _,sorted_score = torch.sort(sorted_indices)
    #     # print(graph.ndata['train_mask'].nonzero().squeeze())
    #     # print(sum(sorted_score[graph.ndata['train_mask'].nonzero().squeeze()]))
    # elif param.mask_strategy == 'degree':
    #     # degree_sorted_value, degree_sorted_indices        = torch.sort(graph.ndata['degree_score'],descending=True)
    #     degree_sorted_indices        = torch.multinomial(graph.ndata['degree_score'], graph.num_nodes())
    #     graph.ndata['degree_sorted'] = degree_sorted_indices
    #     graph.ndata['final_sorted']  = graph.ndata['degree_sorted']

    # elif param.mask_strategy == 'distribution':
    #     dis_sorted_value, dis_sorted_indices = torch.sort(graph.ndata['dis_score'],descending=True)
    #     graph.ndata['dis_sorted']            = dis_sorted_indices
    #     graph.ndata['final_sorted']          = graph.ndata['dis_sorted']
    #     # count函数的作用是统计按照分布排序之后排在前面的节点和设定分布之间的重合度，当lamda值为1的时候完全重合
    #     # def count(a,b):
    #     #     num = 0
    #     #     for i in range(len(a)):
    #     #         if a[i] == b[i]:
    #     #             num+=1
    #     #         else:
    #     #             break
    #     #     return num
    #     # print(count(torch.sort(graph.ndata['dis_sorted'][:len(idx_train)])[0], idx_train))
    #     # quit()
    # elif param.mask_strategy == 'certifyk':
    #     certifyk_sorted_indices         = torch.multinomial(graph.ndata['certifyk_score'], graph.num_nodes())
    #     graph.ndata['certifyk_sorted']  = certifyk_sorted_indices
    #     graph.ndata['final_sorted']     = graph.ndata['certifyk_sorted']
    # elif param.mask_strategy == 'mix':
    #     mix_sorted_indices              = torch.multinomial(graph.ndata['mix_score'], graph.num_nodes())
    #     graph.ndata['mix_sorted']       = mix_sorted_indices
    #     graph.ndata['final_sorted']     = graph.ndata['mix_sorted']
    # elif param.mask_strategy == 'random':
    #     graph.ndata['final_sorted']    = torch.arange(0, graph.num_nodes())
    # else:
    #     raise Exception('node sorting corresponding to other masking strategies will be coming soon..')
    # assert param.num_nodes == len(graph.ndata['final_sorted'])



    acc_list = []
    estp_acc_list = []
    for i, seed in enumerate(seeds):
        print(f"####### Run {i+1} for seed {seed}")
        set_random_seed(seed)

        if logs:
            logger = TBLogger(name=f"{dataset_name}_loss_{loss_fn}_rpr_{replace_rate}_nh_{num_hidden}_nl_{num_layers}_lr_{lr}_mp_{max_epoch}_mpf_{max_epoch_f}_wd_{weight_decay}_wdf_{weight_decay_f}_{encoder_type}_{decoder_type}")
        else:
            logger = None

        model = build_model(param)
        model.to(device)
        optimizer = create_optimizer(optim_type, model, lr, weight_decay)

        if use_scheduler:
            logging.info("Use schedular")
            scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epoch) ) * 0.5
            # scheduler = lambda epoch: epoch / warmup_steps if epoch < warmup_steps \
                    # else ( 1 + np.cos((epoch - warmup_steps) * np.pi / (max_epoch - warmup_steps))) * 0.5
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
        else:
            scheduler = None
        
        ############### add by ssh 
        if use_feat_filter:
            if use_low:
                x = graph.ndata['low_signal']
            else:
                x = graph.ndata['high_signal']
        else:
            x = graph.ndata['feat']
        ############### add by ssh 

  
        if not load_model:
            model = pretrain(model, graph, x, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, logger)
            model = model.cpu()

        if load_model:
            logging.info("Loading Model ... ")
            model.load_state_dict(torch.load("checkpoint.pt"))
        if save_model:
            logging.info("Saveing Model ...")
            torch.save(model.state_dict(), "checkpoint.pt")
        
        model = model.to(device)
        model.eval()

        final_acc, estp_acc = node_classification_evaluation(model, graph, graph.ndata['feat'], num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob)
        acc_list.append(final_acc)
        estp_acc_list.append(estp_acc)

        if logger is not None:
            logger.finish()

    final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
    estp_acc, estp_acc_std = np.mean(estp_acc_list), np.std(estp_acc_list)
    print(f"# final_acc: {final_acc:.4f}±{final_acc_std:.4f}")
    print(f"# early-stopping_acc: {estp_acc:.4f}±{estp_acc_std:.4f}")




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',        type=str,                   default= 'Cora') #['Cora', 'CiteSeer', 'PubMed', 'DBLP', 'WikiCS', 'Coauthor-CS', 'Coauthor-Phy','Amazon-Computers', 'Amazon-Photo', 'ogbn-arxiv','Reddit','Flickr','Yelp']
    parser.add_argument('--attack',         type=str,                   default= 'no') # ['DICE-0.1','Meta_Self-0.05' ,...] 攻击方式-扰动率


    # 同一攻击比例不同划分的图
    # parser.add_argument('--train_size',     type=float,                 default= 0.5,                                                                           help='train rate.')
    # parser.add_argument('--val_size',       type=float,                 default= 0.2,                                                                           help='val rate.')
    # parser.add_argument('--test_size',      type=float,                 default= 0.3,                                                                           help='test rate.')
    # parser.add_argument('--group',          type=int,                   default= 5,                                                                             help='Group TAG: train rate * 10.')
    # parser.add_argument('--use_g1_split',   action='store_true',        default=False,                                                                          help='whether use_g1_split.')  


    margs = parser.parse_args()
    Train_FilterMGAE_nodecls(margs)