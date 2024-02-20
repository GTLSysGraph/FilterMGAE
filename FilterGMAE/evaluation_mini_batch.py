import copy
import logging
from tqdm import tqdm
import os
import numpy as np
import psutil
import torch
import torch.nn as nn
import dgl
from torch.utils.data import DataLoader
from .utils import accuracy,set_random_seed,show_occupied_memory


def evaluete_mini_batch(model, graph, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, device, batch_size, shuffle):
    
    train_mask = graph.ndata["train_mask"]
    val_mask = graph.ndata["val_mask"]
    test_mask = graph.ndata["test_mask"]
    labels = graph.ndata["label"]
    train_lbls, val_lbls, test_lbls = labels[train_mask], labels[val_mask], labels[test_mask]

    print(f"num_train: {train_mask.shape[0]}, num_val: {val_mask.shape[0]}, num_test: {test_mask.shape[0]}")

    # 这里要注意labels的位置不一样 linear在cpu上，而finetune在cuda上
    if linear_prob:
        test_acc = linear_probing_minibatch(model, graph, [train_mask,val_mask,test_mask],[train_lbls, val_lbls, test_lbls], lr_f=lr_f, weight_decay_f=weight_decay_f, max_epoch_f=max_epoch_f, batch_size=batch_size, device=device, shuffle=shuffle)
    else:
        test_acc = finetune(model, graph, labels.to(device), num_classes, lr_f=lr_f, weight_decay_f=weight_decay_f, max_epoch_f=max_epoch_f, use_scheduler=True,batch_size=batch_size, device=device)
    return test_acc



def finetune(model, graph, labels, num_classes, lr_f, weight_decay_f, max_epoch_f, use_scheduler, batch_size, device):
    logging.info("-- Finetuning in downstream tasks ---")

    train_nid = graph.ndata['train_mask'].nonzero().squeeze()
    val_nid = graph.ndata['val_mask'].nonzero().squeeze()
    test_nid = graph.ndata['test_mask'].nonzero().squeeze()

    model = model.get_encoder()
    model.reset_classifier(int(num_classes))
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
    train_loader = dgl.dataloading.DataLoader(
            graph,train_nid, sampler,  # 这里用的是所有的 train val test 为了后面的linear probe和fineturn
            batch_size=batch_size,
            shuffle=False, 
            drop_last=False,
            num_workers=1)    
    val_loader = dgl.dataloading.DataLoader(
            graph,val_nid, sampler,  # 这里用的是所有的 train val test 为了后面的linear probe和fineturn
            batch_size=batch_size,
            shuffle=False, 
            drop_last=False,
            num_workers=1)    
    
    test_loader = dgl.dataloading.DataLoader(
            graph,test_nid, sampler,  # 这里用的是所有的 train val test 为了后面的linear probe和fineturn
            batch_size=batch_size,
            shuffle=False, 
            drop_last=False,
            num_workers=1)    
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_f, weight_decay=weight_decay_f)

    if use_scheduler and max_epoch_f > 0:
        logging.info("Use schedular")
        warmup_epochs = int(max_epoch_f * 0.1)
        # scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epoch_f) ) * 0.5
        scheduler = lambda epoch: epoch / warmup_epochs if epoch < warmup_epochs else ( 1 + np.cos((epoch - warmup_epochs) * np.pi / (max_epoch_f - warmup_epochs))) * 0.5
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
    else:
        scheduler = None


    def eval_with_lc(model, loader):
        pred_counts = []
        model.eval()
        epoch_iter = tqdm(loader)
        with torch.no_grad():
            for input_nodes, output_nodes, _ in epoch_iter:
                subgraph = dgl.node_subgraph(graph, input_nodes,store_ids=True).to(device)
                x = subgraph.ndata.pop("feat")
                prediction, _ = model(subgraph, x) # 注意这里返回的是tuple，所以分开
                prediction = prediction[:output_nodes.shape[0]]
                pred_counts.append((prediction.argmax(1) == labels[output_nodes]))
        pred_counts = torch.cat(pred_counts)
        acc = pred_counts.float().sum() / pred_counts.shape[0]
        return acc


    best_val_acc = 0
    best_model = None
    best_epoch = 0
    test_acc = 0
    early_stop_cnt = 0

    
    for epoch in range(max_epoch_f):
        if early_stop_cnt >= 10:
            break
        
        epoch_iter = tqdm(train_loader)
        losses = []
        model.train()

        for input_nodes, output_nodes, _ in epoch_iter:
            subgraph = dgl.node_subgraph(graph, input_nodes,store_ids=True).to(device)
            x = subgraph.ndata.pop("feat")
            prediction, _ = model(subgraph, x)
            prediction = prediction[:output_nodes.shape[0]]
            loss = criterion(prediction, labels[output_nodes])

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

            epoch_iter.set_description(f"Finetuning | train_loss: {loss.item():.4f}, Memory: {show_occupied_memory():.2f} MB")
            optimizer.step()
            losses.append(loss.item())


        if scheduler is not None:
            scheduler.step()

        print("eval val...")
        val_acc  = eval_with_lc(model, val_loader)

        if val_acc > best_val_acc:
            best_model = copy.deepcopy(model)
            best_val_acc = val_acc
            best_epoch = epoch
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1
        print("val Acc {:.4f}, Best Val Acc {:.4f}".format(val_acc, best_val_acc))
    
    # 用best model
    model = best_model
    print("eval test...")
    test_acc = eval_with_lc(model, test_loader)

    test_acc = np.array(test_acc.cpu())
    print(f"Finetune | TestAcc: {test_acc:.4f},  Best ValAcc: {best_val_acc:.4f} in epoch {best_epoch} --- ")
    return test_acc





def linear_probing_minibatch(model, graph, mask, labels, lr_f, weight_decay_f, max_epoch_f, batch_size, device, shuffle):
    logging.info("-- Linear Probing in downstream tasks ---")

  
    train_lbls, val_lbls, test_lbls = labels
    train_mask, val_mask, test_mask = mask

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
    # 这里用的是所有的 train val test 为了后面的linear probe和fineturn
    train_eval_dataloader = dgl.dataloading.DataLoader(
            graph, torch.arange(0, graph.num_nodes()), sampler,  
            batch_size=batch_size,
            shuffle=False, # 顺序不能变
            drop_last=False,
            num_workers=1)

    print("######## Prepare All Embedding used...")
    with torch.no_grad():
        model.eval()
        embeddings = []

        for input_nodes, output_nodes, _ in tqdm(train_eval_dataloader):
            subgraph = dgl.node_subgraph(graph, input_nodes).to(device)
            x = subgraph.ndata.pop("feat")
            batch_emb, _ = model.embed(subgraph, x) #注意返回的是tuple
            batch_emb = batch_emb[:output_nodes.shape[0]] #这里一定要分开写，因为上面是tuple！
            embeddings.append(batch_emb.cpu())
    embeddings = torch.cat(embeddings, dim=0)
    train_emb, val_emb, test_emb = embeddings[train_mask], embeddings[val_mask], embeddings[test_mask]


    batch_size_LinearProbing = 5120

    acc = []
    seeds = [0]
    for i,_ in enumerate(seeds):
        print(f"######## Run seed {seeds[i]} for LinearProbing...")
        set_random_seed(seeds[i])
        print(f"training sample:{len(train_emb)}")
        test_acc = node_classification_linear_probing(
            (train_emb, val_emb, test_emb), 
            (train_lbls, val_lbls, test_lbls), 
            lr_f, weight_decay_f, max_epoch_f, device, batch_size=batch_size_LinearProbing, shuffle=shuffle) # 这个shuffle可以变
        acc.append(test_acc)

    print(f"# final_acc: {np.mean(acc):.4f}, std: {np.std(acc):.4f}")
    return np.mean(acc)




def node_classification_linear_probing(embeddings, labels, lr, weight_decay, max_epoch, device, mute=False, batch_size=-1, shuffle=True):
    criterion = torch.nn.CrossEntropyLoss()

    train_emb, val_emb, test_emb = embeddings
    train_label, val_label, test_label = labels
    train_label = train_label.to(torch.long)
    val_label = val_label.to(torch.long)
    test_label = test_label.to(torch.long)
    
    best_val_acc = 0
    best_val_epoch = 0
    best_model = None

    if not mute:
        epoch_iter = tqdm(range(max_epoch))
    else:
        epoch_iter = range(max_epoch)

    encoder = LogisticRegression(train_emb.shape[1], int(train_label.max().item() + 1))
    encoder = encoder.to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr, weight_decay=weight_decay)
    
    assert len(train_emb) == len(train_label)
    assert len(val_emb) == len(val_label)
    assert len(test_emb) == len(test_label)

    if batch_size > 0:
        train_loader = LinearProbingDataLoader(np.arange(len(train_emb)), train_emb, train_label, batch_size=batch_size, num_workers=1, shuffle=shuffle)
        # train_loader = DataLoader(np.arange(len(train_emb)), batch_size=batch_size, shuffle=False)
        val_loader = LinearProbingDataLoader(np.arange(len(val_emb)), val_emb, val_label, batch_size=batch_size, num_workers=1,shuffle=False)
        test_loader = LinearProbingDataLoader(np.arange(len(test_emb)), test_emb, test_label, batch_size=batch_size, num_workers=1, shuffle=False)
    else:
        train_loader = [np.arange(len(train_emb))]
        val_loader = [np.arange(len(val_emb))]
        test_loader = [np.arange(len(test_emb))]

    def eval_forward(loader, _label):
        pred_all = []
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            pred = encoder(None, batch_x)
            pred_all.append(pred.cpu())
        pred = torch.cat(pred_all, dim=0)
        acc = accuracy(pred, _label)
        return acc

    for epoch in epoch_iter:
        encoder.train()

        for batch_x, batch_label in train_loader:
            batch_x = batch_x.to(device)
            batch_label = batch_label.to(device)
            pred = encoder(None, batch_x)
            loss = criterion(pred, batch_label)
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
            optimizer.step()

        with torch.no_grad():
            encoder.eval()
            val_acc = eval_forward(val_loader, val_label)
        
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch
            best_model = copy.deepcopy(encoder)

        if not mute:
            epoch_iter.set_description(f"# Epoch: {epoch}, train_loss:{loss.item(): .4f}, val_acc:{val_acc:.4f}")

    best_model.eval()
    encoder = best_model #因为太大了，所以就只用best model了
    with torch.no_grad():
        test_acc = eval_forward(test_loader, test_label)
    if mute:
        print(f"# IGNORE: --- TestAcc: {test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} --- ")
    else:
        print(f"--- TestAcc: {test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} --- ")

    return test_acc




class LinearProbingDataLoader(DataLoader):
    def __init__(self, idx, feats, labels=None, **kwargs):
        self.labels = labels
        self.feats = feats

        kwargs["collate_fn"] = self.__collate_fn__
        super().__init__(dataset=idx, **kwargs)

    def __collate_fn__(self, batch_idx):
        feats = self.feats[batch_idx]
        label = self.labels[batch_idx]

        return feats, label
    


class LogisticRegression(nn.Module):
    def __init__(self, num_dim, num_class):
        super().__init__()
        self.linear = nn.Linear(num_dim, num_class)

    def forward(self, g, x, *args):
        logits = self.linear(x)
        return logits
    
