from pickle import TRUE
import time
import torch
import dgl.data
from datasets_dgl.utils import *
import scipy
from scipy.sparse import load_npz
import pickle
import numpy as np
import random

class SyntheticDGL(torch.utils.data.Dataset):

    def __init__(self, syn_q):
        data_dir = '/home/songsh/FilterGMAE/datasets_dgl/all_data_synthetic/'

        edge = np.loadtxt(data_dir + 'syn-{}.edge'.format(syn_q), dtype=int).tolist()
        labels = np.loadtxt(data_dir + 'syn-{}.lab'.format(syn_q), dtype=int)
        features = np.loadtxt(data_dir + 'syn-{}.feat'.format(syn_q), dtype=float)
       
        n = labels.shape[0]
        idx = [i for i in range(n)]
        random.shuffle(idx)
        idx_train = np.array(idx[:100])
        idx_val   = np.array(idx[100:150])
        idx_test  = np.array(idx[150:])

        U = [e[0] for e in edge] # 生成的时候就是对称的了
        V = [e[1] for e in edge]
        
        g = dgl.graph((U, V))

        c1 = 0
        c2 = 0
        lab = labels.tolist()
        for e in edge:
            if lab[e[0]] == lab[e[1]]:
                c1 += 1
            else:
                c2 += 1
        print(c1/len(edge), c2/len(edge))

        #normalization will make features degenerated
        #features = normalize_features(features)
        features = torch.FloatTensor(features)

        nclass = 2
        self.labels    = torch.LongTensor(labels)
        self.idx_train = torch.LongTensor(idx_train)
        self.idx_val = torch.LongTensor(idx_val)
        self.idx_test  = torch.LongTensor(idx_test)

        self.get_mask()
        g.ndata['feat'] = features
        g.ndata['label'] = self.labels
        g.ndata['train_mask'] =   torch.tensor(self.train_mask)
        g.ndata['val_mask']   =   torch.tensor(self.val_mask)
        g.ndata['test_mask']  =   torch.tensor(self.test_mask)

        self.graph = g




    def get_mask(self):
        def get_mask(idx):
            mask = np.zeros(self.labels.shape[0], dtype=np.bool)
            mask[idx] = 1
            return mask

        self.train_mask = get_mask(self.idx_train)
        self.val_mask = get_mask(self.idx_val)
        self.test_mask = get_mask(self.idx_test)
        

    def __getitem__(self, i):
        assert i == 0
        return self.graph, self.graph.ndata['label']

    def __len__(self):
        return 1

class SyntheticDataset(torch.utils.data.Dataset):

    def __init__(self, syn_q):
        """
            Loading Chameleon Dataset
        """
        start = time.time()
        print("[I] Loading synthetic dataset" )
        base_graph = SyntheticDGL(syn_q)
        self.graph = base_graph.graph

        print('train_mask, test_mask, val_mask sizes :',self.graph.ndata['train_mask'].sum(),self.graph.ndata['val_mask'].sum(),self.graph.ndata['test_mask'].sum())
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time()-start))


    def collate(self, samples):
        return samples[0]

    @property
    def num_classes(self):
        return 2

if __name__ == '__main__':
    dataset = SyntheticDataset(0)
    print(dataset.train.__getitem__(0))

