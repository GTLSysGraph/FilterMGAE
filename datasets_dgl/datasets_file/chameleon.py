from pickle import TRUE
import time
import torch
import dgl.data
from datasets_dgl.utils import *
import scipy
from scipy.sparse import load_npz
import pickle
import numpy as np

class ChameleonDGL(torch.utils.data.Dataset):

    def __init__(self,args):
        data_name = args.data.split('-')[1].lower()

        data_dir = '/home/songsh/datasets_dgl/all_data_attack/%s/%s/' % (args.attack, data_name)

        adj = load_npz(data_dir + '%s_%s_%s.npz' % (args.attack, data_name, args.ptb_rate))
        g = dgl.from_scipy(adj)

        
        with open(data_dir + data_name + '_data.pickle', 'rb') as handle:
            data = pickle.load(handle)
        features  = data["features"]
        self.labels    = data["labels"]
        self.idx_train = data["idx_train"]
        self.idx_val   = data["idx_val"]
        if args.attack == 'nettack':
            self.idx_test  = np.load(data_dir + '%s_%s_test.npy'  % (args.attack, data_name))
        else:
            self.idx_test  = data["idx_test"]

        g.ndata['feat']  = torch.tensor(features)
        g.ndata['label'] = torch.tensor(self.labels)

        self.get_mask()
        g.ndata['train_mask'] =   torch.tensor(self.train_mask)
        g.ndata['val_mask']   =   torch.tensor(self.val_mask)
        g.ndata['test_mask']  =   torch.tensor(self.test_mask)
        
        g  = dgl.remove_self_loop(g)
        g  = dgl.add_self_loop(g) # 不管如何这里要处理一下，因为有的加载的文件中已经包含自环了，但有的文件没有，所以要统一，要不如果有自环的再加就两层自环了！还好自己的代码中对自环处理后再重加，惊险
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

class ChameleonDataset(torch.utils.data.Dataset):

    def __init__(self, args):
        """
            Loading Chameleon Dataset
        """
        start = time.time()
        print("[I] Loading dataset %s..." % (args.data))
        self.name  = args.data.split('-')[1]
        base_graph = ChameleonDGL(args)
        self.graph = base_graph.graph

        print('train_mask, test_mask, val_mask sizes :',self.graph.ndata['train_mask'].sum(),self.graph.ndata['val_mask'].sum(),self.graph.ndata['test_mask'].sum())
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time()-start))


    def collate(self, samples):
        return samples[0]

    @property
    def num_classes(self):
        return 5

if __name__ == '__main__':
    dataset = ChameleonDataset('Chameleon')
    print(dataset.train.__getitem__(0))

