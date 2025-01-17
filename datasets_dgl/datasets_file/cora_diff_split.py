from pickle import TRUE
import time
import torch
import dgl.data
from datasets_dgl.utils import *
import scipy

class CoraDGL(torch.utils.data.Dataset):

    def __init__(self, args):
        data_name = args.data.split('-')[1].lower()

        data_dir = '/home/songsh/FilterGMAE/datasets_dgl/all_data_attack_diff_split/%s/%s/' % (args.attack, data_name)

        adj = torch.load(data_dir + '%s_%s_%s_g%s.pt' % (args.attack, data_name, args.ptb_rate, args.group),map_location= 'cuda')
        g = dgl.from_scipy(to_scipy(adj))
        
        features = to_tensor_features(scipy.sparse.load_npz(data_dir + '%s_features.npz' % (data_name))).to_dense()
        self.labels   = torch.tensor(np.load(data_dir + '%s_labels.npy' % (data_name)))
        g.ndata['feat'] = features
        g.ndata['label'] = self.labels


        if args.use_g1_split:
            self.idx_train = np.load(data_dir + '%s_%s_%s_g1_idx_train_0.1.npy' % (args.attack, data_name, args.ptb_rate))
            self.idx_val   = np.load(data_dir + '%s_%s_%s_g1_idx_val_0.1.npy'   % (args.attack, data_name, args.ptb_rate))
            self.idx_test  = np.load(data_dir + '%s_%s_%s_g1_idx_test_0.8.npy'  % (args.attack, data_name, args.ptb_rate))
        else:
            self.idx_train = np.load(data_dir + '%s_%s_%s_g%s_idx_train_%s.npy' % (args.attack, data_name, args.ptb_rate, args.group, args.train_size))
            self.idx_val   = np.load(data_dir + '%s_%s_%s_g%s_idx_val_%s.npy'   % (args.attack, data_name, args.ptb_rate, args.group, args.val_size))
            self.idx_test  = np.load(data_dir + '%s_%s_%s_g%s_idx_test_%s.npy'  % (args.attack, data_name, args.ptb_rate, args.group, args.test_size))
        
        self.get_mask()
        g.ndata['train_mask'] =   torch.tensor(self.train_mask)
        g.ndata['val_mask']   =   torch.tensor(self.val_mask)
        g.ndata['test_mask']  =   torch.tensor(self.test_mask)
        g  = dgl.remove_self_loop(g)
        g  = dgl.add_self_loop(g) # 统一处理一下，因为有的加载的文件中已经包含自环了，但有的文件没有，要不如果有自环的再加就两层自环了

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

class CoraDataset_DiffSplit(torch.utils.data.Dataset):

    def __init__(self, args):
        """
            Loading Cora Dataset
        """
        start = time.time()
        print("[I] Loading dataset %s..." % (args.data))
        self.name  = args.data.split('-')[1]
        base_graph = CoraDGL(args)
        self.graph = base_graph.graph

        print('train_mask, test_mask, val_mask sizes :',self.graph.ndata['train_mask'].sum(),self.graph.ndata['val_mask'].sum(),self.graph.ndata['test_mask'].sum())
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time()-start))


    def collate(self, samples):
        return samples[0]

    @property
    def num_classes(self):
        return 7

if __name__ == '__main__':
    dataset = CoraDataset_DiffSplit('Cora')
    print(dataset.train.__getitem__(0))

