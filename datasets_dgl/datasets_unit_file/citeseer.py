from pickle import TRUE
import time
import torch
import dgl.data
from datasets_dgl.utils import *
import scipy

class CiteseerUnitDGL(torch.utils.data.Dataset):  # Adaptive attack model

    def __init__(self,args):
        data_name = args.dataset.split('-')[1].lower()

        data_dir = '/home/songsh/datasets_dgl/all_data_unit_test/%s/%s/' % (args.scenario, data_name)

        adj = torch.load(data_dir + '%s/%s/A_perturbed_budget_%s_ptb_%s.pt' % (args.split, args.adaptive_attack_model, args.budget, args.unit_ptb), 
        map_location= 'cuda')
        g = dgl.from_scipy(to_scipy(adj))
        
        features = to_tensor_features(scipy.sparse.load_npz(data_dir + '%s_features.npz' % (data_name))).to_dense()
        self.labels   = torch.tensor(np.load(data_dir + '%s_labels.npy' % (data_name)))
        g.ndata['feat'] = features
        g.ndata['label'] = self.labels

        self.idx_train = np.load(data_dir + '%s/idx_train.npy' % (args.split))
        self.idx_val   = np.load(data_dir + '%s/idx_val.npy'   % (args.split))
        self.idx_test  = np.load(data_dir + '%s/idx_test.npy'  % (args.split))
        
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

class CiteseerUnitDataset(torch.utils.data.Dataset):

    def __init__(self, args):
        """
            Loading Cora Dataset
        """
        start = time.time()
        print("[I] Loading dataset %s..." % (args.dataset))
        self.name  = args.dataset.split('-')[1]
        base_graph = CiteseerUnitDGL(args)
        self.graph = base_graph.graph

        print('train_mask, test_mask, val_mask sizes :',self.graph.ndata['train_mask'].sum(),self.graph.ndata['val_mask'].sum(),self.graph.ndata['test_mask'].sum())
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time()-start))


    def collate(self, samples):
        return samples[0]

    @property
    def num_classes(self):
        return 6

if __name__ == '__main__':
    dataset = CiteseerUnitDataset() # 这个貌似就是cora不是cora ml
    print(dataset.train.__getitem__(0))