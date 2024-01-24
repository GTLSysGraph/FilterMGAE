import dgl
from collections import namedtuple, Counter
from dgl.dataloading import GraphDataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
# attack nodecls dataset
from datasets_dgl.datasets_file.cora     import CoraDataset
from datasets_dgl.datasets_file.cora_diff_split import CoraDataset_DiffSplit

from datasets_dgl.datasets_file.coraml   import CoraMLDataset
from datasets_dgl.datasets_file.citeseer import CiteseerDataset
from datasets_dgl.datasets_file.pubmed   import PubmedDataset
from datasets_dgl.datasets_file.polblogs import PolblogsDataset
from datasets_dgl.datasets_file.chameleon import ChameleonDataset
from datasets_dgl.datasets_file.squirrel import SquirrelDataset
from datasets_dgl.datasets_file.synthetic import SyntheticDataset
from datasets_dgl.datasets_file.ogbnarxiv import OgbnArxivDataset
# raw nodecls dataset 
from dgl.data     import CoraGraphDataset
from dgl.data     import CiteseerGraphDataset
from dgl.data     import PubmedGraphDataset
from dgl.data     import RedditDataset
from dgl.data     import CoauthorCSDataset


# inductive 
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data.ppi import PPIDataset

# graph dataset
from dgl.data import TUDataset


def load_attack_data(args):
    if args.data in ['Attack-Cora']:
        return CoraDataset(args)
        # return CoraDataset_DiffSplit(args)
    elif args.data in ['Attack-Citeseer']:
        return CiteseerDataset(args)
    elif args.data in ['Attack-Pubmed']:
        return PubmedDataset(args)
    elif args.data in ['Attack-Polblogs']:
        return PolblogsDataset(args)
    elif args.data in ['Attack-Cora_ml']:
        return CoraMLDataset(args)
    elif args.data in ['Attack-ogbn-arxiv']:
        return OgbnArxivDataset(args)
    elif args.data in ['Attack-Chameleon']:
        return ChameleonDataset(args)
    elif args.data in ['Attack-Squirrel']:
        return SquirrelDataset(args)
    else:
        raise Exception('Unknown dataset!')

def load_data(dataname):
    if dataname in ['Cora']:
        return CoraGraphDataset()
    elif dataname in ['Citeseer']:
        return CiteseerGraphDataset()
    elif dataname in ['Pubmed']:
        return PubmedGraphDataset()
    if dataname in ['CoauthorCS']:
        return CoauthorCSDataset()
    elif dataname in ['ogbn-arxiv']:
        return DglNodePropPredDataset('ogbn-arxiv',root = '/home/songsh/.dgl')
    elif dataname in ['Reddit']:
        return RedditDataset()
    elif dataname in ['PPI']:
        return PPIDataset()
    elif dataname.split('-')[0] in ['syn']:
        return SyntheticDataset(dataname.split('-')[1])
    else:
        raise Exception('Unknown dataset!')



def load_graph_data(dataset_name):
    assert dataset_name in ['IMDB-BINARY' ,'IMDB-MULTI' ,'PROTEINS', 'COLLAB', 'MUTAG', 'REDDIT-BINARY', 'NCI1']
    return TUDataset(dataset_name)


def load_inductive_dataset(dataset_name):
    if dataset_name == "ppi":
        batch_size = 2
        # define loss function
        # create the dataset
        train_dataset = PPIDataset(mode='train')
        valid_dataset = PPIDataset(mode='valid')
        test_dataset = PPIDataset(mode='test')
        train_dataloader = GraphDataLoader(train_dataset, batch_size=batch_size)
        valid_dataloader = GraphDataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        test_dataloader = GraphDataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        eval_train_dataloader = GraphDataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        g = train_dataset[0]
        num_classes = train_dataset.num_labels
        num_features = g.ndata['feat'].shape[1]
    else:
        batch_size = 1
        dataset = load_data(dataset_name)
        num_classes = dataset.num_classes
        g = dataset[0] 
        num_features = g.ndata["feat"].shape[1]

        train_mask = g.ndata['train_mask']
        feat = g.ndata["feat"]
        feat = scale_feats(feat)
        g.ndata["feat"] = feat

        g = g.remove_self_loop()
        g = g.add_self_loop()

        train_nid = np.nonzero(train_mask.data.numpy())[0].astype(np.int64)
        train_g = dgl.node_subgraph(g, train_nid)
        train_dataloader = [train_g]
        valid_dataloader = [g]
        test_dataloader = valid_dataloader
        eval_train_dataloader = [train_g]
        
    return train_dataloader, valid_dataloader, test_dataloader, eval_train_dataloader, num_features, num_classes



def process_OGB(dataset):
    graph, labels = dataset[0]
    num_nodes = graph.num_nodes()

    split_idx = dataset.get_idx_split()
    train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    graph = preprocess(graph)

    if not torch.is_tensor(train_idx):
        train_idx = torch.as_tensor(train_idx)
        val_idx = torch.as_tensor(val_idx)
        test_idx = torch.as_tensor(test_idx)

    feat = graph.ndata["feat"]
    feat = scale_feats(feat)
    graph.ndata["feat"] = feat

    train_mask = torch.full((num_nodes,), False).index_fill_(0, train_idx, True)
    val_mask = torch.full((num_nodes,), False).index_fill_(0, val_idx, True)
    test_mask = torch.full((num_nodes,), False).index_fill_(0, test_idx, True)
    graph.ndata["label"] = labels.view(-1)
    graph.ndata["train_mask"], graph.ndata["val_mask"], graph.ndata["test_mask"] = train_mask, val_mask, test_mask
    return graph



def preprocess(graph):
    feat = graph.ndata["feat"]
    graph = dgl.to_bidirected(graph)
    graph.ndata["feat"] = feat

    graph = graph.remove_self_loop().add_self_loop()
    graph.create_formats_()
    return graph


def scale_feats(x):
    scaler = StandardScaler()
    feats = x.numpy()
    scaler.fit(feats)
    feats = torch.from_numpy(scaler.transform(feats)).float()
    return feats
