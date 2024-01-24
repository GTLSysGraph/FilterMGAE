import dgl
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv
from scipy.sparse import lil_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from datasets_dgl.utils import to_scipy

def GCNAdjNorm(adj, order=-0.5):
    r"""

    Description
    -----------
    Normalization of adjacency matrix proposed in `GCN <https://arxiv.org/abs/1609.02907>`__.

    Parameters
    ----------
    adj : scipy.sparse.csr.csr_matrix or torch.FloatTensor
        Adjacency matrix in form of ``N * N`` sparse matrix (or in form of ``N * N`` dense tensor).
    order : float, optional
        Order of degree matrix. Default: ``-0.5``.


    Returns
    -------
    adj : scipy.sparse.csr.csr_matrix
        Normalized adjacency matrix in form of ``N * N`` sparse matrix.

    """
    if sp.issparse(adj):
        adj = sp.eye(adj.shape[0]) + adj
        adj.data[np.where((adj.data > 0) * (adj.data == 1))[0]] = 1
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum, order).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        adj = d_mat_inv @ adj @ d_mat_inv
    else:
        rowsum = torch.sparse.mm(adj, torch.ones((adj.shape[0], 1), device=adj.device)) + 1
        d_inv = torch.pow(rowsum, order).flatten()
        d_inv[torch.isinf(d_inv)] = 0.

        self_loop_idx = torch.stack((
            torch.arange(adj.shape[0], device=adj.device),
            torch.arange(adj.shape[0], device=adj.device)
        ))
        self_loop_val = torch.ones_like(self_loop_idx[0], dtype=adj.dtype)
        indices = torch.cat((self_loop_idx, adj.indices()), dim=1)
        values = torch.cat((self_loop_val, adj.values()))
        values = d_inv[indices[0]] * values * d_inv[indices[1]]
        adj = torch.sparse.FloatTensor(indices, values, adj.shape).coalesce()

    return adj



class GCNGuard(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 hidden_features,
                 n_layers,
                 activation=F.relu,
                 layer_norm=False,
                 dropout=True,
                 feat_norm=None,
                 adj_norm_func=GCNAdjNorm,
                 drop=0.0,
                 attention=True):
        super(GCNGuard, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.feat_norm = feat_norm
        self.adj_norm_func = adj_norm_func
        if type(hidden_features) is int:
            hidden_features = [hidden_features] * (n_layers - 1)
        elif type(hidden_features) is list or type(hidden_features) is tuple:
            assert len(hidden_features) == (n_layers - 1), "Incompatible sizes between hidden_features and n_layers."
        n_features = [in_features] + hidden_features + [out_features]

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            if layer_norm:
                self.layers.append(nn.LayerNorm(n_features[i]))
            self.layers.append(GCNConv(in_features=n_features[i],
                                       out_features=n_features[i + 1],
                                       activation=activation if i != n_layers - 1 else None,
                                       dropout=dropout if i != n_layers - 1 else 0.0))
        self.reset_parameters()
        self.drop = drop
        self.drop_learn = torch.nn.Linear(2, 1)
        self.attention = attention

    @property
    def model_type(self):
        return "torch"

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, adj):
        for layer in self.layers:
            if isinstance(layer, nn.LayerNorm):
                x = layer(x)
            else:
                if self.attention:
                    adj = self.att_coef(x, adj)
                x = layer(x, adj)

        return x

    def att_coef(self, features, adj):
        edge_index = adj._indices()

        n_node = features.shape[0]
        row, col = edge_index[0].cpu().data.numpy()[:], edge_index[1].cpu().data.numpy()[:]

        features_copy = features.cpu().data.numpy()
        sim_matrix = cosine_similarity(X=features_copy, Y=features_copy)  # try cosine similarity
        sim = sim_matrix[row, col]
        sim[sim < 0.1] = 0

        """build a attention matrix"""
        att_dense = lil_matrix((n_node, n_node), dtype=np.float32)
        att_dense[row, col] = sim
        if att_dense[0, 0] == 1:
            att_dense = att_dense - sp.diags(att_dense.diagonal(), offsets=0, format="lil")
        # normalization, make the sum of each row is 1
        att_dense_norm = normalize(att_dense, axis=1, norm='l1')

        """add learnable dropout, make character vector"""
        if self.drop:
            character = np.vstack((att_dense_norm[row, col].A1,
                                   att_dense_norm[col, row].A1))
            character = torch.from_numpy(character.T).to(features.device)
            drop_score = self.drop_learn(character)
            drop_score = torch.sigmoid(drop_score)  # do not use softmax since we only have one element
            mm = torch.nn.Threshold(0.5, 0)
            drop_score = mm(drop_score)
            mm_2 = torch.nn.Threshold(-0.49, 1)
            drop_score = mm_2(-drop_score)
            drop_decision = drop_score.clone().requires_grad_()
            drop_matrix = lil_matrix((n_node, n_node), dtype=np.float32)
            drop_matrix[row, col] = drop_decision.cpu().data.numpy().squeeze(-1)
            att_dense_norm = att_dense_norm.multiply(drop_matrix.tocsr())  # update, remove the 0 edges

        if att_dense_norm[0, 0] == 0:  # add the weights of self-loop only add self-loop at the first layer
            degree = (att_dense_norm != 0).sum(1).A1
            lam = 1 / (degree + 1)  # degree +1 is to add itself
            self_weight = sp.diags(np.array(lam), offsets=0, format="lil")
            att = att_dense_norm + self_weight  # add the self loop
        else:
            att = att_dense_norm

        row, col = att.nonzero()
        att_adj = np.vstack((row, col))
        att_edge_weight = att[row, col]
        att_edge_weight = np.exp(att_edge_weight)  # exponent, kind of softmax
        att_edge_weight = torch.tensor(np.array(att_edge_weight)[0], dtype=torch.float32).to(features.device)
        att_adj = torch.tensor(att_adj, dtype=torch.int64).to(features.device)

        shape = (n_node, n_node)
        new_adj = torch.sparse.FloatTensor(att_adj, att_edge_weight, shape)

        return new_adj


class GATGuard(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 hidden_features,
                 n_layers,
                 n_heads,
                 activation=F.leaky_relu,
                 layer_norm=False,
                 feat_norm=None,
                 adj_norm_func=GCNAdjNorm,
                 drop=False,
                 attention=True,
                 dropout=0.0):

        super(GATGuard, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.feat_norm = feat_norm
        self.adj_norm_func = adj_norm_func
        if type(hidden_features) is int:
            hidden_features = [hidden_features] * (n_layers - 1)
        elif type(hidden_features) is list or type(hidden_features) is tuple:
            assert len(hidden_features) == (n_layers - 1), "Incompatible sizes between hidden_features and n_layers."
        n_features = [in_features] + hidden_features + [out_features]
  

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            if layer_norm:
                if i == 0:
                    self.layers.append(nn.LayerNorm(n_features[i]))
                else:
                    self.layers.append(nn.LayerNorm(n_features[i] * n_heads))
            self.layers.append(GATConv(in_feats=n_features[i] * n_heads if i != 0 else n_features[i],
                                       out_feats=n_features[i + 1],
                                       num_heads=n_heads,
                                       activation=activation if i != n_layers - 1 else None))
        self.drop = drop
        self.drop_learn = torch.nn.Linear(2, 1)
        self.attention = attention
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.head = nn.Identity()

    @property
    def model_type(self):
        return "dgl"

    def forward(self, g, x, weight=None, return_hidden=None):
        adj = to_scipy(g.adj().to_dense())
        # graph = dgl.from_scipy(adj).to(x.device)
        # graph.ndata['features'] = x
        h = x
        hidden_list= []
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.LayerNorm):
                h = layer(h)
            else:
                if self.attention:
                    adj = self.att_coef(h, adj)
                    # graph = dgl.from_scipy(adj).to(x.device)
                    # graph.ndata['features'] = x
                h = layer(g, h).flatten(1)
                if i != len(self.layers) - 1:
                    if self.dropout is not None:
                        h = self.dropout(h)
            hidden_list.append(h)

        if return_hidden:
            return self.head(h), hidden_list
        else:
            return self.head(h)


    def reset_classifier(self, num_classes):
        self.head = nn.Linear(self.num_heads * self.out_dim, num_classes)


    def att_coef(self, features, adj):
        adj = adj.tocoo()
        n_node = features.shape[0]
        row, col = adj.row, adj.col

        features_copy = features.cpu().data.numpy()
        sim_matrix = cosine_similarity(X=features_copy, Y=features_copy)  # try cosine similarity
        sim = sim_matrix[row, col]
        sim[sim < 0.1] = 0

        """build a attention matrix"""
        att_dense = lil_matrix((n_node, n_node), dtype=np.float32)
        att_dense[row, col] = sim
        if att_dense[0, 0] == 1:
            att_dense = att_dense - sp.diags(att_dense.diagonal(), offsets=0, format="lil")
        # normalization, make the sum of each row is 1
        att_dense_norm = normalize(att_dense, axis=1, norm='l1')

        """add learnable dropout, make character vector"""
        if self.drop:
            character = np.vstack((att_dense_norm[row, col].A1,
                                   att_dense_norm[col, row].A1))
            character = torch.from_numpy(character.T).to(features.device)
            drop_score = self.drop_learn(character)
            drop_score = torch.sigmoid(drop_score)  # do not use softmax since we only have one element
            mm = torch.nn.Threshold(0.5, 0)
            drop_score = mm(drop_score)
            mm_2 = torch.nn.Threshold(-0.49, 1)
            drop_score = mm_2(-drop_score)
            drop_decision = drop_score.clone().requires_grad_()
            drop_matrix = lil_matrix((n_node, n_node), dtype=np.float32)
            drop_matrix[row, col] = drop_decision.cpu().data.numpy().squeeze(-1)
            att_dense_norm = att_dense_norm.multiply(drop_matrix.tocsr())  # update, remove the 0 edges

        if att_dense_norm[0, 0] == 0:  # add the weights of self-loop only add self-loop at the first layer
            degree = (att_dense_norm != 0).sum(1).A1
            lam = 1 / (degree + 1)  # degree +1 is to add itself
            self_weight = sp.diags(np.array(lam), offsets=0, format="lil")
            att = att_dense_norm + self_weight  # add the self loop
        else:
            att = att_dense_norm

        row, col = att.nonzero()
        att_edge_weight = att[row, col]
        att_edge_weight = np.exp(att_edge_weight)
        att_edge_weight = np.asarray(att_edge_weight.ravel())[0]
        new_adj = sp.csr_matrix((att_edge_weight, (row, col)))

        return new_adj
    



class GCNConv(nn.Module):
    r"""

    Description
    -----------
    GCN convolutional layer.

    Parameters
    ----------
    in_features : int
        Dimension of input features.
    out_features : int
        Dimension of output features.
    activation : func of torch.nn.functional, optional
        Activation function. Default: ``None``.
    residual : bool, optional
        Whether to use residual connection. Default: ``False``.
    dropout : float, optional
        Dropout rate during training. Default: ``0.0``.

    """

    def __init__(self,
                 in_features,
                 out_features,
                 activation=None,
                 residual=False,
                 dropout=0.0):
        super(GCNConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)

        if residual:
            self.residual = nn.Linear(in_features, out_features)
        else:
            self.residual = None
        self.activation = activation

        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.reset_parameters()

    def reset_parameters(self):
        """Reset parameters."""
        if self.activation == F.leaky_relu:
            gain = nn.init.calculate_gain('leaky_relu')
        else:
            gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.linear.weight, gain=gain)

    def forward(self, x, adj):
        r"""

        Parameters
        ----------
        x : torch.Tensor
            Tensor of input features.
        adj : torch.SparseTensor
            Sparse tensor of adjacency matrix.

        Returns
        -------
        x : torch.Tensor
            Output of layer.

        """

        x = self.linear(x)
        x = torch.sparse.mm(adj, x)
        if self.activation is not None:
            x = self.activation(x)
        if self.residual is not None:
            x = x + self.residual(x)
        if self.dropout is not None:
            x = self.dropout(x)

        return x
    


