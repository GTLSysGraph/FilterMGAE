import numpy as np
import random
import scipy.sparse as sp
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score as acc
import torch
from datasets_dgl.utils import sparse_mx_to_torch_sparse_tensor
# 因为版本问题报了上述提示性错误 忽略一下
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

def compute_graph_homophily_score(g, label):
    edges = g.edges()
    return sum(label[edges[0]] == label[edges[1]]) / g.num_edges()


def high_dim_gaussian(mu, sigma):
    if mu.ndim > 1:
        d = len(mu)
        res = np.zeros(d)
        for i in range(d):
            res[i] = np.random.normal(mu[i], sigma[i])
    else:
        d = 1
        res = np.zeros(d)
        res = np.random.normal(mu, sigma)
    return res


def generate_uniform_theta(Y, c):
    theta = np.zeros(len(Y), dtype='float')
    for i in range(c):
        idx = np.where(Y == i)
        sample = np.random.uniform(low=0, high=1, size=len(idx[0]))
        sample_sum = np.sum(sample)
        for j in range(len(idx[0])):
            theta[idx[0][j]] = sample[j] * len(idx[0]) / sample_sum
    return theta


def generate_theta_dirichlet(Y, c):
    theta = np.zeros(len(Y), dtype='float')
    for i in range(c):
        idx = np.where(Y == i)
        temp = np.random.uniform(low=0, high=1, size=len(idx[0]))
        sample = np.random.dirichlet(temp, 1)
        sample_sum = np.sum(sample)
        for j in range(len(idx[0])):
            theta[idx[0][j]] = sample[0][j] * len(idx[0]) / sample_sum
    return theta
    
def SBM(sizes, probs, mus, sigmas, noise,
        radius, feats_type='gaussian', selfloops=True):
    # -----------------------------------------------
    #     step1: get c,d,n
    # -----------------------------------------------
    c = len(sizes)
    if mus.ndim > 1:
        d = mus.shape[1]
    else:
        d = 1
    n = sizes.sum() # 100个第1类的点和100个第二类的点，一共200个
    all_node_ids = [ids for ids in range(0, n)]
    # -----------------------------------------------
    #     step2: generate Y with sizes
    # -----------------------------------------------
    Y = np.zeros(n, dtype='int')
    for i in range(c):
        class_i_ids = random.sample(all_node_ids, sizes[i])
        Y[class_i_ids] = i
        for item in class_i_ids:
            all_node_ids.remove(item)
    # -----------------------------------------------
    #     step3: generate A with Y and probs
    # -----------------------------------------------
    if selfloops:
        A = np.diag(np.ones(n, dtype='int'))
    else:
        A = np.zeros((n, n), dtype='int')
    for i in range(n):
        for j in range(i + 1, n):
            prob_ = probs[Y[i]][Y[j]] # 如果Y[i]]和[Y[j]都为0或都为1，对应于概率p，不同类型对应概率q
            rand_ = random.random()
            if rand_ <= prob_:
                A[i][j] = 1
                A[j][i] = 1
    # -----------------------------------------------
    #     step4: generate X with Y and mus, sigmas
    # -----------------------------------------------
    X = np.zeros((n, d), dtype='float')
    for i in range(n):
        mu = mus[Y[i]]
        sigma = sigmas[Y[i]]
        X[i] = high_dim_gaussian(mu, sigma)

    return A, X, Y


def generate(p, q, idx):
    A, X, Y = \
        SBM(sizes=np.array([100, 100]),
        probs=np.array([[p, q], [q, p]]),
        mus=np.array([[-0.5]*20, [0.5]*20]), # 多维高斯向量 mus
        sigmas=np.array([[1]*20, [1]*20]),   # 多维高斯向量 sigmas
        noise=[],
        radius=[],
        selfloops=False)
        
    return A, X, Y



def save_generate(p ,q):
    A, X, Y = \
        SBM(sizes=np.array([100, 100]),
        probs=np.array([[p, q], [q, p]]),
        mus=np.array([[-0.5]*20, [0.5]*20]), # 多维高斯向量 mus
        sigmas=np.array([[1]*20, [1]*20]),   # 多维高斯向量 sigmas
        noise=[],
        radius=[],
        selfloops=False)
    
    src = A.nonzero()[0]
    dst = A.nonzero()[1]
    edges = np.stack((src,dst)).T
    np.savetxt('./datasets_dgl/all_data_synthetic/syn-10.edge', edges, fmt='%.0f')
    np.savetxt('./datasets_dgl/all_data_synthetic/syn-10.feat', X,     fmt='%.6f')
    np.savetxt('./datasets_dgl/all_data_synthetic/syn-10.lab',  Y,     fmt='%.0f')

    

        
def calculate(A, X):
    A = sp.coo_matrix(A)
    A = A + A.T.multiply(A.T > A) - A.multiply(A.T > A) # 对称操作
    rowsum = np.array(A.sum(1)).clip(min=1)
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    A = A.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

    low = 0.5 * sp.eye(A.shape[0]) + A
    high = 0.5 * sp.eye(A.shape[0]) - A
    # 因为graph在cpu上，只把feat放在cuda上为了方便计算，算好后low和high重新存到cpu上
    sp_tensor_low  = sparse_mx_to_torch_sparse_tensor(low).cuda()
    sp_tensor_high = sparse_mx_to_torch_sparse_tensor(high).cuda()
    low_signal  = torch.sparse.mm(torch.sparse.mm(sp_tensor_low, sp_tensor_low), X)
    high_signal = torch.sparse.mm(torch.sparse.mm(sp_tensor_high, sp_tensor_high), X)
    return low_signal.cpu(), high_signal.cpu()

    # trash code! low
    # low = low.todense()
    # high = high.todense()
    # low_signal = np.dot(np.dot(low, low), X)
    # high_signal = np.dot(np.dot(high, high), X)
    # return low_signal, high_signal
    



if __name__ == '__main__':
    save_generate(0.05, 0.15)