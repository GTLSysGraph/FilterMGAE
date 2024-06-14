import scipy.sparse as sp
import numpy as np
from numba import njit

def drop_dissimilar_edges(features, adj, threshold, binary_feature=True, metric='similarity'):
    """Drop dissimilar edges.(Faster version using numba)
    """
    if not sp.issparse(adj):
        adj = sp.csr_matrix(adj)

    adj_triu = sp.triu(adj, format='csr')

    if sp.issparse(features):
        features = features.todense().A # make it easier for njit processing

    if metric == 'distance':
        removed_cnt = dropedge_dis(adj_triu.data, adj_triu.indptr, adj_triu.indices, features, threshold)
    else:
        if binary_feature:
            removed_cnt = dropedge_jaccard(adj_triu.data, adj_triu.indptr, adj_triu.indices, features, threshold)
        else:
            removed_cnt = dropedge_cosine(adj_triu.data, adj_triu.indptr, adj_triu.indices, features, threshold)
    print('removed %s edges in the original graph' % removed_cnt)
    modified_adj = adj_triu + adj_triu.transpose()
    return modified_adj


@njit
def dropedge_jaccard(A, iA, jA, features, threshold):
    removed_cnt = 0
    for row in range(len(iA)-1):
        for i in range(iA[row], iA[row+1]):
            # print(row, jA[i], A[i])
            n1 = row
            n2 = jA[i]
            a, b = features[n1], features[n2]
            intersection = np.count_nonzero(a*b)
            J = intersection * 1.0 / (np.count_nonzero(a) + np.count_nonzero(b) - intersection)

            if J < threshold:
                A[i] = 0
                # A[n2, n1] = 0
                removed_cnt += 1
    return removed_cnt


@njit
def dropedge_cosine(A, iA, jA, features, threshold):
    removed_cnt = 0
    for row in range(len(iA)-1):
        for i in range(iA[row], iA[row+1]):
            # print(row, jA[i], A[i])
            n1 = row
            n2 = jA[i]
            a, b = features[n1], features[n2]
            inner_product = (a * b).sum()
            C = inner_product / (np.sqrt(np.square(a).sum()) * np.sqrt(np.square(b).sum()) + 1e-8)

            if C < threshold:
                A[i] = 0
                # A[n2, n1] = 0
                removed_cnt += 1
    return removed_cnt

@njit
def dropedge_dis(A, iA, jA, features, threshold):
    removed_cnt = 0
    for row in range(len(iA)-1):
        for i in range(iA[row], iA[row+1]):
            # print(row, jA[i], A[i])
            n1 = row
            n2 = jA[i]
            C = np.linalg.norm(features[n1] - features[n2])
            if C > threshold:
                A[i] = 0
                # A[n2, n1] = 0
                removed_cnt += 1

    return removed_cnt



def truncatedSVD(data, k=50):
    """Truncated SVD on input data.

    Parameters
    ----------
    data :
        input matrix to be decomposed
    k : int
        number of singular values and vectors to compute.

    Returns
    -------
    numpy.array
        reconstructed matrix.
    """
    print('=== GCN-SVD: rank={} ==='.format(k))
    if sp.issparse(data):
        data = data.asfptype()
        U, S, V = sp.linalg.svds(data, k=k)
        print("rank_after = {}".format(len(S.nonzero()[0])))
        diag_S = np.diag(S)
    else:
        U, S, V = np.linalg.svd(data)
        U = U[:, :k]
        S = S[:k]
        V = V[:k, :]
        print("rank_before = {}".format(len(S.nonzero()[0])))
        diag_S = np.diag(S)
        print("rank_after = {}".format(len(diag_S.nonzero()[0])))

    return U @ diag_S @ V
