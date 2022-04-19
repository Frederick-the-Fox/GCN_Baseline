import numpy as np
import scipy.sparse as sp
import torch


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(dataset, metapath):
    path="../data/" + dataset + '/'
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = np.genfromtxt("{}{}.label".format(path, dataset), dtype=np.dtype(str))
    labels = encode_onehot(labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, metapath),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    # idx_train = range(140)
    # idx_val = range(200, 500)
    # idx_test = range(500, 1500)

    idx_test_20 = np.load('/home/hangni/HeCo-main/data/my_data/' + dataset + '/test_20.npy').tolist()
    idx_train_20 = np.load('/home/hangni/HeCo-main/data/my_data/' + dataset + '/train_20.npy').tolist()
    idx_val_20 = np.load('/home/hangni/HeCo-main/data/my_data/' + dataset + '/val_20.npy').tolist()
    idx_test_40 = np.load('/home/hangni/HeCo-main/data/my_data/' + dataset + '/test_40.npy').tolist()
    idx_train_40 = np.load('/home/hangni/HeCo-main/data/my_data/' + dataset + '/train_40.npy').tolist()
    idx_val_40 = np.load('/home/hangni/HeCo-main/data/my_data/' + dataset + '/val_40.npy').tolist()
    idx_test_60 = np.load('/home/hangni/HeCo-main/data/my_data/' + dataset + '/test_60.npy').tolist()
    idx_train_60 = np.load('/home/hangni/HeCo-main/data/my_data/' + dataset + '/train_60.npy').tolist()
    idx_val_60 = np.load('/home/hangni/HeCo-main/data/my_data/' + dataset + '/val_60.npy').tolist()

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train_20 = torch.LongTensor(idx_train_20)
    idx_val_20 = torch.LongTensor(idx_val_20)
    idx_test_20 = torch.LongTensor(idx_test_20)
    idx_train_40 = torch.LongTensor(idx_train_40)
    idx_val_40 = torch.LongTensor(idx_val_40)
    idx_test_40 = torch.LongTensor(idx_test_40)
    idx_train_60 = torch.LongTensor(idx_train_60)
    idx_val_60 = torch.LongTensor(idx_val_60)
    idx_test_60 = torch.LongTensor(idx_test_60)
    
    return adj, features, labels, idx_train_20, idx_val_20, idx_test_20, idx_train_40, idx_val_40, idx_test_40, idx_train_60, idx_val_60, idx_test_60


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
