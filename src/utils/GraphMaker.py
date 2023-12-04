import numpy as np
import scipy.sparse as sp
import torch
import codecs
import os
import pickle

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


class GraphMaker(object):
    def __init__(self, opt, datasetname):
        self.opt = opt
        self.user = set()
        self.item = set()
        user_map = {}
        item_map = {}
        data=[]
        filename = "../dataset/" + datasetname + "/train.txt"
        self.datasetname = datasetname
        with codecs.open(filename) as infile:
            for line in infile:
                line = line.strip().split("\t")
                line[0] = int(line[0])
                line[1] = int(line[1])
                if line[0] not in user_map.keys():
                    user_map[line[0]] = len(user_map)
                if line[1] not in item_map.keys():
                    item_map[line[1]] = len(item_map)
                line[0] = user_map[line[0]]
                line[1] = item_map[line[1]]
                data.append((int(line[0]),int(line[1]),float(line[2])))
                self.user.add(int(line[0]))
                self.item.add(int(line[1]))

        opt["number_user"] = len(self.user)
        opt["number_item"] = len(self.item)

        print("number_user", len(self.user))
        print("number_item", len(self.item))
        
        self.raw_data = data
        if not os.path.exists(os.path.join(os.getcwd(), '../dataset', datasetname, f'all_adj.pkl')):
            self.UV,self.VU, self.adj = self.preprocess(data, opt)
        else:
            print("real graph loaded!")
            self.adj = pickle.load(open(os.path.join('../dataset', datasetname, f'all_adj.pkl'), 'rb'))
            self.UV = pickle.load(open(os.path.join('../dataset', datasetname, f'UV_adj.pkl'), 'rb'))
            self.VU = pickle.load(open(os.path.join('../dataset', datasetname, f'VU_adj.pkl'), 'rb'))

    def preprocess(self,data,opt):
        UV_edges = []
        VU_edges = []
        all_edges = []
        real_adj = {}

        user_real_dict = {}
        item_real_dict = {}
        for edge in data:
            UV_edges.append([edge[0],edge[1]])
            if edge[0] not in user_real_dict.keys():
                user_real_dict[edge[0]] = set()
            user_real_dict[edge[0]].add(edge[1])

            VU_edges.append([edge[1], edge[0]])
            if edge[1] not in item_real_dict.keys():
                item_real_dict[edge[1]] = set()
            item_real_dict[edge[1]].add(edge[0])

            all_edges.append([edge[0],edge[1] + opt["number_user"]])
            all_edges.append([edge[1] + opt["number_user"], edge[0]])
            if edge[0] not in real_adj :
                real_adj[edge[0]] = {}
            real_adj[edge[0]][edge[1]] = 1

        UV_edges = np.array(UV_edges)
        VU_edges = np.array(VU_edges)
        all_edges = np.array(all_edges)
        UV_adj = sp.coo_matrix((np.ones(UV_edges.shape[0]), (UV_edges[:, 0], UV_edges[:, 1])),
                               shape=(opt["number_user"], opt["number_item"]),
                               dtype=np.float32)
        VU_adj = sp.coo_matrix((np.ones(VU_edges.shape[0]), (VU_edges[:, 0], VU_edges[:, 1])),
                               shape=(opt["number_item"], opt["number_user"]),
                               dtype=np.float32)
        all_adj = sp.coo_matrix((np.ones(all_edges.shape[0]), (all_edges[:, 0], all_edges[:, 1])),shape=(opt["number_item"]+opt["number_user"], opt["number_item"]+opt["number_user"]),dtype=np.float32)
        # UV_adj = normalize(UV_adj)
        # VU_adj = normalize(VU_adj)
        all_adj = normalize(all_adj)
        # UV_adj = sparse_mx_to_torch_sparse_tensor(UV_adj)
        # VU_adj = sparse_mx_to_torch_sparse_tensor(VU_adj)
        all_adj = sparse_mx_to_torch_sparse_tensor(all_adj)
        pickle.dump(all_adj, open(os.path.join('../dataset', self.datasetname, f'all_adj.pkl'), 'wb'))
        pickle.dump(UV_adj, open(os.path.join('../dataset', self.datasetname, f'UV_adj.pkl'), 'wb'))
        pickle.dump(VU_adj, open(os.path.join('../dataset', self.datasetname, f'VU_adj.pkl'), 'wb'))
        print("real graph saved!")
        return UV_adj, VU_adj, all_adj

def create_cross_Graph(source_UV, source_VU, target_UV, target_VU, datasetname):
    if not os.path.exists(os.path.join(os.getcwd(), '../dataset', datasetname, 'cross_adj.pkl')):
        n_users = source_UV.shape[0]
        n_items1 = source_UV.shape[1]
        n_items2 = target_UV.shape[1]
        cross_adj = sp.dok_matrix((n_users + n_items1 + n_items2, n_users + n_items1 + n_items2), dtype=np.float32)

        cross_adj = cross_adj.tolil()
        cross_adj[:n_users, n_users:n_items1+n_users] = source_UV.tolil()
        cross_adj[:n_users, n_users+n_items1:] = target_UV.tolil()
        cross_adj[n_users:n_items1+n_users, :n_users] = source_VU.tolil()
        cross_adj[n_users+n_items1:, :n_users] = target_VU.tolil()

        cross_adj = cross_adj.todok()
        cross_adj = normalize(cross_adj)
        cross_adj = sparse_mx_to_torch_sparse_tensor(cross_adj)
        pickle.dump(cross_adj, open(os.path.join('../dataset', datasetname, 'cross_adj.pkl'), 'wb'))
        print("cross graph saved!")    
    else:
        cross_adj = pickle.load(open(os.path.join('../dataset', datasetname, 'cross_adj.pkl'), 'rb'))
        print("real cross graph loaded!")
    return cross_adj
