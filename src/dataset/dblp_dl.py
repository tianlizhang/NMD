import dgl
import torch
from utils import utils
import scipy.sparse as sp
import os
import numpy as np
from IPython import embed


def sparse_to_tuple(sparse_mx):
    '''
    Function to transfer sparse matrix to tuple format
    :param sparse_mx: original sparse matrix
    :return: corresponding tuple format
    '''
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx): # sp.sparse.isspmatrix_coo(mx)
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape
    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


class DBLP_Dataset():
    def __init__(self, args):
        # dataset = 'dblp'
        # k = 32
        # core_dg = dgl.load_graphs(f'../preprocess/data/{dataset}/{k}-core_graph.bin')[0][0]
        args.dblp_args = utils.Namespace(args.dblp_args)
        
        gpath = os.path.join(args.dblp_args.folder, args.dblp_args.graph_file)
        core_dg = dgl.load_graphs(gpath)[0][0]
        print(f'core_dg:', core_dg)
        
        self.num_nodes = core_dg.number_of_nodes()
        data = torch.vstack([core_dg.edges()[0], core_dg.edges()[1], core_dg.edata['ts']]).t()
        
        cols = utils.Namespace({'source': 0, 'target': 1, 'time': 2})
        data[:,cols.time] = utils.aggregate_by_time(data[:,cols.time], args.dblp_args.aggr_time)
        self.max_time = data[:,cols.time].max()
        self.min_time = data[:,cols.time].min()
        
        ids = data[:,cols.source] * self.num_nodes + data[:,cols.target]
        self.num_non_existing = float(self.num_nodes**2 - ids.unique().size(0))

        idx = data[:,[cols.source, cols.target, cols.time]]
        
        self.nodes_feats = core_dg.ndata['feat']
        self.feats_per_node = self.nodes_feats.shape[1]
        self.edges = {'idx': idx, 'vals': torch.ones(idx.size(0))}
    
    
    def prepare_node_feats(self, nfeat):
        nfeat_sp = sp.coo_matrix(nfeat[0]) # ignore batch dim
        
        coords, values, shape = sparse_to_tuple(nfeat_sp)
        idxs = torch.LongTensor(coords.astype(float))
        vals = torch.FloatTensor(values)
        nfeat_ts = torch.sparse.FloatTensor(idxs.t(), vals, shape).float()
        return nfeat_ts
        
        