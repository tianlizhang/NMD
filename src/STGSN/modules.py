import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import GraphNeuralNetwork
from .layers import Attention
from utils import taskers_utils
import scipy.sparse as sp
import numpy as np


def get_gnn_sup_d(adj):
    '''
    Function to get GNN support (normalized adjacency matrix w/ self-connected edges)
    :param adj: original adjacency matrix
    :return: GNN support
    '''
    # ====================
    num_nodes = len(adj)
    adj = adj + taskers_utils.make_sparse_eye(num_nodes).to(adj.device)
    degs = torch.sparse.sum(adj, dim=1).to_dense()
    
    idx = adj._indices()
    vals = adj._values()
    vals = vals / degs[idx[0]]
    return torch.sparse.FloatTensor(idx, vals, adj.shape).float().to(adj.device)


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


class STGSN(nn.Module):
    '''
    Class to define DDNE
    '''
    def __init__(self, args, dropout_rate):
        super(STGSN, self).__init__()
        # ====================
        # self.enc_dims = end_dims # Layer configuration of encoder
        self.enc_dims = [args.feats_per_node]
        self.enc_dims.extend(args.enc_dims)
        # self.enc_dims = [args.feats_per_node,
        #          args.layer_1_feats,
        #          args.layer_2_feats, args.layer_2_feats//2]
        self.dropout_rate = dropout_rate # Dropout rate
        # ==========
        # Encoder
        self.enc = STGSN_Enc(self.enc_dims, self.dropout_rate)
        # self.dec = STGSN_Dec(self.enc_dims[-1], self.dropout_rate)
        self.theta = 0.1
        

    def forward(self, A_list, Nodes_list, mask=None):
        '''
        Rewrite the forward function
        :param sup_list: list of GNN supports (i.e., normalized adjacency matrices)
        :param feat_list: list of GNN feature inputs (i.e., node attributes)
        :param gbl_sup: global GNN support
        :param gbl_feat: global GNN feature input
        :param num_nodes: number of associated nodes
        :return: prediction result
        '''
        # ====================
        num_nodes = len(A_list[0])
        sup_list = [] # List of GNN support (tensor)
        col_net = torch.zeros((num_nodes, num_nodes)).to(A_list[0].device)
        coef_sum, tau = 0.0, len(A_list)
        
        for t in range(tau):
            sup = get_gnn_sup_d(A_list[t])
            sup_list.append(sup)
            # ==========
            coef = (1-self.theta)**(tau-t)
            col_net += coef*A_list[t]
            coef_sum += coef
        # ==========
        col_net /= coef_sum
        
        col_net_sp = sp.coo_matrix(col_net.cpu())
        col_sup_sp = sparse_to_tuple(col_net_sp)
        idxs = torch.LongTensor(col_sup_sp[0].astype(float)).to(A_list[0].device)
        vals = torch.FloatTensor(col_sup_sp[1]).to(A_list[0].device)
        col_net_ts = torch.sparse.FloatTensor(idxs.t(), vals, col_sup_sp[2]).float().to(A_list[0].device)
        
        col_sup_tnr = get_gnn_sup_d(col_net_ts)
        feat_list = [item.to_dense() for item in Nodes_list]
        dyn_emb = self.enc(sup_list, feat_list, col_sup_tnr, feat_list[-1], num_nodes)

        return dyn_emb


class STGSN_Enc(nn.Module):
    '''
    Class to define the encoder of STGSN
    '''
    def __init__(self, enc_dims, dropout_rate):
        super(STGSN_Enc, self).__init__()
        # ====================
        self.enc_dims = enc_dims # Layer configuration of encoder
        self.dropout_rate = dropout_rate # Dropout rate
        # ==========
        # Structural encoder
        self.num_struc_layers = len(self.enc_dims)-1  # Number of GNN layers
        self.struc_enc = nn.ModuleList()
        for l in range(self.num_struc_layers):
            self.struc_enc.append(
                GraphNeuralNetwork(self.enc_dims[l], self.enc_dims[l+1], dropout_rate=self.dropout_rate))
        # ===========
        # Temporal encoder
        self.att = Attention(self.enc_dims[-1])

    def forward(self, sup_list, feat_list, gbl_sup, gbl_feat, num_nodes):
        '''
        Rewrite the forward function
        :param sup_list: list of GNN supports (i.e., normalized adjacency matrices)
        :param feat_list: list of GNN feature inputs (i.e., node attributes)
        :param gbl_sup: global GNN support
        :param gbl_feat: global GNN feature input
        :param num_nodes: number of associated nodes
        :return: dynamic node embedding
        '''
        # ====================
        win_size = len(sup_list) # Window size, i.e., #historical snapshots
        # ====================
        # Structural encoder
        ind_input_list = feat_list # List of attribute inputs w.r.t. historical snapshots
        gbl_input = gbl_feat
        ind_output_list = None # List of embedding outputs w.r.t. historical snapshots
        gbl_output = None
        for l in range(self.num_struc_layers):
            gbl_output = self.struc_enc[l](gbl_input, gbl_sup)
            gbl_input = gbl_output
            # ==========
            ind_output_list = []
            for i in range(win_size):
                ind_input = ind_input_list[i]
                ind_sup = sup_list[i]
                ind_output = self.struc_enc[l](ind_input, ind_sup)
                ind_output_list.append(ind_output)
            ind_input_list = ind_output_list
        gbl_emb = gbl_output
        ind_emb_list = ind_output_list
        # ==========
        # Temporal encoder
        agg_emb = self.att(ind_emb_list, gbl_emb, num_nodes)
        dyn_emb = torch.cat((agg_emb, gbl_emb), dim=1) # Dynamic node embedding

        return dyn_emb


class STGSN_Dec(nn.Module):
    '''
    Class to define the decoder of STGSN
    '''
    def __init__(self, emb_dim, dropout_rate):
        super(STGSN_Dec, self).__init__()
        # ====================
        self.emb_dim = emb_dim # Dimensionality of dynamic embedding
        self.dropout_rate = dropout_rate # Dropout rate
        # ==========
        self.dec = nn.Linear(4*self.emb_dim, 1)

    def forward(self, dyn_emb, num_nodes):
        '''
        Rewrite the forward function
        :param dyn_emb: dynamic embedding given by encoder
        :param num_nodes: number of associated nodes
        :return: prediction result
        '''
        # ====================
        adj_est = None
        for i in range(num_nodes):
            cur_emb = dyn_emb[i, :]
            cur_emb = torch.reshape(cur_emb, (1, self.emb_dim*2)) # (1, emb_dim*2)
            cur_emb = cur_emb.repeat(num_nodes, 1) # (num_nodes, emb_dim*2)
            cat_emb = torch.cat((cur_emb, dyn_emb), dim=1) # (num_nodes, emb_dim*4)
            col_est = torch.sigmoid(self.dec(cat_emb))
            if i == 0:
                adj_est = col_est
            else:
                adj_est = torch.cat((adj_est, col_est), dim=1)

        return adj_est
