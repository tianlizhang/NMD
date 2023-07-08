import torch
import torch.nn as nn
from .layers import mat_GRU_cell
import math
from utils import utils
from torch.nn.parameter import Parameter


class EGCN(nn.Module):
    '''
    Class to define dyngraph2vec (AERNN variant)
    '''
    def __init__(self, args, activation, device='cpu', skipfeats=False):
        super(EGCN, self).__init__()
        
        self.device = device
        self.skipfeats = skipfeats
        self.enc = EGCN_Enc(args, activation)
        
        self._parameters = nn.ParameterList()
        self._parameters.extend(list(self.enc.parameters()))
        

    def parameters(self):
        return self._parameters


    def forward(self, A_list, Nodes_list, nodes_mask_list):
        node_feats= Nodes_list[-1]

        Nodes_list = self.enc(A_list, Nodes_list, nodes_mask_list)
        out = Nodes_list[-1]
        
        if self.skipfeats:
            out = torch.cat((out, node_feats), dim=1)   # use node_feats.to_dense() if 2hot encoded input 
        return out



class EGCN_Enc(nn.Module):
    def __init__(self, args, activation):
        super(EGCN_Enc, self).__init__()
        feats = [args.feats_per_node,
                 args.layer_1_feats,
                 args.layer_2_feats]
        self.evolve_weights = nn.ModuleList()
        cell_args = utils.Namespace({})
        cell_args.rows = feats[i-1]
        cell_args.cols = feats[i]
        
        self.activation = activation
        self.GCN_init_weights = Parameter(torch.Tensor(cell_args.rows, cell_args.cols))
        self.reset_param(self.GCN_init_weights)
        
        for i in range (1, len(feats)):
            self.evolve_weights.append(mat_GRU_cell(cell_args))
            self.reset_param(self.GCN_init_weights)
    
        
    def reset_param(self,t):
        #Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv,stdv)
    
    
    def forward(self, A_list, node_embs_list, mask_list):
        for l, unit in enumerate(self.evolve_weights):
            GCN_weights = self.GCN_init_weights
            out_seq = []

            for t, Ahat in enumerate(A_list):
                node_embs = node_embs_list[t]
                
                GCN_weights = unit(GCN_weights, node_embs, mask_list[t])
                node_embs = self.activation(Ahat.matmul(node_embs.matmul(GCN_weights)))
                
                out_seq.append(node_embs)
            node_embs_list = out_seq
        return out_seq
            
        