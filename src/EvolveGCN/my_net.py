# import utils as u
from utils import utils
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import math
import torch.nn.functional as F
# from torch_geometric.nn import SAGEConv
from collections import defaultdict
import pandas as pd
import numpy as np
from loader import cites_loader
from .models import Attention
from IPython import embed

# def label_normorlization(labels):
#     maximum = labels.max()
#     minimum = labels.min()
#     new_value = (labels-minimum)/(maximum-minimum)
#     return new_value, (maximum,minimum)


# def build_cites(adj_list, pred_ts=1):
#     ts_nid2cites = {}
#     for ts in range(len(adj_list)):
#         tgt_lst = adj_list[ts]._indices()[1, :].tolist()
        
#         ts_nid2cites[ts] = defaultdict(int)
#         for tgt in tgt_lst:
#             ts_nid2cites[ts][tgt] += 1

#     cites_infos = []
#     for ts in range(len(adj_list)-1):
#         nids = adj_list[ts]._indices()[0, :].unique().tolist()
        
#         pdf = pd.DataFrame({'nid': nids})
#         for yy in range(ts+1, len(adj_list)):
#             cdf = pd.DataFrame({'nid': list(ts_nid2cites[yy].keys()), str(yy): list(ts_nid2cites[yy].values())})
#             cdf[str(yy)] = cdf[str(yy)].astype('float32')
            
#             pdf = pd.merge(pdf, cdf, how='left', on='nid')
#         pdf.fillna(0, inplace=True)
        
#         cites = pdf.iloc[:, 1:1+pred_ts].values
#         cites_cum_log = np.log(cites.cumsum(axis=1) + 1)
#         cites_norm, _ = label_normorlization(cites_cum_log)
        
#         sids_in_ngh = pdf['nid'].tolist()
#         key_nodes = torch.tensor(sids_in_ngh).unique()  # 按照label的顺序排列成
#         sid2pos = -torch.ones(max(key_nodes) + 1, dtype=torch.long) # [108073]
#         sid2pos[key_nodes] = torch.arange(len(key_nodes))

#         cites_infos.append({'cites_norm': torch.from_numpy(cites_norm), 'sid2pos': sid2pos})
#     return cites_infos


class EGCN(torch.nn.Module):
    def __init__(self, args, fusion_mode, activation, device='cpu', skipfeats=False):
        super().__init__()
        GRCU_args = utils.Namespace({})

        feats = [args.feats_per_node,
                 args.layer_1_feats,
                 args.layer_2_feats]
        self.device = device
        self.skipfeats = skipfeats
        self.GRCU_layers, self.cites_layers = [], []
        self._parameters = nn.ParameterList()
        for i in range(1,len(feats)):
            GRCU_args = utils.Namespace({'in_feats' : feats[i-1],
                                        'out_feats': feats[i],
                                        'activation': activation})

            grcu_i = GRCU(GRCU_args)
            #print (i,'grcu_i', grcu_i)
            self.GRCU_layers.append(grcu_i.to(self.device))
            self._parameters.extend(list(self.GRCU_layers[-1].parameters()))

            self.cites_layers.append(GRCU_cites(GRCU_args).to(self.device))
            self._parameters.extend(list(self.cites_layers[-1].parameters()))

        self.pred_year = 1
        hidden_channels = args.layer_1_feats
        self.node_predictor = cites_loader.FcPredictor(hidden_channels, node_year = self.pred_year).to(self.device)

        self.fusion_mode = fusion_mode
        print(f'fusion_mode: {self.fusion_mode}')
        if self.fusion_mode == 'cat':
            self.mlp = nn.Linear(2*hidden_channels, hidden_channels).to(self.device)
        elif self.fusion_mode == 'add':
            self.mlp = nn.Linear(hidden_channels, hidden_channels).to(self.device)
        elif self.fusion_mode == 'mlp_add':
            self.mlp = nn.Linear(hidden_channels, hidden_channels).to(self.device)
            self.mlp_edge = nn.Linear(hidden_channels, hidden_channels).to(self.device)
            self.mlp_node = nn.Linear(hidden_channels, hidden_channels).to(self.device)
        elif self.fusion_mode == 'attn':
            self.mlp = nn.Linear(hidden_channels, hidden_channels).to(self.device)
            self.attention = Attention(hidden_channels).to(self.device)
        
    def parameters(self):
        return self._parameters

    def forward(self,A_list, Nodes_list, nodes_mask_list):
        node_feats= Nodes_list[-1]
        for ii, unit in enumerate(self.GRCU_layers):
            if ii == 0:
                cnfeat_list = self.cites_layers[ii](A_list, Nodes_list)
            else:
                cnfeat_list = self.cites_layers[ii](A_list, cnfeat_list)
                
            Nodes_list = unit(A_list, Nodes_list, nodes_mask_list)
        
        cites_list = cites_loader.build_cites(A_list)
        node_loss = self.calc_cites(cnfeat_list, cites_list)
        
        out = Nodes_list[-1]
        out_cites = cnfeat_list[-1]
        
        edge_h = self.mlp(self.fusion(out, out_cites))
        # node_h = self.mlp_node(self.fusion(out_cites, out))
        
        if self.skipfeats:
            out = torch.cat((out,node_feats), dim=1)   # use node_feats.to_dense() if 2hot encoded input 
        return edge_h, node_loss

    
    def fusion(self, edge_h, node_h):
        if self.fusion_mode == 'cat':
            return torch.hstack([edge_h, node_h.detach()])
        elif self.fusion_mode == 'add':
            return edge_h + node_h.detach()
        elif self.fusion_mode == 'mlp_add':
            return self.mlp_edge(edge_h) + self.mlp_node(node_h.detach())
        elif self.fusion_mode == 'attn':
            emb = torch.stack([edge_h, node_h], dim=1) # [n, 2, d]
            emb, att = self.attention(emb)
            return emb
    
    
    def calc_cites(self, cnfeat_list, cites_list):
        # cite_mask = torch.zeros((len(cnfeat_list[0]), 5), dtype=torch.bool).to(self.device)
        node_loss = 0
        for tt, cites in enumerate(cites_list):
            nfeat = cnfeat_list[tt]
            gt = cites['cites_norm'][cites['sid2pos'][np.arange(len(nfeat))]].to(self.device)
            # cite_mask[:, 0:self.node_year-(tt - (0 + self.node_year))] = True
            pred = self.node_predictor(nfeat)
            node_loss += torch.mean(torch.sum(torch.square(pred - gt), dim=1))
            # gdim = gt_list[tt].shape[1]
            # node_loss += torch.mean(torch.sum(torch.square\
            #     (pred*cite_mask[0:len(pred), 0:gdim] - gt_list[tt]*cite_mask[0:len(pred), 0:gdim] ), dim=1))
        return node_loss / len(cites_list)
    
    
class GRCU_cites(torch.nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        cell_args = utils.Namespace({})
        cell_args.rows = args.in_feats
        cell_args.cols = args.out_feats

        self.activation = self.args.activation
        self.GCN_init_weights = Parameter(torch.Tensor(self.args.in_feats,self.args.out_feats))
        self.reset_param(self.GCN_init_weights)

    def reset_param(self,t):
        #Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv,stdv)

    def forward(self, A_list, node_embs_list):
        GCN_weights = self.GCN_init_weights
        out_seq = []
        for t, Ahat in enumerate(A_list):
            node_embs = node_embs_list[t]
            #first evolve the weights from the initial and use the new weights with the node_embs
            node_embs = self.activation(Ahat.matmul(node_embs.matmul(GCN_weights)))
            out_seq.append(node_embs)
        return out_seq
    
    
class GRCU(torch.nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        cell_args = utils.Namespace({})
        cell_args.rows = args.in_feats
        cell_args.cols = args.out_feats

        self.evolve_weights = mat_GRU_cell(cell_args)

        self.activation = self.args.activation
        self.GCN_init_weights = Parameter(torch.Tensor(self.args.in_feats,self.args.out_feats))
        self.reset_param(self.GCN_init_weights)

    def reset_param(self,t):
        #Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv,stdv)

    def forward(self,A_list,node_embs_list,mask_list):
        GCN_weights = self.GCN_init_weights
        out_seq = []
        for t,Ahat in enumerate(A_list):
            node_embs = node_embs_list[t]
            #first evolve the weights from the initial and use the new weights with the node_embs
            GCN_weights = self.evolve_weights(GCN_weights,node_embs,mask_list[t])
            node_embs = self.activation(Ahat.matmul(node_embs.matmul(GCN_weights)))

            out_seq.append(node_embs)

        return out_seq

class mat_GRU_cell(torch.nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.update = mat_GRU_gate(args.rows,
                                   args.cols,
                                   torch.nn.Sigmoid())

        self.reset = mat_GRU_gate(args.rows,
                                   args.cols,
                                   torch.nn.Sigmoid())

        self.htilda = mat_GRU_gate(args.rows,
                                   args.cols,
                                   torch.nn.Tanh())
        
        self.choose_topk = TopK(feats = args.rows,
                                k = args.cols)

    def forward(self,prev_Q,prev_Z,mask):
        z_topk = self.choose_topk(prev_Z,mask)

        update = self.update(z_topk,prev_Q)
        reset = self.reset(z_topk,prev_Q)

        h_cap = reset * prev_Q
        h_cap = self.htilda(z_topk, h_cap)

        new_Q = (1 - update) * prev_Q + update * h_cap

        return new_Q

        

class mat_GRU_gate(torch.nn.Module):
    def __init__(self,rows,cols,activation):
        super().__init__()
        self.activation = activation
        #the k here should be in_feats which is actually the rows
        self.W = Parameter(torch.Tensor(rows,rows))
        self.reset_param(self.W)

        self.U = Parameter(torch.Tensor(rows,rows))
        self.reset_param(self.U)

        self.bias = Parameter(torch.zeros(rows,cols))

    def reset_param(self,t):
        #Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv,stdv)

    def forward(self,x,hidden):
        out = self.activation(self.W.matmul(x) + \
                              self.U.matmul(hidden) + \
                              self.bias)

        return out

class TopK(torch.nn.Module):
    def __init__(self,feats,k):
        super().__init__()
        self.scorer = Parameter(torch.Tensor(feats,1))
        self.reset_param(self.scorer)
        
        self.k = k

    def reset_param(self,t):
        #Initialize based on the number of rows
        stdv = 1. / math.sqrt(t.size(0))
        t.data.uniform_(-stdv,stdv)

    def forward(self,node_embs,mask):
        scores = node_embs.matmul(self.scorer) / self.scorer.norm()
        scores = scores + mask

        vals, topk_indices = scores.view(-1).topk(self.k)
        topk_indices = topk_indices[vals > -float("Inf")]

        if topk_indices.size(0) < self.k:
            topk_indices = utils.pad_with_last_val(topk_indices,self.k, node_embs.device)
            
        tanh = torch.nn.Tanh()

        if isinstance(node_embs, torch.sparse.FloatTensor) or \
           isinstance(node_embs, torch.cuda.sparse.FloatTensor):
            node_embs = node_embs.to_dense()

        out = node_embs[topk_indices] * tanh(scores[topk_indices].view(-1,1))

        #we need to transpose the output
        return out.t()
