import torch
# import utils as u
from utils import utils
from argparse import Namespace
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import torch.nn as nn
import math
from loader import cites_loader
from EvolveGCN import models


class Sp_GCN(torch.nn.Module):
    def __init__(self,args,activation):
        super().__init__()
        self.activation = activation
        self.num_layers = args.num_layers

        self.w_list = nn.ParameterList()
        for i in range(self.num_layers):
            if i==0:
                w_i = Parameter(torch.Tensor(args.feats_per_node, args.layer_1_feats))
                utils.reset_param(w_i)
            else:
                w_i = Parameter(torch.Tensor(args.layer_1_feats, args.layer_2_feats))
                utils.reset_param(w_i)
            self.w_list.append(w_i)


    def forward(self,A_list, Nodes_list, nodes_mask_list):
        node_feats = Nodes_list[-1]
        #A_list: T, each element sparse tensor
        #take only last adj matrix in time
        Ahat = A_list[-1]
        #Ahat: NxN ~ 30k
        #sparse multiplication

        # Ahat NxN
        # self.node_embs = Nxk
        #
        # note(bwheatman, tfk): change order of matrix multiply
        last_l = self.activation(Ahat.matmul(node_feats.matmul(self.w_list[0])))
        for i in range(1, self.num_layers):
            last_l = self.activation(Ahat.matmul(last_l.matmul(self.w_list[i])))
        return last_l


class Sp_Skip_GCN(Sp_GCN):
    def __init__(self,args,activation):
        super().__init__(args,activation)
        self.W_feat = Parameter(torch.Tensor(args.feats_per_node, args.layer_1_feats))

    def forward(self,A_list, Nodes_list = None):
        node_feats = Nodes_list[-1]
        #A_list: T, each element sparse tensor
        #take only last adj matrix in time
        Ahat = A_list[-1]
        #Ahat: NxN ~ 30k
        #sparse multiplication

        # Ahat NxN
        # self.node_feats = Nxk
        #
        # note(bwheatman, tfk): change order of matrix multiply
        l1 = self.activation(Ahat.matmul(node_feats.matmul(self.W1)))
        l2 = self.activation(Ahat.matmul(l1.matmul(self.W2)) + (node_feats.matmul(self.W3)))

        return l2

class Sp_Skip_NodeFeats_GCN(Sp_GCN):
    def __init__(self,args,activation):
        super().__init__(args,activation)

    def forward(self,A_list, Nodes_list = None):
        node_feats = Nodes_list[-1]
        Ahat = A_list[-1]
        last_l = self.activation(Ahat.matmul(node_feats.matmul(self.w_list[0])))
        for i in range(1, self.num_layers):
            last_l = self.activation(Ahat.matmul(last_l.matmul(self.w_list[i])))
        skip_last_l = torch.cat((last_l,node_feats), dim=1)   # use node_feats.to_dense() if 2hot encoded input
        return skip_last_l

class Sp_GCN_LSTM_A(Sp_GCN):
    def __init__(self,args,activation):
        super().__init__(args,activation)
        self.rnn = nn.LSTM(
                input_size=args.layer_2_feats,
                hidden_size=args.lstm_l2_feats,
                num_layers=args.lstm_l2_layers
                )

    def forward(self ,A_list, Nodes_list = None, nodes_mask_list = None):
        last_l_seq=[]
        for t,Ahat in enumerate(A_list):
            node_feats = Nodes_list[t]
            #A_list: T, each element sparse tensor
            #note(bwheatman, tfk): change order of matrix multiply
            last_l = self.activation(Ahat.matmul(node_feats.matmul(self.w_list[0])))
            for i in range(1, self.num_layers):
                last_l = self.activation(Ahat.matmul(last_l.matmul(self.w_list[i])))
            last_l_seq.append(last_l)

        last_l_seq = torch.stack(last_l_seq)

        out, _ = self.rnn(last_l_seq, None)
        return out[-1]


class Sp_GCN_LSTM_A_tim(Sp_GCN):
    def __init__(self,args, fusion_mode, activation):
        super().__init__(args, activation)
        
        self.w_cites_list = nn.ParameterList()
        for i in range(self.num_layers):
            if i==0:
                w_i = Parameter(torch.Tensor(args.feats_per_node, args.layer_1_feats))
                utils.reset_param(w_i)
            else:
                w_i = Parameter(torch.Tensor(args.layer_1_feats, args.layer_2_feats))
                utils.reset_param(w_i)
            self.w_cites_list.append(w_i)


        self.rnn = nn.LSTM(
                input_size=args.layer_2_feats,
                hidden_size=args.lstm_l2_feats,
                num_layers=args.lstm_l2_layers
                )

        hidden_channels = args.lstm_l2_feats
        self.pred_year = 1
        self.node_predictor = cites_loader.FcPredictor(hidden_channels, node_year = self.pred_year)
        
        self.fusion_mode = fusion_mode
        print(f'fusion_mode: {self.fusion_mode}')
        if self.fusion_mode == 'cat':
            self.mlp = nn.Linear(2*hidden_channels, hidden_channels)
        elif self.fusion_mode == 'add':
            self.mlp = nn.Linear(hidden_channels, hidden_channels)
        elif self.fusion_mode == 'mlp_add':
            self.mlp = nn.Linear(hidden_channels, hidden_channels)
            self.mlp_edge = nn.Linear(hidden_channels, hidden_channels)
            self.mlp_node = nn.Linear(hidden_channels, hidden_channels)
        elif self.fusion_mode == 'attn':
            self.mlp = nn.Linear(hidden_channels, hidden_channels)
            self.attention = models.Attention(hidden_channels)
        

    def forward(self ,A_list, Nodes_list = None, nodes_mask_list = None):
        last_l_seq, cites_seq = [], []
        for t,Ahat in enumerate(A_list):
            node_feats = Nodes_list[t]
            #A_list: T, each element sparse tensor
            #note(bwheatman, tfk): change order of matrix multiply
            last_l = self.activation(Ahat.matmul(node_feats.matmul(self.w_list[0])))
            for i in range(1, self.num_layers):
                last_l = self.activation(Ahat.matmul(last_l.matmul(self.w_list[i])))
            last_l_seq.append(last_l)
            
            cites_l = self.activation(Ahat.matmul(node_feats.matmul(self.w_cites_list[0])))
            for i in range(1, self.num_layers):
                cites_l = self.activation(Ahat.matmul(cites_l.matmul(self.w_cites_list[i])))
            cites_seq.append(cites_l)
        ## Fuseion Layer
        last_l_seq = [self.mlp(self.fusion(last_l, cites_l)) for last_l, cites_l in zip(last_l_seq, cites_seq)]
        cites_list = cites_loader.build_cites(A_list)
        node_loss = self.calc_cites(cites_seq, cites_list)
        ## Temporal 
        last_l_seq = torch.stack(last_l_seq)
        out, _ = self.rnn(last_l_seq, None)
        return out[-1], node_loss
    
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
        node_loss = 0
        for tt, cites in enumerate(cites_list):
            nfeat = cnfeat_list[tt]
            gt = cites['cites_norm'][cites['sid2pos'][torch.arange(len(nfeat))]].to(nfeat.device)
            pred = self.node_predictor(nfeat)
            node_loss += torch.mean(torch.sum(torch.square(pred - gt), dim=1))
        return node_loss / len(cites_list)


class Sp_GCN_LSTM_A_tim_shared(Sp_GCN):
    def __init__(self,args, fusion_mode, activation):
        super().__init__(args, activation)
        
        # self.w_cites_list = nn.ParameterList()
        for i in range(self.num_layers):
            if i==0:
                w_i = Parameter(torch.Tensor(args.feats_per_node, args.layer_1_feats))
                utils.reset_param(w_i)
            else:
                w_i = Parameter(torch.Tensor(args.layer_1_feats, args.layer_2_feats))
                utils.reset_param(w_i)
            # self.w_cites_list.append(w_i)


        self.rnn = nn.LSTM(
                input_size=args.layer_2_feats,
                hidden_size=args.lstm_l2_feats,
                num_layers=args.lstm_l2_layers
                )

        hidden_channels = args.lstm_l2_feats
        self.pred_year = 1
        self.node_predictor = cites_loader.FcPredictor(hidden_channels, node_year = self.pred_year)
        
        self.fusion_mode = fusion_mode
        print(f'fusion_mode: {self.fusion_mode}')
        if self.fusion_mode == 'cat':
            self.mlp = nn.Linear(2*hidden_channels, hidden_channels)
        elif self.fusion_mode == 'add':
            self.mlp = nn.Linear(hidden_channels, hidden_channels)
        elif self.fusion_mode == 'mlp_add':
            self.mlp = nn.Linear(hidden_channels, hidden_channels)
            self.mlp_edge = nn.Linear(hidden_channels, hidden_channels)
            self.mlp_node = nn.Linear(hidden_channels, hidden_channels)
        elif self.fusion_mode == 'attn':
            self.mlp = nn.Linear(hidden_channels, hidden_channels)
            self.attention = models.Attention(hidden_channels)
        

    def forward(self ,A_list, Nodes_list = None, nodes_mask_list = None):
        last_l_seq, cites_seq = [], []
        for t,Ahat in enumerate(A_list):
            node_feats = Nodes_list[t]
            #A_list: T, each element sparse tensor
            #note(bwheatman, tfk): change order of matrix multiply
            last_l = self.activation(Ahat.matmul(node_feats.matmul(self.w_list[0])))
            for i in range(1, self.num_layers):
                last_l = self.activation(Ahat.matmul(last_l.matmul(self.w_list[i])))
            last_l_seq.append(last_l)
            
            cites_l = self.activation(Ahat.matmul(node_feats.matmul(self.w_list[0])))
            for i in range(1, self.num_layers):
                cites_l = self.activation(Ahat.matmul(cites_l.matmul(self.w_list[i])))
            cites_seq.append(cites_l)
        ## Fuseion Layer
        last_l_seq = [self.mlp(self.fusion(last_l, cites_l)) for last_l, cites_l in zip(last_l_seq, cites_seq)]
        cites_list = cites_loader.build_cites(A_list)
        node_loss = self.calc_cites(cites_seq, cites_list)
        ## Temporal 
        last_l_seq = torch.stack(last_l_seq)
        out, _ = self.rnn(last_l_seq, None)
        return out[-1], node_loss
    
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
        node_loss = 0
        for tt, cites in enumerate(cites_list):
            nfeat = cnfeat_list[tt]
            gt = cites['cites_norm'][cites['sid2pos'][torch.arange(len(nfeat))]].to(nfeat.device)
            pred = self.node_predictor(nfeat)
            node_loss += torch.mean(torch.sum(torch.square(pred - gt), dim=1))
        return node_loss / len(cites_list)
    


class Sp_GCN_LSTM_A_tim_fuse2(Sp_GCN):
    def __init__(self,args, fusion_mode, activation):
        super().__init__(args, activation)
        
        self.w_cites_list = nn.ParameterList()
        for i in range(self.num_layers):
            if i==0:
                w_i = Parameter(torch.Tensor(args.feats_per_node, args.layer_1_feats))
                utils.reset_param(w_i)
            else:
                w_i = Parameter(torch.Tensor(args.layer_1_feats, args.layer_2_feats))
                utils.reset_param(w_i)
            self.w_cites_list.append(w_i)


        self.rnn = nn.LSTM(
                input_size=args.layer_2_feats,
                hidden_size=args.lstm_l2_feats,
                num_layers=args.lstm_l2_layers
                )

        hidden_channels = args.lstm_l2_feats
        self.pred_year = 1
        self.node_predictor = cites_loader.FcPredictor(hidden_channels, node_year = self.pred_year)
        
        self.fusion_mode = fusion_mode
        print(f'fusion_mode: {self.fusion_mode}')
        if self.fusion_mode == 'cat':
            self.mlp = nn.Linear(2*hidden_channels, hidden_channels)
        elif self.fusion_mode == 'add':
            self.mlp = nn.Linear(hidden_channels, hidden_channels)
        elif self.fusion_mode == 'mlp_add':
            self.mlp = nn.Linear(hidden_channels, hidden_channels)
            self.mlp_edge = nn.Linear(hidden_channels, hidden_channels)
            self.mlp_node = nn.Linear(hidden_channels, hidden_channels)
        elif self.fusion_mode == 'attn':
            self.mlp = nn.Linear(hidden_channels, hidden_channels)
            self.attention = models.Attention(hidden_channels)
        

    def forward(self ,A_list, Nodes_list = None, nodes_mask_list = None):
        last_l_seq, cites_l_seq = [], []
        for t,Ahat in enumerate(A_list):
            node_feats = Nodes_list[t]
            #A_list: T, each element sparse tensor
            #note(bwheatman, tfk): change order of matrix multiply
            last_l = self.activation(Ahat.matmul(node_feats.matmul(self.w_list[0])))
            for i in range(1, self.num_layers):
                last_l = self.activation(Ahat.matmul(last_l.matmul(self.w_list[i])))
            last_l_seq.append(last_l)
            
            cites_l = self.activation(Ahat.matmul(node_feats.matmul(self.w_cites_list[0])))
            for i in range(1, self.num_layers):
                cites_l = self.activation(Ahat.matmul(cites_l.matmul(self.w_cites_list[i])))
            cites_l_seq.append(cites_l)
        ## Fuseion Layer
        # last_l_seq = [self.mlp(self.fusion(last_l, cites_l)) for last_l, cites_l in zip(last_l_seq, cites_seq)]
        
        last_seq, cites_seq = [], []
        for last_l, cites_l in zip(last_l_seq, cites_l_seq):
            last_fuse = self.mlp(self.fusion(last_l, cites_l))
            cites_fuse =  self.mlp(self.fusion(cites_l, last_l))
            
            last_seq.append(last_fuse)
            cites_seq.append(cites_fuse)
        
        cites_list = cites_loader.build_cites(A_list)
        node_loss = self.calc_cites(cites_seq, cites_list)
        ## Temporal 
        last_seq = torch.stack(last_seq)
        out, _ = self.rnn(last_seq, None)
        return out[-1], node_loss
    
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
        node_loss = 0
        for tt, cites in enumerate(cites_list):
            nfeat = cnfeat_list[tt]
            gt = cites['cites_norm'][cites['sid2pos'][torch.arange(len(nfeat))]].to(nfeat.device)
            pred = self.node_predictor(nfeat)
            node_loss += torch.mean(torch.sum(torch.square(pred - gt), dim=1))
        return node_loss / len(cites_list)
    
    
class Sp_GCN_GRU_A(Sp_GCN_LSTM_A):
    def __init__(self,args,activation):
        super().__init__(args,activation)
        self.rnn = nn.GRU(
                input_size=args.layer_2_feats,
                hidden_size=args.lstm_l2_feats,
                num_layers=args.lstm_l2_layers
                )

class Sp_GCN_LSTM_B(Sp_GCN):
    def __init__(self,args,activation):
        super().__init__(args,activation)
        assert args.num_layers == 2, 'GCN-LSTM and GCN-GRU requires 2 conv layers.'
        self.rnn_l1 = nn.LSTM(
                input_size=args.layer_1_feats,
                hidden_size=args.lstm_l1_feats,
                num_layers=args.lstm_l1_layers
                )

        self.rnn_l2 = nn.LSTM(
                input_size=args.layer_2_feats,
                hidden_size=args.lstm_l2_feats,
                num_layers=args.lstm_l2_layers
                )
        self.W2 = Parameter(torch.Tensor(args.lstm_l1_feats, args.layer_2_feats))
        utils.reset_param(self.W2)

    def forward(self,A_list, Nodes_list = None, nodes_mask_list = None):
        l1_seq=[]
        l2_seq=[]
        for t,Ahat in enumerate(A_list):
            node_feats = Nodes_list[t]
            l1 = self.activation(Ahat.matmul(node_feats.matmul(self.w_list[0])))
            l1_seq.append(l1)

        l1_seq = torch.stack(l1_seq)

        out_l1, _ = self.rnn_l1(l1_seq, None)

        for i in range(len(A_list)):
            Ahat = A_list[i]
            out_t_l1 = out_l1[i]
            #A_list: T, each element sparse tensor
            l2 = self.activation(Ahat.matmul(out_t_l1).matmul(self.w_list[1]))
            l2_seq.append(l2)

        l2_seq = torch.stack(l2_seq)

        out, _ = self.rnn_l2(l2_seq, None)
        return out[-1]


class Sp_GCN_GRU_B(Sp_GCN_LSTM_B):
    def __init__(self,args,activation):
        super().__init__(args,activation)
        self.rnn_l1 = nn.GRU(
                input_size=args.layer_1_feats,
                hidden_size=args.lstm_l1_feats,
                num_layers=args.lstm_l1_layers
               )

        self.rnn_l2 = nn.GRU(
                input_size=args.layer_2_feats,
                hidden_size=args.lstm_l2_feats,
                num_layers=args.lstm_l2_layers
                )

class Classifier(torch.nn.Module):
    def __init__(self,args,out_features=2, in_features = None):
        super(Classifier,self).__init__()
        activation = torch.nn.ReLU()

        if in_features is not None:
            num_feats = in_features
        elif args.experiment_type in ['sp_lstm_A_trainer', 'sp_lstm_B_trainer',
                                    'sp_weighted_lstm_A', 'sp_weighted_lstm_B'] :
            num_feats = args.gcn_parameters['lstm_l2_feats'] * 2
        else:
            num_feats = args.gcn_parameters['layer_2_feats'] * 2
        print ('CLS num_feats',num_feats)

        self.mlp = torch.nn.Sequential(torch.nn.Linear(in_features = num_feats,
                                                       out_features =args.cls_feats),
                                       activation,
                                       torch.nn.Linear(in_features = args.cls_feats,
                                                       out_features = out_features))

    def forward(self,x):
        return self.mlp(x)



class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z): # [n, 2, d]
        w = self.project(z) # [n, 2, 1]
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta # [n, 2, d] -> [n, d]