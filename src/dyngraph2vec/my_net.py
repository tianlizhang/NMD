import torch
import torch.nn as nn
from loader import cites_loader
from IPython import embed
import numpy as np
from EvolveGCN import models


class dyngraph2vec(nn.Module):
    '''
    Class to define dyngraph2vec (AERNN variant)
    '''
    def __init__(self, args, num_nodes, fusion_mode, dropout_rate):
        super(dyngraph2vec, self).__init__()
        # ====================
        self.struc_dims = [num_nodes]
        self.struc_dims.extend(args.struc_dims)
        self.temp_dims = args.temp_dims
        # self.struc_dims = [num_nodes, args.layer_1_feats] # Layer configuration of structural encoder
        # self.temp_dims = [self.struc_dims[-1], args.layer_2_feats, args.layer_2_feats]  # Layer configuration of temporal encoder
        # self.dec_dims = [self.temp_dims[-1], args.layer_1_feats, num_nodes] # Layer configuration of decoder
        self.dropout_rate = dropout_rate # Dropout rate
        # ==========
        # Encoder
        self.enc = dyngraph2vec_Enc(self.struc_dims, self.temp_dims, self.dropout_rate, fusion_mode)
        # Decoder
        # self.dec = dyngraph2vec_Dec(self.dec_dims, self.dropout_rate)

    def forward(self, adj_list, node_list=None, mask_list=None):
        '''
        Rewrite the forward function
        :param adj_list: list of historical adjacency matrices (i.e., input)
        :return: prediction result
        '''
        # ====================
        dyn_emb, node_loss = self.enc(adj_list)
        # adj_est = self.dec(dyn_emb)
        return dyn_emb, node_loss


class dyngraph2vec_Enc(nn.Module):
    '''
    Class to define the encoder of dyngraph2vec (AERNN variant)
    '''
    def __init__(self, struc_dims, temp_dims, dropout_rate, fusion_mode):
        super(dyngraph2vec_Enc, self).__init__()
        # ====================
        self.struc_dims = struc_dims # Layer configuration of structural encoder
        self.temp_dims = temp_dims # Layer configuration of temporal encoder
        self.dropout_rate = dropout_rate # Dropout rate
        # ==========
        # Structural encoder
        self.num_struc_layers = len(self.struc_dims)-1 # Number of FC layers
        self.struc_enc = nn.ModuleList()
        self.struc_drop = nn.ModuleList()
        self.cites_enc = nn.ModuleList()
        self.cites_drop = nn.ModuleList()
        for l in range(self.num_struc_layers):
            self.struc_enc.append(
                nn.Linear(in_features=self.struc_dims[l], out_features=self.struc_dims[l+1]))
            self.cites_enc.append(
                nn.Linear(in_features=self.struc_dims[l], out_features=self.struc_dims[l+1]))
            
        for l in range(self.num_struc_layers):
            self.struc_drop.append(nn.Dropout(p=self.dropout_rate))
            self.cites_drop.append(nn.Dropout(p=self.dropout_rate))
        # ==========
        # Temporal encoder
        self.num_temp_layers = len(self.temp_dims)-1 # Numer of LSTM layers
        self.temp_enc = nn.ModuleList()
        for l in range(self.num_temp_layers):
            self.temp_enc.append(
                nn.LSTM(input_size=self.temp_dims[l], hidden_size=self.temp_dims[l+1]))
        # ==========
        # cites predictior
        self.pred_year = 1
        hidden_channels = self.struc_dims[-1]
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
        

    def forward(self, adj_list):
        '''
        Rewrite the forward function
        :param adj_list: list of historical adjacency matrices (i.e., input)
        :return: dynamic node embedding
        '''
        # ====================
        cites_list = cites_loader.build_cites(adj_list)
        
        adj_list = [adj.to_dense() for adj in adj_list]
        win_size = len(adj_list)  # Window size, i.e., #historical snapshots
        num_nodes, _ = adj_list[0].shape
        # ====================
        # Structural encoder
        temp_input_tnr = None
        cites_input_tnr = None
        for t in range(win_size):
            adj = adj_list[t]
            struc_input = adj
            cites_input = adj
            
            struc_output = None
            cites_output = None
            for l in range(self.num_struc_layers):
                struc_output = self.struc_enc[l](struc_input) # (num_nodes, num_nodes) -> (num_nodes, d)
                struc_output = self.struc_drop[l](struc_output)
                struc_output = torch.relu(struc_output)
                struc_input = struc_output
                
                cites_output = self.cites_enc[l](cites_input)
                cites_output = self.struc_drop[l](cites_output)
                cites_output = torch.relu(cites_output)
                cites_input = cites_output
            if t==0:
                temp_input_tnr = struc_output
                cites_input_tnr = cites_output
            else:
                temp_input_tnr = torch.cat((temp_input_tnr, struc_output), dim=0) # (num_nodes*win_size, d)
                cites_input_tnr = torch.cat((cites_input_tnr, cites_output), dim=0)
        # ==========
        # Fusion layer
        temp_input_tnr = self.mlp(self.fusion(temp_input_tnr, cites_input_tnr))
        cites_input_tnr = torch.reshape(cites_input_tnr, (win_size, int(num_nodes), self.temp_dims[0]))
        cites_output_list = [cites_input_tnr[ii] for ii in range(len(cites_input_tnr))]
        node_loss = self.calc_cites(cites_output_list, cites_list)
        
        # ==========
        # Temporal encoder
        temp_input = torch.reshape(temp_input_tnr, (win_size, int(num_nodes), self.temp_dims[0]))
        temp_output = None
        for l in range(self.num_temp_layers):
            temp_output, _ = self.temp_enc[l](temp_input)
            temp_input = temp_output
        dyn_emb = temp_output[-1, :] # Dynamic node embedding (N*d) # [win, N, d]
        return dyn_emb, node_loss


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


class dyngraph2vec_Dec(nn.Module):
    '''
    Class to define the decoder of dyngraph2vec (AERNN variant)
    '''
    def __init__(self, dec_dims, dropout_rate):
        super(dyngraph2vec_Dec, self).__init__()
        # ====================
        self.dec_dims = dec_dims # Layer configuration of decoder
        self.dropout_rate = dropout_rate # Dropout rate
        # ==========
        # Decoder
        self.num_dec_layers = len(self.dec_dims)-1 # Number of FC layers
        self.dec = nn.ModuleList()
        self.dec_drop = nn.ModuleList()
        for l in range(self.num_dec_layers):
            self.dec.append(
                nn.Linear(in_features=self.dec_dims[l], out_features=self.dec_dims[l+1]))
        for l in range(self.num_dec_layers-1):
            self.dec_drop.append(nn.Dropout(p=self.dropout_rate))

    def forward(self, dyn_emb):
        '''
        Rewrite the forward function
        :param dyn_emb: dynamic embedding given by encoder
        :return: prediction result
        '''
        # ====================
        dec_input = dyn_emb
        dec_output = None
        for l in range(self.num_dec_layers-1):
            dec_output = self.dec[l](dec_input)
            dec_output = self.dec_drop[l](dec_output)
            dec_output = torch.relu(dec_output)
            dec_input = dec_output
        dec_output = self.dec[-1](dec_input)
        dec_output = torch.sigmoid(dec_output)
        adj_est = dec_output

        return adj_est
