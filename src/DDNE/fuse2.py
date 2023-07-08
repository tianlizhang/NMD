import torch
import torch.nn as nn
from IPython import embed
from loader import cites_loader
from EvolveGCN import models


class DDNE(nn.Module):
    '''
    Class to define DDNE
    '''
    def __init__(self, args, win_size, num_nodes, fusion_mode, dropout_rate):
        super(DDNE, self).__init__()
        # ====================
        # self.enc_dims = end_dims # Layer configuration of encoder
        # self.dec_dims = dec_dims # Layer configuration of decoder
        # self.enc_dims = [num_nodes, args.layer_1_feats, args.layer_2_feats]
        self.enc_dims = [num_nodes]
        self.enc_dims.extend(args.enc_dims)
        # self.dec_dims = [2*self.enc_dims[-1]*(win_size+1), num_nodes]
        self.dropout_rate = dropout_rate # Dropout rate
        # ==========
        # Encoder
        self.enc = DDNE_Enc(self.enc_dims, self.dropout_rate, win_size, fusion_mode)
        # Decoder
        # self.dec = DDNE_Dec(self.dec_dims, self.dropout_rate)

    def forward(self, adj_list, node_list=None, mask_list=None):
        '''
        Rewrite the forward function
        :param adj_list: list of historical adjacency matrices (i.e., input)
        :return: prediction result
        '''
        # ====================
        dyn_emb, node_loss = self.enc(adj_list)
        return dyn_emb, node_loss
        # adj_est = self.dec(dyn_emb)
        # return adj_est, dyn_emb

class DDNE_Enc(nn.Module):
    '''
    Class to define the encoder of DDNE
    '''
    def __init__(self, enc_dims, dropout_rate, win_size, fusion_mode):
        super(DDNE_Enc, self).__init__()
        # ====================
        self.enc_dims = enc_dims # Layer configuration of encoder
        self.dropout_rate = dropout_rate # Dropout rate
        # ==========
        # Define the encoder, i.e., multi-layer RNN
        self.num_enc_layers = len(self.enc_dims)-1 # Number of RNNs
        self.for_RNN_layer_list = nn.ModuleList() # Forward RNN encoder
        self.rev_RNN_layer_list = nn.ModuleList() # Reverse RNN encoder
        
        self.for_cites_layer_list = nn.ModuleList() # Forward cites encoder
        self.rev_cites_layer_list = nn.ModuleList() # Reverse cites encoder
        for l in range(self.num_enc_layers):
            self.for_RNN_layer_list.append(
                nn.GRU(input_size=self.enc_dims[l], hidden_size=self.enc_dims[l+1]))
            self.rev_RNN_layer_list.append(
                nn.GRU(input_size=self.enc_dims[l], hidden_size=self.enc_dims[l+1]))
            
            self.for_cites_layer_list.append(
                nn.GRU(input_size=self.enc_dims[l], hidden_size=self.enc_dims[l+1]))
            self.rev_cites_layer_list.append(
                nn.GRU(input_size=self.enc_dims[l], hidden_size=self.enc_dims[l+1]))
        
        # ==========
        # cites predictior
        self.pred_year = 1
        hidden_channels = 2*self.enc_dims[-1]
        self.node_predictor = cites_loader.FcPredictor(hidden_channels, node_year = self.pred_year)
        
        self.fusion_mode = fusion_mode
        print(f'fusion_mode: {self.fusion_mode}')
        if self.fusion_mode == 'cat' or self.fusion_mode == 'cat_tach':
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

        # ==========
        # Structural encoder
        for_RNN_input = torch.stack([adj for adj in adj_list])
        rev_RNN_input = torch.stack([adj for adj in adj_list[::-1]])
        for_cites_input, rev_cites_input = for_RNN_input, rev_RNN_input
        
        for l in range(self.num_enc_layers):
            # ==========
            for_RNN_output, _ = self.for_RNN_layer_list[l](for_RNN_input)
            for_RNN_input = for_RNN_output
            # ==========
            rev_RNN_output, _ = self.rev_RNN_layer_list[l](rev_RNN_input)
            rev_RNN_input = rev_RNN_output
            # ==========
            for_cites_output, _ = self.for_cites_layer_list[l](for_cites_input)
            for_cites_input = for_cites_output
            # ==========
            rev_cites_output, _ = self.rev_cites_layer_list[l](rev_cites_input)
            rev_cites_input = rev_cites_output
        # ==========
        # Fusion layer
        # RNN_out_list_tmp = [self.mlp(self.fusion(torch.hstack([for_out, rev_out]), torch.hstack([for_cite, rev_cite]))) \
        #     for for_out, rev_out, for_cite, rev_cite in zip(for_RNN_output, rev_RNN_output, for_cites_output, rev_cites_output)]
        
        # cites_out_list = [torch.hstack([for_cite, rev_cite]) for for_cite, rev_cite in zip(for_cites_output, rev_cites_output)]
        # cites_out_list_tmp = [self.mlp(self.fusion(torch.hstack([for_cite, rev_cite]), torch.hstack([for_out, rev_out]))) \
        #     for for_out, rev_out, for_cite, rev_cite in zip(for_RNN_output, rev_RNN_output, for_cites_output, rev_cites_output)]
        
        RNN_out_list, cites_out_list = [], []
        for for_out, rev_out, for_cite, rev_cite in zip(for_RNN_output, rev_RNN_output, for_cites_output, rev_cites_output):
            hh_rnn, hh_cites = torch.hstack([for_out, rev_out]),  torch.hstack([for_cite, rev_cite])
            RNN_out = self.mlp(self.fusion(hh_rnn, hh_cites))
            cites_out = self.mlp(self.fusion(hh_cites, hh_rnn))
            
            RNN_out_list.append(RNN_out)
            cites_out_list.append(cites_out)
        
        node_loss = self.calc_cites(cites_out_list, cites_list)
        # ==========
        # Temporal encoder
        dyn_emb = torch.hstack(RNN_out_list)
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
        elif self.fusion_mode == 'cat_tach':
            return torch.hstack([edge_h, node_h])
    
    
    def calc_cites(self, cnfeat_list, cites_list):
        node_loss = 0
        for tt, cites in enumerate(cites_list):
            nfeat = cnfeat_list[tt]
            
            gt = cites['cites_norm'][cites['sid2pos'][torch.arange(len(nfeat))]].to(nfeat.device)
            pred = self.node_predictor(nfeat)
            node_loss += torch.mean(torch.sum(torch.square(pred - gt), dim=1))
        return node_loss / len(cites_list)
    
    

class DDNE_Dec(nn.Module):
    '''
    Class to define the decoder of DDNE
    '''
    def __init__(self, dec_dims, dropout_rate):
        super(DDNE_Dec, self).__init__()
        # ====================
        self.dec_dims = dec_dims # Layer configuration of decoder
        self.dropout_rate = dropout_rate # Dropout rate
        # ==========
        # Decoder
        self.num_dec_layers = len(self.dec_dims)-1  # Number of FC layers
        self.dec = nn.ModuleList()
        for l in range(self.num_dec_layers):
            self.dec.append(
                nn.Linear(in_features=self.dec_dims[l], out_features=self.dec_dims[l+1]))
        self.dec_drop = nn.ModuleList()
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