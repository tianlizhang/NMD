import torch
import torch.nn as nn
from IPython import embed


class DDNE(nn.Module):
    '''
    Class to define DDNE
    '''
    def __init__(self, args, win_size, num_nodes, dropout_rate):
        super(DDNE, self).__init__()
        # ====================
        # self.enc_dims = [num_nodes, args.layer_1_feats, args.layer_2_feats]
        self.enc_dims = [num_nodes]
        self.enc_dims.extend(args.enc_dims)
        # self.dec_dims = [2*self.enc_dims[-1]*(win_size+1), num_nodes]
        self.dropout_rate = dropout_rate # Dropout rate
        # ==========
        # Encoder
        self.enc = DDNE_Enc(self.enc_dims, self.dropout_rate)
        # Decoder
        # self.dec = DDNE_Dec(self.dec_dims, self.dropout_rate)

    def forward(self, adj_list, node_list=None, mask_list=None):
        '''
        Rewrite the forward function
        :param adj_list: list of historical adjacency matrices (i.e., input)
        :return: prediction result
        '''
        # ====================
        dyn_emb = self.enc(adj_list)
        return dyn_emb
        # adj_est = self.dec(dyn_emb)
        # return adj_est, dyn_emb

class DDNE_Enc(nn.Module):
    '''
    Class to define the encoder of DDNE
    '''
    def __init__(self, enc_dims, dropout_rate):
        super(DDNE_Enc, self).__init__()
        # ====================
        self.enc_dims = enc_dims # Layer configuration of encoder
        self.dropout_rate = dropout_rate # Dropout rate
        # ==========
        # Define the encoder, i.e., multi-layer RNN
        self.num_enc_layers = len(self.enc_dims)-1 # Number of RNNs
        self.for_RNN_layer_list = nn.ModuleList() # Forward RNN encoder
        self.rev_RNN_layer_list = nn.ModuleList() # Reverse RNN encoder
        for l in range(self.num_enc_layers):
            self.for_RNN_layer_list.append(
                nn.GRU(input_size=self.enc_dims[l], hidden_size=self.enc_dims[l+1]))
            self.rev_RNN_layer_list.append(
                nn.GRU(input_size=self.enc_dims[l], hidden_size=self.enc_dims[l+1]))

    def forward(self, adj_list):
        '''
        Rewrite the forward function
        :param adj_list: list of historical adjacency matrices (i.e., input)
        :return: dynamic node embedding
        '''
        # ====================
        adj_list = [adj.to_dense() for adj in adj_list]
        
        # ====================
        # Structural encoder
        for_RNN_input = torch.stack([adj for adj in adj_list]) # [win_size, num_nodes, num_nodes]
        rev_RNN_input = torch.stack([adj for adj in adj_list[::-1]])
        for l in range(self.num_enc_layers):
            # ==========
            for_RNN_output, _ = self.for_RNN_layer_list[l](for_RNN_input) # [win_size, num_nodes, d]
            for_RNN_input = for_RNN_output
            # ==========
            rev_RNN_output, _ = self.rev_RNN_layer_list[l](rev_RNN_input)
            rev_RNN_input = rev_RNN_output
        # ==========
        # from IPython import embed
        # embed()
        RNN_out_list = [torch.hstack([for_out, rev_out]) for for_out, rev_out in zip(for_RNN_output, rev_RNN_output)] # list((num_nodes, 2*d))
        dyn_emb = torch.hstack(RNN_out_list) # [num_nodes, 2*d*win_size]

        return dyn_emb


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