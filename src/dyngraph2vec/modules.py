import torch
import torch.nn as nn

class dyngraph2vec(nn.Module):
    '''
    Class to define dyngraph2vec (AERNN variant)
    '''
    def __init__(self, args, num_nodes, dropout_rate):
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
        self.enc = dyngraph2vec_Enc(self.struc_dims, self.temp_dims, self.dropout_rate)
        # Decoder
        # self.dec = dyngraph2vec_Dec(self.dec_dims, self.dropout_rate)

    def forward(self, adj_list, node_list=None, mask_list=None):
        '''
        Rewrite the forward function
        :param adj_list: list of historical adjacency matrices (i.e., input)
        :return: prediction result
        '''
        # ====================
        dyn_emb = self.enc(adj_list)
        # adj_est = self.dec(dyn_emb)
        return dyn_emb
        # return adj_est, dyn_emb

class dyngraph2vec_Enc(nn.Module):
    '''
    Class to define the encoder of dyngraph2vec (AERNN variant)
    '''
    def __init__(self, struc_dims, temp_dims, dropout_rate):
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
        for l in range(self.num_struc_layers):
            self.struc_enc.append(
                nn.Linear(in_features=self.struc_dims[l], out_features=self.struc_dims[l+1]))
        for l in range(self.num_struc_layers):
            self.struc_drop.append(nn.Dropout(p=self.dropout_rate))
        # ==========
        # Temporal encoder
        self.num_temp_layers = len(self.temp_dims)-1 # Numer of LSTM layers
        self.temp_enc = nn.ModuleList()
        for l in range(self.num_temp_layers):
            self.temp_enc.append(
                nn.LSTM(input_size=self.temp_dims[l], hidden_size=self.temp_dims[l+1]))

    def forward(self, adj_list):
        '''
        Rewrite the forward function
        :param adj_list: list of historical adjacency matrices (i.e., input)
        :return: dynamic node embedding
        '''
        # ====================
        adj_list = [adj.to_dense() for adj in adj_list]
        win_size = len(adj_list)  # Window size, i.e., #historical snapshots
        num_nodes, _ = adj_list[0].shape
        # ====================
        # Structural encoder
        temp_input_tnr = None
        for t in range(win_size):
            adj = adj_list[t]
            struc_input = adj
            struc_output = None
            for l in range(self.num_struc_layers):
                struc_output = self.struc_enc[l](struc_input)
                struc_output = self.struc_drop[l](struc_output)
                struc_output = torch.relu(struc_output)
                struc_input = struc_output
            if t==0:
                temp_input_tnr = struc_output
            else:
                temp_input_tnr = torch.cat((temp_input_tnr, struc_output), dim=0)
        # ==========
        # Temporal encoder
        temp_input = torch.reshape(temp_input_tnr, (win_size, int(num_nodes), self.temp_dims[0]))
        temp_output = None
        for l in range(self.num_temp_layers):
            temp_output, _ = self.temp_enc[l](temp_input)
            temp_input = temp_output
        dyn_emb = temp_output[-1, :] # Dynamic node embedding (N*d)

        return dyn_emb

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
