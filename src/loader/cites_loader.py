from collections import defaultdict
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from IPython import embed


def label_normorlization(labels):
    maximum = labels.max()
    minimum = labels.min()
    new_value = (labels-minimum)/(maximum-minimum)
    return new_value, (maximum,minimum)


def build_cites(adj_list, pred_ts=1):
    ts_nid2cites = {}
    for ts in range(len(adj_list)):
        tgt_lst = adj_list[ts]._indices()[1, :].tolist()
        
        ts_nid2cites[ts] = defaultdict(int)
        for tgt in tgt_lst:
            ts_nid2cites[ts][tgt] += 1

    cites_infos = []
    for ts in range(len(adj_list)-1):
        nids = adj_list[ts]._indices()[0, :].unique().tolist()
        
        pdf = pd.DataFrame({'nid': nids})
        for yy in range(ts+1, len(adj_list)):
            cdf = pd.DataFrame({'nid': list(ts_nid2cites[yy].keys()), str(yy): list(ts_nid2cites[yy].values())})
            cdf[str(yy)] = cdf[str(yy)].astype('float32')
            
            pdf = pd.merge(pdf, cdf, how='left', on='nid')
        pdf.fillna(0, inplace=True)
        
        cites = pdf.iloc[:, 1:1+pred_ts].values
        cites_cum_log = np.log(cites.cumsum(axis=1) + 1)
        cites_norm, _ = label_normorlization(cites_cum_log)
        
        sids_in_ngh = pdf['nid'].tolist()
        key_nodes = torch.tensor(sids_in_ngh).unique()  # 按照label的顺序排列成
        sid2pos = -torch.ones(max(key_nodes) + 1, dtype=torch.long) # [108073]
        sid2pos[key_nodes] = torch.arange(len(key_nodes))

        cites_infos.append({'cites_norm': torch.from_numpy(cites_norm), 'sid2pos': sid2pos})
    return cites_infos


class FcPredictor(nn.Module):
    def __init__(self, hidden_channels, hidden_dims = [20, 10, 20, 8], node_year=5):
        super(FcPredictor, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Linear(hidden_channels, hidden_dims[0]), 
            nn.ReLU(), 
            nn.Linear(hidden_dims[0], hidden_dims[1])
        ))
        self.linear1 = nn.Sequential(*modules)

        modules = []
        modules.append(nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[2]), 
            nn.ReLU(), 
            nn.Linear(hidden_dims[2], hidden_dims[3]),
            nn.ReLU(),
            nn.Linear(hidden_dims[3], node_year),
        ))
        self.linear2 = nn.Sequential(*modules)

    def forward(self, c):
        z = self.linear1(c) # 128 -> 64 -> 64
        pred = self.linear2(z) # 64->32->16->5
        return pred