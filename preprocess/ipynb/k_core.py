import dgl
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, remove_self_loops,  from_networkx
import networkx as nx
from tqdm import tqdm, trange
import json
import os
import pickle as pkl
from IPython import embed

def retrieve_name_ex(var):
    frame = sys._getframe(2)
    while(frame):
        for item in frame.f_locals.items():
            if (var is item[1]):
                return item[0]
        frame = frame.f_back
    return ""

def myout(*para, threshold=10):
    def get_mode(var):
        if isinstance(var, (list, dict, set)):
            return 'len'
        elif isinstance(var, (np.ndarray, torch.Tensor)):
            return 'shape'
        else: return ''

    for var in para:
        name = retrieve_name_ex(var)
        mode = get_mode(var)
        if mode=='len':
            len_var = len(var)
            if isinstance(var, list) and len_var>threshold and threshold>6:
                print(f'{name} : len={len_var}, list([{var[0]}, {var[1]}, {var[2]}, ..., {var[-3]}, {var[-2]}, {var[-1]}])')
            elif isinstance(var, set) and len_var>threshold and threshold>6:
                var = list(var)
                print(f'{name} : len={len_var}, set([{var[0]}, {var[1]}, {var[2]}, ..., {var[-3]}, {var[-2]}, {var[-1]}])')
            elif isinstance(var, dict) and len_var>threshold and threshold>6:
                tmp = []
                for kk, vv in var.items():
                    tmp.append(f'{kk}: {vv}')
                    if len(tmp) > threshold: break
                print(f'{name} : len={len_var}, dict([{tmp[0]}, {tmp[1]}, {tmp[2]}, {tmp[3]}, {tmp[4]}, {tmp[5]}, ...])')
            else:
                print(f'{name} : len={len_var}, {var}')
        elif mode=='shape':
            sp = var.shape
            if len(sp)<2:
                print(f'{name} : shape={sp}, {var}')
            else:
                print(f'{name} : shape={sp}')
                print(var)
        else:
            print(f"{name} = {var}")

def core_number(G):
    if nx.number_of_selfloops(G) > 0:
        msg = (
            "Input graph has self loops which is not permitted; "
            "Consider using G.remove_edges_from(nx.selfloop_edges(G))."
        )
        raise ValueError(msg)
    degrees = dict(G.degree())
    # Sort nodes by degree.
    nodes = sorted(degrees, key=degrees.get)
    bin_boundaries = [0]
    curr_degree = 0
    for i, v in enumerate(nodes):
        if degrees[v] > curr_degree:
            bin_boundaries.extend([i] * (degrees[v] - curr_degree))
            curr_degree = degrees[v]
    node_pos = {v: pos for pos, v in enumerate(nodes)}
    # The initial guess for the core number of a node is its degree.
    core = degrees
    nbrs = {v: list(nx.all_neighbors(G, v)) for v in G}
    for v in tqdm(nodes):
        for u in nbrs[v]:
            if core[u] > core[v]:
                nbrs[u].remove(v)
                pos = node_pos[u]
                bin_start = bin_boundaries[core[u]]
                node_pos[u] = bin_start
                node_pos[nodes[bin_start]] = pos
                nodes[bin_start], nodes[pos] = nodes[pos], nodes[bin_start]
                bin_boundaries[core[u]] += 1
                core[u] -= 1
    return core


def main(dataset, k=30, start=2000, end=2022):
    gpath = f'../data/{dataset}/nx_graph_{start}_{end}.pkl'
    if os.path.exists(gpath):
        print(f'Loading G from {gpath}')
        G = pkl.load(open(gpath, 'rb'))
    else:
        graph = dgl.load_graphs(f'../../../62_seal_hints/preprocess/data/{dataset}/raw_graph.bin')[0][0]
        print(dataset, graph.edata['ts'].unique())
        
        ts_eids = graph.filter_edges(lambda x: (x.data['ts']>=start) & (x.data['ts']<end))
        ts_graph = dgl.edge_subgraph(graph, ts_eids)
        print(ts_graph)

        g = Data(feat=ts_graph.ndata['feat'], edge_index=torch.stack(ts_graph.edges()), ts=ts_graph.edata['ts'], \
            raw_nid=ts_graph.ndata['raw_nid'])
        G = to_networkx(g, node_attrs=['feat', 'raw_nid'], edge_attrs=['ts'], remove_self_loops=True)
        pkl.dump(G, open(gpath, 'wb'))
        
    
    cpath = f'../data/{dataset}/core_dict_{start}_{end}.json'
    if os.path.exists(cpath):
        print(f'Loading core_dict from {cpath}')
        core_dict = json.load(open(cpath, 'r'))
        core_dict = {int(kk): vv for kk,vv in core_dict.items()}
    else:
        core_dict = core_number(G)
        json.dump(core_dict, open(cpath, 'w'))
        
    g_core = nx.k_core(G, core_number=core_dict, k=k)
    core_dg = dgl.from_networkx(g_core, node_attrs=['feat', 'raw_nid'], edge_attrs=['ts'])
    print(core_dg, core_dg.edata['ts'].unique())
    embed()
    dgl.save_graphs(f'../data/{dataset}/{k}-core_graph.bin', [core_dg])


if __name__ == '__main__':
    k = 33
    main('dblp', k, 2000, 2021)
    # main('aps', k, 2000, 2020)
    # main('aminer', k, 2000, 2020)
    