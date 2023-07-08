import argparse
import yaml
import torch
import numpy as np
import time
import random
import math

def pad_with_last_col(matrix,cols):
    out = [matrix]
    pad = [matrix[:,[-1]]] * (cols - matrix.size(1))
    out.extend(pad)
    return torch.cat(out,dim=1)

def pad_with_last_val(vect, k, device):
    # device = 'cuda:0' if vect.is_cuda else 'cpu'
    pad = torch.ones(k - vect.size(0),
                         dtype=torch.long,
                         device = device) * vect[-1]
    vect = torch.cat([vect,pad])
    return vect



def sparse_prepare_tensor(tensor,torch_size, ignore_batch_dim = True):
    if ignore_batch_dim:
        tensor = sp_ignore_batch_dim(tensor)
    tensor = make_sparse_tensor(tensor,
                                tensor_type = 'float',
                                torch_size = torch_size)
    return tensor

def sp_ignore_batch_dim(tensor_dict):
    tensor_dict['idx'] = tensor_dict['idx'][0]
    tensor_dict['vals'] = tensor_dict['vals'][0]
    return tensor_dict

def aggregate_by_time(time_vector,time_win_aggr):
        time_vector = time_vector - time_vector.min()
        time_vector = time_vector // time_win_aggr
        return time_vector

def sort_by_time(data,time_col):
        _, sort = torch.sort(data[:,time_col])
        data = data[sort]
        return data

def print_sp_tensor(sp_tensor,size):
    print(torch.sparse.FloatTensor(sp_tensor['idx'].t(),sp_tensor['vals'],torch.Size([size,size])).to_dense())

def reset_param(t):
    stdv = 2. / math.sqrt(t.size(0))
    t.data.uniform_(-stdv,stdv)

def make_sparse_tensor(adj,tensor_type,torch_size):
    if len(torch_size) == 2:
        tensor_size = torch.Size(torch_size)
    elif len(torch_size) == 1:
        tensor_size = torch.Size(torch_size*2)

    if tensor_type == 'float':
        test = torch.sparse.FloatTensor(adj['idx'].t(),
                                      adj['vals'].type(torch.float),
                                      tensor_size)
        return torch.sparse.FloatTensor(adj['idx'].t(),
                                      adj['vals'].type(torch.float),
                                      tensor_size)
    elif tensor_type == 'long':
        return torch.sparse.LongTensor(adj['idx'].t(),
                                      adj['vals'].type(torch.long),
                                      tensor_size)
    else:
        raise NotImplementedError('only make floats or long sparse tensors')

def sp_to_dict(sp_tensor):
    return  {'idx': sp_tensor._indices().t(),
             'vals': sp_tensor._values()}

class Namespace(object):
    '''
    helps referencing object in a dictionary as dict.key instead of dict['key']
    '''
    def __init__(self, adict):
        self.__dict__.update(adict)

def set_seeds(rank):
    seed = int(time.time())+rank
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def random_param_value(param, param_min, param_max, type='int'):
    if str(param) is None or str(param).lower()=='none':
        if type=='int':
            return random.randrange(param_min, param_max+1)
        elif type=='logscale':
            interval=np.logspace(np.log10(param_min), np.log10(param_max), num=100)
            return np.random.choice(interval,1)[0]
        else:
            return random.uniform(param_min, param_max)
    else:
        return param

def load_data(file):
    with open(file) as file:
        file = file.read().splitlines()
    data = torch.tensor([[float(r) for r in row.split(',')] for row in file[1:]])
    return data

def load_data_from_tar(file, tar_archive, replace_unknow=False, starting_line=1, sep=',', type_fn = float, tensor_const = torch.DoubleTensor):
    f = tar_archive.extractfile(file)
    lines = f.read()#
    lines=lines.decode('utf-8')
    if replace_unknow:
        lines=lines.replace('unknow', '-1')
        lines=lines.replace('-1n', '-1')

    lines=lines.splitlines()

    data = [[type_fn(r) for r in row.split(sep)] for row in lines[starting_line:]]
    data = tensor_const(data)
    #print (file,'data size', data.size())
    return data

def create_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--config_file', '--c', default='experiments/parameters_example.yaml', type=argparse.FileType(mode='r'), help='optional, yaml file containing parameters to be used, overrides command line parameters')
    parser.add_argument('--dataset', '--d',default='', type=str)
    parser.add_argument('--model', '--m',default='', type=str)
    parser.add_argument('--fusion_mode', default='cat', type=str)
    parser.add_argument('--log', default='', type=str)
    parser.add_argument('--gid', default=-1, type=int)
    return parser

def parse_args(parser):
    args = parser.parse_args()
    if args.config_file:
        data = yaml.load(args.config_file, Loader=yaml.FullLoader)
        delattr(args, 'config_file')
        # print(data)
        arg_dict = args.__dict__
        for key, value in data.items():
            arg_dict[key] = value

    # args.learning_rate =random_param_value(args.learning_rate, args.learning_rate_min, args.learning_rate_max, type='logscale')
    # # args.adj_mat_time_window = random_param_value(args.adj_mat_time_window, args.adj_mat_time_window_min, args.adj_mat_time_window_max, type='int')
    # args.num_hist_steps = random_param_value(args.num_hist_steps, args.num_hist_steps_min, args.num_hist_steps_max, type='int')
    # args.gcn_parameters['feats_per_node'] =random_param_value(args.gcn_parameters['feats_per_node'], args.gcn_parameters['feats_per_node_min'], args.gcn_parameters['feats_per_node_max'], type='int')
    # args.gcn_parameters['layer_1_feats'] =random_param_value(args.gcn_parameters['layer_1_feats'], args.gcn_parameters['layer_1_feats_min'], args.gcn_parameters['layer_1_feats_max'], type='int')
    # if args.gcn_parameters['layer_2_feats_same_as_l1'] or args.gcn_parameters['layer_2_feats_same_as_l1'].lower()=='true':
    #     args.gcn_parameters['layer_2_feats'] = args.gcn_parameters['layer_1_feats']
    # else:
    #     args.gcn_parameters['layer_2_feats'] =random_param_value(args.gcn_parameters['layer_2_feats'], args.gcn_parameters['layer_1_feats_min'], args.gcn_parameters['layer_1_feats_max'], type='int')
    # args.gcn_parameters['lstm_l1_feats'] =random_param_value(args.gcn_parameters['lstm_l1_feats'], args.gcn_parameters['lstm_l1_feats_min'], args.gcn_parameters['lstm_l1_feats_max'], type='int')
    # if args.gcn_parameters['lstm_l2_feats_same_as_l1'] or args.gcn_parameters['lstm_l2_feats_same_as_l1'].lower()=='true':
    #     args.gcn_parameters['lstm_l2_feats'] = args.gcn_parameters['lstm_l1_feats']
    # else:
    #     args.gcn_parameters['lstm_l2_feats'] =random_param_value(args.gcn_parameters['lstm_l2_feats'], args.gcn_parameters['lstm_l1_feats_min'], args.gcn_parameters['lstm_l1_feats_max'], type='int')
    # args.gcn_parameters['cls_feats'] =random_param_value(args.gcn_parameters['cls_feats'], args.gcn_parameters['cls_feats_min'], args.gcn_parameters['cls_feats_max'], type='int')

    return args



def get_free_device(gid):
    def get_free_gpu():
        import gpustat
        stats = gpustat.GPUStatCollection.new_query()
        GPU_usage = {stats[ii].entry['index']: stats[ii].entry['memory.used'] for ii in range(len(stats))}
        bestGPU = sorted(GPU_usage.items(), key=lambda item: item[1])[0][0]
        print(f"setGPU: Setting GPU to: {bestGPU}, GPU_usage: {GPU_usage}")
        return bestGPU
    try:
        if gid<0:
            gid = get_free_gpu()
    except:
        gid = get_free_gpu()
    return torch.device(f'cuda:{gid}') if torch.cuda.is_available() else torch.device('cpu')


class EarlyStopMonitor(object):
    def __init__(self, num_val, win_size=5):
        self.vals = [[] for _ in range(num_val)]
        self.num_val = num_val
        self.win_size = win_size
    
    def early_stop_check(self, val_list):
        assert len(val_list) == self.num_val
        for ii in range(len(val_list)):
            self.vals[ii].append(val_list[ii]) # 记录最新值
            
            if len(self.vals[ii])> self.win_size: # 只保留最后win_size个epoch的值
                self.vals[ii] = self.vals[ii][-self.win_size:]

        for lst in self.vals:
            if len(self.vals[ii]) != self.win_size: # 若数量不足win_size，则不stop
                return False
            if not all(a >= b for a, b in zip(lst, lst[1:])): # 若不是所有的元素都下降，则不stop
                return False
        return True # 所有的val都下降了，可以结束


if __name__ == '__main__':
    from IPython import embed
    # early_stop = EarlyStopMonitor(2, win_size=5)
    # lst = [1, 2]
    # early_stop.early_stop_check(lst)
    early_stop = EarlyStopMonitor(2, win_size=5)
    early_stop.vals = [[39.5342, 40.2696, 39.11, 40.1901, 39.7501], [19.4435, 19.8535, 19.0356, 19.9972], 73.8488, 71.7835, 74.]
    
    embed()