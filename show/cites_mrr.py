import torch
import sys
import easydict
import yaml
from collections import Counter, defaultdict
from scipy.sparse import coo_matrix
import numpy as np
from tqdm import tqdm
import argparse
import pandas as pd
from IPython import embed


sys.path.append('../src/')
from engine import trainer
from loader import spliter
from run_exp import build_dataset, build_tasker, build_gcn, build_classifier, build_loss
from utils import utils


def gather_node_embs(nodes_embs,node_indices):
    cls_input = []
    for node_set in node_indices: # [2, e] ->  list([e], [e])
        cls_input.append(nodes_embs[node_set]) # [e, d]
    return torch.cat(cls_input,dim = 1) # [e, 2d]
        
        
def get_row_MRR(probs,true_classes):
    existing_mask = true_classes == 1
    ordered_indices = np.flip(probs.argsort()) #descending in probability
    ordered_existing_mask = existing_mask[ordered_indices]
    existing_ranks = np.arange(1, true_classes.shape[0]+1, dtype=np.float)[ordered_existing_mask]
    MRR = (1/existing_ranks).sum()/existing_ranks.shape[0]
    return MRR


def load_model(model, mpath, device):
    param_dict = torch.load(mpath, map_location=device)
    if not isinstance(param_dict, dict):
        return param_dict
    try:
        model.load_state_dict(param_dict)
    except:
        for key in param_dict:
            try:
                model.state_dict()[key].copy_(param_dict[key])
            except:
                try:
                    model.state_dict()[key.replace('_edge', '')].copy_(param_dict[key])
                except:
                    print(key)
    return model
            

def calc_df(gcn, classifier, gcn_path, cls_path, splitter_, trainer_):
    gcn = load_model(gcn, gcn_path, args.device)
    classifier = load_model(classifier, cls_path, args.device)
    
    cites2mrrs = defaultdict(lambda: [])
    # for ss in tqdm(splitter_.train):
    for ss in tqdm(splitter_.test):
        ss = trainer_.prepare_sample(ss)
        results = gcn(ss.hist_adj_list, ss.hist_ndFeats_list, ss.node_mask_list)
        
        try: nodes_embs, _ = results
        except: nodes_embs, _ = results, 0.0
        
        # predictions = classifier(torch.cat([nodes_embs[node_set] for node_set in ss.label_sp['idx']], dim=1))
        node_indices =  ss.label_sp['idx']
        predict_batch_size = 2048
        predictions=[]
        for i in range(1 +(node_indices.size(1)//predict_batch_size)):
            cls_input = gather_node_embs(nodes_embs, node_indices[:, i*predict_batch_size:(i+1)*predict_batch_size])
            prediction = classifier(cls_input)
            predictions.append(prediction.cpu().detach())
        predictions=torch.cat(predictions, dim=0) # [num_edges, out_dim=2], 二分类问题
        
        probs = torch.softmax(predictions,dim=1)[:,1].cpu().detach().numpy() # 分类为1的概率
        true_classes = ss.label_sp['vals'].cpu().detach().numpy()
        
        adj = ss.label_sp['idx'].cpu().detach().numpy()
        pred_matrix = coo_matrix((probs,(adj[0], adj[1]))).toarray()
        true_matrix = coo_matrix((true_classes,(adj[0],adj[1]))).toarray()
        
        cnts = Counter(ss.label_sp['idx'][1, torch.where(ss.label_sp['vals']==1)[0]].tolist())
        cites = torch.zeros(len(nodes_embs))
        for kk,vv in dict(cnts).items():
            cites[kk] = vv
            
        for ii, pred_row in enumerate(pred_matrix):
            if np.isin(1, true_matrix[ii]):
                cites2mrrs[cites[ii].item()].append( get_row_MRR(pred_row, true_matrix[ii]) )
    
    cites2mrrs = {k: v for k, v in sorted(cites2mrrs.items(), key=lambda item: item[1])}
    cites2mrr = {kk: round(np.array(vv).mean()*100, 4) for kk,vv in cites2mrrs.items()}
    cites2num = {kk: len(vv) for kk,vv in cites2mrrs.items()}
    return list(cites2mrr.keys()), list(cites2mrr.values()), list(cites2num.values())
    
           
def main(args):
    dataset = build_dataset(args)
    tasker = build_tasker(args,dataset)
    splitter_ = spliter.splitter(args,tasker)
    gcn = build_gcn(args, tasker)
    classifier = build_classifier(args,tasker)
    comp_loss = build_loss(args, dataset).to(args.device)
    
    trainer_ = trainer.Trainer(args,
                            splitter = splitter_,
                            gcn = gcn,
                            classifier = classifier,
                            comp_loss = comp_loss,
                            dataset = dataset,
                            num_classes = tasker.num_classes)
    
    gcn_path = f'../ckpt/{args.model}/{args.data}_{args.log}_gcn_{args.epoch}e.pkl'
    cls_path = f'../ckpt/{args.model}/{args.data}_{args.log}_cls_{args.epoch}e.pkl'
    cites, mrr, num = calc_df(gcn, classifier, gcn_path, cls_path, splitter_, trainer_)
    
    df = pd.DataFrame({'cites': cites, 'mrr': mrr, 'num': num})
    df = df.sort_values(by=['cites'])
    
    dic = {'cites': df['cites'].tolist(), 
           'mrrs':  df['mrr'].tolist(), 
           'nums': df['num'].tolist()
           }
    # print('cites = np.array(', df['cites'].tolist(), ')')
    # print('mrr = np.array(', df['mrr'].tolist(), ')')
    # print('num = np.array(', df['num'].tolist(), ')')
    print(f"{args.model} = ", dic)
    print(('python ' + ' '.join(sys.argv)))
    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--config_file', '--c', default='../src/configs/default.yaml', type=argparse.FileType(mode='r'), help='optional, yaml file containing parameters to be used, overrides command line parameters')
    parser.add_argument('--data', '--d',default='bca', type=str)
    parser.add_argument('--model', '--m',default='gcn_lstm', type=str)
    parser.add_argument('--epoch', '--e',default=0, type=int)
    
    parser.add_argument('--fusion_mode', default='cat', type=str)
    parser.add_argument('--alpha', default=1.0, type=float)
    parser.add_argument('--num_hist_steps', '--w', default=10, type=int)
    
    parser.add_argument('--cls_feats', '--f', default=128, type=int)
    parser.add_argument('--learning_rate', '--lr', default=0.005, type=float)
    parser.add_argument('--negative_mult_training', '--k', default=100, type=int)
    
    parser.add_argument('--log', default='cat', type=str)
    parser.add_argument('--gid', default=-1, type=int)
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()
    
    # data = yaml.safe_load(open(args.config_file))/
    data = yaml.load(args.config_file, Loader=yaml.FullLoader)
    for key, value in data.items():
        if key not in args:
            args.__dict__[key] = value
    
    seed=args.seed#+rank
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    args.device = utils.get_free_device(args.gid)
    main(args)