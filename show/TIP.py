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
from sklearn.linear_model import LogisticRegression, LinearRegression


sys.path.append('../src/')
from engine import trainer
from loader import spliter
from run_exp import build_dataset, build_tasker, build_gcn, build_classifier, build_loss
from utils import utils


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
    

def label_normorlization(labels):
    maximum = labels.max()
    minimum = labels.min()
    new_value = (labels-minimum)/(maximum-minimum)
    return new_value, (maximum,minimum)

       
def get_X_y(dataloader, trainer_, gcn, mode = 'list'):
    X_lst, y_lst = [], []
    for ss in tqdm(dataloader):
        ss = trainer_.prepare_sample(ss)
        results = gcn(ss.hist_adj_list, ss.hist_ndFeats_list, ss.node_mask_list)
        
        try: nodes_embs, _ = results
        except: nodes_embs, _ = results, 0.0
        
        cnts = Counter(ss.label_sp['idx'][1, torch.where(ss.label_sp['vals']==1)[0]].tolist())
        cites = torch.zeros(len(nodes_embs))
        for kk,vv in dict(cnts).items():
            cites[kk] = vv
        
        # if args.norm:
        cites_log = np.log(cites+1)
        cites_norm, _ = label_normorlization(cites_log)
        
        X_lst.append(nodes_embs.detach().cpu().numpy())
        y_lst.append(cites_norm)
    if mode == 'array':
        return np.vstack(X_lst), np.hstack(y_lst) # trainL 0.2607722697547187, test_score: 0.043349578149195045
    else:
        return X_lst, y_lst
        
        

def TIP_score_old(gcn, classifier, gcn_path, cls_path, splitter_, trainer_):
    gcn = load_model(gcn, gcn_path, args.device)
    # classifier = load_model(classifier, cls_path, args.device)
    
    X_train, y_train = get_X_y(splitter_.train, trainer_, gcn, mode='array')
    X_test, y_test = get_X_y(splitter_.test, trainer_, gcn, mode='array')
    
    sk_model = LinearRegression()
    # embed()
    sk_model.fit(X_train, y_train)
    test_score = sk_model.score(X_test, y_test)
    train_score = sk_model.score(X_train, y_train)
    
    return train_score, test_score


def TIP_score(gcn, classifier, gcn_path, cls_path, splitter_, trainer_):
    gcn = load_model(gcn, gcn_path, args.device)
    # classifier = load_model(classifier, cls_path, args.device)
    
    # X_train, y_train = get_X_y(splitter_.train, trainer_, gcn, mode='array')
    X_lst, y_lst = get_X_y(splitter_.test, trainer_, gcn, mode='list')
    scores = []
    for ii in range(len(X_lst)):
        X_train, y_train = X_lst[ii], y_lst[ii]
        sk_model = LinearRegression()
        sk_model.fit(X_train, y_train)
        train_score = sk_model.score(X_train, y_train)
        scores.append(train_score)
    return round(np.mean(np.array(scores))*100, 3)

      
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
    # train_score, test_score = TIP_score(gcn, classifier, gcn_path, cls_path, splitter_, trainer_)
    # print(f'train_score: {train_score}, test_score: {test_score}')
    
    train_score = TIP_score(gcn, classifier, gcn_path, cls_path, splitter_, trainer_)
    print('score:', train_score) # 0.4799, 0.5482
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