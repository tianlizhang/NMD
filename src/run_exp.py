from utils import utils
import torch
import torch.distributed as dist
import numpy as np
import random
from IPython import embed
import argparse
import yaml
import os

from loader import link_pred_tasker, spliter
from EvolveGCN import models


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

def build_random_hyper_params(args):
	if args.model == 'all':
		model_types = ['gcn', 'egcn_o', 'egcn_h', 'gruA', 'gruB','egcn','lstmA', 'lstmB']
		args.model=model_types[args.rank]
	elif args.model == 'all_nogcn':
		model_types = ['egcn_o', 'egcn_h', 'gruA', 'gruB','egcn','lstmA', 'lstmB']
		args.model=model_types[args.rank]
	elif args.model == 'all_noegcn3':
		model_types = ['gcn', 'egcn_h', 'gruA', 'gruB','egcn','lstmA', 'lstmB']
		args.model=model_types[args.rank]
	elif args.model == 'all_nogruA':
		model_types = ['gcn', 'egcn_o', 'egcn_h', 'gruB','egcn','lstmA', 'lstmB']
		args.model=model_types[args.rank]
		args.model=model_types[args.rank]
	elif args.model == 'saveembs':
		model_types = ['gcn', 'gcn', 'skipgcn', 'skipgcn']
		args.model=model_types[args.rank]

	args.learning_rate =random_param_value(args.learning_rate, args.learning_rate_min, args.learning_rate_max, type='logscale')
	# args.adj_mat_time_window = random_param_value(args.adj_mat_time_window, args.adj_mat_time_window_min, args.adj_mat_time_window_max, type='int')

	if args.model == 'gcn':
		args.num_hist_steps = 0
	else:
		args.num_hist_steps = random_param_value(args.num_hist_steps, args.num_hist_steps_min, args.num_hist_steps_max, type='int')

	args.gcn_parameters['feats_per_node'] =random_param_value(args.gcn_parameters['feats_per_node'], args.gcn_parameters['feats_per_node_min'], args.gcn_parameters['feats_per_node_max'], type='int')
	args.gcn_parameters['layer_1_feats'] =random_param_value(args.gcn_parameters['layer_1_feats'], args.gcn_parameters['layer_1_feats_min'], args.gcn_parameters['layer_1_feats_max'], type='int')
	if args.gcn_parameters['layer_2_feats_same_as_l1'] or args.gcn_parameters['layer_2_feats_same_as_l1'].lower()=='true':
		args.gcn_parameters['layer_2_feats'] = args.gcn_parameters['layer_1_feats']
	else:
		args.gcn_parameters['layer_2_feats'] =random_param_value(args.gcn_parameters['layer_2_feats'], args.gcn_parameters['layer_1_feats_min'], args.gcn_parameters['layer_1_feats_max'], type='int')
	args.gcn_parameters['lstm_l1_feats'] =random_param_value(args.gcn_parameters['lstm_l1_feats'], args.gcn_parameters['lstm_l1_feats_min'], args.gcn_parameters['lstm_l1_feats_max'], type='int')
	if args.gcn_parameters['lstm_l2_feats_same_as_l1'] or args.gcn_parameters['lstm_l2_feats_same_as_l1'].lower()=='true':
		args.gcn_parameters['lstm_l2_feats'] = args.gcn_parameters['lstm_l1_feats']
	else:
		args.gcn_parameters['lstm_l2_feats'] =random_param_value(args.gcn_parameters['lstm_l2_feats'], args.gcn_parameters['lstm_l1_feats_min'], args.gcn_parameters['lstm_l1_feats_max'], type='int')
	args.gcn_parameters['cls_feats']=random_param_value(args.gcn_parameters['cls_feats'], args.gcn_parameters['cls_feats_min'], args.gcn_parameters['cls_feats_max'], type='int')
	return args

def build_dataset(args):
	if args.data == 'bco' or args.data == 'bca':
		from dataset import bitcoin_dl
		if args.data == 'bco':
			args.bitcoin_args = args.bitcoinotc_args
		elif args.data == 'bca':
			args.bitcoin_args = args.bitcoinalpha_args
		return bitcoin_dl.bitcoin_dataset(args)
	# elif args.data == 'aml_sim':
	# 	return aml.Aml_Dataset(args)
	# elif args.data == 'elliptic':
	# 	return ell.Elliptic_Dataset(args)
	# elif args.data == 'elliptic_temporal':
	# 	return ell_temp.Elliptic_Temporal_Dataset(args)
	elif args.data == 'uci':
		from dataset import uci_dl
		return uci_dl.Uc_Irvine_Message_Dataset(args)
	# elif args.data == 'dbg':
	# 	return dbg.dbg_dataset(args)
	# elif args.data == 'colored_graph':
	# 	return cg.Colored_Graph(args)
	elif args.data == 'as':
		from dataset import auto_syst_dl
		return auto_syst_dl.Autonomous_Systems_Dataset(args)
	elif args.data == 'dblp':
		from dataset import dblp_dl
		return dblp_dl.DBLP_Dataset(args)
	elif args.data == 'aps':
		args.dblp_args = args.aps_args
		from dataset import dblp_dl
		return dblp_dl.DBLP_Dataset(args)
	# elif args.data == 'reddit':
	# 	return rdt.Reddit_Dataset(args)
	# elif args.data.startswith('sbm'):
	# 	from dataset import sbm_dl
	# 	if args.data == 'sbm20':
	# 		args.sbm_args = args.sbm20_args
	# 	elif args.data == 'sbm50':
	# 		args.sbm_args = args.sbm50_args
	# 	return sbm_dl.sbm_dataset(args)
	else:
		raise NotImplementedError('only arxiv has been implemented')

def build_tasker(args,dataset):
	if args.task == 'link_pred':
		return link_pred_tasker.Link_Pred_Tasker(args, dataset)
	elif args.task == 'edge_cls':
		from loader import edge_cls_tasker
		return edge_cls_tasker.Edge_Cls_Tasker(args,dataset)
	# elif args.task == 'node_cls':
	# 	return nct.Node_Cls_Tasker(args,dataset)
	# elif args.task == 'static_node_cls':
	# 	return nct.Static_Node_Cls_Tasker(args,dataset)
	else:
		raise NotImplementedError('still need to implement the other tasks')

# def build_gcn(args,tasker):
#     gcn_args = utils.Namespace(args.gcn_parameters)
#     gcn_args.feats_per_node = tasker.feats_per_node
#     if args.model == 'gcn':
#         return models.Sp_GCN(gcn_args,activation = torch.nn.RReLU()).to(args.device)
#     elif args.model == 'skipgcn':
#         return models.Sp_Skip_GCN(gcn_args,activation = torch.nn.RReLU()).to(args.device)
#     elif args.model == 'skipfeatsgcn':
#         return models.Sp_Skip_NodeFeats_GCN(gcn_args,activation = torch.nn.RReLU()).to(args.device)
#     else:
#         assert args.num_hist_steps > 0, 'more than one step is necessary to train LSTM'
#         if args.model == 'lstmA':
#             return models.Sp_GCN_LSTM_A(gcn_args,activation = torch.nn.RReLU()).to(args.device)
#         elif args.model == 'gruA':
#             return models.Sp_GCN_GRU_A(gcn_args,activation = torch.nn.RReLU()).to(args.device)
#         elif args.model == 'lstmB':
#             return models.Sp_GCN_LSTM_B(gcn_args,activation = torch.nn.RReLU()).to(args.device)
#         elif args.model == 'gruB':
#             return models.Sp_GCN_GRU_B(gcn_args,activation = torch.nn.RReLU()).to(args.device)
#         elif args.model == 'egcn_h':
#             return egcn_h.EGCN(gcn_args, activation = torch.nn.RReLU(), device = args.device)
#         elif args.model == 'skipfeatsegcn_h':
#             return egcn_h.EGCN(gcn_args, activation = torch.nn.RReLU(), device = args.device, skipfeats=True)
#         elif args.model == 'egcn_o':
#             return egcn_o.EGCN(gcn_args, activation = torch.nn.RReLU(), device = args.device)
#         # elif args.model == 'my_net':
#         # 	return my_net.EGCN(gcn_args, activation = torch.nn.RReLU(), device = args.device)
#         elif args.model == 'stgsn':
#             from STGSN import modules
#             return modules.STGSN(gcn_args, dropout_rate = 0.2).to(args.device)
#         elif args.model == 'stgsn_tim':
#             from STGSN import my_net
#             return my_net.STGSN(gcn_args, dropout_rate = 0.2).to(args.device)
#         elif args.model == 'ddne':
#             from DDNE import modules
#             return modules.DDNE(gcn_args, args.num_hist_steps, tasker.data.num_nodes, dropout_rate = 0.2).to(args.device)
#         elif args.model == 'ddne_tim':
#             from DDNE import my_net
#             return my_net.DDNE(gcn_args, args.num_hist_steps, tasker.data.num_nodes, dropout_rate = 0.2).to(args.device)
#         elif args.model == 'd2v':
#             from dyngraph2vec import modules, my_net
#             return modules.dyngraph2vec(gcn_args, tasker.data.num_nodes, dropout_rate = 0.2).to(args.device)
#             # return my_net.dyngraph2vec(gcn_args, tasker.data.num_nodes, dropout_rate = 0.2).to(args.device)
#         elif args.model == 'gcn_gan':
#             from GCN_GAN import modules
#             return [modules.GCN_GAN(gcn_args, tasker.data.num_nodes, dropout_rate = 0.2).to(args.device), \
#             modules.DiscNet(gcn_args, tasker.data.num_nodes, dropout_rate = 0.2).to(args.device)]
#             # return my_net.dyngraph2vec(gcn_args, tasker.data.num_nodes, dropout_rate = 0.2).to(args.device)
#         else:
#             raise NotImplementedError('need to finish modifying the models')
			# return my_net.EGCN(gcn_args, activation = torch.nn.RReLU(), device = args.device)

def build_gcn(args, tasker):
    assert args.num_hist_steps > 0, 'more than one step is necessary to train LSTM'
    if args.model == 'gcn_lstm':
        gcn_args = utils.Namespace(args.gcn_parameters)
        gcn_args.feats_per_node = tasker.feats_per_node
        return models.Sp_GCN_LSTM_A(gcn_args, activation = torch.nn.RReLU()).to(args.device)
    elif args.model == 'gcn_lstm_tim':
        gcn_args = utils.Namespace(args.gcn_parameters)
        gcn_args.feats_per_node = tasker.feats_per_node
        return models.Sp_GCN_LSTM_A_tim(gcn_args, args.fusion_mode, activation = torch.nn.RReLU()).to(args.device)
    elif args.model == 'gcn_lstm_tim_shared':
        gcn_args = utils.Namespace(args.gcn_parameters)
        gcn_args.feats_per_node = tasker.feats_per_node
        return models.Sp_GCN_LSTM_A_tim_shared(gcn_args, args.fusion_mode, activation = torch.nn.RReLU()).to(args.device)
    elif args.model == 'gcn_lstm_tim_fuse2':
        gcn_args = utils.Namespace(args.gcn_parameters)
        gcn_args.feats_per_node = tasker.feats_per_node
        return models.Sp_GCN_LSTM_A_tim_fuse2(gcn_args, args.fusion_mode, activation = torch.nn.RReLU()).to(args.device)
    
    elif args.model == 'd2v':
        gcn_args = utils.Namespace(args.d2v_parameters)
        from dyngraph2vec import modules, my_net
        return modules.dyngraph2vec(gcn_args, tasker.data.num_nodes, dropout_rate = 0.2).to(args.device)
    elif args.model == 'd2v_tim':
        gcn_args = utils.Namespace(args.d2v_parameters)
        from dyngraph2vec import my_net
        return my_net.dyngraph2vec(gcn_args, tasker.data.num_nodes, args.fusion_mode, dropout_rate = 0.2).to(args.device)
    elif args.model == 'd2v_tim_shared':
        gcn_args = utils.Namespace(args.d2v_parameters)
        from dyngraph2vec import shared_SE
        return shared_SE.dyngraph2vec(gcn_args, tasker.data.num_nodes, args.fusion_mode, dropout_rate = 0.2).to(args.device)
    elif args.model == 'd2v_tim_fuse2':
        gcn_args = utils.Namespace(args.d2v_parameters)
        from dyngraph2vec import fuse2
        return fuse2.dyngraph2vec(gcn_args, tasker.data.num_nodes, args.fusion_mode, dropout_rate = 0.2).to(args.device)
    
    elif args.model == 'ddne':
        gcn_args = utils.Namespace(args.ddne_parameters)
        from DDNE import modules
        return modules.DDNE(gcn_args, args.num_hist_steps, tasker.data.num_nodes, dropout_rate = 0.2).to(args.device)
    elif args.model == 'ddne_tim':
        gcn_args = utils.Namespace(args.ddne_parameters)
        from DDNE import my_net
        return my_net.DDNE(gcn_args, args.num_hist_steps, tasker.data.num_nodes, args.fusion_mode, dropout_rate = 0.2).to(args.device)
    elif args.model == 'ddne_tim_shared':
        gcn_args = utils.Namespace(args.ddne_parameters)
        from DDNE import shared_SE
        return shared_SE.DDNE(gcn_args, args.num_hist_steps, tasker.data.num_nodes, args.fusion_mode, dropout_rate = 0.2).to(args.device)
    elif args.model == 'ddne_tim_fuse2':
        gcn_args = utils.Namespace(args.ddne_parameters)
        from DDNE import fuse2
        return fuse2.DDNE(gcn_args, args.num_hist_steps, tasker.data.num_nodes, args.fusion_mode, dropout_rate = 0.2).to(args.device)
    
    
    elif args.model == 'egcn':
        gcn_args = utils.Namespace(args.egcn_parameters)
        gcn_args.feats_per_node = tasker.feats_per_node
        from EvolveGCN import egcn_h
        return egcn_h.EGCN(gcn_args, activation = torch.nn.RReLU(), device = args.device)
    elif args.model == 'egcn_tim':
        gcn_args = utils.Namespace(args.egcn_parameters)
        gcn_args.feats_per_node = tasker.feats_per_node
        from EvolveGCN import my_net
        return my_net.EGCN(gcn_args, args.fusion_mode, activation = torch.nn.RReLU(), device = args.device)
    elif args.model == 'egcn_tim_shared':
        gcn_args = utils.Namespace(args.egcn_parameters)
        gcn_args.feats_per_node = tasker.feats_per_node
        from EvolveGCN import shared_SE
        return shared_SE.EGCN(gcn_args, args.fusion_mode, activation = torch.nn.RReLU(), device = args.device)
    elif args.model == 'egcn_tim_fuse2':
        gcn_args = utils.Namespace(args.egcn_parameters)
        gcn_args.feats_per_node = tasker.feats_per_node
        from EvolveGCN import fuse2
        return fuse2.EGCN(gcn_args, args.fusion_mode, activation = torch.nn.RReLU(), device = args.device)
    
    elif args.model == 'stgsn':
        gcn_args = utils.Namespace(args.stgsn_parameters)
        gcn_args.feats_per_node = tasker.feats_per_node
        from STGSN import modules
        return modules.STGSN(gcn_args, dropout_rate = 0.2).to(args.device)
    elif args.model == 'stgsn_tim':
        gcn_args = utils.Namespace(args.stgsn_parameters)
        gcn_args.feats_per_node = tasker.feats_per_node
        from STGSN import my_net
        return my_net.STGSN(gcn_args, args.fusion_mode, dropout_rate = 0.2).to(args.device)
    elif args.model == 'stgsn_tim_shared':
        gcn_args = utils.Namespace(args.stgsn_parameters)
        gcn_args.feats_per_node = tasker.feats_per_node
        from STGSN import shared_SE
        return shared_SE.STGSN(gcn_args, args.fusion_mode, dropout_rate = 0.2).to(args.device)
    elif args.model == 'stgsn_tim_fuse2':
        gcn_args = utils.Namespace(args.stgsn_parameters)
        gcn_args.feats_per_node = tasker.feats_per_node
        from STGSN import fuse2
        return fuse2.STGSN(gcn_args, args.fusion_mode, dropout_rate = 0.2).to(args.device)
    
    elif args.model == 'gcn_gan':
        from GCN_GAN import modules
        return [modules.GCN_GAN(gcn_args, tasker.data.num_nodes, dropout_rate = 0.2).to(args.device), \
        modules.DiscNet(gcn_args, tasker.data.num_nodes, dropout_rate = 0.2).to(args.device)]
        # return my_net.dyngraph2vec(gcn_args, tasker.data.num_nodes, dropout_rate = 0.2).to(args.device)
    else:
        raise NotImplementedError('need to finish modifying the models')
        
        
def build_classifier(args,tasker):
	if 'node_cls' == args.task or 'static_node_cls' == args.task:
		mult = 1
	else:
		mult = 2
	if 'gru' in args.model or 'lstm' in args.model:
		in_feats = args.gcn_parameters['lstm_l2_feats'] * mult
	elif 'ddne' in args.model:
		in_feats = args.ddne_parameters['enc_dims'][-1] * mult * (args.num_hist_steps+1) * 2
	elif 'd2v' in args.model:
		in_feats = args.d2v_parameters['temp_dims'][-1] * mult
	elif 'stgsn' in args.model:    
		in_feats = args.stgsn_parameters['enc_dims'][-1] * mult * 2 # global + local
	elif 'egcn' in args.model:  
		in_feats = args.egcn_parameters['layer_2_feats'] * mult
	else:
		in_feats = args.gcn_parameters['layer_2_feats'] * mult

	return models.Classifier(args, in_features = in_feats, out_features = tasker.num_classes).to(args.device)
    
    
def build_loss(args, dataset):
    # if 'egcn' in args.model:
    #     from EvolveGCN import loss
    #     return loss.Cross_Entropy(args, dataset)
    # elif args.model == 'stgsn':
    #     from EvolveGCN import loss
    #     return loss.Cross_Entropy(args, dataset)
    # elif args.model == 'ddne':
    #     from DDNE import loss
    #     return loss.DDNE_loss(args)
    # elif args.model == 'd2v':
    if args.model == 'gcn_gan':
        from GCN_GAN import loss
        return loss.gcn_gan_loss(args)
    else:
        from EvolveGCN import loss
        return loss.Cross_Entropy(args, dataset)
        # from dyngraph2vec import loss
        # return loss.d2v_loss(args)
        # from STGSN import loss
        # return loss.STGSN_loss()


def main():
    # Assign the requested random hyper parameters
    # args = build_random_hyper_params(args)

    #build the dataset
    dataset = build_dataset(args)
    #build the tasker
    tasker = build_tasker(args,dataset)
    #build the splitter
    splitter_ = spliter.splitter(args,tasker)
    #build the models
    gcn = build_gcn(args, tasker)
    classifier = build_classifier(args,tasker)
    # build a loss
    comp_loss = build_loss(args, dataset).to(args.device)

    #trainer
    if args.model == 'gcn_gan':
        from GCN_GAN import trainer
    # elif args.model == 'd2v':
    #     # from dyngraph2vec import trainer
    #     from engine import trainer
    else:
        from engine import trainer
        
    trainer = trainer.Trainer(args,
                            splitter = splitter_,
                            gcn = gcn,
                            classifier = classifier,
                            comp_loss = comp_loss,
                            dataset = dataset,
                            num_classes = tasker.num_classes)
    trainer.train()
    # if args.save:
    #     if not os.path.exists(f'../ckpt/{args.model}'):
    #         os.mkdir(f'../ckpt/{args.model}')
    #     spath_gcn = f'../ckpt/{args.model}/{args.data}_{args.fusion_mode}_gcn.pkl'
    #     spath_cls = f'../ckpt/{args.model}/{args.data}_{args.fusion_mode}_cls.pkl'
    #     torch.save(gcn.state_dict(), spath_gcn)
    #     torch.save(classifier.state_dict(), spath_cls)


if __name__ == '__main__':
    # parser = utils.create_parser()
    # args = utils.parse_args(parser)
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--config_file', '--c', default='configs/default.yaml', type=argparse.FileType(mode='r'), help='optional, yaml file containing parameters to be used, overrides command line parameters')
    parser.add_argument('--data', '--d',default='bca', type=str)
    parser.add_argument('--model', '--m',default='gcn_lstm', type=str)
    
    parser.add_argument('--fusion_mode', default='cat', type=str)
    parser.add_argument('--alpha', default=1.0, type=float)
    parser.add_argument('--num_hist_steps', '--w', default=10, type=int)
    
    parser.add_argument('--cls_feats', '--f', default=128, type=int)
    parser.add_argument('--learning_rate', '--lr', default=0.005, type=float)
    parser.add_argument('--negative_mult_training', '--k', default=100, type=int)
    
    parser.add_argument('--log', default='', type=str)
    parser.add_argument('--gid', default=-1, type=int)
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()
    
    if args.config_file:
        data = yaml.load(args.config_file, Loader=yaml.FullLoader)
        delattr(args, 'config_file')
        arg_dict = args.__dict__
        # embed()
        for key, value in data.items():
            if key not in arg_dict:
                arg_dict[key] = value
            
    args.use_logfile = False
    if args.log:
        args.use_logfile = True
    
    global rank, wsize, use_cuda
    args.use_cuda = (torch.cuda.is_available() and args.use_cuda)
    args.device = utils.get_free_device(args.gid)

    print ("use CUDA:", args.use_cuda, "- device:", args.device)
    try:
        dist.init_process_group(backend='mpi') #, world_size=4
        rank = dist.get_rank()
        wsize = dist.get_world_size()
        print('Hello from process {} (out of {})'.format(dist.get_rank(), dist.get_world_size()))
        if args.use_cuda:
            torch.cuda.set_device(rank )  # are we sure of the rank+1????
            print('using the device {}'.format(torch.cuda.current_device()))
    except:
        rank = 0
        wsize = 1
        print(('MPI backend not preset. Set process rank to {} (out of {})'.format(rank, wsize)))

    if args.seed is None and args.seed!='None':
        seed = 123+rank#int(time.time())+rank
    else:
        seed=args.seed#+rank
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    args.seed=seed
    args.rank=rank
    args.wsize=wsize
    
    main()

