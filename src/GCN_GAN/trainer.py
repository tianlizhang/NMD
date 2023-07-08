import torch
# import utils as u
from utils import utils, logger, taskers_utils
# import logger
import time
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from IPython import embed

def get_gnn_sup(adj):
    '''
    Function to get GNN support (normalized adjacency matrix w/ self-connected edges)
    :param adj: original adjacency matrix
    :return: GNN support
    '''
    # ====================
    num_nodes = len(adj)
    adj = adj + taskers_utils.make_sparse_eye(num_nodes).to(adj.device)
    degs = torch.sqrt(torch.sparse.sum(adj, dim=1).to_dense())
    
    idx = adj._indices()
    vals = adj._values()
    vals = vals / degs[idx[0]]
    vals = vals / degs[idx[1]]
    return torch.sparse.FloatTensor(idx, vals, adj.shape).float().to(adj.device)


class Trainer():
	def __init__(self, args, splitter, gcn, classifier, comp_loss, dataset, num_classes):
		self.args = args
		self.splitter = splitter
		self.tasker = splitter.tasker
		self.gcn = gcn[0]
		self.disc_net = gcn[1]
		self.classifier = classifier
		self.comp_loss = comp_loss

		self.num_nodes = dataset.num_nodes
		self.data = dataset
		self.num_classes = num_classes

		self.logger = logger.Logger(args, self.num_classes)
		self.grad_clip = 0.01
		self.init_optimizers(args)

		if self.tasker.is_static:
			adj_matrix = utils.sparse_prepare_tensor(self.tasker.adj_matrix, torch_size = [self.num_nodes], ignore_batch_dim = False)
			self.hist_adj_list = [adj_matrix]
			self.hist_ndFeats_list = [self.tasker.nodes_feats.float()]

	def init_optimizers(self,args):
		params = self.gcn.parameters()
		self.gcn_opt = torch.optim.Adam(params, lr = args.learning_rate)
		params = self.classifier.parameters()
		self.classifier_opt = torch.optim.Adam(params, lr = args.learning_rate)
		self.disc_opt = torch.optim.Adam(self.disc_net.parameters(), lr = args.learning_rate)
		self.gcn_opt.zero_grad()
		self.classifier_opt.zero_grad()
		self.disc_opt.zero_grad()

	def save_checkpoint(self, state, filename='checkpoint.pth.tar'):
		torch.save(state, filename)

	def load_checkpoint(self, filename, model):
		if os.path.isfile(filename):
			print("=> loading checkpoint '{}'".format(filename))
			checkpoint = torch.load(filename)
			epoch = checkpoint['epoch']
			self.gcn.load_state_dict(checkpoint['gcn_dict'])
			self.classifier.load_state_dict(checkpoint['classifier_dict'])
			self.gcn_opt.load_state_dict(checkpoint['gcn_optimizer'])
			self.classifier_opt.load_state_dict(checkpoint['classifier_optimizer'])
			self.logger.log_str("=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']))
			return epoch
		else:
			self.logger.log_str("=> no checkpoint found at '{}'".format(filename))
			return 0

	def train(self):
		self.tr_step = 0

		for e in range(self.args.num_epochs):
			eval_train, nodes_embs = self.run_epoch(self.splitter.train, e, 'TRAIN', grad = True)
			if len(self.splitter.dev)>0 and e>self.args.eval_after_epochs: # eval_after_epochs = 5
				eval_valid, _ = self.run_epoch(self.splitter.dev, e, 'VALID', grad = False)
				eval_test, _ = self.run_epoch(self.splitter.test, e, 'TEST', grad = False)

				to_print = f'epoch: {e}: train: {eval_train}, valid: {eval_valid}, test: {eval_test}'
				print(to_print)
				self.logger.log_text(to_print)
    
				if self.args.save_node_embeddings:
					self.save_node_embs_csv(nodes_embs, self.splitter.train_idx, log_file+'_train_nodeembs.csv.gz')
					self.save_node_embs_csv(nodes_embs, self.splitter.dev_idx, log_file+'_valid_nodeembs.csv.gz')
					self.save_node_embs_csv(nodes_embs, self.splitter.test_idx, log_file+'_test_nodeembs.csv.gz')


	def run_epoch(self, split, epoch, set_name, grad):
		log_interval=999
		if set_name=='TEST':
			log_interval=1
		self.logger.log_epoch_start(epoch, len(split), set_name, minibatch_log_interval=log_interval)

		torch.set_grad_enabled(grad)
		for s in tqdm(split):
			if self.tasker.is_static:
				s = self.prepare_static_sample(s)
			else:
				s = self.prepare_sample(s)
            
			loss, predictions, nodes_embs = self.predict(s, grad)        
			# loss = self.comp_loss(predictions, s.label_sp['vals']) + node_loss
			# self.logger.log_minibatch(predictions, s.label_sp['vals'], loss.detach(), adj = s.label_sp['idx'])
			if set_name in ['TEST', 'VALID'] and self.args.task == 'link_pred':
				self.logger.log_minibatch(predictions, s.label_sp['vals'], loss.detach(), adj = s.label_sp['idx'])
			else:
				self.logger.log_minibatch(predictions.detach(), s.label_sp['vals'], loss.detach(), adj = s.label_sp['idx'])
			# if grad:
			# 	self.optim_step(loss)

		torch.set_grad_enabled(True)
		eval_measure = self.logger.log_epoch_done(epoch)

		return eval_measure, nodes_embs

	def predict(self, sample, grad):
		num_nodes = len(sample.hist_adj_list[0])
		sup_list = [] # List of GNN support (tensor)
		noise_list = [] # List of noise input
        
		for t in range(len(sample.hist_adj_list)):
			sup_tnr = get_gnn_sup(sample.hist_adj_list[t])
			sup_list.append(sup_tnr)
            # =========
            # Generate random noise
			noise_feat = torch.rand([num_nodes, sample.hist_ndFeats_list[0].shape[1]])
			noise_feat_tnr = torch.FloatTensor(noise_feat).to(sample.hist_adj_list[0].device)
			noise_list.append(noise_feat_tnr)

		num_nodes, _ = noise_list[0].shape
        # adj_list: list([np.array(num_nodes, num_nodes)]), hist_ndFeats_list: list([np.array(num_nodes, nfeat_dim)])
        # mask_list:  list([np.array(num_nodes, 1)])
		gnd_tnr = torch.sparse.FloatTensor(sample.grn_adj['idx'][0].t(), sample.grn_adj['vals'][0].type(torch.float), \
                torch.Size([self.num_nodes]*2)).to( sample.hist_adj_list[0].device).to_dense()
        # ====================
        # Train the discriminator
		loss = torch.tensor(0.0)
		adj_est, nodes_embs = self.gcn(sup_list, noise_list)

		if grad:
			disc_real, disc_fake = self.disc_net(gnd_tnr, adj_est, self.num_nodes)

			disc_loss = self.comp_loss.get_disc_loss(disc_real, disc_fake)
			self.disc_opt.zero_grad()
			disc_loss.backward()
			self.disc_opt.step()
    
			for param in self.disc_net.parameters():
				param.data.clamp_(-self.grad_clip, self.grad_clip)
            # ==========
            # Train the generative network
			adj_est, nodes_embs = self.gcn(sup_list, noise_list)
			disc_real, disc_fake = self.disc_net(gnd_tnr, adj_est, self.num_nodes)
			loss = self.comp_loss.get_gen_loss(adj_est, gnd_tnr, disc_fake)

			self.gcn_opt.zero_grad()
			loss.backward()
			self.gcn_opt.step()

		pred_1 = adj_est[sample.label_sp['idx'][0], sample.label_sp['idx'][1]] # (num_edges, )
		gather_predictions = torch.stack([1-pred_1, pred_1]).t() # (num_edges, 2)
		return loss, gather_predictions, nodes_embs


	def gather_node_embs(self,nodes_embs, node_indices):
		cls_input = []

		for node_set in node_indices: # [2, e] ->  list([e], [e])
			cls_input.append(nodes_embs[node_set]) # [e, d]
		return torch.cat(cls_input,dim = 1) # [e, 2d]

	def optim_step(self, loss):
		self.tr_step += 1
		loss.backward()

		if self.tr_step % self.args.steps_accum_gradients == 0:
			self.gcn_opt.step()
			self.classifier_opt.step()

			self.gcn_opt.zero_grad()
			self.classifier_opt.zero_grad()


	def prepare_sample(self,sample):
		sample = utils.Namespace(sample)
		for i,adj in enumerate(sample.hist_adj_list):
			adj = utils.sparse_prepare_tensor(adj, torch_size = [self.num_nodes])
			sample.hist_adj_list[i] = adj.to(self.args.device)

			nodes = self.tasker.prepare_node_feats(sample.hist_ndFeats_list[i])

			sample.hist_ndFeats_list[i] = nodes.to(self.args.device)
			node_mask = sample.node_mask_list[i]
			sample.node_mask_list[i] = node_mask.to(self.args.device).t() #transposed to have same dimensions as scorer

		label_sp = self.ignore_batch_dim(sample.label_sp)

		if self.args.task in ["link_pred", "edge_cls"]:
			label_sp['idx'] = label_sp['idx'].to(self.args.device).t()   
   ####### ALDO TO CHECK why there was the .t() -----> because I concatenate embeddings when there are pairs of them,\
       # the embeddings are row vectors after the transpose
		else:
			label_sp['idx'] = label_sp['idx'].to(self.args.device)

		label_sp['vals'] = label_sp['vals'].type(torch.long).to(self.args.device)
		sample.label_sp = label_sp

		return sample

	def prepare_static_sample(self,sample):
		sample = utils.Namespace(sample)

		sample.hist_adj_list = self.hist_adj_list

		sample.hist_ndFeats_list = self.hist_ndFeats_list

		label_sp = {}
		label_sp['idx'] =  [sample.idx]
		label_sp['vals'] = sample.label
		sample.label_sp = label_sp

		return sample

	def ignore_batch_dim(self,adj):
		if self.args.task in ["link_pred", "edge_cls"]:
			adj['idx'] = adj['idx'][0]
		adj['vals'] = adj['vals'][0]
		return adj

	def save_node_embs_csv(self, nodes_embs, indexes, file_name):
		csv_node_embs = []
		for node_id in indexes:
			orig_ID = torch.DoubleTensor([self.tasker.data.contID_to_origID[node_id]])

			csv_node_embs.append(torch.cat((orig_ID,nodes_embs[node_id].double())).detach().numpy())

		pd.DataFrame(np.array(csv_node_embs)).to_csv(file_name, header=None, index=None, compression='gzip')
		#print ('Node embs saved in',file_name)
