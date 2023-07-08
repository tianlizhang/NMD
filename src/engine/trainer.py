import torch
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from IPython import embed
from utils import utils, logger
import sys


def get_id_max(nums):
    idd = np.argmax(nums)
    return idd, nums[idd]

def metric_by(nums, name='mrr'):
    valid_idd, valid_max = get_id_max(nums[:, 1])
    test_idd, test_max = get_id_max(nums[:, 2])
    test_val = nums[:, 2][valid_idd]
    to_print = f'{name}: valid_max is {valid_max} at epoch {valid_idd}(test = {test_val}), test_max is {test_max} at epoch {test_idd}'
    return to_print
    print(f'{name}: valid_max is {valid_max} at epoch {valid_idd}(test = {test_val}), test_max is {test_max} at epoch {test_idd}')
    # return valid_max, test_max, nums[:, 2][valid_idd]



class Trainer():
	def __init__(self,args, splitter, gcn, classifier, comp_loss, dataset, num_classes):
		self.args = args
		self.splitter = splitter
		self.tasker = splitter.tasker
		self.gcn = gcn
		self.classifier = classifier
		self.comp_loss = comp_loss

		self.num_nodes = dataset.num_nodes
		self.data = dataset
		self.num_classes = num_classes

		self.logger = logger.Logger(args, self.num_classes)
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
		self.gcn_opt.zero_grad()
		self.classifier_opt.zero_grad()

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
		self.logger.log_text('python ' + ' '.join(sys.argv))
		self.tr_step = 0
		# early_stop = utils.EarlyStopMonitor(num_val=6, win_size=5)
		early_stop = utils.EarlyStopMonitor(num_val=4, win_size=4)
		for e in range(self.args.num_epochs):
			eval_train, nodes_embs = self.run_epoch(self.splitter.train, e, 'TRAIN', grad = True)
			if len(self.splitter.dev)>0 and e>self.args.eval_after_epochs: # eval_after_epochs = 5
				eval_valid, _ = self.run_epoch(self.splitter.dev, e, 'VALID', grad = False)
				eval_test, _ = self.run_epoch(self.splitter.test, e, 'TEST', grad = False)

				to_print = f'epoch: {e}: train: {eval_train}, valid: {eval_valid}, test: {eval_test}'
				print(to_print)
				self.logger.log_text(to_print)
    
				# if early_stop.early_stop_check([eval_valid['MRR'], eval_valid['MAP'], eval_valid['AUC'], \
				# 						eval_test['MRR'], eval_test['MAP'], eval_test['AUC']]):
				if early_stop.early_stop_check([eval_valid['MRR'], eval_valid['MAP'], \
										eval_test['MRR'], eval_test['MAP']]):
					print(f'Early stop at epoch {e}.(Values={early_stop.vals})')
					break
				if self.args.save_node_embeddings:
					self.save_node_embs_csv(nodes_embs, self.splitter.train_idx, log_file+'_train_nodeembs.csv.gz')
					self.save_node_embs_csv(nodes_embs, self.splitter.dev_idx, log_file+'_valid_nodeembs.csv.gz')
					self.save_node_embs_csv(nodes_embs, self.splitter.test_idx, log_file+'_test_nodeembs.csv.gz')

			if self.args.save:
				if not os.path.exists(f'../ckpt/{self.args.model}'):
					os.mkdir(f'../ckpt/{self.args.model}')
				spath_gcn = f'../ckpt/{self.args.model}/{self.args.data}_{self.args.fusion_mode}_gcn_{e}e.pkl'
				spath_cls = f'../ckpt/{self.args.model}/{self.args.data}_{self.args.fusion_mode}_cls_{e}e.pkl'
				if self.args.model == 'egcn' or self.args.model == 'egcn_tim':
					torch.save(self.gcn, spath_gcn)
					torch.save(self.classifier, spath_cls)
				else:
					torch.save(self.gcn.state_dict(), spath_gcn)
					torch.save(self.classifier.state_dict(), spath_cls)
				
    
			torch.cuda.empty_cache()
			torch.cuda.empty_cache()
			torch.cuda.empty_cache()
		self.show_result()

 
	def show_result(self):
		fr = open(self.logger.get_log_file_name(), 'rb')
		lines = fr.readlines()
		fr.close()

		mrrs = []
		maps = []
		aucs = []
		for line in lines:
			line = line.decode('utf-8').strip()
			if line.startswith('INFO:root:epoch'):
				mrrs.append(np.array([float(item.split(', ')[0][0:-1]) for item in line.split("'MRR': ")[1:]]))
				maps.append(np.array([float(item.split(', ')[0][0:-1]) for item in line.split("'MAP': ")[1:]]))
				aucs.append(np.array([float(item.split(', ')[0][0:-1]) for item in line.split("'AUC': ")[1:]]))
		mrr = np.vstack(mrrs)
		map = np.vstack(maps)
		auc = np.vstack(aucs)

		self.logger.log_text(metric_by(mrr))
		self.logger.log_text(metric_by(map, 'map'))
		self.logger.log_text(metric_by(auc, 'auc'))
		return


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

			predictions, nodes_embs, node_loss = self.predict(s.hist_adj_list, s.hist_ndFeats_list, s.label_sp['idx'], s.node_mask_list)

			loss = self.comp_loss(predictions, s.label_sp['vals']) + self.args.alpha * node_loss
			# self.logger.log_minibatch(predictions, s.label_sp['vals'], loss.detach(), adj = s.label_sp['idx'])
			if set_name in ['TEST', 'VALID'] and self.args.task == 'link_pred':
				self.logger.log_minibatch(predictions, s.label_sp['vals'], loss.detach(), adj = s.label_sp['idx'])
			else:
				self.logger.log_minibatch(predictions.detach(), s.label_sp['vals'], loss.detach(), adj = s.label_sp['idx'])
			if grad:
				self.optim_step(loss)

		torch.set_grad_enabled(True)
		eval_measure = self.logger.log_epoch_done(epoch)

		return eval_measure, nodes_embs

	def predict(self,hist_adj_list,hist_ndFeats_list,node_indices,mask_list):
		results = self.gcn(hist_adj_list, # list([np.array(num_nodes, num_nodes)])
								hist_ndFeats_list, # list([np.array(num_nodes, nfeat_dim)])
								mask_list) # list([np.array(num_nodes, 1)])
		try:
			nodes_embs, node_loss = results
		except:
			nodes_embs, node_loss = results, 0.0

		predict_batch_size = 100000
		gather_predictions=[]
		for i in range(1 +(node_indices.size(1)//predict_batch_size)):
			cls_input = self.gather_node_embs(nodes_embs, node_indices[:, i*predict_batch_size:(i+1)*predict_batch_size])
			predictions = self.classifier(cls_input)
			gather_predictions.append(predictions)
		gather_predictions=torch.cat(gather_predictions, dim=0) # [num_edges, out_dim=2], 二分类问题
		return gather_predictions, nodes_embs, node_loss

	def gather_node_embs(self,nodes_embs,node_indices):
		cls_input = []

		for node_set in node_indices: # [2, e] ->  list([e], [e])
			cls_input.append(nodes_embs[node_set]) # [e, d]
		return torch.cat(cls_input,dim = 1) # [e, 2d]

	def optim_step(self,loss):
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
