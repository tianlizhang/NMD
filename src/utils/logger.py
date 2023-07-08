import logging
import pprint
import sys
import datetime
import torch
# import utils
from .utils import Namespace
import matplotlib.pyplot as plt
import time
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.sparse import coo_matrix
import numpy as np
from IPython import embed



class Logger():
    def __init__(self, args, num_classes, minibatch_log_interval=10):

        if args is not None:
            # currdate=str(datetime.datetime.today().strftime('%Y%m%d%H%M%S'))
            # self.log_name= '../log/log_'+args.data+'_'+args.task+'_'+args.model+'_'+currdate+'_r'+str(args.rank)+'.log'
            currdate=str(datetime.datetime.today().strftime('%m.%d-%H:%M'))
            self.log_name= '../log/' + currdate + '_' + args.data + '_' + args.model + '_' + str(args.log)+'.log'
            
            if args.use_logfile:
                print ("Log file:", self.log_name)
                logging.basicConfig(filename=self.log_name, level=logging.INFO)
            else:
                print ("Log: STDOUT")
                logging.basicConfig(stream=sys.stdout, level=logging.INFO)

            logging.info ('*** PARAMETERS ***')
            logging.info (pprint.pformat(args.__dict__)) # displays the string
            logging.info ('')
        else:
            print ("Log: STDOUT")
            logging.basicConfig(stream=sys.stdout, level=logging.INFO)

        self.num_classes = num_classes
        self.minibatch_log_interval = minibatch_log_interval
        self.eval_k_list = [10, 100, 1000]
        self.args = args


    def get_log_file_name(self):
        return self.log_name

    def log_epoch_start(self, epoch, num_minibatches, set, minibatch_log_interval=None):
        #ALDO
        self.epoch = epoch
        ######
        self.set = set
        self.losses = []
        self.MRRs = []
        self.MAPs = []
        self.AUCs = []

        self.batch_sizes=[]
        self.minibatch_done = 0
        self.num_minibatches = num_minibatches
        if minibatch_log_interval is not None:
            self.minibatch_log_interval = minibatch_log_interval
        # logging.info('################ '+set+' epoch '+str(epoch)+' ###################')


    def log_minibatch(self, predictions, true_classes, loss, **kwargs):
        # predictions: [e, 2] 二分类任务
        probs = torch.softmax(predictions,dim=1)[:,1] # 分类为1的概率
        # if self.set in ['TEST', 'VALID'] and self.args.task == 'link_pred':
        if self.args.task == 'link_pred':
            MRR = self.get_MRR(probs, true_classes, kwargs['adj'],do_softmax=False)
            MAP, AUC = torch.tensor(self.get_MAP_AUC(probs, true_classes, do_softmax=False))
        else:
            MRR = AUC = torch.tensor([0.0])
            MAP = torch.tensor(self.get_MAP(probs, true_classes, do_softmax=False))
        
        batch_size = predictions.size(0)
        self.batch_sizes.append(batch_size)
        self.losses.append(loss)
        self.MRRs.append(MRR)
        self.MAPs.append(MAP)
        self.AUCs.append(AUC)

        self.minibatch_done+=1


    def log_epoch_done(self, epoch):
        eval_measure = 0

        self.losses = torch.stack(self.losses)
        # logging.info(self.set+' mean losses '+ str(self.losses.mean()))
        if self.args.target_measure=='loss' or self.args.target_measure=='Loss':
            eval_measure = self.losses.mean()

        epoch_MRR = self.calc_epoch_metric(self.batch_sizes, self.MRRs)
        epoch_MAP = self.calc_epoch_metric(self.batch_sizes, self.MAPs)
        epoch_AUC = self.calc_epoch_metric(self.batch_sizes, self.AUCs)
        # logging.info(self.set+' mean MRR '+ str(epoch_MRR)+' - mean MAP '+ str(epoch_MAP))
        names = ['MRR', 'MAP', 'AUC', 'LOSS']
        metric_dic = {name: round(val*100, 4) for name, val in zip(names, [epoch_MRR, epoch_MAP, epoch_AUC, self.losses.mean().item()])}
        
        # if len(self.metric_dic) == 3:
        #     to_print = f'epoch {epoch}: {self.metric_dic}'
        #     print(to_print)
        #     logging.info(to_print)
        # logging.info(f'epoch {epoch}: {self.set} MRR: {epoch_MRR}, MAP: {epoch_MAP}, AUC: {epoch_AUC}, Loss: {self.losses.mean()}')
        return metric_dic

    def log_text(self, text):
        logging.info(text)
        
    
    def get_MRR(self,predictions,true_classes, adj ,do_softmax=False):
        if do_softmax:
            probs = torch.softmax(predictions,dim=1)[:,1]
        else:
            probs = predictions

        probs = probs.cpu().numpy()
        true_classes = true_classes.cpu().numpy()
        adj = adj.cpu().numpy()

        pred_matrix = coo_matrix((probs,(adj[0],adj[1]))).toarray()
        true_matrix = coo_matrix((true_classes,(adj[0],adj[1]))).toarray()

        row_MRRs = []
        for i,pred_row in enumerate(pred_matrix):
            #check if there are any existing edges
            if np.isin(1,true_matrix[i]):
                row_MRRs.append(self.get_row_MRR(pred_row,true_matrix[i]))

        avg_MRR = torch.tensor(row_MRRs).mean()
        return avg_MRR

    def get_row_MRR(self,probs,true_classes):
        """_summary_

        Args:
            probs (np.array(dtype=float32)): shape = (num_nodes, )
            true_classes (np.array()): shape = (num_nodes)

        Returns:
            _type_: _description_
        """
        existing_mask = true_classes == 1
        #descending in probability
        ordered_indices = np.flip(probs.argsort())

        ordered_existing_mask = existing_mask[ordered_indices]

        existing_ranks = np.arange(1, true_classes.shape[0]+1, dtype=np.float)[ordered_existing_mask]

        MRR = (1/existing_ranks).sum()/existing_ranks.shape[0]
        return MRR


    def get_MAP_AUC(self,predictions,true_classes, do_softmax=False):
        if do_softmax:
            probs = torch.softmax(predictions,dim=1)[:,1]
        else:
            probs = predictions

        predictions_np = probs.detach().cpu().numpy()
        true_classes_np = true_classes.detach().cpu().numpy()
        return average_precision_score(true_classes_np, predictions_np), roc_auc_score(true_classes_np, predictions_np)
    
    
    def get_MAP(self,predictions,true_classes, do_softmax=False):
        if do_softmax:
            probs = torch.softmax(predictions,dim=1)[:,1]
        else:
            probs = predictions

        predictions_np = probs.detach().cpu().numpy()
        true_classes_np = true_classes.detach().cpu().numpy()
        return average_precision_score(true_classes_np, predictions_np)
    

    def eval_predicitions(self, predictions, true_classes, num_classes):
        predicted_classes = predictions.argmax(dim=1)
        failures = (predicted_classes!=true_classes).sum(dtype=torch.float)
        error = failures/predictions.size(0)

        conf_mat_per_class = Namespace({})
        conf_mat_per_class.true_positives = {}
        conf_mat_per_class.false_negatives = {}
        conf_mat_per_class.false_positives = {}

        for cl in range(num_classes):
            cl_indices = true_classes == cl

            pos = predicted_classes == cl
            hits = (predicted_classes[cl_indices] == true_classes[cl_indices])

            tp = hits.sum()
            fn = hits.size(0) - tp
            fp = pos.sum() - tp

            conf_mat_per_class.true_positives[cl] = tp
            conf_mat_per_class.false_negatives[cl] = fn
            conf_mat_per_class.false_positives[cl] = fp
        return error, conf_mat_per_class


    def eval_predicitions_at_k(self, predictions, true_classes, num_classes, k):
        conf_mat_per_class = Namespace({})
        conf_mat_per_class.true_positives = {}
        conf_mat_per_class.false_negatives = {}
        conf_mat_per_class.false_positives = {}

        if predictions.size(0)<k:
            k=predictions.size(0)

        for cl in range(num_classes):
            # sort for prediction with higher score for target class (cl)
            _, idx_preds_at_k = torch.topk(predictions[:,cl], k, dim=0, largest=True, sorted=True)
            predictions_at_k = predictions[idx_preds_at_k]
            predicted_classes = predictions_at_k.argmax(dim=1)

            cl_indices_at_k = true_classes[idx_preds_at_k] == cl
            cl_indices = true_classes == cl

            pos = predicted_classes == cl
            hits = (predicted_classes[cl_indices_at_k] == true_classes[idx_preds_at_k][cl_indices_at_k])

            tp = hits.sum()
            fn = true_classes[cl_indices].size(0) - tp # This only if we want to consider the size at K -> hits.size(0) - tp
            fp = pos.sum() - tp

            conf_mat_per_class.true_positives[cl] = tp
            conf_mat_per_class.false_negatives[cl] = fn
            conf_mat_per_class.false_positives[cl] = fp
        return conf_mat_per_class


    def calc_microavg_eval_measures(self, tp, fn, fp):
        tp_sum = sum(tp.values()).item()
        fn_sum = sum(fn.values()).item()
        fp_sum = sum(fp.values()).item()

        p = tp_sum*1.0 / (tp_sum+fp_sum)
        r = tp_sum*1.0 / (tp_sum+fn_sum)
        if (p+r)>0:
            f1 = 2.0 * (p*r) / (p+r)
        else:
            f1 = 0
        return p, r, f1

    def calc_eval_measures_per_class(self, tp, fn, fp, class_id):
        #ALDO
        if type(tp) is dict:
            tp_sum = tp[class_id].item()
            fn_sum = fn[class_id].item()
            fp_sum = fp[class_id].item()
        else:
            tp_sum = tp.item()
            fn_sum = fn.item()
            fp_sum = fp.item()
        ########
        if tp_sum==0:
            return 0,0,0

        p = tp_sum*1.0 / (tp_sum+fp_sum)
        r = tp_sum*1.0 / (tp_sum+fn_sum)
        if (p+r)>0:
            f1 = 2.0 * (p*r) / (p+r)
        else:
            f1 = 0
        return p, r, f1

    def calc_epoch_metric(self,batch_sizes, metric_val):
        """_summary_

        Args:
            batch_sizes (list): list(int), e.g [15756, 18180, 21412] is #nodes in each timestamp
            metric_val (list): list(torch.tensor(int)), e.g. [torch.tensor(0.0099), torch.tensor(0.0099)]

        Returns:
            _type_: _description_
        """
        batch_sizes = torch.tensor(batch_sizes, dtype = torch.float)
        epoch_metric_val = torch.stack(metric_val).cpu() * batch_sizes # stack: 0dim -> 1dim, [0.009, 0.0099]
        epoch_metric_val = epoch_metric_val.sum()/batch_sizes.sum()

        return epoch_metric_val.detach().item()
