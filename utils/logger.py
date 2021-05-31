
import numpy as np
import logging
from collections import OrderedDict
import torch
import math
from pdb import set_trace as stop
import os
from models.utils import custom_replace
from utils.metrics import *
import torch.nn.functional as F 
import warnings
warnings.filterwarnings("ignore")


class LossLogger:
    def __init__(self,model_name):
        self.model_name = model_name
        open(model_name+'/train.log',"w").close()
        open(model_name+'/valid.log',"w").close()
        open(model_name+'/test.log',"w").close()

    def log_losses(self,file_name,epoch,loss,metrics,loss_unk=''):
        log_file = open(self.model_name+'/'+file_name,"a")
        log_file.write(str(epoch)+','+str(loss)+','+str(loss_unk)+','+str(metrics['mAP'])+'\n')
        log_file.close()


class Logger:
    def __init__(self,args):
        self.model_name = args.model_name
        self.best_mAP = 0
        self.best_class_acc = 0

        if args.model_name:
            try:
                os.makedirs(args.model_name)
            except OSError as exc:
                pass

            try:
                os.makedirs(args.model_name+'/epochs/')
            except OSError as exc:
                pass

            self.file_names = {}
            self.file_names['train'] = os.path.join(args.model_name,'train_results.csv')
            self.file_names['valid'] = os.path.join(args.model_name,'valid_results.csv')
            self.file_names['test'] = os.path.join(args.model_name,'test_results.csv')

            self.file_names['valid_all_aupr'] = os.path.join(args.model_name,'valid_all_aupr.csv')
            self.file_names['valid_all_auc'] = os.path.join(args.model_name,'valid_all_auc.csv')
            self.file_names['test_all_aupr'] = os.path.join(args.model_name,'test_all_aupr.csv')
            self.file_names['test_all_auc'] = os.path.join(args.model_name,'test_all_auc.csv')
            

            f = open(self.file_names['train'],'w+'); f.close()
            f = open(self.file_names['valid'],'w+'); f.close()
            f = open(self.file_names['test'],'w+'); f.close()
            f = open(self.file_names['valid_all_aupr'],'w+'); f.close()
            f = open(self.file_names['valid_all_auc'],'w+'); f.close()
            f = open(self.file_names['test_all_aupr'],'w+'); f.close()
            f = open(self.file_names['test_all_auc'],'w+'); f.close()
            os.utime(args.model_name,None)
        
        self.best_valid = {'loss':1000000,'mAP':0,'ACC':0,'HA':0,'ebF1':0,'OF1':0,'CF1':0,'meanAUC':0,'medianAUC':0,'meanAUPR':0,'medianAUPR':0,'meanFDR':0,'medianFDR':0,'allAUC':None,'allAUPR':None,
        'concept_acc':0,'class_acc':0}

        self.best_test = {'loss':1000000,'mAP':0,'ACC':0,'HA':0,'ebF1':0,'OF1':0,'CF1':0,'meanAUC':0,'medianAUC':0,'meanAUPR':0,'medianAUPR':0,'meanFDR':0,'medianFDR':0,'allAUC':None,'allAUPR':None,'epoch':0,
        'concept_acc':0,'class_acc':0}


    def evaluate(self,train_metrics,valid_metrics,test_metrics,epoch,num_params,model,valid_loss,test_loss,all_preds,all_targs,all_ids,args):

        
        if args.dataset == 'cub':
            for metric in valid_metrics.keys():
                if not 'all' in metric and not 'time'in metric:
                    if  valid_metrics[metric] >= self.best_valid[metric]:
                        self.best_valid[metric]= valid_metrics[metric]
                        self.best_test[metric]= test_metrics[metric]
                        if metric == 'ACC':
                            self.best_test['epoch'] = epoch


            if valid_metrics['class_acc'] >= self.best_class_acc:
                self.best_class_acc = valid_metrics['class_acc']

                print('> Saving Model\n')
                save_dict =  {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'valid_mAP': valid_metrics['mAP'],
                    'test_mAP': test_metrics['mAP'],
                    'valid_loss': valid_loss,
                    'test_loss': test_loss
                    }
                torch.save(save_dict, args.model_name+'/best_model.pt')

            
            print('\n')
            print('**********************************')
            print('best mAP:  {:0.3f}'.format(self.best_test['mAP']))
            print('best CF1:  {:0.3f}'.format(self.best_test['CF1']))
            print('best OF1:  {:0.3f}'.format(self.best_test['OF1']))
            print('best Concept ACC:  {:0.3f}'.format(self.best_test['concept_acc']))
            print('best Class ACC:  {:0.3f}'.format(self.best_test['class_acc']))
            print('**********************************')

        else:

            if valid_metrics['mAP'] >= self.best_mAP:
                self.best_mAP = valid_metrics['mAP']
                self.best_test['epoch'] = epoch

                for metric in valid_metrics.keys():
                    if not 'all' in metric and not 'time'in metric:
                        self.best_valid[metric]= valid_metrics[metric]
                        self.best_test[metric]= test_metrics[metric]    

                print('> Saving Model\n')
                save_dict =  {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'valid_mAP': valid_metrics['mAP'],
                    'test_mAP': test_metrics['mAP'],
                    'valid_loss': valid_loss,
                    'test_loss': test_loss
                    }
                torch.save(save_dict, args.model_name+'/best_model.pt')

        
            print('\n')
            print('**********************************')
            print('best mAP:  {:0.1f}'.format(self.best_test['mAP']*100))
            print('best CF1:  {:0.1f}'.format(self.best_test['CF1']*100))
            print('best OF1:  {:0.1f}'.format(self.best_test['OF1']*100))
            print('**********************************')



        return self.best_valid,self.best_test
