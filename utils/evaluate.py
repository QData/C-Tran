
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


def compute_metrics(args,all_predictions,all_targets,all_masks,loss,loss_unk,elapsed,known_labels=0,all_metrics=False,verbose=True):
    
    all_predictions = F.sigmoid(all_predictions)

    if args.dataset =='cub':
        all_preds_concepts = all_predictions[:,0:112].clone()
        all_targets_concepts = all_targets[:,0:112].clone()
        all_preds_concepts[all_preds_concepts >= 0.5] = 1
        all_preds_concepts[all_preds_concepts < 0.5] = 0

        concept_accs = []
        for i in range(all_preds_concepts.size(1)): 
            concept_accs.append(metrics.accuracy_score(all_targets_concepts[:,i],all_preds_concepts[:,i]))
        concept_acc = np.array(concept_accs).mean()

        all_preds_classes = all_predictions[:,112:].clone()
        all_targets_classes = all_targets[:,112:].clone()
        pred_max_val,pred_max_idx = torch.max(all_preds_classes,1)
        _,target_max_idx = torch.max(all_targets_classes,1)

        class_acc = (pred_max_idx==target_max_idx).sum().item()/pred_max_idx.size(0)

    else:
        concept_acc = 0
        class_acc = 0
        

    unknown_label_mask = custom_replace(all_masks,1,0,0)


    if known_labels > 0:
        meanAP = custom_mean_avg_precision(all_targets,all_predictions,unknown_label_mask)
    else:
        meanAP = metrics.average_precision_score(all_targets,all_predictions, average='macro', pos_label=1)

    optimal_threshold = 0.5 

    all_targets = all_targets.numpy()
    all_predictions = all_predictions.numpy()

    top_3rd = np.sort(all_predictions)[:,-3].reshape(-1,1)
    all_predictions_top3 = all_predictions.copy()
    all_predictions_top3[all_predictions_top3<top_3rd] = 0
    all_predictions_top3[all_predictions_top3<optimal_threshold] = 0
    all_predictions_top3[all_predictions_top3>=optimal_threshold] = 1

    CP_top3 = metrics.precision_score(all_targets, all_predictions_top3, average='macro')
    CR_top3 = metrics.recall_score(all_targets, all_predictions_top3, average='macro')
    CF1_top3 = (2*CP_top3*CR_top3)/(CP_top3+CR_top3)
    OP_top3 = metrics.precision_score(all_targets, all_predictions_top3, average='micro')
    OR_top3 = metrics.recall_score(all_targets, all_predictions_top3, average='micro')
    OF1_top3 = (2*OP_top3*OR_top3)/(OP_top3+OR_top3)

    
    all_predictions_thresh = all_predictions.copy()
    all_predictions_thresh[all_predictions_thresh < optimal_threshold] = 0
    all_predictions_thresh[all_predictions_thresh >= optimal_threshold] = 1
    CP = metrics.precision_score(all_targets, all_predictions_thresh, average='macro')
    CR = metrics.recall_score(all_targets, all_predictions_thresh, average='macro')
    CF1 = (2*CP*CR)/(CP+CR)
    OP = metrics.precision_score(all_targets, all_predictions_thresh, average='micro')
    OR = metrics.recall_score(all_targets, all_predictions_thresh, average='micro')
    OF1 = (2*OP*OR)/(OP+OR)  

    acc_ = list(subset_accuracy(all_targets, all_predictions_thresh, axis=1, per_sample=True))
    hl_ = list(hamming_loss(all_targets, all_predictions_thresh, axis=1, per_sample=True))
    exf1_ = list(example_f1_score(all_targets, all_predictions_thresh, axis=1, per_sample=True))        
    acc = np.mean(acc_)
    hl = np.mean(hl_)
    exf1 = np.mean(exf1_)

    eval_ret = OrderedDict([('Subset accuracy', acc),
                        ('Hamming accuracy', 1 - hl),
                        ('Example-based F1', exf1),
                        ('Label-based Micro F1', OF1),
                        ('Label-based Macro F1', CF1)])

    
    ACC = eval_ret['Subset accuracy']
    HA = eval_ret['Hamming accuracy']
    ebF1 = eval_ret['Example-based F1']
    OF1 = eval_ret['Label-based Micro F1']
    CF1 = eval_ret['Label-based Macro F1']

    if verbose:
        print('loss:  {:0.3f}'.format(loss))
        print('lossu: {:0.3f}'.format(loss_unk))
        print('----')
        print('mAP:   {:0.1f}'.format(meanAP*100))
        print('----')
        print('CP:    {:0.1f}'.format(CP*100))
        print('CR:    {:0.1f}'.format(CR*100))
        print('CF1:   {:0.1f}'.format(CF1*100))
        print('OP:    {:0.1f}'.format(OP*100))
        print('OR:    {:0.1f}'.format(OR*100))
        print('OF1:   {:0.1f}'.format(OF1*100))
        if args.dataset in ['coco','vg']:
            print('----')
            print('CP_t3: {:0.1f}'.format(CP_top3*100))
            print('CR_t3: {:0.1f}'.format(CR_top3*100))
            print('CF1_t3:{:0.1f}'.format(CF1_top3*100))
            print('OP_t3: {:0.1f}'.format(OP_top3*100))
            print('OR_t3: {:0.1f}'.format(OR_top3*100))
            print('OF1_t3:{:0.1f}'.format(OF1_top3*100)) 

    metrics_dict = {}
    metrics_dict['mAP'] = meanAP
    metrics_dict['ACC'] = ACC
    metrics_dict['HA'] = HA
    metrics_dict['ebF1'] = ebF1
    metrics_dict['OF1'] = OF1
    metrics_dict['CF1'] = CF1
    metrics_dict['loss'] = loss
    metrics_dict['time'] = elapsed

    if args.dataset =='cub':
        print('Concept Acc:    {:0.3f}'.format(concept_acc))
        print('Class Acc:    {:0.3f}'.format(class_acc))
        metrics_dict['concept_acc'] = concept_acc
        metrics_dict['class_acc'] = class_acc

    print('')

    return metrics_dict