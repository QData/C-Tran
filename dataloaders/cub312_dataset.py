

from __future__ import print_function, division
import os
import torch
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
import pickle
from pdb import set_trace as stop
from PIL import Image
import json, string, sys
import torchvision.transforms.functional as TF
import random
import time
from dataloaders.data_utils import get_unk_mask_indices
from dataloaders.data_utils import image_loader


def get_unk_mask_indices_cub(image,testing,num_labels,known_labels,group_unk,group_dict,concept_certainty):
    if testing:
        if known_labels > 0:
            uncertain_indices = np.argwhere(concept_certainty[0:112].numpy()==1).reshape(-1)
            group_unk[uncertain_indices] = 1
            unk_mask_indices = np.argwhere(group_unk==1).reshape(-1).tolist()
        else:
            unk_mask_indices = range(num_labels)
    else:
        # known_indices = []
        # n_groups = np.random.choice(28, 1)[0]
        # for group in np.random.choice(28, n_groups, replace=False):
        #     known_indices += group_dict[group] 
        # group_unk = np.ones(112)
        # group_unk[known_indices] = 0

        # if known_labels > 0:
        #     uncertain_indices = np.argwhere(concept_certainty[0:112].numpy()==1).reshape(-1)
        #     group_unk[uncertain_indices] = 1
        #     unk_mask_indices = np.argwhere(group_unk==1).reshape(-1).tolist()
        # else:
        #     unk_mask_indices = range(num_labels) 

        # sample random number of known labels during training
        if known_labels > 0:
            random.seed()
            num_known = random.randint(0,int(num_labels*0.75))
            unk_mask_indices = random.sample(range(num_labels), (num_labels-num_known))
        else:
            unk_mask_indices = range(num_labels)

    return unk_mask_indices


class CUBDataset(Dataset):
    def __init__(self, img_dir, img_list, image_transform,known_labels=0,attr_group_dict=None,testing=False,n_groups=1):
        with open(img_list, "rb" ) as f:
            self.labels = pickle.load(f)

        self.image_transform = image_transform
        self.img_dir = img_dir
        self.num_concepts= 112
        self.num_labels= 200


        # np.random.seed()
        self.attr_group_dict = attr_group_dict

        known_indices = []
        for group in np.random.choice(28, n_groups, replace=False):
            known_indices += attr_group_dict[group]

        
        self.group_unk_mask = np.ones(self.num_concepts)
        self.group_unk_mask[known_indices] = 0


        self.known_labels = known_labels
        self.testing=testing
        self.epoch = 1

    
    def __getitem__(self, index):
        name = self.labels[index]['img_path']
        

        name = name.replace('/juice/scr/scr102/scr/thaonguyen/CUB_supervision/datasets/CUB_200_2011/images/','')

        img_path = os.path.join(self.img_dir, name)

        image = image_loader(img_path,self.image_transform)

        concept = torch.Tensor(self.labels[index]['attribute_label'])
        class_label = torch.Tensor([self.labels[index]['class_label']])
        concept_certainty = torch.Tensor(self.labels[index]['attribute_certainty'])


        unk_mask_indices = get_unk_mask_indices_cub(image,self.testing,self.num_concepts,self.known_labels,np.copy(self.group_unk_mask),self.attr_group_dict,concept_certainty)


        mask = concept.clone()
        mask.scatter_(0,torch.Tensor(unk_mask_indices).long() , -1)

        class_mask = torch.Tensor(self.num_labels).fill_(-1)

        mask = torch.cat((mask,class_mask),0) 

        sample = {}
        sample['image'] = image
        sample['labels'] = concept
        sample['class_label'] = class_label
        sample['concept_certainty'] = concept_certainty
        sample['mask'] = mask
        sample['imageIDs'] = name

        return sample

    def __len__(self):
        return len(self.labels)
