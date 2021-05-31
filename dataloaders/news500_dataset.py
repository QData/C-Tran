
from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle
from pdb import set_trace as stop
import json, string, sys
from dataloaders.data_utils import get_unk_mask_indices
from dataloaders.data_utils import image_loader

class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, ann_dir,split='train',transform=None,known_labels=0,testing=False):
        # Load training data.
        self.ann_dir = ann_dir  # the path you save train_split.p and caption_test_space.npy
        self.split = split
        self.transform = transform

        self.num_labels = 500
        self.known_labels = known_labels
        self.testing=testing

        # Load annotations.
        print(('Loading %s Label annotations...') % self.split)
        self.annData = pickle.load(open(os.path.join(ann_dir, '%s_split.p' % self.split),'rb'))
        self.targets = torch.Tensor(np.load(open(os.path.join(ann_dir, 'caption_%s_space.npy' % self.split),'rb')))
        self.epoch = 1

    def __getitem__(self, index):
        sample = self.annData[index]
        img_path = sample['file_path']
        image_id = sample['image_id']

        img_path = img_path.replace('/localtmp/data/data/',self.ann_dir)

        image = image_loader(img_path,self.transform)
        
        labels = self.targets[index, :]

        mask = labels.clone()
        unk_mask_indices = get_unk_mask_indices(image,self.testing,self.num_labels,self.known_labels)
        mask.scatter_(0,torch.Tensor(unk_mask_indices).long() , -1)

        sample = {}
        sample['image'] = image
        sample['labels'] = labels
        sample['mask'] = mask
        sample['imageIDs'] = str(image_id)

        return sample

    def __len__(self):
        return len(self.annData)
