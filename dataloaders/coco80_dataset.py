
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle
from pdb import set_trace as stop
from dataloaders.data_utils import get_unk_mask_indices,image_loader

class Coco80Dataset(Dataset):

    def __init__(self, split,num_labels,data_file,img_root,annotation_dir,max_samples=-1,transform=None,known_labels=0,testing=False,analyze=False):
        self.split=split
        self.split_data = pickle.load(open(data_file,'rb'))
        
        if max_samples != -1:
            self.split_data = self.split_data[0:max_samples]

        self.img_root = img_root
        self.transform = transform
        self.num_labels = num_labels
        self.known_labels = known_labels
        self.testing=testing
        self.epoch = 1

    def __len__(self):
        return len(self.split_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_ID = self.split_data[idx]['file_name']

        img_name = os.path.join(self.img_root,image_ID)
        image = image_loader(img_name,self.transform)

        labels = self.split_data[idx]['objects']
        labels = torch.Tensor(labels)

        unk_mask_indices = get_unk_mask_indices(image,self.testing,self.num_labels,self.known_labels)
        
        mask = labels.clone()
        mask.scatter_(0,torch.Tensor(unk_mask_indices).long() , -1)

        sample = {}
        sample['image'] = image
        sample['labels'] = labels
        sample['mask'] = mask
        sample['imageIDs'] = image_ID
        return sample


