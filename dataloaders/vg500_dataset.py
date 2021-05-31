
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle
from pdb import set_trace as stop
from dataloaders.data_utils import get_unk_mask_indices,image_loader

class VGDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, img_list, image_transform,label_path,known_labels=40,testing=False):
        with open(img_list, 'r') as f:
            self.img_names = f.readlines()
        with open(label_path, 'r') as f:
            self.labels = json.load(f) 
        
        self.image_transform = image_transform
        self.img_dir = img_dir
        self.num_labels= 500

        self.known_labels = known_labels
        self.testing=testing
        self.epoch = 1
    
    def __getitem__(self, index):
        name = self.img_names[index][:-1]
        img_path = os.path.join(self.img_dir, name)

        image = image_loader(img_path,self.image_transform)

        label = np.zeros(self.num_labels).astype(np.float32)
        label[self.labels[name]] = 1.0
        label = torch.Tensor(label)

        unk_mask_indices = get_unk_mask_indices(image,self.testing,self.num_labels,self.known_labels)
        mask = label.clone()
        mask.scatter_(0,torch.Tensor(unk_mask_indices).long() , -1)

        sample = {}
        sample['image'] = image
        sample['labels'] = label
        sample['mask'] = mask
        sample['imageIDs'] = name

        return sample

    def __len__(self):
        return len(self.img_names)