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
import csv
from dataloaders.data_utils import get_unk_mask_indices



from xml.dom.minidom import parse 
import xml.dom.minidom
category_info = {'aeroplane':0, 'bicycle':1, 'bird':2, 'boat':3, 'bottle':4,
                 'bus':5, 'car':6, 'cat':7, 'chair':8, 'cow':9,
                 'diningtable':10, 'dog':11, 'horse':12, 'motorbike':13, 'person':14,
                 'pottedplant':15, 'sheep':16, 'sofa':17, 'train':18, 'tvmonitor':19}

class Voc07Dataset(torch.utils.data.Dataset):
    def __init__(self, img_dir='./data/VOCdevkit/VOC2007/JPEGImages', anno_path='./data/VOCdevkit/VOC2007/Main/trainval.txt', image_transform=None, labels_path='./data/VOCdevkit/VOC2007/Annotations',known_labels=0,testing=False,use_difficult=False):
        self.img_names  = []
        with open(anno_path, 'r') as f:
             self.img_names = f.readlines()
        self.img_dir = img_dir

        self.num_labels = 20
        self.known_labels = known_labels
        self.testing=testing
        
        self.labels = []
        for name in self.img_names:
            label_file = os.path.join(labels_path,name[:-1]+'.xml')
            label_vector = np.zeros(self.num_labels)
            DOMTree = xml.dom.minidom.parse(label_file)
            root = DOMTree.documentElement
            objects = root.getElementsByTagName('object')  
            for obj in objects:
                if (not use_difficult) and (obj.getElementsByTagName('difficult')[0].firstChild.data) == '1':
                    continue
                tag = obj.getElementsByTagName('name')[0].firstChild.data.lower()
                label_vector[int(category_info[tag])] = 1.0
            self.labels.append(label_vector)
        
        # self.labels = np.array(self.labels).astype(np.float32)
        self.labels = np.array(self.labels).astype(int)
        self.image_transform = image_transform
        self.epoch = 1
        

    def __getitem__(self, index):
        name = self.img_names[index][:-1]+'.jpg'
        image = Image.open(os.path.join(self.img_dir, name)).convert('RGB')
          
        if self.image_transform:
            image = self.image_transform(image)

        labels = torch.Tensor(self.labels[index])

        unk_mask_indices = get_unk_mask_indices(image,self.testing,self.num_labels,self.known_labels,self.epoch)

        mask = labels.clone()
        mask.scatter_(0,torch.Tensor(unk_mask_indices).long() , -1)

        sample = {}
        sample['image'] = image
        sample['labels'] = labels
        sample['mask'] = mask
        sample['imageIDs'] = str(name)

        return sample


    def __len__(self):
        return len(self.img_names)


