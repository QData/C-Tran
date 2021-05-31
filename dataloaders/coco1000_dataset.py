
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
import hashlib
import time
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import csv
from dataloaders.data_utils import get_unk_mask_indices
from dataloaders.data_utils import image_loader,pil_loader

def get_vocab(objData):
    spunctuation = set(string.punctuation)
    swords = set(stopwords.words('english'))
    print('Building vocabulary of words...')
    lem = WordNetLemmatizer()
    word_counts = dict()
    for (i, entry) in enumerate(objData['annotations']):
        if i % 10000 == 0: print('.'),
        caption = entry['caption']
        for word in word_tokenize(caption.lower()):
            word = lem.lemmatize(word)
            if word not in swords and word not in spunctuation:
                word_counts[word] = 1 + word_counts.get(word, 0)
    sword_counts = sorted(word_counts.items(), key = lambda x: -x[1])
    id2word = {idx: word for (idx, (word, count)) in enumerate(sword_counts[:1000])}
    id2count = {idx: count for (idx, (word, count)) in enumerate(sword_counts[:1000])}
    word2id = {word: idx for (idx, word) in id2word.items()}
    vocabulary = (id2word, word2id, id2count)
    pickle.dump(vocabulary, open('data/coco/coco_words_vocabulary.p', 'wb'))

    return vocabulary

class Coco1000Dataset(torch.utils.data.Dataset):
    def __init__(self, annotation_dir,image_dir,split='train',transform = None,known_labels=0,testing=False):
        # Load training data.
        self.split = split
        self.image_dir = image_dir
        self.transform = transform

        self.testing=testing
        self.num_labels = 1000#num_labels
        self.epoch = 1

        self.known_labels = known_labels

        # Load annotations.
        print(('\nLoading %s object annotations...') % self.split)
        self.objData = json.load(open(os.path.join(annotation_dir, 'captions_' + self.split + '2014.json')))
        self.imageIds = [entry['id'] for entry in self.objData['images']]
        self.imageNames = [entry['file_name'] for entry in self.objData['images']]
        self.imageId2index = {image_id: idx for (idx, image_id) in enumerate(self.imageIds)}

        if os.path.exists("data/coco/coco_words_vocabulary.p"):
            self.vocabulary = pickle.load(open('data/coco/coco_words_vocabulary.p', 'rb'))
        else:
            self.vocabulary = get_vocab(self.objData)

        
        label_file_path = os.path.join(annotation_dir, '1000_labels_' + self.split + '2014.npy')
        if os.path.exists(label_file_path):
            print('Loading labels')
            self.labels = np.load(label_file_path)
        else:
            print('Preparing label space')
            lem = WordNetLemmatizer()
            self.labels = np.zeros((len(self.objData['images']), len(self.vocabulary[0])))
            for (i, entry) in enumerate(self.objData['annotations']):
                # if i % 10000 == 0: print('.'),
                image_id = entry['image_id']
                caption = entry['caption']
                for word in word_tokenize(caption.lower()):
                    word = lem.lemmatize(word)
                    if word in self.vocabulary[1].keys():
                        self.labels[self.imageId2index[image_id], self.word2id(word)] = 1
            np.save(label_file_path, self.labels)

    def getLabelWeights(self):
        return (self.labels == 0).sum(axis = 0) / self.labels.sum(axis = 0)

    def decodeCategories(self, labelVector):
        return [self.id2word(idx) for idx in np.nonzero(labelVector)[0]]

    def id2word(self, idx):
        return self.vocabulary[0][idx]

    def word2id(self, word):
        return self.vocabulary[1][word]

    def imageName(self, index):
        return self.split + '2014/' + self.imageNames[index]

    def __getitem__(self, index):
        split_str = self.split if (self.split != 'test') else 'val'
        imageName_ = split_str + '2014/' + self.imageNames[index]
        
        image = pil_loader(os.path.join(self.image_dir, imageName_))
        if self.transform is not None:
            image = self.transform(image)
        
        sample = {'image': image,'labels':torch.Tensor(self.labels[index, :])} 
        

        mask = sample['labels'].clone()
        

        unk_mask_indices = get_unk_mask_indices(image,self.testing,self.num_labels,self.known_labels)


        mask.scatter_(0,torch.Tensor(unk_mask_indices).long() , -1)
        sample['mask'] = mask
        sample['imageIDs'] = imageName_

        return sample

    def __len__(self):
        return len(self.imageIds)

    def numCategories(self):
        return len(self.vocabulary[0])
