#!/usr/bin/env python
# -*- coding: utf-8 -*-
# movie-review polarity dataset for experiment
import torch
import torchvision
import re               # Regular Expressions
import random
import unicodedata
import numpy as np

# Some Constants
POSITIVE=1
NEGATIVE=0

# Path to Dataset
DATASET_PATH = {
    'positive_train': '../dataset/positive_train.txt',
    'negative_train': '../dataset/negative_train.txt',
    'positive_test': '../dataset/positive_test.txt',
    'negative_test': '../dataset/negative_test.txt'
}

def unicodeToAscii(sentence):
    """
    A function for encoding change
    Fields
    - sentence: unicode string
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', sentence)
        if unicodedata.category(c) != 'Mn'
    )

def sentence_to_word(sentence):
    """
    tokenize string into words.
    Fields
    - sentence: python string
    """
    word_list = []
    for word in sentence.split(" "):
        if len(word)> 0:
            word_list.append(word)
    return word_list

def normalizeString(sentence):
    """
    Delete some unuseful words from the given sentence
    Fields
    - sentence: python string
    """
    #s = unicodeToAscii(sentence)
    s = re.sub(r'([\.|!|?|"])+', r" \1", sentence)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s

# movie-review polarity dataset
class Polarity_dataset(torch.utils.data.Dataset):
    def __init__(self,pos_path,neg_path,one_hot=True):
        """
        Dataset class for Polarity dataset
        Fields
        - pos_data: file path to positive dataset txt file
        - neg_data: file path to negative dataset txt file
        - one_hot: if true, polarity will be encoded by one-hot scheme.
        """
        pos_file = open(pos_path, encoding='utf8')
        neg_file = open(neg_path, encoding='utf8')
        self.one_hot = one_hot

        pos = pos_file.readlines()
        neg = neg_file.readlines()

        self.dataset = []

        # Data Reading
        for sent in pos:
            sent_normalized = normalizeString(sent)
            sent_tokenized = sentence_to_word(sent_normalized)
            self.dataset.append((sent_tokenized,POSITIVE))
        for sent in neg:
            sent_normalized = normalizeString(sent)
            sent_tokenized = sentence_to_word(sent_normalized)
            self.dataset.append((sent_tokenized,NEGATIVE))

        self.shuffle()
        pos_file.close()
        neg_file.close()

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self,index):
        return self.sentences[index], self.labels[index]

    def shuffle(self):
        """
        Shuffles dataset. Then set the sentence and labels/
        """
        random.shuffle(self.dataset)
        self._set_data()

    def _set_data(self):
        """
        Set data with self.dataset
        """
        self.sentences = []
        self.labels = []
        for data in self.dataset:
            self.sentences.append(data[0])
            if self.one_hot == True:
                if data[1] == POSITIVE:
                    one_hot = [0,1]
                elif data[1] == NEGATIVE:
                    one_hot = [1,0]
                self.labels.append(one_hot)
            else:
                self.labels.append(data[1])

# movie review polarity dataloader
class Polarity_dataloader():
    def __init__(self,dataset,batch_size=1):
        """
        Fields
        - dataset: an instance ofr Polarity_dataset
        - batch_size: batch size
        """
        self.dataset = dataset
        self.batch_size=batch_size

    def __getitem__(self,index):
        sentences = []
        labels = []
        for i in range(self.batch_size):
            sentences.append(self.dataset[index*self.batch_size+i][0])
            labels.append(self.dataset[index*self.batch_size+i][1])
        labels = torch.Tensor(labels)
        labels = labels.type(torch.LongTensor)
        return sentences,labels

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def shuffle(self):
        self.dataset.shuffle()

    def set_batch_size(self,batch_size):
        self.batch_size = batch_size

    def get_batch_size(self):
        return self.batch_size

class IDMB_dataloader(torch.utils.data.DataLoader):
    def __init__(self):
        NotImplemented