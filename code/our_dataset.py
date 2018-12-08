#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torchvision
import re               # Regular Expressions
from random import shuffle
import unicodedata

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

class Polarity_dataset(torch.utils.data.Dataset):
    def __init__(self,pos_path,neg_path):
        """
        Dataset class for Polarity dataset
        Fields
        - pos_data: file path to positive dataset txt file
        - neg_data: file path to negative dataset txt file
        """
        pos_file = open(pos_path, encoding='utf8')
        neg_file = open(neg_path, encoding='utf8')

        pos = pos_file.readlines()
        neg = neg_file.readlines()

        dataset = []

        for sent in pos:
            sent_normalized = normalizeString(sent)
            sent_tokenized = sentence_to_word(sent_normalized)
            dataset.append((sent_tokenized,POSITIVE))

        for sent in neg:
            sent_normalized = normalizeString(sent)
            sent_tokenized = sentence_to_word(sent_normalized)
            dataset.append((sent_tokenized,NEGATIVE))
        
        shuffle(dataset)
        
        self.sentences = []
        self.labels = []
        for data in dataset:
            self.sentences.append(data[0])
            if data[1] == POSITIVE:
                one_hot = torch.Tensor([0,1])
            elif data[1] == NEGATIVE:
                one_hot = torch.Tensor([1,0])
            self.labels.append(one_hot)

        pos_file.close()
        neg_file.close()

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self,index):
        return self.sentences[index], self.labels[index]

class Polarity_dataloader():
    def __init__(self,dataset):
        self.dataset = dataset
    def __getitem__(self,index):
        return [self.dataset[index][0]],self.dataset[index][1]

