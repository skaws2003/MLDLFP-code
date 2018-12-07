
"""
======================================================================
Word embedding.
Most of the code this part is based on the chatbot tutoral of pytorch
======================================================================
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torchvision
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math
import pandas as pd


# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token
UNK_token = 4  # Unknown token

# Some variables
MAX_LENGTH = 8
DATASET_PATH = {
    'train': '../dataset/train.csv',
    'test': '../dataset/test.csv'
}

# Library
print("Reading file...")
all_data = pd.read_csv(DATASET_PATH['train'],names=['polarity','id','date','query','user','text'])
all_text = all_data['text']
print(all_text.size)

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

class Voc:
    """
    A class for voca indexing
    """
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", UNK_token: "UNK"}
        self.num_words = 4  # Count SOS, EOS, PAD

    def getIndexOfWord(self,word):
        """
        Returns index of word
        Field:
        - word: a word
        """
        try:
            return self.word2index[word]
        except KeyError:
            return UNK_token

    def getIndexOfSentence(self,sentence):
        """
        Returns index list of sentence
        Field:
        - sentence: a sentence string
        """
        index_list = []
        for word in sentence.split(' '):
            index_list.append(self.getIndexOfWord(word))
        return index_list

    def addSentence(self, sentence):
        """
        Add words entry by addSentence
        """
        for word in sentence.split(' '):
            self.addWord(word)


    def addWord(self, word):
        """
        Add word to entry
        """
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        """
        Removes words below a certain count threshold
        """
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", UNK_token: "UNK"}
        self.num_words = 4 # Count default tokens

        for word in keep_words:
            self.addWord(word)

for i,text in enumerate(all_text):
    if i<30 and i>20:
        print(text)
    if i>30:
        break
