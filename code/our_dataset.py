import torch
import torchvision
import utils
import re               # Regular Expressions
from random import shuffle

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
    s = re.sub(r'([.!?"])', r" \1", sentence)
    s = re.sub(r"[^a-zA-Z.!?]+", r" \1", s)
    return s

class Polarity_dataset(torch.utils.data.Dataset):
    def __init__(self,pos_path,neg_path):
        """
        Dataset class for Polarity dataset
        Fields
        - pos_data: file path to positive dataset txt file
        - neg_data: file path to negative dataset txt file
        """
        pos_file = open(pos_path)
        neg_file = open(neg_path)

        pos = pos_file.readlines()
        neg = neg_file.readlines()

        self.dataset = []

        for sent in pos:
            sent_normalized = normalizeString(sent)
            sent_tokenized = sentence_to_word(sent_normalized)
            self.dataset.append((sent_tokenized,POSITIVE))

        for sent in neg:
            sent_normalized = utils.normalizeString(sent)
            sent_tokenized = sentence_to_word(sent_normalized)
            self.dataset.append((sent_tokenized,NEGATIVE))
        
        shuffle(self.dataset)
        
        pos_file.close()
        neg_file.close()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,index):
        return self.dataset[index]

