'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
from __future__ import unicode_literals, print_function, division
import os
import sys
import time
import math

import torch.nn as nn
import torch.nn.init as init
import torch


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


term_width = 90 #os.get_terminal_size().columns

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

from torchvision import datasets
from torch.utils import data

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


from io import open
import unicodedata
import string
import re
import random

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1
UNKNOWN = 3

MAX_LENGTH = 50
MIN_LENGTH = 3
SKIP_P = 0.999
UNKNOWN_P = 0.000005
MAX_SIZE = 5000


class Lang:  # ont-hot vector 인코딩 다른 embedding 필요 language
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2:"UNKNOWN"}
        self.n_words = 3  # Count SOS and EOS

    def addSentence(self, sentence):  # tokenizer 한국어
        for word in sentence:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:  # initializer
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word]

def normalizeString(s):
    s = re.sub(r'([.!?"])', r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" \1", s)
    return s

def readLangs(lang):
    print("Reading lines...")
    data = []
    n = 0
    # Read the file and split into lines
    with open('data/%s.txt' % (lang), 'r',encoding='utf-8') as datafile:
        line = datafile.readline()
        while line:
            if n > MAX_SIZE:
                break
            if random.random() < SKIP_P and n > 300:
                datafile.readline()
                continue
            data.append(normalizeString(line))
            line = datafile.readline()
            n += 1


    return data

MAX_CHAR = 50
MIN_CHAR = 5

def filtersen(p):
    return len(p) < MAX_CHAR and re.compile(r'[^가-힣0-9\.,?! ]').match(p) is None\
               and len(p) > MIN_CHAR    #Maxlen 보다 작거나 숫자+. 으로 끝나지 않음, Minlen 보다 큼

def filtersens(sens):
    return [sen for sen in sens if filtersen(sen)]

def prepareData(lang):   #데이터 준비
    data = readLangs(lang)
    sens = filtersens(data)#filter

    print("Trimmed to %s sentence sens" % len(sens))
    print("Counting words...")
    lan = Lang(lang)
    for sen in sens:
        lan.addSentence(sen)
    print("Counted words:")
    print(lan.name, lan.n_words)
    return lan, sens

def preparepairs(name):
    print("Reading Pairs")
    linenum = 1
    pairs = []
    with open('data/%s.txt' % (name), 'r',encoding='utf-8') as pairfile:
        line = pairfile.readline()
        while line:
            pairs.append([line, linenum])
            line = pairfile.readline()
            linenum += 1
    return pairs

def preparetestset(name):
    print("Reading Pairs")
    linenum = 1
    pairs = []
    with open('data/%s.txt' % (name), 'r',encoding='utf-8') as pairfile:
        line = pairfile.readline()
        while line:
            pairs.append([line.split('\t')[0], int(line.split('\t')[1])])
            line = pairfile.readline()
    return pairs

#helper
def indexesFromSentence(lang, sentence):
    list = []
    for word in sentence:
        try:
            if random.random() < UNKNOWN_P:
                list.append(lang.word2vec['UNKNOWN'])
            else:
                list.append(lang.word2vec[word])
        except KeyError:
            list.append(lang.word2vec['UNKNOWN'])

    return list

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(lang.word2vec['EOS'])
    return torch.tensor(indexes, dtype=torch.float, device=device)


class WEmbedding():
    def __init__(self, filename, length = 200):
        self.name = filename
        self.word2vec = {"SOS":[0]*length, "EOS":[0]*length, "UNKNOWN":[0]*length}
        self.n_words = 0
        self.dim = 0
        self.load(filename)

    def getvec(self, word):
        return self.word2vec[word]

    def setvec(self, word, vec): #word = 단어 'word' vec = 벡터 [1, 2, 3, 4, ..., 200]
        self.word2vec[word] = vec

    def load(self, file):
        with open(file, 'r', encoding='utf8') as f:
            line = f.readline()
            print(line)
            s = line.split()
            self.n_words = int(s[0])
            self.dim = int(s[1])
            line = f.readline()
            while line:
                s = line.split()
                self.word2vec[re.findall('(?<=\')(.*?)(?=\')', s[0])[0]] = [float(a) for a in s[1:]]
                line = f.readline()

    def inittokens(self, sens):
        first = self.word2vec["SOS"]
        last = self.word2vec["EOS"]
        ran = self.word2vec["UNKNOWN"]
        a = 0
        for sen in sens:
            try:
                tags = sen
                first = [sum(x) for x in zip(self.word2vec[tags[0]], first)]
                last = [sum(x) for x in zip(self.word2vec[tags[-3]], last)]
                ran = [sum(x) for x in zip(self.word2vec[random.choice(tags)], ran)]
            except KeyError:
                a+=1


        self.word2vec["SOS"] = [x/(len(sens)-a) for x in first]
        self.word2vec["EOS"] = [x/(len(sens)-a) for x in last]
        self.word2vec["UNKNOWN"] = [x/(len(sens)-a) for x in ran]
        
# download pretrained glove model from link https://nlp.stanford.edu/projects/glove/
# GloveSaver(glove_path)
# saves the downloaded text file to pickle and bcolz object
# only has to be done once for each text file
class GloveSaver():
    def __init__(self, glove_path):
        self.words = []
        idx = 0
        self.word2idx = {}

        self.vectors = bcolz.carray(np.zeros(1), rootdir=f'{glove_path}/6b.50.dat', mode='w')
        with open(f'{glove_path}/glove.6B.50d.txt', 'rb') as f:
            for l in f:
                line = l.decode().split()
                word = line[0]
                self.words.append(word)
                self.word2idx[word] = idx
                idx += 1
                vect = np.array(line[1:]).astype(np.float)
                self.vectors.append(vect)
            self.vectors = bcolz.carray(self.vectors[1:].reshape((400000,50)), rootdir=f'{glove_path}/6B.50.dat', mode='w')
            self.vectors.flush()
            pickle.dump(self.words, open(f'{glove_path}/6B.50_words.pkl', 'wb'))
            pickle.dump(self.word2idx, open(f'{glove_path}/6B.50_idx.pkl', 'wb'))

# embedder which reads from objects generated by GloveSaver
# embedder = GloveEmbedder(glove_path)
# embedder.getvec('word') returns vector
# embedder.getseq(wordseq) returns list of vectors
# has to be called out each time
class GloveEmbedder():
    def __init__(self, glove_path):
        self.vectors = bcolz.open(f'{glove_path}/6B.50.dat')[:]
        self.words = pickle.load(open(f'{glove_path}/6B.50_words.pkl', 'rb'))
        self.word2idx = pickle.load(open(f'{glove_path}/6B.50_idx.pkl', 'rb'))
        self.mean = np.zeros(50)
        for i in range(2000):
            self.mean += self.vectors[i] / 1000
        self.glove = {w: self.vectors[self.word2idx[w]] for w in self.words}

    def getvec(self, word):
        if not word in self.glove.keys():
            return self.mean
        return self.glove[word]

    def getseq(self, wordseq):
        returnseq = []
        for words in wordseq:
            returnseq.append(self.getvec(words))
        return returnseq
