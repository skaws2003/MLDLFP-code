from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torch.optim as optim

from PIL import Image

best_acc = 0
start_epoch = 0

thisnet = ResNet18()
thisnet = thisnet.to(device)
if device =='cuda':
  thisnet = torch.nn.DataParallel(thisnet)
  cudnn.benchmark = True

if resume == True:
    # resume from checkpoint
    print('resuming from checkpoint')
    assert os.path.isdir(checkpoint_dir), 'No checkpoint!'
    checkpoint = torch.load(checkpoint_dir + checkpoint_file)
    thisnet.load_state_dict(checkpoint['thisnet'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(thisnet.parameters(), lr=0.001, momentum = 0.9, weight_decay=5e-4)


def train(epoch):
    print('\nThis Epoch: %d' % epoch)
    thisnet.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_i, (inputs, labels) in enumerate(train_dl):
        inputs, labels = inputs.to(device), labels.to(device)
        # print(inputs.shape)
        optimizer.zero_grad()
        outputs = thisnet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()


def test(epoch, checkpoint_loc):
    global best_acc
    thisnet.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_i, (inputs, labels) in enumerate(test_dl):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = thisnet(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    acc = 100. * correct / total
    print(acc)
    if acc > best_acc:
        print('saving best model')
        state = {
            'thisnet': thisnet.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if os.path.isdir(checkpoint_dir):
            torch.save(state,  checkpoint_dir + checkpoint_file)
        best_acc = acc

# actual training phase
for epoch in range(start_epoch, start_epoch + 10):
    train(epoch)
    test(epoch, "resnet18f.t7")

print(best_acc)