'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse


from models import *
import out_dataset as ds
import csv

# trainer class for training
class Trainer():

    def __init__(self, learning_rate, checkpoint_loc, csv_loc, batch_size):
        self.learning_rate = learning_rate
        self.checkpoint_loc = checkpoint_loc
        self.csv_loc = csv_loc
        self.batch_size = batch_size

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.best_acc = 0  # best test accuracy
        self.start_epoch = 0  # start from epoch 0 or last checkpoint epoch
        self.this_acc = 0
        self.this_loss = 0
        # Data
        print('==> Preparing data..')

        self.datasets = {x: ds.Sentiment_dataset(dataset_path=ds.DATASET_PATH[x]) for x in ['test', 'train']}
        self.trainloader = torch.utils.data.DataLoader(self.datasets['train'], batch_size=batch_size, shuffle=True, num_workers=2)
        self.valloader = torch.utils.data.DataLoader(self.datasets['test'], batch_size=batch_size, shuffle=False, num_workers=2)

        # Model
        print('==> Building model..')
        self.net = RNN()
        self.net = self.net.to(self.device)


        if self.device == 'cuda':
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    def resume(self):
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(self.checkpoint_loc)
        self.net.load_state_dict(checkpoint['net'])
        self.best_acc = checkpoint['acc']
        self.start_epoch = checkpoint['epoch']
        print('net acc :', self.best_acc, 'epoch :', self.start_epoch)

    # Training
    def train(self, epoch):
        print('\nEpoch: %d' % epoch)
        self.net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            print(batch_idx, len(self.trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        print(correct/total)

    def test(self, epoch):
        global best_acc
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.valloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                print(batch_idx, len(self.valloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        # Save checkpoint.
        acc = 100.*correct/total
        self.this_loss = test_loss / enumerate(self.valloader)
        self.this_acc = acc
        print(acc)
        if acc > best_acc:
            print('Saving..  %f' % acc)
            state = {
                'net': self.net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir(self.checkpoint_loc):
                os.mkdir(self.checkpoint_loc)
            torch.save(state, self.checkpoint_loc)
            best_acc = acc

    #actual training
    def trainepoch(self, epoch_num):
        with open(self.csv_loc, 'wb') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow(['epoch', 'loss', 'accuracy'])
            for epoch in range(self.start_epoch, self.start_epoch + epoch_num):
                self.train(epoch)
                self.test(epoch)
                filewriter.writerow([epoch, self.this_loss, self.this_acc])

        print('finished training')
        self.start_epoch += epoch_num
