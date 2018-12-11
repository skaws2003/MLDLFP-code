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
import time
import copy

from models import *
from dataloader import *
from utils import Lang
from models.seq2seq2 import *
from DALoss import DALoss
import pickle

parser = argparse.ArgumentParser(description='PyTorch ERNN Training')
parser.add_argument('--lr', default=0.01, type=float, help='Initial learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--predict', action='store_true', help='forward prop')
parser.add_argument('--batch_size', default=200, type=int, help='define batch size')
parser.add_argument('--epoch', default=200, type=int, help='define epoch')
parser.add_argument('--silent', action='store_false', help='Only print test result')
parser.add_argument('--hidden_size', default=512, type=int, help='Hidden Layer size')
#parser.add_argument('--arch', default='ernn', help='Network architecture')
parser.add_argument('--num_split', default=3, type=int, help='Number of split RNN')
parser.add_argument('--cuda', default=0,type=int,help='gpu num')

args = parser.parse_args()

if args.cuda==0:
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
elif args.cuda==1:
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
else:
    print("Not a valid cuda")

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Model
print('==> Building model..')

input_size = 128  #same as embedding size
num_layers = 2      ###
num_split = args.num_split
hidden_size = args.hidden_size
output_size = 2
batch_size = args.batch_size

net=darnn.DARNN

# Log files
logfileAcc = open("log_da_acc%d.txt"%args.hidden_size,'w')
logfileLoss = open("log_da_loss%d.txt"%args.hidden_size,'w')

# Set batch size to 1 for embedding
dataloaders['train'].set_batch_size(1)
dataloaders['test'].set_batch_size(1)

# Word embedding
lang = Lang('eng')
for _, (text, _) in enumerate(dataloaders['train']):
    for i in range(len(text)):
        lang.addSentence(text[i])
for _, (text, _) in enumerate(dataloaders['test']):
    for i in range(len(text)):
        lang.addSentence(text[i])

dataloaders['train'].set_batch_size(batch_size)
dataloaders['test'].set_batch_size(batch_size)

embedding = nn.Embedding(lang.n_words, input_size, padding_idx=0)

encoder = EncoderRNN(input_size, hidden_size, embedding, net, n_layers=num_layers, num_split=num_split, dropout=0, device=device).to(device)
# attn_model = Attn('general', hidden_size)
# decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, output_size, EfficientRNN, n_layers=1, num_split=3, dropout=0.1)
decoder = linear_decoder('general', embedding, hidden_size, output_size, n_layers=num_layers, num_split=3, dropout=0.1).to(device)

#checkpoint = torch.load('checkpoint/ERNN.t7')
#net.load_state_dict(checkpoint['net'])
#best_acc = checkpoint['acc']
#start_epoch = checkpoint['epoch']

'''
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
'''

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load('checkpoint/path/here/check.t7')
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    print('net acc :', best_acc, 'epoch :', start_epoch)

criterion = DALoss(1, 10)
encoder_optimizer = optim.SGD(encoder.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=args.lr)

#encoder_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=encoder_optimizer, factor=0.5, patience=10)
#decoder_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=decoder_optimizer,factor=0.5,patience=10)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    encoder.train()
    decoder.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (text, semantic) in enumerate(dataloaders['train']):
        
        inputs, targets = text, semantic.to(device)
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        inputs = [[lang.word2index[word] for word in text[i]] for i in range(len(text))]
        inputs = torch.LongTensor(zeroPadding(inputs)).transpose(0, 1).to(device)


        outputs, hidden, domain_weight = encoder(inputs, torch.LongTensor([len(seq) for seq in inputs])) # domain weight here!
        outputs = outputs.transpose(0, 1)
        output, hidden = decoder(hidden, outputs)


        loss = criterion(domain_weight, output, targets)
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        train_loss += loss.item()
        _, predicted = output.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx%(len(dataloaders['train'])// 2) == 0 and args.silent: #print every 50%
            print(batch_idx, len(dataloaders['train']), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        

    #encoder_scheduler.step(metrics=train_loss)      # Learning rate decay
    #decoder_scheduler.step(metrics=train_loss)

def test(epoch):
    global best_acc
    encoder.eval()
    decoder.eval()
    test_loss = 0
    correct = 0
    total = 0
    domain_weight = None
    with torch.no_grad():
        for batch_idx, (text, semantic) in enumerate(dataloaders['test']):
            inputs, targets = text, semantic.to(device)

            inputs = [[lang.word2index[word] for word in text[i]] for i in range(len(text))]
            inputs = torch.LongTensor(zeroPadding(inputs)).transpose(0, 1).to(device)

            outputs, hidden, domain_weight = encoder(inputs, torch.LongTensor(
                [len(seq) for seq in inputs]))  # domain weight here!
            outputs = outputs.transpose(0, 1)
            output, hidden = decoder(hidden, outputs)

            loss = criterion(domain_weight, output, targets)

            test_loss += loss.item()
            _, predicted = output.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % (len(dataloaders['test']) // 2) == 0 and args.silent:  # print every 50%
                print(batch_idx, len(dataloaders['test']), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                  % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    print(acc)
    # Write logs
    logfileAcc.write(str(epoch) + '\t' + str(acc) + '\n')
    logfileLoss.write(str(epoch) + '\t' + str(test_loss) + '\n')

    """
    if acc > best_acc:
        print('Saving..  %f' % acc)
        with open('Rpweights.pickle', 'wb') as handle:
            pickle.dump(domain_weight, handle, protocol=pickle.HIGHEST_PROTOCOL)
        best_acc = acc
    """

    '''
        state = {
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        #if not os.path.isdir('checkpoint'):
            #os.mkdir('checkpoint')
        torch.save(state, 'checkpoint/ERNN.t7')
    '''
# not use yet
'''
def predict():
    testset = (root='ts', transform=torchvision.transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)
    net.eval()
    data = "filename,classid\n"
    with torch.no_grad():
        for batch_idx, (inputs, targets, path) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            images = inputs.reshape(-1, 32, 32).to(device)
            outputs = net(inputs)

            _, predicted = outputs.max(1)
            print(path[0].replace("/0/", "/") + ",{0:02d}".format(predicted.data.tolist()[0]))
            data += path[0].replace("/0/", "/")  + ',{0:02d}'.format(predicted.data.tolist()[0]) + '\n'

    with open('/content/gdrive/My Drive/Colab Notebooks/result.csv', 'w') as result:
        result.write(data)
'''

if __name__ == '__main__':
    learning_rate = args.lr
    all_time = time.time()
    for epoch in range(start_epoch, start_epoch+args.epoch):
        state_bfore = copy.deepcopy(encoder.model.state_dict())
        epoch_time = time.time()
        dataloaders['train'].shuffle()
        train(epoch)
        test(epoch)
        state_after = encoder.model.state_dict()
        """
        if epoch%2 == 0 and epoch != 0:
            if dataloaders['train'].get_batch_size() > 1:
                dataloaders['train'].set_batch_size(dataloaders['train'].get_batch_size()//2)
                print("batch decay. Now size: %d"%dataloaders['train'].get_batch_size())
        """
        print("time took for epoch: %f"%(time.time()-epoch_time))
    print("time took for all: %f"%(time.time()-all_time))
    logfileAcc.close()
    logfileLoss.close()