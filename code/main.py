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
from dataloader import *
from utils import Lang
from models.seq2seq import *

parser = argparse.ArgumentParser(description='PyTorch ERNN Training')
parser.add_argument('--lr', default=0.1, type=float, help='Initial learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--predict', action='store_true', help='forward prop')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
batch_size = 1 # should be 1

# Model
print('==> Building model..')

input_size = 128  #same as embedding size
num_layers = 1
num_split = -1
hidden_size = 512
output_size = 2
#net =  ernn.EfficientRNN
net =  rnn.RNN


lang = Lang('eng')
for _, (text, _) in enumerate(dataloaders['train']):
    lang.addSentence(text[0])
for _, (text, _) in enumerate(dataloaders['test']):
    lang.addSentence(text[0])

embedding = nn.Embedding(lang.n_words, input_size)

encoder = EncoderRNN(input_size, hidden_size, embedding, net, n_layers=1, num_split=num_split, dropout=0, device=device).to(device)
# attn_model = Attn('general', hidden_size)
# decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, output_size, EfficientRNN, n_layers=1, num_split=3, dropout=0.1)
decoder = linear_decoder('general', embedding, hidden_size, output_size, n_layers=1, num_split=3, dropout=0.1).to(device)

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

criterion = nn.MSELoss()
encoder_optimizer = optim.SGD(encoder.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=args.lr)


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

        inputs = torch.tensor([[lang.word2index[word] for word in text[0]]]).to(device)
        inputs = torch.LongTensor(zeroPadding(inputs)).transpose(0, 1)

        outputs, hidden = encoder(inputs, torch.LongTensor([len(seq) for seq in inputs])) #second input is for packed sequence. not used yet
        outputs = outputs.transpose(0, 1)
        output, hidden = decoder(hidden, outputs)


        loss = criterion(output[0], targets)
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        train_loss += loss.item()
        _, predicted = output[0].max(1)
        total += targets.size(0)
        _, t_index = targets.max(1)
        correct += predicted.eq(t_index).sum().item()

        if batch_idx%(len(dataloaders['train'])// 5)==0: #print every 20%
            print(batch_idx, len(dataloaders['train']), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    encoder.eval()
    decoder.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (text, semantic) in enumerate(dataloaders['test']):
            inputs, targets = text, semantic.to(device)

            inputs = torch.tensor([[lang.word2index[word] for word in text[0]]]).to(device)
            outputs, hidden = encoder(inputs, torch.LongTensor([len(seq) for seq in [[3, 3, 3, 3]]])) #second input is for packed sequence. not used yet
            output, hidden = decoder(hidden, outputs)

            loss = criterion(output[0], targets)

            test_loss += loss.item()
            _, predicted = output[0].max(1)
            total += targets.size(0)
            _, t_index = targets.max(1)
            correct += predicted.eq(t_index).sum().item()

            if batch_idx % (len(dataloaders['test']) // 5) == 0:  # print every 10%
                print(batch_idx, len(dataloaders['test']), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                  % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    print(acc)
    with open('log.txt', 'a', encoding='utf8') as logfile:
        logfile.write("epoch :" + str(epoch) + ', accuracy :' + str(acc)+'\n')
    '''
    if acc > best_acc:
        print('Saving..  %f' % acc)
        state = {
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        #if not os.path.isdir('checkpoint'):
            #os.mkdir('checkpoint')
        torch.save(state, 'checkpoint/ERNN.t7')
        best_acc = acc
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

    for epoch in range(start_epoch, start_epoch+200):
        train(epoch)
        test(epoch)