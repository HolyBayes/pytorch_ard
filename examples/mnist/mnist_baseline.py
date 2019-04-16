import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np

import os, sys
sys.path.append('../')
import time

from models import LeNet_MNIST


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt_file = 'checkpoint/ckpt_baseline.t7'

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = datasets.MNIST('./data', train=True, download=True,
               transform=transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Normalize((0.1307,), (0.3081,))
               ]))
trainloader = torch.utils.data.DataLoader(trainset,batch_size=128, shuffle=True)

testset = datasets.MNIST('./data', train=False, transform=transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Normalize((0.1307,), (0.3081,))
               ]))
testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=True)


n_classes = 10

# Model
print('==> Building model..')
model = LeNet_MNIST(1, n_classes).to(device)

if device.type == 'cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

if os.path.isfile(ckpt_file):
    checkpoint = torch.load(ckpt_file)
    model.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = []
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        # scheduler.step(loss)
        optimizer.step(loss)

        train_loss.append(loss.item())
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    print('Train loss: %.3f' % np.mean(train_loss))
    print('Train accuracy: %.3f%%' % (correct * 100.0/total))

def test(epoch):
    global best_acc
    model.eval()
    test_loss = []
    correct = 0
    total = 0
    inference_time_seconds = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            start_ts = time.time()
            outputs = model(inputs)
            inference_time_seconds += time.time() - start_ts
            loss = criterion(outputs, targets)

            test_loss.append(loss.item())
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # Save checkpoint.
    acc = 100.*correct/total
    print('Test loss: %.3f' % np.mean(test_loss))
    print('Test accuracy: %.3f%%' % acc)
    print('Inference time: %.2f seconds' % inference_time_seconds)
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, ckpt_file)
        best_acc = acc


for epoch in range(start_epoch, start_epoch+100):
    train(epoch)
    test(epoch)
