import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import numpy as np

import os, sys
sys.path.append('../')

from models import LeNetARD
from torch_ard import get_ard_reg, get_dropped_params_ratio


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt_file = 'checkpoint/ckpt_ard.t7'

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
reg_factor = 1e-4

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

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
model = LeNetARD(3, len(classes)).to(device)

# if device.type == 'cuda':
#     model = torch.nn.DataParallel(model)
#     cudnn.benchmark = True

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
        loss = criterion(outputs, targets) + \
            trainloader.batch_size * reg_factor * get_ard_reg(model) / len(trainset)
        loss.backward()
        # scheduler.step(loss)
        optimizer.step()

        train_loss.append(loss.item())
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    print('Train loss: %.2f' % np.mean(train_loss))
    print('Train accuracy: %.2f%%' % (correct * 100.0/total))

def test(epoch):
    global best_acc
    model.eval()
    test_loss = []
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model.predict(inputs)
            loss = criterion(outputs, targets)

            test_loss.append(loss.item())
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # Save checkpoint.
    acc = 100.*correct/total
    print('Test loss: %.2f' % np.mean(test_loss))
    print('Test accuracy: %.2f%%' % acc)
    print('Compression: %.2f%%' % (100.*get_dropped_params_ratio(model)))
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


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
