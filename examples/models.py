import torch_ard as nn_ard
from torch import nn
import torch.nn.functional as F
import torch

class DenseModelARD(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_size=150, activation=None):
        super(DenseModelARD, self).__init__()
        self.l1 = nn_ard.LinearARD(input_shape, hidden_size)
        self.l2 = nn_ard.LinearARD(hidden_size, output_shape)
        self.activation = activation
        self._init_weights()

    def forward(self, input):
        return self._forward(input)

    def predict(self, input, clip=False, deterministic=True, thresh=3):
        with torch.no_grad():
            return self._forward(input, clip, deterministic, thresh=thresh)

    def _forward(self, input, clip=False, deterministic=False, thresh=3):
        x = input
        x = self.l1._forward(x, clip=clip, deterministic=deterministic, thresh=thresh)
        x = nn.functional.tanh(x)
        x = self.l2._forward(x, clip=clip, deterministic=deterministic, thresh=thresh)
        if self.activation: x = self.activation(x)
        return x

    def _init_weights(self):
        for layer in self.children():
            if hasattr(layer, 'weight'): nn.init.xavier_uniform(layer.weight, gain=nn.init.calculate_gain('relu'))


class DenseModel(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_size=150, activation=None):
        super(DenseModel, self).__init__()
        self.l1 = nn.Linear(input_shape, hidden_size)
        self.l2 = nn.Linear(hidden_size, output_shape)
        self.activation = activation
        self._init_weights()

    def forward(self, input):
        x = input
        x = self.l1(x)
        x = nn.functional.tanh(x)
        x = self.l2(x)
        if self.activation: x = self.activation(x)
        return x

    def predict(self, input):
        with torch.no_grad():
            return self.forward(input)

    def _init_weights(self):
        for layer in self.children():
            if hasattr(layer, 'weight'): nn.init.xavier_uniform(layer.weight, gain=nn.init.calculate_gain('relu'))

class LeNet(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(input_shape, 20, 5)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.l1 = nn.Linear(50*5*5, 500)
        self.l2 = nn.Linear(500, output_shape)
        self._init_weights()

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.shape[0], -1)
        out = F.relu(self.l1(out))
        return F.log_softmax(self.l2(out), dim=1)

    def _init_weights(self):
        for layer in self.children():
            if hasattr(layer, 'weight'): nn.init.xavier_uniform(layer.weight, gain=nn.init.calculate_gain('relu'))

class LeNetARD(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(LeNetARD, self).__init__()
        self.conv1 = nn_ard.Conv2dARD(input_shape, 20, 5)
        self.conv2 = nn_ard.Conv2dARD(20, 50, 5)
        self.l1 = nn_ard.LinearARD(50*5*5, 500)
        self.l2 = nn_ard.LinearARD(500, output_shape)
        self._init_weights()

    def forward(self, input):
        return self._forward(input)

    def predict(self, input, clip=False, deterministic=True, thresh=3):
        with torch.no_grad():
            return self._forward(input, clip, deterministic, thresh=thresh)

    def _forward(self, x, clip=False, deterministic=False, thresh=3):
        out = F.relu(self.conv1._forward(x, clip, deterministic, thresh))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2._forward(out, clip, deterministic, thresh))
        out = F.max_pool2d(out, 2)
        out = out.view(out.shape[0], -1)
        out = F.relu(self.l1._forward(out, clip, deterministic, thresh))
        return F.log_softmax(self.l2._forward(out, clip, deterministic, thresh), dim=1)

    def _init_weights(self):
        for layer in self.children():
            if hasattr(layer, 'weight'): nn.init.xavier_uniform(layer.weight, gain=nn.init.calculate_gain('relu'))


class LeNet_MNIST(LeNet):
    def __init__(self, input_shape, output_shape):
        super(LeNet_MNIST, self).__init__(input_shape, output_shape)
        self.l1 = nn.Linear(50*4*4, 500)
        super(LeNet_MNIST, self)._init_weights()

class LeNetARD_MNIST(LeNetARD):
    def __init__(self, input_shape, output_shape):
        super(LeNetARD_MNIST, self).__init__(input_shape, output_shape)
        self.l1 = nn_ard.LinearARD(50*4*4, 500)
        super(LeNetARD_MNIST, self)._init_weights()
