import sys
sys.path.append('../')
from layers import LinearARD
from torch import nn
import torch

class DenseModel(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_size=150, device=None, activation=None):
        super(DenseModel, self).__init__()
        self.l1 = LinearARD(input_shape, hidden_size)
        self.l2 = LinearARD(hidden_size, output_shape)
        self.device = device
        self.activation = activation

    def forward(self, input):
        return self._forward(input)

    def predict(self, input, clip=False, deterministic=False):
        with torch.no_grad():
            return self._forward(input, clip, deterministic)

    def _forward(self, input, clip=False, deterministic=False):
        x = input
        if self.device: x = x.to(device)
        x = self.l1._forward(x, clip=clip, deterministic=deterministic)
        x = nn.functional.tanh(x)
        x = self.l2._forward(x, clip=clip, deterministic=deterministic)
        if self.activation: x = self.activation(x)
        return x

    def eval_reg(self):
        return self.l1.eval_reg() + self.l2.eval_reg()

    def eval_compression(self):
        l1_params_cnt_dropped, l1_params_cnt_all = self.l1.get_ard()
        l2_params_cnt_dropped, l2_params_cnt_all = self.l2.get_ard()
        return (l1_params_cnt_dropped + l2_params_cnt_dropped) / \
            (l1_params_cnt_all + l2_params_cnt_all)
