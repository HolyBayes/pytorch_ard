import torch
from torch import nn
from torch.nn import Parameter
from functools import reduce
import operator

class LinearARD(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearARD, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.log_sigma2 = Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.normal_(std=0.01)
        if self.bias is not None:
            self.bias.data.uniform_(0, 0)
        self.log_sigma2.data.uniform_(-10,-10)

    @staticmethod
    def clip(tensor, to=8):
        return torch.clamp(tensor, -to, to)

    def forward(self, input):
        return self._forward(input)

    def forward_w_clip(self, input):
        return self._forward(input, clip=True)

    def forward_deterministic(self, input):
        return self._forward(input, deterministic=True)

    def _forward(self, input, clip=False, deterministic=False, thresh=3):
        log_alpha = self.clip(self.log_sigma2 - torch.log(self.weight ** 2))
        clip_mask = torch.ge(log_alpha, thresh)
        W = self.weight
        if deterministic:
            activation = input.matmul(torch.where(clip_mask,
                torch.Tensor([0]), W).t())
        else:
            if clip:
                W = torch.where(clip_mask, torch.Tensor([0]), self.weight)
            mu = input.matmul(W.t())
            si = torch.sqrt((input * input)\
                .matmul(((torch.exp(log_alpha) * self.weight * self.weight)+1e-8).t()))
            activation = mu + torch.normal(torch.zeros_like(mu), torch.ones_like(mu)) * si
        return activation + self.bias


    def eval_reg(self, **kwargs):
        k1, k2, k3 = 0.63576, 1.8732, 1.48695; C = -k1
        log_alpha = self.clip(self.log_sigma2 - torch.log(self.weight ** 2))
        mdkl = k1 * torch.sigmoid(k2 + k3 * log_alpha) - 0.5 * torch.log1p(torch.exp(-log_alpha)) + C
        return -torch.sum(mdkl)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

    def get_ard(self, thresh=3, **kwargs):
        log_alpha = self.log_sigma2 - 2 * torch.log(torch.abs(self.weight))
        params_cnt_dropped = int((log_alpha > thresh).sum().cpu().numpy())
        params_cnt_all = reduce(operator.mul, log_alpha.shape, 1)
        return params_cnt_dropped, params_cnt_all

    def get_reg(self):
        log_alpha = self.log_sigma2 - 2 * torch.log(torch.abs(self.weight))
        return log_alpha.min(), log_alpha.max()
