import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from functools import reduce
import operator


class LinearARD(nn.Module):
    """
    Dense layer implementation with weights ARD-prior (arxiv:1701.05369)
    """

    def __init__(self, in_features, out_features, bias=True, thresh=3, ard_init=-10):
        super(LinearARD, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.thresh = thresh
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.ard_init = ard_init
        self.log_sigma2 = Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def forward(self, input):
        if self.training:
            epsilon = self.weight.new(self.weight.shape).normal_()
            W = self.weight + epsilon * torch.exp(self.log_sigma2 / 2)
        else:
            W = self.weights_clipped
        return F.linear(input, W) + self.bias

    @property
    def weights_clipped(self):
        clip_mask = self.get_clip_mask()
        return torch.where(clip_mask, torch.zeros_like(self.weight), self.weight)

    def reset_parameters(self):
        self.weight.data.normal_(0, 0.02)
        if self.bias is not None:
            self.bias.data.zero_()
        self.log_sigma2.data.fill_(self.ard_init)

    def get_clip_mask(self):
        log_alpha = self.log_alpha
        return torch.ge(log_alpha, self.thresh)

    def get_reg(self, **kwargs):
        """
        Get weights regularization (KL(q(w)||p(w)) approximation)
        """
        k1, k2, k3 = 0.63576, 1.8732, 1.48695
        C = -k1
        mdkl = k1 * torch.sigmoid(k2 + k3 * self.log_alpha) - \
            0.5 * torch.log1p(torch.exp(-self.log_alpha)) + C
        return -torch.sum(mdkl)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

    def get_dropped_params_cnt(self):
        """
        Get number of dropped weights (with log alpha greater than "thresh" parameter)

        :returns (number of dropped weights, number of all weight)
        """
        return self.get_clip_mask().sum().cpu().numpy()

    @property
    def log_alpha(self):
        log_alpha = self.log_sigma2 - 2 * \
            torch.log(torch.abs(self.weight) + 1e-15)
        return torch.clamp(log_alpha, -10, 10)


class Conv2dARD(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, ard_init=-10, thresh=3):
        bias = False  # Goes to nan if bias = True
        super(Conv2dARD, self).__init__(in_channels, out_channels, kernel_size, stride,
                                        padding, dilation, groups, bias)
        self.bias = None
        self.thresh = thresh
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ard_init = ard_init
        self.log_sigma2 = Parameter(ard_init * torch.ones_like(self.weight))
        # self.log_sigma2 = Parameter(2 * torch.log(torch.abs(self.weight) + eps).clone().detach()+ard_init*torch.ones_like(self.weight))

    def forward(self, input):
        """
        Forward with all regularized connections and random activations (Beyesian mode). Typically used for train
        """
        if self.training == False:
            return F.conv2d(input, self.weights_clipped,
                            self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        W = self.weight
        zeros = torch.zeros_like(W)
        clip_mask = self.get_clip_mask()
        conved_mu = F.conv2d(input, W, self.bias, self.stride,
                             self.padding, self.dilation, self.groups)
        log_alpha = self.log_alpha
        conved_si = torch.sqrt(1e-15 + F.conv2d(input * input,
                                                torch.exp(log_alpha) * W *
                                                W, self.bias, self.stride,
                                                self.padding, self.dilation, self.groups))
        conved = conved_mu + \
            conved_si * \
            torch.normal(torch.zeros_like(conved_mu),
                         torch.ones_like(conved_mu))
        return conved

    @property
    def weights_clipped(self):
        clip_mask = self.get_clip_mask()
        return torch.where(clip_mask, torch.zeros_like(self.weight), self.weight)

    def get_clip_mask(self):
        log_alpha = self.log_alpha
        return torch.ge(log_alpha, self.thresh)

    def get_reg(self, **kwargs):
        """
        Get weights regularization (KL(q(w)||p(w)) approximation)
        """
        k1, k2, k3 = 0.63576, 1.8732, 1.48695
        C = -k1
        log_alpha = self.log_alpha
        mdkl = k1 * torch.sigmoid(k2 + k3 * log_alpha) - \
            0.5 * torch.log1p(torch.exp(-log_alpha)) + C
        return -torch.sum(mdkl)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_channels, self.out_channels, self.bias is not None
        )

    def get_dropped_params_cnt(self):
        """
        Get number of dropped weights (greater than "thresh" parameter)

        :returns (number of dropped weights, number of all weight)
        """
        return self.get_clip_mask().sum().cpu().numpy()

    @property
    def log_alpha(self):
        log_alpha = self.log_sigma2 - 2 * \
            torch.log(torch.abs(self.weight) + 1e-15)
        return torch.clamp(log_alpha, -8, 8)


class ELBOLoss(nn.Module):
    def __init__(self, net, loss_fn):
        super(ELBOLoss, self).__init__()
        self.loss_fn = loss_fn
        self.net = net

    def forward(self, input, target, loss_weight=1., kl_weight=1.):
        assert not target.requires_grad
        # Estimate ELBO
        return loss_weight * self.loss_fn(input, target)  \
            + kl_weight * get_ard_reg(self.net)


def get_ard_reg(module):
    """
    :param module: model to evaluate ard regularization for
    :param reg: auxilary cumulative variable for recursion
    :return: total regularization for module
    """
    if isinstance(module, LinearARD) or isinstance(module, Conv2dARD):
        return module.get_reg()
    elif hasattr(module, 'children'):
        return sum([get_ard_reg(submodule) for submodule in module.children()])
    return 0


def _get_dropped_params_cnt(module):
    if hasattr(module, 'get_dropped_params_cnt'):
        return module.get_dropped_params_cnt()
    elif hasattr(module, 'children'):
        return sum([_get_dropped_params_cnt(submodule) for submodule in module.children()])
    return 0


def _get_params_cnt(module):
    if any([isinstance(module, l) for l in [LinearARD, Conv2dARD]]):
        return reduce(operator.mul, module.weight.shape, 1)
    elif hasattr(module, 'children'):
        return sum(
            [_get_params_cnt(submodule) for submodule in module.children()])
    return sum(p.numel() for p in module.parameters())


def get_dropped_params_ratio(model):
    return _get_dropped_params_cnt(model) * 1.0 / _get_params_cnt(model)
