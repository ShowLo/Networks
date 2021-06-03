import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.nn.init as init
from math import exp

class MaskConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, init, softmax, learn_mask, mask):
        super(MaskConv, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.softmax = softmax

        self.original_weight = torch.rand(out_channels, in_channels//groups, kernel_size, kernel_size)
        nn.init.kaiming_normal_(self.original_weight, mode='fan_out')
        self.original_weight = nn.Parameter(self.original_weight)
        if mask is None:
            if init == 'all-ones':
                self.mask_weight = torch.ones(kernel_size, kernel_size)
            elif init == 'cross':
                tmp = torch.zeros(kernel_size, kernel_size)
                for i in range(kernel_size):
                    tmp[i][kernel_size//2] = 1
                    tmp[kernel_size//2][i] = 1
                self.mask_weight = tmp
            elif init == 'gauss':
                tmp = torch.rand(kernel_size, kernel_size)
                # nn.init.kaiming_normal_(tmp, mode='fan_out')
                for i in range(kernel_size):
                    for j in range(kernel_size):
                        tmp[i][j] = exp(-(abs(i - kernel_size//2) + abs(j - kernel_size//2))**2/2)

                self.mask_weight = tmp
                # print(self.mask_weight)
            else:
                self.mask_weight = torch.rand(kernel_size, kernel_size)
                nn.init.kaiming_normal_(self.mask_weight, mode='fan_out')
            if learn_mask:
                self.mask_weight = nn.Parameter(self.mask_weight)
            else:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.mask_weight = self.mask_weight.to(device)
        else:
            self.mask_weight = mask

    def forward(self, x):
        if self.softmax:
            softmax_weight = F.softmax(self.mask_weight.view(1, -1)).reshape(self.kernel_size, self.kernel_size)
            weight = self.original_weight * softmax_weight
        else:
            # self.mask_weight = nn.Parameter(torch.clamp(self.mask_weight, 0, 1))
            # print(self.mask_weight)
            # self.mask_weight.clamp_(0, 1)
            self.mask_weight.data.clamp_(0, 1)
            weight = self.original_weight * self.mask_weight
        return F.conv2d(input=x, weight=weight, bias=self.bias, stride=self.stride, padding=self.padding, 
                        dilation=self.dilation, groups=self.groups)

class ESBlock(nn.Module):
    def __init__(self, branch_nums, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, 
                 bias=False, deploy=False, gamma_init=None, init='', softmax=False, learn_mask=False, mask=None, 
                 BN_momentum=0.1, BN_eps=1e-5):
        super(ESBlock, self).__init__()

        self.deploy = deploy

        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, 
                      padding=padding, dilation=dilation, groups=groups, bias=bias),
            nn.BatchNorm2d(num_features=out_channels, momentum=BN_momentum, eps=BN_eps)
        )
        self.parallel_convs = nn.ModuleList()
        for i in range(branch_nums):
            self.parallel_convs.append(
                nn.Sequential(
                    MaskConv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, 
                             padding=padding, dilation=dilation, groups=groups, bias=bias, init=init, softmax=softmax, 
                             learn_mask=learn_mask, mask=mask[i] if mask is not None else None),
                    nn.BatchNorm2d(num_features=out_channels, momentum=BN_momentum, eps=BN_eps)
                )
            )
        
        if gamma_init is not None:
            self.init_gamma(gamma_init)

    def forward(self, x):
        y = self.primary_conv(x)
        if self.deploy:
            return y
        for pc in self.parallel_convs:
            y += pc(x)
        return y

    def init_gamma(self, gamma_value):
        init.constant_(self.primary_conv[1].weight, gamma_value)
        for pc in self.parallel_convs:
            init.constant_(pc[1].weight, gamma_value)
        print('init gamma of of all conv as ', gamma_value)

    def single_init(self):
        init.constant_(self.primary_conv[1].weight, 1)
        for pc in self.parallel_convs:
            init.constant_(pc[1].weight, 0)
        print('init gamma of primary conv as 1, parallel conv as 0')
