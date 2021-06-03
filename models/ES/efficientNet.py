'''
EfficientNet
Ref: https://github.com/ansleliu/EfficientNet.PyTorch/blob/master/efficientnet.py
'''

from torch.nn import functional as F
from collections import OrderedDict
from torch import nn
import torch
from math import ceil, exp

from ..ensure_divisible import ensure_divisible

from .ESBlock import ESBlock

class Swish(nn.Module):
    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x.mul_(x.sigmoid()) if self.inplace else x.mul(x.sigmoid())


class ConvBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, momentum, eps, branch_nums, deploy, gamma_init, init, softmax, learn_mask, mask):

        super(ConvBnAct, self).__init__()

        self.convBnAct = nn.Sequential(OrderedDict([
            ('convBN', ESBlock(branch_nums=branch_nums, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
                               stride=stride, padding=(kernel_size-1)//2, groups=groups, deploy=deploy, gamma_init=gamma_init, init=init, 
                               softmax=softmax, learn_mask=learn_mask, mask=mask, BN_momentum=momentum, BN_eps=eps)),
            # ("conv", nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
            #                    stride=stride, padding=(kernel_size-1)//2, groups=groups, bias=False)),
            # ("bn", nn.BatchNorm2d(num_features=out_channels, momentum=momentum, eps=eps)),
            ("act", Swish(inplace=True))
        ]))

    def forward(self, x):
        return self.convBnAct(x)


class SEModule(nn.Module):
    '''
    SE Module
    '''
    def __init__(self, in_channels, mid_channels):
        super(SEModule, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1, padding=0, bias=True),
            Swish(inplace=True),
            nn.Conv2d(in_channels=mid_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x):
        return x * torch.sigmoid(self.se(self.avg_pool(x)))


class Bottleneck(nn.Module):
    '''
    The basic unit
    '''
    def __init__(self, in_channels, out_channels, expand_ratio, kernel_size, stride, reduction_ratio, BN_momentum, BN_eps,
                 branch_nums, deploy, gamma_init, init, softmax, learn_mask, mask):
        super(Bottleneck, self).__init__()
        
        # Whether to use residual structure
        self.use_residual = (stride == 1 and in_channels == out_channels)
        # Whether to use SE Moudule
        self.use_se = reduction_ratio is not None and reduction_ratio > 1

        self.expand_ratio = expand_ratio

        exp_size = int(in_channels * expand_ratio)

        # step 1. Expansion phase/Pointwise convolution
        if expand_ratio != 1:
            self.expansion = nn.Sequential(OrderedDict([
                ("conv", nn.Conv2d(in_channels=in_channels, out_channels=exp_size, kernel_size=1, 
                                stride=1, padding=0, groups=1, bias=False)),
                ("bn", nn.BatchNorm2d(num_features=exp_size, momentum=BN_momentum, eps=BN_eps)),
                ("act", Swish(inplace=True))
            ]))

        # step 2. Depthwise convolution phase
        self.depthwise = ConvBnAct(in_channels=exp_size, out_channels=exp_size, kernel_size=kernel_size,
                                   stride=stride, groups=exp_size, momentum=BN_momentum, eps=BN_eps, 
                                   branch_nums=branch_nums, deploy=deploy, gamma_init=gamma_init, 
                                   init=init, softmax=softmax, learn_mask=learn_mask, mask=mask)
        
        # step 3. Squeeze and Excitation
        if self.use_se:
            mid_channels = max(1, in_channels//reduction_ratio)
            self.se = SEModule(in_channels=exp_size, mid_channels=mid_channels)

        # step 4. Linear Pointwise convolution phase
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels=exp_size, out_channels=out_channels, kernel_size=1,
                      stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels, eps=BN_eps, momentum=BN_momentum)
        )

    def forward(self, x):
        out = x
        if self.expand_ratio != 1:
            x = self.expansion(x)
        x = self.depthwise(x)
        if self.use_se:
            x = self.se(x)
        x = self.pointwise(x)
        if self.use_residual:
            # do not use drop connect, according to https://github.com/zsef123/EfficientNets-PyTorch/blob/master/models/effnet.py#L40
            return out + x
        else:
            return x


class EfficientNet(nn.Module):
    def __init__(self, mode="b0", num_classes=1000, input_size=224, width_multiplier=1.0, BN_momentum=0.01, BN_eps=1e-3,
                 branch_nums=1, deploy=False, gamma_init=None, init='', softmax=False, learn_mask=False, same_mask=False):
        super(EfficientNet, self).__init__()

        arch_params = {
            # arch width_multi depth_multi input_size dropout_rate
            'b0': (1.0, 1.0, 224, 0.2),
            'b1': (1.0, 1.1, 240, 0.2),
            'b2': (1.1, 1.2, 260, 0.3),
            'b3': (1.2, 1.4, 300, 0.3),
            'b4': (1.4, 1.8, 380, 0.4),
            'b5': (1.6, 2.2, 456, 0.4),
            'b6': (1.8, 2.6, 528, 0.5),
            'b7': (2.0, 3.1, 600, 0.5),
        }
        width_multi, depth_multi, input_s, dropout_rate = arch_params[mode]

        width_multiplier *= width_multi

        s = 2
        if input_size == 32 or input_size == 56:
            # using cifar-10, cifar-100 or Tiny-ImageNet
            s = 1

        mask3, mask5 = None, None
        if same_mask:
            if init == 'all-ones':
                mask3, mask5 = [torch.ones(3, 3) for _ in range(branch_nums)], [torch.ones(5, 5) for _ in range(branch_nums)]
            elif init == 'cross':
                mask3, mask5 = [torch.zeros(3, 3) for _ in range(branch_nums)], [torch.zeros(5, 5) for _ in range(branch_nums)]
                for idx in range(branch_nums):
                    for i in range(3):
                        mask3[idx][i][1] = 1
                        mask3[idx][1][i] = 1
                    for i in range(5):
                        mask5[idx][i][2] = 1
                        mask5[idx][2][i] = 1
            elif init == 'gauss':
                mask3, mask5 = [torch.rand(3, 3) for _ in range(branch_nums)], [torch.rand(5, 5) for _ in range(branch_nums)]
                for idx in range(branch_nums):
                    for i in range(3):
                        for j in range(3):
                            mask3[idx][i][j] = exp(-(abs(i - 1) + abs(j - 1))**2/2)
                    for i in range(5):
                        for j in range(5):
                            mask5[idx][i][j] = exp(-(abs(i - 2) + abs(j - 2))**2/2)
            else:
                mask3, mask5 = [torch.rand(3, 3) for _ in range(branch_nums)], [torch.rand(5, 5) for _ in range(branch_nums)]
                for idx in range(branch_nums):
                    nn.init.kaiming_normal_(mask3[idx], mode='fan_out')
                    nn.init.kaiming_normal_(mask5[idx], mode='fan_out')
            if learn_mask:
                for idx in range(branch_nums):
                    mask3[idx], mask5[idx] = nn.Parameter(mask3[idx]), nn.Parameter(mask5[idx])
            else:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                for idx in range(branch_nums):
                    mask3, mask5 = mask3[idx].to(device), mask5[idx].to(device)

        settings = [
            # expand_ratio, channels_num, layers_num, kernel_size, stride, reduction_ratio
            [1, 16, 1, 3, 1, 4],   # 3x3, 112 -> 112
            [6, 24, 2, 3, s, 4],   # 3x3, 112 ->  56
            [6, 40, 2, 5, 2, 4],   # 5x5, 56  ->  28
            [6, 80, 3, 3, 2, 4],   # 3x3, 28  ->  14
            [6, 112, 3, 5, 1, 4],  # 5x5, 14  ->  14
            [6, 192, 4, 5, 2, 4],  # 5x5, 14  ->   7
            [6, 320, 1, 3, 1, 4],  # 3x3, 7   ->   7
        ]
        
        first_channels = 32
        last_channels = 1280
        out_channels = self._round_channels(first_channels, width_multiplier)

        feature_extraction_layers = []

        stem = ConvBnAct(in_channels=3, out_channels=out_channels, kernel_size=3, stride=s, groups=1, 
                         momentum=BN_momentum, eps=BN_eps, branch_nums=branch_nums, deploy=deploy, 
                         gamma_init=gamma_init, init=init, softmax=softmax, learn_mask=learn_mask, 
                         mask=mask3)
        feature_extraction_layers.append(stem)

        in_channels = out_channels
        for expand_ratio, channels_num, layers_num, kernel_size, stride, reduction_ratio in settings:
            out_channels = self._round_channels(channels_num, width_multiplier)
            repeats_times = self._round_repeats(layers_num, depth_multi)

            mask = mask3 if kernel_size == 3 else mask5

            for _ in range(repeats_times):
                feature_extraction_layers.append(
                    Bottleneck(in_channels=in_channels, out_channels=out_channels, expand_ratio=expand_ratio, kernel_size=kernel_size, 
                               stride=stride, reduction_ratio=reduction_ratio, BN_momentum=BN_momentum, BN_eps=BN_eps, branch_nums=branch_nums, 
                               deploy=deploy, gamma_init=gamma_init, init=init, softmax=softmax, learn_mask=learn_mask, mask=mask)
                )
                in_channels = out_channels
                stride = 1
        
        # the last stage
        last_channels_num = self._round_channels(last_channels, width_multiplier)
        feature_extraction_layers.append(
            nn.Sequential(OrderedDict([
                ("conv", nn.Conv2d(in_channels=in_channels, out_channels=last_channels_num, kernel_size=1, 
                                stride=1, padding=0, groups=1, bias=False)),
                ("bn", nn.BatchNorm2d(num_features=last_channels_num, momentum=BN_momentum, eps=BN_eps)),
                ("act", Swish(inplace=True))
            ]))
        )
        feature_extraction_layers.append(nn.AdaptiveAvgPool2d(1))

        self.features = nn.Sequential(*feature_extraction_layers)

        ########################################################################################################################
        # Classification part
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(last_channels_num, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        '''
        Initialize the weights
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _round_channels(self, channels, width_multiplier):
        if width_multiplier == 1.0:
            return channels
        return ensure_divisible(channels * width_multiplier)

    @staticmethod
    def _round_repeats(repeats, depth_multi):
        if depth_multi == 1.0:
            return repeats
        return int(ceil(depth_multi * repeats))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def EfficientNetB0(args):
    return EfficientNet(mode='b0', num_classes=args.num_classes, input_size=args.input_size, 
                        width_multiplier=args.width_multiplier, BN_momentum=0.01, BN_eps=1e-3,
                        branch_nums=args.branch_nums, deploy=args.deploy, gamma_init=1/(args.branch_nums + 1),
                        init=args.init, softmax=args.softmax, learn_mask=args.learn_mask, same_mask=args.same_mask)


if __name__ == "__main__":
    from torchsummaryX import summary
    
    model = EfficientNet(mode='b0')
    model.eval()
    summary(model, torch.zeros((1, 3, 224, 224)))
