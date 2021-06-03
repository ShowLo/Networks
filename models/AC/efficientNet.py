'''
EfficientNet
Ref: https://github.com/ansleliu/EfficientNet.PyTorch/blob/master/efficientnet.py
'''

from torch.nn import functional as F
from collections import OrderedDict
from torch import nn
import torch
import math

from ..ensure_divisible import ensure_divisible

from .acblock import ACBlock


class Swish(nn.Module):
    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x.mul_(x.sigmoid()) if self.inplace else x.mul(x.sigmoid())


class ConvBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, momentum, eps, deploy, gamma_init):

        super(ConvBnAct, self).__init__()

        self.convBnAct = nn.Sequential(OrderedDict([
            ("convBn", ACBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, 
                               padding=(kernel_size-1)//2, groups=groups, BN_momentum=momentum, BN_eps=eps, deploy=deploy, 
                               gamma_init=gamma_init)),
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
    def __init__(self, in_channels, out_channels, expand_ratio, kernel_size, stride, reduction_ratio, BN_momentum, BN_eps, deploy, gamma_init):
        super(Bottleneck, self).__init__()
        
        # Whether to use residual structure
        self.use_residual = (stride == 1 and in_channels == out_channels)
        # Whether to use SE Moudule
        self.use_se = reduction_ratio is not None and reduction_ratio > 1

        self.expand_ratio = expand_ratio

        exp_size = int(in_channels * expand_ratio)

        # step 1. Expansion phase/Pointwise convolution
        if expand_ratio != 1:
            self.expansion = ConvBnAct(in_channels=in_channels, out_channels=exp_size, kernel_size=1, 
                                       stride=1, groups=1, momentum=BN_momentum, eps=BN_eps, deploy=deploy, 
                                       gamma_init=gamma_init)

        # step 2. Depthwise convolution phase
        self.depthwise = ConvBnAct(in_channels=exp_size, out_channels=exp_size, kernel_size=kernel_size,
                                   stride=stride, groups=exp_size, momentum=BN_momentum, eps=BN_eps, deploy=deploy, 
                                   gamma_init=gamma_init)
        
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
    def __init__(self, mode="b0", num_classes=1000, input_size=224, width_multiplier=1.0, BN_momentum=0.01, BN_eps=1e-3, deploy=False, gamma_init=1/3):
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
                         momentum=BN_momentum, eps=BN_eps, deploy=deploy, gamma_init=gamma_init)
        feature_extraction_layers.append(stem)

        in_channels = out_channels
        for expand_ratio, channels_num, layers_num, kernel_size, stride, reduction_ratio in settings:
            out_channels = self._round_channels(channels_num, width_multiplier)
            repeats_times = self._round_repeats(layers_num, depth_multi)

            for _ in range(repeats_times):
                feature_extraction_layers.append(
                    Bottleneck(in_channels=in_channels, out_channels=out_channels, expand_ratio=expand_ratio, kernel_size=kernel_size, 
                               stride=stride, reduction_ratio=reduction_ratio, BN_momentum=BN_momentum, BN_eps=BN_eps, deploy=deploy, 
                               gamma_init=gamma_init)
                )
                in_channels = out_channels
                stride = 1
        
        # the last stage
        last_channels_num = self._round_channels(last_channels, width_multiplier)
        feature_extraction_layers.append(
            ConvBnAct(in_channels=in_channels, out_channels=last_channels_num, kernel_size=1, 
                      stride=1, groups=1, momentum=BN_momentum, eps=BN_eps, deploy=deploy, 
                      gamma_init=gamma_init)
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
        return int(math.ceil(depth_multi * repeats))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    from torchsummaryX import summary
    
    model = EfficientNet(mode='b0')
    model.eval()
    summary(model, torch.zeros((1, 3, 224, 224)))
