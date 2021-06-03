'''
EfficientNetV2
Ref: https://github.com/d-li14/efficientnetv2.pytorch
'''

from torch.nn import functional as F
from collections import OrderedDict
from torch import nn
import torch
import math

# from ..ensure_divisible import ensure_divisible
def ensure_divisible(number, divisor=8, min_value=None):
    '''
    Ensure that 'number' can be 'divisor' divisible
    Reference from original tensorflow repo:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    '''
    if min_value is None:
        min_value = divisor
    new_num = max(min_value, int(number + divisor / 2) // divisor * divisor)
    if new_num < 0.9 * number:
        new_num += divisor
    return new_num


class Swish(nn.Module):
    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x.mul_(x.sigmoid()) if self.inplace else x.mul(x.sigmoid())


class ConvBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, momentum, eps):

        super(ConvBnAct, self).__init__()

        self.convBnAct = nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
                               stride=stride, padding=(kernel_size-1)//2, groups=groups, bias=False)),
            ("bn", nn.BatchNorm2d(num_features=out_channels, momentum=momentum, eps=eps)),
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


class MBConv(nn.Module):
    '''
    The basic unit
    '''
    def __init__(self, in_channels, out_channels, expand_ratio, kernel_size, stride, reduction_ratio, BN_momentum, BN_eps):
        super(MBConv, self).__init__()
        
        # Whether to use residual structure
        self.use_residual = (stride == 1 and in_channels == out_channels)
        # Whether to use SE Moudule
        self.use_se = reduction_ratio is not None and reduction_ratio > 1

        self.expand_ratio = expand_ratio

        exp_size = int(in_channels * expand_ratio)

        # step 1. Expansion phase/Pointwise convolution
        if expand_ratio != 1:
            self.expansion = ConvBnAct(in_channels=in_channels, out_channels=exp_size, kernel_size=1, 
                                       stride=1, groups=1, momentum=BN_momentum, eps=BN_eps)

        # step 2. Depthwise convolution phase
        self.depthwise = ConvBnAct(in_channels=exp_size, out_channels=exp_size, kernel_size=kernel_size,
                                   stride=stride, groups=exp_size, momentum=BN_momentum, eps=BN_eps)
        
        # step 3. Squeeze and Excitation
        if self.use_se:
            mid_channels = max(1, ensure_divisible(in_channels//reduction_ratio))
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

class FusedMBConv(nn.Module):
    '''
    The basic unit
    '''
    def __init__(self, in_channels, out_channels, expand_ratio, kernel_size, stride, reduction_ratio, BN_momentum, BN_eps):
        super(FusedMBConv, self).__init__()
        
        # Whether to use residual structure
        self.use_residual = (stride == 1 and in_channels == out_channels)
        # Whether to use SE Moudule
        self.use_se = reduction_ratio is not None and reduction_ratio > 1

        self.expand_ratio = expand_ratio

        exp_size = int(in_channels * expand_ratio)

        # step 1
        self.expansion = ConvBnAct(in_channels=in_channels, out_channels=exp_size, kernel_size=kernel_size, 
                                   stride=stride, groups=1, momentum=BN_momentum, eps=BN_eps)

        # step 2. Squeeze and Excitation
        if self.use_se:
            mid_channels = max(1, ensure_divisible(in_channels//reduction_ratio))
            self.se = SEModule(in_channels=exp_size, mid_channels=mid_channels)

        # step 4. Linear Pointwise convolution phase
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels=exp_size, out_channels=out_channels, kernel_size=1,
                      stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels, eps=BN_eps, momentum=BN_momentum)
        )

    def forward(self, x):
        out = x
        x = self.expansion(x)
        if self.use_se:
            x = self.se(x)
        x = self.pointwise(x)
        if self.use_residual:
            return out + x
        else:
            return x


class EfficientNet(nn.Module):
    def __init__(self, mode="b0", num_classes=1000, input_size=224, width_multiplier=1.0, BN_momentum=0.01, BN_eps=1e-3):
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

        stem = ConvBnAct(in_channels=3, out_channels=out_channels, kernel_size=3, stride=s, groups=1, momentum=BN_momentum, eps=BN_eps)
        feature_extraction_layers.append(stem)

        in_channels = out_channels
        for expand_ratio, channels_num, layers_num, kernel_size, stride, reduction_ratio in settings:
            out_channels = self._round_channels(channels_num, width_multiplier)
            repeats_times = self._round_repeats(layers_num, depth_multi)

            for _ in range(repeats_times):
                feature_extraction_layers.append(
                    MBConv(in_channels=in_channels, out_channels=out_channels, expand_ratio=expand_ratio, kernel_size=kernel_size, 
                               stride=stride, reduction_ratio=reduction_ratio, BN_momentum=BN_momentum, BN_eps=BN_eps)
                )
                in_channels = out_channels
                stride = 1
        
        # the last stage
        last_channels_num = self._round_channels(last_channels, width_multiplier)
        feature_extraction_layers.append(
            ConvBnAct(in_channels=in_channels, out_channels=last_channels_num, kernel_size=1, 
                      stride=1, groups=1, momentum=BN_momentum, eps=BN_eps)
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


class EfficientNetV2(nn.Module):
    def __init__(self, mode='S', num_classes=1000, width_multiplier=1.0, BN_momentum=0.01, BN_eps=1e-3, dropout_rate=0.2):
        super(EfficientNetV2, self).__init__()

        arch_params = {
            # expand_ratio, channels_num, layers_num, kernel_size, stride, reduction_ratio, conv_type(0: MBConv, 1: FusedMBConv)
            'S': [
                [1,  24,  2, 3, 1, None, 1],
                [4,  48,  4, 3, 2, None, 1],
                [4,  64,  4, 3, 2, None, 1],
                [4, 128,  6, 3, 2,    4, 0],
                [6, 160,  9, 3, 1,    4, 0],
                [6, 256, 15, 3, 2,    4, 0]
            ],
            'M': [
                [1,  24,  3, 3, 1, None, 1],
                [4,  48,  5, 3, 2, None, 1],
                [4,  80,  5, 3, 2, None, 1],
                [4, 160,  7, 3, 2,    4, 0],
                [6, 176, 14, 3, 1,    4, 0],
                [6, 304, 18, 3, 2,    4, 0],
                [6, 512,  5, 3, 1,    4, 0]
            ],
            'L': [
                [1,  32,  4, 3, 1, None, 1],
                [4,  64,  7, 3, 2, None, 1],
                [4,  96,  7, 3, 2, None, 1],
                [4, 192, 10, 3, 2,    4, 0],
                [6, 224, 19, 3, 1,    4, 0],
                [6, 384, 25, 3, 2,    4, 0],
                [6, 640,  7, 3, 1,    4, 0]
            ],
            'XL': [
                [1,  32,  4, 3, 1, None, 1],
                [4,  64,  8, 3, 2, None, 1],
                [4,  96,  8, 3, 2, None, 1],
                [4, 192, 16, 3, 2,    4, 0],
                [6, 256, 24, 3, 1,    4, 0],
                [6, 512, 32, 3, 2,    4, 0],
                [6, 640,  8, 3, 1,    4, 0]
            ]
        }

        settings = arch_params[mode]
        
        first_channels = 24
        last_channels = 1280
        out_channels = self._round_channels(first_channels, width_multiplier)

        feature_extraction_layers = []

        stem = ConvBnAct(in_channels=3, out_channels=out_channels, kernel_size=3, stride=2, groups=1, momentum=BN_momentum, eps=BN_eps)
        feature_extraction_layers.append(stem)

        in_channels = out_channels
        for expand_ratio, channels_num, layers_num, kernel_size, stride, reduction_ratio, conv_type in settings:
            out_channels = self._round_channels(channels_num, width_multiplier)

            for _ in range(layers_num):
                if conv_type == 0:
                    feature_extraction_layers.append(
                        MBConv(in_channels=in_channels, out_channels=out_channels, expand_ratio=expand_ratio, kernel_size=kernel_size, 
                                stride=stride, reduction_ratio=reduction_ratio, BN_momentum=BN_momentum, BN_eps=BN_eps)
                    )
                elif conv_type == 1:
                    feature_extraction_layers.append(
                        FusedMBConv(in_channels=in_channels, out_channels=out_channels, expand_ratio=expand_ratio, kernel_size=kernel_size,
                                    stride=stride, reduction_ratio=reduction_ratio, BN_momentum=BN_momentum, BN_eps=BN_eps)
                    )
                else:
                    raise ValueError('conv_type should be 0 or 1!')
                in_channels = out_channels
                stride = 1
        
        # the last stage
        last_channels_num = self._round_channels(last_channels, width_multiplier)
        feature_extraction_layers.append(
            ConvBnAct(in_channels=in_channels, out_channels=last_channels_num, kernel_size=1, 
                      stride=1, groups=1, momentum=BN_momentum, eps=BN_eps)
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

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    from torchsummaryX import summary
    
    # model = EfficientNet(mode='b3')
    # model.eval()
    # summary(model, torch.zeros((1, 3, 300, 300)))
    model = EfficientNetV2(mode='S')
    model.eval()
    summary(model, torch.zeros((1, 3, 384, 384)))
