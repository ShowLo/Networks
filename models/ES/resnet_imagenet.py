'''
ResNet for ImageNet in PyTorch.
Ref: https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..ensure_divisible import ensure_divisible
# from collections import OrderedDict

from .ESBlock import ESBlock


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, branch_nums=2, deploy=False, gamma_init=None, init='', softmax=False, learn_mask=False):
        super(BasicBlock, self).__init__()
        self.convBn1 = ESBlock(branch_nums=branch_nums, in_channels=in_planes, out_channels=planes, kernel_size=3, 
                               stride=stride, padding=1, deploy=deploy, gamma_init=gamma_init, init=init, 
                               softmax=softmax, learn_mask=learn_mask)
        self.convBn2 = ESBlock(branch_nums=branch_nums, in_channels=planes, out_channels=planes, kernel_size=3, 
                               stride=1, padding=1, deploy=deploy, gamma_init=gamma_init, init=init, 
                               softmax=softmax, learn_mask=learn_mask)
        # self.conv1 = nn.Conv2d(in_channels=in_planes, out_channels=planes, kernel_size=3, stride=stride, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(num_features=planes)
        # self.conv2 = nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn2 = nn.Sequential(OrderedDict([('lastBN', nn.BatchNorm2d(num_features=planes))]))
        #nn.BatchNorm2d(num_features=planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_planes, out_channels=self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_features=self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.convBn1(x))
        out = self.convBn2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, branch_nums=2, deploy=False, gamma_init=None, init='', softmax=False, learn_mask=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_planes, out_channels=planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=planes)
        self.convBn2 = ESBlock(branch_nums=branch_nums, in_channels=planes, out_channels=planes, kernel_size=3, 
                               stride=stride, padding=1, deploy=deploy, gamma_init=gamma_init, init=init, 
                               softmax=softmax, learn_mask=learn_mask)
        # self.conv2 = nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=stride, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(num_features=planes)
        self.conv3 = nn.Conv2d(in_channels=planes, out_channels=self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_planes, out_channels=self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_features=self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.convBn2(out))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000, width_multiplier=1.0, branch_nums=2, deploy=False, 
                 gamma_init=None, init='', softmax=False):
        super(ResNet, self).__init__()
        
        divisor = 8
        
        self.in_planes = ensure_divisible(64*width_multiplier, divisor)

        self.convBn1 = ESBlock(branch_nums=branch_nums, in_channels=3, out_channels=ensure_divisible(64*width_multiplier, divisor), kernel_size=7, 
                               stride=2, padding=3, deploy=deploy, gamma_init=gamma_init, init=init, softmax=softmax, learn_mask=learn_mask)
        # self.conv1 = nn.Conv2d(in_channels=3, out_channels=ensure_divisible(64*width_multiplier, divisor), kernel_size=7, stride=2, padding=3, bias=False)
        # self.bn1 = nn.BatchNorm2d(num_features=ensure_divisible(64*width_multiplier, divisor))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, ensure_divisible(64*width_multiplier, divisor), num_blocks[0], stride=1, 
                                       branch_nums=branch_nums, deploy=deploy, gamma_init=gamma_init, init=init, 
                                       softmax=softmax, learn_mask=learn_mask)
        self.layer2 = self._make_layer(block, ensure_divisible(128*width_multiplier, divisor), num_blocks[1], stride=2, 
                                       branch_nums=branch_nums, deploy=deploy, gamma_init=gamma_init, init=init, 
                                       softmax=softmax, learn_mask=learn_mask)
        self.layer3 = self._make_layer(block, ensure_divisible(256*width_multiplier, divisor), num_blocks[2], stride=2, 
                                       branch_nums=branch_nums, deploy=deploy, gamma_init=gamma_init, init=init, 
                                       softmax=softmax, learn_mask=learn_mask)
        self.layer4 = self._make_layer(block, ensure_divisible(512*width_multiplier, divisor), num_blocks[3], stride=2, 
                                       branch_nums=branch_nums, deploy=deploy, gamma_init=gamma_init, init=init, 
                                       softmax=softmax, learn_mask=learn_mask)
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(ensure_divisible(512*width_multiplier*block.expansion, divisor), num_classes)

        self._initialize_weights()

    def _make_layer(self, block, planes, num_blocks, stride, branch_nums, deploy, gamma_init, init, softmax, learn_mask):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, branch_nums, deploy, gamma_init, init, softmax, learn_mask))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.convBn1(x))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pooling(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def _initialize_weights(self):
        '''
        Initialize the weights
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.BatchNorm2d):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def ResNet18(args):
    return ResNet(BasicBlock, [2, 2, 2, 2], args.num_classes, args.width_multiplier, args.branch_nums, 
                  args.deploy, 1/(args.branch_nums + 1), args.init, args.softmax, args.learn_mask)


def ResNet34(args):
    return ResNet(BasicBlock, [3, 4, 6, 3], args.num_classes, args.width_multiplier, args.branch_nums, 
                  args.deploy, 1/(args.branch_nums + 1), args.init, args.softmax, args.learn_mask)


def ResNet50(args):
    return ResNet(Bottleneck, [3, 4, 6, 3], args.num_classes, args.width_multiplier, args.branch_nums, 
                  args.deploy, 1/(args.branch_nums + 1), args.init, args.softmax, args.learn_mask)


def ResNet101(args):
    return ResNet(Bottleneck, [3, 4, 23, 3], args.num_classes, args.width_multiplier, args.branch_nums, 
                  args.deploy, 1/(args.branch_nums + 1), args.init, args.softmax, args.learn_mask)


def ResNet152(args):
    return ResNet(Bottleneck, [3, 8, 36, 3], args.num_classes, args.width_multiplier, args.branch_nums, 
                  args.deploy, 1/(args.branch_nums + 1), args.init, args.softmax, args.learn_mask)