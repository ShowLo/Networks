'''
ResNet for CIFAR
Ref: https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

from collections import OrderedDict

from .ESBlock import ESBlock

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1, option='A', branch_nums=2, deploy=False, gamma_init=None, init='', softmax=False, learn_mask=False):
        super(BasicBlock, self).__init__()

        self.convBn1 = ESBlock(branch_nums=branch_nums, in_channels=in_planes, out_channels=planes, kernel_size=3, 
                               stride=stride, padding=1, deploy=deploy, gamma_init=gamma_init, init=init, 
                               softmax=softmax, learn_mask=learn_mask)
        self.convBn2 = ESBlock(branch_nums=branch_nums, in_channels=planes, out_channels=planes, kernel_size=3, 
                               stride=1, padding=1, deploy=deploy, gamma_init=gamma_init, init=init, 
                               softmax=softmax, learn_mask=learn_mask)
        #nn.Conv2d(in_channels=in_planes, out_channels=planes, kernel_size=3, stride=stride, padding=1, bias=False)
        #self.bn1 = nn.BatchNorm2d(num_features=planes)
        # self.conv2 = nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn2 = nn.Sequential(OrderedDict([('lastBN', nn.BatchNorm2d(num_features=planes))]))#nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                '''
                For CIFAR10 ResNet paper uses option A.
                '''
                print('using option A')
                num = planes - in_planes
                self.shortcut = LambdaLayer(lambda x : F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, num//2, num//2), 'constant', 0))
            elif option == 'B':
                print('using option B')
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_channels=in_planes, out_channels=planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(num_features=planes)
                )

    def forward(self, x):
        #out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.convBn1(x))
        out = self.convBn2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, branch_nums=2, option='A', deploy=False, gamma_init=None, init='', softmax=False, learn_mask=False):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.convBn1 = ESBlock(branch_nums=branch_nums, in_channels=3, out_channels=16, kernel_size=3, 
                               stride=1, padding=1, deploy=deploy, gamma_init=gamma_init, init=init, 
                               softmax=softmax, learn_mask=learn_mask)
        # self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(num_features=16)

        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, branch_nums=branch_nums, option=option, 
                                       deploy=deploy, gamma_init=gamma_init, init=init, softmax=softmax, 
                                       learn_mask=learn_mask)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, branch_nums=branch_nums, option=option, 
                                       deploy=deploy, gamma_init=gamma_init, init=init, softmax=softmax,
                                       learn_mask=learn_mask)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, branch_nums=branch_nums, option=option, 
                                       deploy=deploy, gamma_init=gamma_init, init=init, softmax=softmax,
                                       learn_mask=learn_mask)
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(64, num_classes)

        self._initialize_weights()

    def _make_layer(self, block, planes, num_blocks, stride, branch_nums, option, deploy, gamma_init, init, softmax, learn_mask):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, branch_nums=branch_nums, option=option, 
                                deploy=deploy, gamma_init=gamma_init, init=init, softmax=softmax, 
                                learn_mask=learn_mask))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.convBn1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
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


def ResNet20(args):
    return ResNet(BasicBlock, [3, 3, 3], args.num_classes, args.branch_nums, args.option, args.deploy, 
                  1/(args.branch_nums + 1), args.init, args.softmax, args.learn_mask)


def ResNet32(args):
    return ResNet(BasicBlock, [5, 5, 5], args.num_classes, args.branch_nums, args.option, args.deploy, 
                  1/(args.branch_nums + 1), args.init, args.softmax, args.learn_mask)


def ResNet44(args):
    return ResNet(BasicBlock, [7, 7, 7], args.num_classes, args.branch_nums, args.option, args.deploy, 
                  1/(args.branch_nums + 1), args.init, args.softmax, args.learn_mask)


def ResNet56(args):
    return ResNet(BasicBlock, [9, 9, 9], args.num_classes, args.branch_nums, args.option, args.deploy, 
                  1/(args.branch_nums + 1), args.init, args.softmax, args.learn_mask)


def ResNet110(args):
    return ResNet(BasicBlock, [18, 18, 18], args.num_classes, args.branch_nums, args.option, args.deploy, 
                  1/(args.branch_nums + 1), args.init, args.softmax, args.learn_mask)


def ResNet1202(args):
    return ResNet(BasicBlock, [200, 200, 200], args.num_classes, args.branch_nums, args.option, args.deploy, 
                  1/(args.branch_nums + 1), args.init, args.softmax, args.learn_mask)
