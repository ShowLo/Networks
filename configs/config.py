from collections import namedtuple

dataset = namedtuple('dataset', ['input_size', 'num_classes'])
CONFIG = {
    'imagenet' : dataset(input_size=224, num_classes=1000),
    'tinyimagenet' : dataset(input_size=56, num_classes=200),
    'cifar100' : dataset(input_size=32, num_classes=100),
    'cifar10' : dataset(input_size=32, num_classes=10),
    'svhn' : dataset(input_size=32, num_classes=10)
}