from .mobileNetV3 import MobileNetV3
from .resnet_imagenet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from .resnet_cifar import ResNet20, ResNet32, ResNet44, ResNet56, ResNet110, ResNet1202
from .efficientNet import EfficientNet

def get_model(args):
    if args.model.lower() == 'mobilenetv3-large':
        return MobileNetV3(mode='large', classes_num=args.num_classes, input_size=args.input_size, width_multiplier=args.width_multiplier)
    elif args.model.lower() == 'mobilenetv3-small':
        return MobileNetV3(mode='small', classes_num=args.num_classes, input_size=args.input_size, width_multiplier=args.width_multiplier) 
    # ResNet for ImageNet
    elif args.model.lower() == 'resnet-18':
        return ResNet18(args)
    elif args.model.lower() == 'resnet-34':
        return ResNet34(args)
    elif args.model.lower() == 'resnet-50':
        return ResNet50(args)
    elif args.model.lower() == 'resnet-101':
        return ResNet101(args)
    elif args.model.lower() == 'resnet-152':
        return ResNet152(args)
    # ResNet for CIFAR
    elif args.model.lower() == 'resnet-20':
        return ResNet20(args)
    elif args.model.lower() == 'resnet-32':
        return ResNet32(args)
    elif args.model.lower() == 'resnet-44':
        return ResNet44(args)
    elif args.model.lower() == 'resnet-56':
        return ResNet56(args)
    elif args.model.lower() == 'resnet-110':
        return ResNet110(args)
    elif args.model.lower() == 'resnet-1202':
        return ResNet1202(args)
    # EfficientNet
    elif args.model.lower() == 'efficientnet-b0':
        return EfficientNet(mode='b0', num_classes=args.num_classes, input_size=args.input_size, width_multiplier=args.width_multiplier)
    else:
        raise NotImplementedError('Model: ' + args.model + ' not implemented yet!')