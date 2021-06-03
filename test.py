import argparse

from models.get_model import get_model
import torch
from utils.Regularization import Regularization

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch implementation')
    # Root catalog of images
    parser.add_argument('--data-dir', type=str, default='/media/data2/chenjiarong/ImageData')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--print-freq', type=int, default=1000)
    parser.add_argument('--save-epoch-freq', type=int, default=10)
    parser.add_argument('--save-path', type=str, default='/media/data2/chenjiarong/saved-model/Networks')
    parser.add_argument('-save', default=False, action='store_true', help='save model or not')
    parser.add_argument('-write-csv', default=False, action='store_true', help='write into csv or not')
    parser.add_argument('--resume', type=str, default='', help='For training from one checkpoint')
    parser.add_argument('--start-epoch', type=int, default=0, help='Corresponding to the epoch of resume')
    parser.add_argument('--ema-decay', type=float, default=0, help='The decay of exponential moving average ')
    parser.add_argument('--dataset', type=str, default='ImageNet', help='The dataset to be trained')
    parser.add_argument('--num-classes', type=int, default=1000)
    parser.add_argument('--input-size', type=int, default=224)
    parser.add_argument('--width-multiplier', type=float, default=1.0, help='width multiplier')
    parser.add_argument('-dali', default=False, action='store_true', help='Using DALI or not')
    parser.add_argument('--model', type=str, default='resnet-18', help='model to train')
    parser.add_argument('--mode', type=str, default='es', help='normal, ac or es')
    parser.add_argument('-deploy', default=False, action='store_true', help='deploy or not')
    parser.add_argument('--branch-nums', type=int, default=2, help='number of branches used in ESBlock')
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('-all-ones', default=False, action='store_true', help='init mask_weight with all ones or not')
    parser.add_argument('-softmax', default=False, action='store_true', help='using softmax for mask weight or not')

    args = parser.parse_args()

    model = get_model(args)

    ave = torch.zeros(3,3)
    count = 0
    for name, param in model.named_parameters():
        if 'mask_weight' in name:
            if len(param) == 3:
                ave += param.data
                count += 1
    print(ave/count)

    '''
    reg = Regularization(model, 1e-4, 1e-4)

    params = []
    weight_list_l1 = []
    weight_list_l2 = []
    weight_no = []
    for name, param in model.named_parameters():
        print(name)
        if 'mask_weight' in name:
            # mask conv
            weight_list_l1.append((name, param))

        elif 'weight' in name and ('linear' in name or len(param.size()) == 4):
            # weight of normal conv or linear
            weight_list_l2.append(name)
        # do not apply regularization on bias, BN ...
        else:
            weight_no.append(name)
    print('L1')
    print(weight_list_l1)
    '''
    # print('L2')
    # print(weight_list_l2)
    # print('No')
    # print(weight_no)
