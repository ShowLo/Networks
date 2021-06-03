'''
L1 or L2 Regularization
Ref: https://github.com/PanJinquan/pytorch-learning-notes/blob/master/image_classification/train_resNet.py
'''

import torch

class Regularization(torch.nn.Module):

    def __init__(self, model, weight_decay_l2, weight_decay_l1=0, bias_and_bn=False):
        ''' 
        p=2: L2 Regularization, p=1: L1 Regularization
        '''
        super(Regularization, self).__init__()
        self.weight_decay_l1 = weight_decay_l1
        self.weight_decay_l2 = weight_decay_l2
        self.bias_and_bn = bias_and_bn
        self.weight_info(model)

    def to(self, device):
        '''
        :param device: cuda or cpu
        '''
        self.device = device
        super().to(device)
        return self

    def forward(self, model):
        # get the latest weight
        weight_list_l1, weight_list_l2 = self.get_weight(model)
        reg_loss = self.regularization_loss(weight_list_l1, weight_list_l2)
        return reg_loss

    def get_weight(self, model):
        '''
        the weight with L1 Regularization to be stored in `weight_list_l1`
        the weight with L2 Regularization to be stored in `weight_list_l2`
        '''
        weight_list_l1 = []
        weight_list_l2 = []
        for name, param in model.named_parameters():
            if 'mask_weight' in name:
                # mask conv
                weight_list_l1.append((name, param))
            elif 'weight' in name and ('linear' in name or len(param.size()) == 4):
                # weight of normal conv or linear
                weight_list_l2.append((name, param))
            # do not apply regularization on bias, BN ...
            else:
                # test, add bias and bn
                if self.bias_and_bn:
                    weight_list_l2.append((name, param))
                
        return weight_list_l1, weight_list_l2

    def regularization_loss(self, weight_list_l1, weight_list_l2):
        '''
        if weight_decay_l1 is 0, the weight in `weight_list_l1` will be applied L2 Regularization
        otherwise, it will be applied L1 Regularization
        '''
        reg_loss = 0

        if self.weight_decay_l1 == 0:
            loss = 0
            weight_list = weight_list_l1 + weight_list_l2
            for name, w in weight_list:
                # loss += torch.norm(w, p=2)
                loss += torch.sum(torch.pow(w, 2))
            reg_loss = self.weight_decay_l2/2*loss
        else:
            loss1, loss2 = 0, 0
            for name, w in weight_list_l1:
                # loss1 += torch.norm(w, p=1)
                loss1 += torch.sum(torch.abs(w))
            for name, w in weight_list_l2:
                # loss2 += torch.norm(w, p=2)
                loss2 += torch.sum(torch.pow(w, 2))
            reg_loss = self.weight_decay_l1*loss1 + self.weight_decay_l2/2*loss2

        return reg_loss

    def weight_info(self, model):
        weight_list_l1, weight_list_l2 = self.get_weight(model)
        print('---------------regularization weight---------------')
        print('------------------------L1-------------------------')
        for name, w in weight_list_l1:
            print(name)
        print('------------------------L2-------------------------')
        for name, w in weight_list_l2:
            print(name)
        print('---------------------------------------------------')