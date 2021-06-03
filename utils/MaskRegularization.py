'''
L1 or L2 Regularization
Ref: https://github.com/PanJinquan/pytorch-learning-notes/blob/master/image_classification/train_resNet.py
'''

import torch

class MaskRegularization(torch.nn.Module):

    def __init__(self, model, weight_decay=0, p=1):
        '''
        p=2: L2 Regularization, p=1: L1 Regularization
        '''
        super(MaskRegularization, self).__init__()
        self.weight_decay = weight_decay
        self.p = p
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
        mask_weight = self.get_mask_weight(model)
        reg_loss = self.regularization_loss(mask_weight)
        return reg_loss

    def get_mask_weight(self, model):
        mask_weight = []
        for name, param in model.named_parameters():
            if 'mask_weight' in name:
                mask_weight.append((name, param))
        
        return mask_weight

    def regularization_loss(self, mask_weight):
        '''
        '''
        reg_loss = 0

        loss = 0
        for name, w in mask_weight:
            if self.p == 1:
                loss += torch.sum(torch.abs(w))
            else:
                loss += torch.sum(torch.pow(w, self.p))
        reg_loss = self.weight_decay/self.p*loss

        return reg_loss

    def weight_info(self, model):
        mask_weight = self.get_mask_weight(model)
        print('---------------mask weight---------------')
        for name, w in mask_weight:
            print(name)
        print('---------------------------------------------------')



class PolarLoss(torch.nn.Module):

    def __init__(self, model, t):
        ''' 
        '''
        super(PolarLoss, self).__init__()
        self.t = t
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
        mask_weight = self.get_weight(model)
        polar_loss = self.polar_loss(mask_weight)
        return polar_loss

    def get_weight(self, model):
        '''
        '''
        mask_weight = []
        for name, param in model.named_parameters():
            if 'mask_weight' in name:
                # mask conv
                mask_weight.append((name, param))
                
        return mask_weight

    def polar_loss(self, mask_weight):
        '''
        '''
        loss = 0
        for name, w in mask_weight:
            mean = torch.mean(w)
            loss += self.t * torch.sum(torch.abs(w)) - torch.sum(torch.abs(w - mean))
        return loss

    def weight_info(self, model):
        mask_weight = self.get_weight(model)
        print('--------------------polar weight-------------------')
        for name, w in mask_weight:
            print(name)
        print('---------------------------------------------------')