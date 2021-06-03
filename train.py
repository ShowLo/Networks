# -*- coding: UTF-8 -*-

'''
Train the model
Ref: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import time
import os
import argparse
import copy
from math import cos, pi

from utils.statistics import *
from tricks.EMA import EMA
from tricks.LabelSmoothing import LabelSmoothingLoss
# from DataLoader import dataloaders
from utils.ResultWriter import ResultWriter
from tricks.CosineLR import CosineWarmupLR, CosineAnnealingWarmRestarts
from tricks.Mixup import mixup_data, mixup_criterion
from models.get_model import get_model
from configs.config import CONFIG
from utils.Regularization import Regularization
from utils.MaskRegularization import MaskRegularization, PolarLoss

def train(args, model, dataloader, loader_len, criterion, reg_loss, optimizer, scheduler, use_gpu, epoch, ema=None, save_file_name='train.csv'):
    '''
    train the model
    '''
    if args.write_csv:
        # save result every epoch
        resultWriter = ResultWriter(args.save_path, save_file_name)
        if epoch == 0:
            resultWriter.create_csv(['epoch', 'loss', 'top-1', 'top-5', 'lr'])

    # use gpu or not
    device = torch.device('cuda' if use_gpu else 'cpu')

    # statistical information
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    total_losses = AverageMeter('TotalLoss', ':.2e')
    model_losses = AverageMeter('MoldelLoss', ':.2e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        loader_len,
        [batch_time, data_time, total_losses, model_losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))
    
    # update lr here if using stepLR
    if args.lr_decay == 'step':
        scheduler.step(epoch)
    
    # Set model to training mode
    model.train()

    end = time.time()

    # Iterate over data
    for i, (inputs, labels) in enumerate(dataloader):
        # measure data loading time
        data_time.update(time.time() - end)

        inputs = inputs.to(device)
        labels = labels.to(device)

        if args.mixup:
            # using mixup
            inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, args.mixup_alpha)
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            acc1_a, acc5_a = accuracy(outputs, labels_a, topk=(1, 5))
            acc1_b, acc5_b = accuracy(outputs, labels_b, topk=(1, 5))
            # measure accuracy and record loss
            acc1 = lam * acc1_a + (1 - lam) * acc1_b
            acc5 = lam * acc5_a + (1 - lam) * acc5_b
        else:
            # normal forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # measure accuracy and record loss
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
        
        model_losses.update(loss.item(), inputs.size(0))

        if reg_loss is not None:
            reg = reg_loss(model)
            loss += reg

        # zero the parameter gradients
        optimizer.zero_grad()

        total_losses.update(loss.item(), inputs.size(0))
        top1.update(acc1[0], inputs.size(0))
        top5.update(acc5[0], inputs.size(0))
            
        # backward + optimize
        loss.backward()

        if args.lr_decay == 'cos':
            # update lr here if using cosine lr decay
            scheduler.step(epoch * loader_len + i)
        elif args.lr_decay == 'sgdr':
            # update lr here if using sgdr
            scheduler.step(epoch + i / loader_len)
        optimizer.step()
        if args.ema_decay > 0:
            # EMA update after training(every iteration)
            ema.update()
                
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    if args.write_csv:   
        # write training result to file
        resultWriter.write_csv([epoch, total_losses.avg, top1.avg.item(), top5.avg.item(), scheduler.optimizer.param_groups[0]['lr']])
    
    print()
    # there is a bug in get_lr() if using pytorch 1.1.0, see https://github.com/pytorch/pytorch/issues/22107
    # so here we don't use get_lr()
    # print('lr:%.6f' % scheduler.get_lr()[0])
    print('lr:%.6f' % scheduler.optimizer.param_groups[0]['lr'])
    print('Train ***    TotalLoss:{total_losses.avg:.2e}    ModelLoss:{model_losses.avg:.2e}    Acc@1:{top1.avg:.2f}    Acc@5:{top5.avg:.2f}'.format(total_losses=total_losses, model_losses=model_losses, top1=top1, top5=top5))

    if args.save and epoch % args.save_epoch_freq == 0 and epoch != 0:
        torch.save(model.state_dict(), os.path.join(args.save_path, "epoch_" + str(epoch) + ".pth"))

def validate(args, model, dataloader, loader_len, criterion, use_gpu, epoch, ema=None, save_file_name='val.csv'):
    '''
    validate the model
    '''

    if args.write_csv:
        # save result every epoch
        resultWriter = ResultWriter(args.save_path, save_file_name)
        if epoch == 0:
            resultWriter.create_csv(['epoch', 'loss', 'top-1', 'top-5'])

    device = torch.device('cuda' if use_gpu else 'cpu')

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        loader_len,
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))
    if args.ema_decay > 0:
        # apply EMA at validation stage
        ema.apply_shadow()
    # Set model to evaluate mode
    model.eval()

    end = time.time()

    # Iterate over data
    for i, (inputs, labels) in enumerate(dataloader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # measure accuracy and record loss
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            

    if args.ema_decay > 0:
        # restore the origin parameters after val
        ema.restore()

    if args.write_csv:
        # write val result to file
        resultWriter.write_csv([epoch, losses.avg, top1.avg.item(), top5.avg.item()])

    print(' Val  ***    Loss:{losses.avg:.2e}    Acc@1:{top1.avg:.2f}    Acc@5:{top5.avg:.2f}'.format(losses=losses, top1=top1, top5=top5))

    if args.save and epoch % args.save_epoch_freq == 0 and epoch != 0:
        torch.save(model.state_dict(), os.path.join(args.save_path, "epoch_" + str(epoch) + ".pth"))

    top1_acc = top1.avg.item()
    top5_acc = top5.avg.item()
    
    return top1_acc, top5_acc

def train_model(args, model, dataloader, loaders_len, criterion, reg_loss, optimizer, scheduler, use_gpu):
    '''
    train the model
    '''
    since = time.time()

    ema = None
    # exponential moving average
    if args.ema_decay > 0:
        ema = EMA(model, decay=args.ema_decay)
        ema.register()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    correspond_top5 = 0.0

    for epoch in range(args.start_epoch, args.num_epochs):

        epoch_time = time.time()
        train(args, model, dataloader['train'], loaders_len['train'], criterion, reg_loss, optimizer, scheduler, use_gpu, epoch, ema)
        top1_acc, top5_acc = validate(args, model, dataloader['val'], loaders_len['val'], criterion, use_gpu, epoch, ema)
        epoch_time = time.time() - epoch_time
        print('Time of epoch-[{:d}/{:d}] : {:.0f}h {:.0f}m {:.0f}s\n'.format(epoch, args.num_epochs, epoch_time // 3600, (epoch_time % 3600) // 60, epoch_time % 60))

        # deep copy the model if it has higher top-1 accuracy
        if top1_acc > best_acc:
            best_acc = top1_acc
            correspond_top5 = top5_acc
            if args.ema_decay > 0:
                ema.apply_shadow()
            best_model_wts = copy.deepcopy(model.state_dict())
            if args.ema_decay > 0:
                ema.restore()

    print(os.path.split(args.save_path)[-1])
    print('Best val top-1 Accuracy: {:4f}'.format(best_acc))
    print('Corresponding top-5 Accuracy: {:4f}'.format(correspond_top5))
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}d {:.0f}h {:.0f}m {:.0f}s'.format(time_elapsed // 86400, (time_elapsed % 86400) // 3600, (time_elapsed % 3600) // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(best_model_wts)
    # save best model weights
    if args.save:
        torch.save(model.state_dict(), os.path.join(args.save_path, 'best_model_wts-' + '{:.2f}'.format(best_acc) + '.pth'))
    return model

if __name__ == '__main__':

    import warnings
    warnings.filterwarnings('ignore')

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
    parser.add_argument('--model', type=str, default='resnet18', help='model to train')
    parser.add_argument('--mode', type=str, default='normal', help='normal, ac or es')
    parser.add_argument('--init', type=str, default='', help='method to init mask_weight')
    parser.add_argument('--option', type=str, default='A', help='option used in resnet-cifar')
    parser.add_argument('-deploy', default=False, action='store_true', help='deploy or not')
    parser.add_argument('--branch-nums', type=int, default=0, help='number of branches used in ESBlock')
    # parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='label smoothing')
    parser.add_argument('--lr-decay', type=str, default='step', help='learning rate decay method, step, cos or sgdr')
    parser.add_argument('--step-size', type=int, default=3, help='step size in stepLR()')
    parser.add_argument('--gamma', type=float, default=0.99, help='gamma in stepLR()')
    parser.add_argument('--lr-min', type=float, default=0, help='minium lr using in CosineWarmupLR')
    parser.add_argument('--warmup-epochs', type=int, default=0, help='warmup epochs using in CosineWarmupLR')
    parser.add_argument('--T-0', type=int, default=10, help='T_0 in CosineAnnealingWarmRestarts')
    parser.add_argument('--T-mult', type=int, default=2, help='T_mult in CosineAnnealingWarmRestarts')
    parser.add_argument('--decay-rate', type=float, default=1, help='decay rate in CosineAnnealingWarmRestarts')
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='weight decay')
    parser.add_argument('--l1-reg', type=float, default=0, help='L1 Regularization')
    parser.add_argument('--l2-reg', type=float, default=0, help='L2 Regularization')
    parser.add_argument('-l1-mask', default=False, action='store_true', help='L1 Regularization for mask weight')
    parser.add_argument('-polar-mask', default=False, action='store_true', help='Polar Loss for mask weight')
    parser.add_argument('--polar-lambda', type=float, default=1e-5, help='lambda for polar loss')
    parser.add_argument('--polar-t', type=float, default=1, help='t in polar loss')
    parser.add_argument('-softmax', default=False, action='store_true', help='using softmax for mask weight or not')
    parser.add_argument('-learn-mask', default=False, action='store_true', help='learn mask weight or not')
    parser.add_argument('-same-mask', default=False, action='store_true', help='use the same mask weight for all conv or not')
    # parser.add_argument('--bn-momentum', type=float, default=0.1, help='momentum in BatchNorm2d')
    parser.add_argument('-use-seed', default=False, action='store_true', help='using fixed random seed or not')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('-deterministic', default=False, action='store_true', help='torch.backends.cudnn.deterministic')
    parser.add_argument('-nbd', default=False, action='store_true', help='no bias decay')
    # parser.add_argument('-zero-gamma', default=False, action='store_true', help='zero gamma in BatchNorm2d when init')
    parser.add_argument('-mixup', default=False, action='store_true', help='mixup or not')
    parser.add_argument('--mixup-alpha', type=float, default=0.2, help='alpha used in mixup')
    parser.add_argument('-bias-and-bn', default=False, action='store_true', help='whether to use weight decay for bias and bn layer')
    args = parser.parse_args()

    args.lr_decay = args.lr_decay.lower()
    args.dataset = args.dataset.lower()
    args.optimizer = args.optimizer.lower()

    # folder to save
    folder_name = [args.model+(('-'+args.option) if 'resnet' in args.model else ''), args.mode+(('-branch'+str(args.branch_nums)+'-init|'+str(args.init))+'|learnMask'+str(args.learn_mask)+'|sameMask'+str(args.same_mask) if args.mode=='es' else ''), 'deploy'+str(args.deploy), args.dataset, 'lr'+str(args.lr), 'bs'+str(args.batch_size), 'ed'+str(args.ema_decay), 'ls'+str(args.label_smoothing), args.optimizer+str(args.weight_decay), 'L1-'+str(args.l1_reg), 'L2-'+str(args.l2_reg), 'softmax'+str(args.softmax),  'epochs'+str(args.num_epochs), 'seed'+(str(args.seed) if args.use_seed else 'None'), 'determin'+str(args.deterministic), 'NoBiasDecay'+str(args.nbd), 'mixup'+(str(args.mixup_alpha) if args.mixup else 'False'), 'biasBn'+str(args.bias_and_bn)]
    if args.lr_decay == 'step':
        folder_name.append('-'.join([args.lr_decay,str(args.step_size),str(args.gamma)]))
    elif args.lr_decay == 'cos':
        folder_name.append('-'.join([args.lr_decay,str(args.warmup_epochs),str(args.lr_min)]))
    elif args.lr_decay == 'sgdr':
        folder_name.append('-'.join([args.lr_decay,str(args.T_0),str(args.T_mult),str(args.warmup_epochs),str(args.decay_rate)]))
    folder_name = '_'.join(folder_name)
    print(folder_name)
    args.save_path = os.path.join(args.save_path, folder_name)
    if args.save and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # use gpu or not
    use_gpu = torch.cuda.is_available()
    print("use_gpu:{}".format(use_gpu))

    # set random seed
    if args.use_seed:
        print('Using fixed random seed')
        torch.manual_seed(args.seed)
    else:
        print('do not use fixed random seed')
    if use_gpu:
        if args.use_seed:
            torch.cuda.manual_seed(args.seed)
            if torch.cuda.device_count() > 1:
                torch.cuda.manual_seed_all(args.seed)
        if args.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
        print('torch.backends.cudnn.deterministic:' + str(args.deterministic))

    # read data
    # dataloaders = dataloaders(args)
    if args.dali and (args.dataset == 'tinyimagenet' or args.dataset == 'imagenet'):
        if args.dataset == 'imagenet':
            from datasets.DALIDataLoader import get_dali_imageNet_train_loader, get_dali_imageNet_val_loader
            train_loader, train_loader_len = get_dali_imageNet_train_loader(data_path=args.data_dir, batch_size=args.batch_size, seed=args.seed, num_threads=args.num_workers)
            val_loader, val_loader_len = get_dali_imageNet_val_loader(data_path=args.data_dir, batch_size=args.batch_size, seed=args.seed, num_threads=args.num_workers)
            dataloaders = {'train' : train_loader, 'val' : val_loader}
            loaders_len = {'train': train_loader_len, 'val' : val_loader_len}
        elif args.dataset == 'tinyimagenet':
            from datasets.DALIDataLoader import get_dali_tinyImageNet_train_loader, get_dali_tinyImageNet_val_loader
            train_loader, train_loader_len = get_dali_tinyImageNet_train_loader(data_path=args.data_dir, batch_size=args.batch_size, seed=args.seed, num_threads=args.num_workers)
            val_loader, val_loader_len = get_dali_tinyImageNet_val_loader(data_path=args.data_dir, batch_size=args.batch_size, seed=args.seed, num_threads=args.num_workers)
            dataloaders = {'train' : train_loader, 'val' : val_loader}
            loaders_len = {'train': train_loader_len, 'val' : val_loader_len}
    else:
        from datasets.DataLoader import dataloaders
        loaders = dataloaders(args)
        train_loader = loaders['train']
        train_loader_len = len(train_loader)
        val_loader = loaders['val']
        val_loader_len = len(val_loader)
        dataloaders = {'train' : train_loader, 'val' : val_loader}
        loaders_len = {'train': train_loader_len, 'val' : val_loader_len}

    # different input size and number of classes for different datasets
    args.input_size = CONFIG[args.dataset].input_size
    args.num_classes = CONFIG[args.dataset].num_classes
    
    # get model
    # model = MobileNetV3(mode=args.mode, classes_num=num_class, input_size=input_size, 
    #                 width_multiplier=args.width_multiplier, dropout=args.dropout, 
    #                 BN_momentum=args.bn_momentum, zero_gamma=args.zero_gamma)
    model = get_model(args)

    if use_gpu:
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        device = torch.device('cuda')
        model.to(device)
    else:
        device = torch.device('cpu')
        model.to(device)

    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            model.load_state_dict(torch.load(args.resume))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))
            exit()

    if args.label_smoothing > 0:
        # using Label Smoothing
        criterion = LabelSmoothingLoss(args.num_classes, label_smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    reg_loss = None
    if args.weight_decay == 0:
        # using regularization to replace when `weight decay = 0`
        reg_loss = Regularization(model=model, weight_decay_l2=args.l2_reg, 
                                  weight_decay_l1=args.l1_reg, bias_and_bn=args.bias_and_bn).to(device)
    elif args.l1_mask:
        reg_loss = MaskRegularization(model=model, weight_decay=args.l1_reg, p=1)
    elif args.polar_mask:
        reg_loss = PolarLoss(model=model, t=args.polar_t)

    
    if args.optimizer == 'sgd':
        if args.nbd:
            from tricks.NoBiasDecay import noBiasDecay
            optimizer_ft = optim.SGD(
                # no bias decay
                noBiasDecay(model, args.lr, args.weight_decay), 
                momentum=0.9)
        elif args.l1_mask or args.polar_mask:
            params = []
            for key, value in model.named_parameters():
                if not value.requires_grad:
                    continue
                lr = args.lr
                # 2*lr for bias, lr for others
                if 'bias' in key:
                    apply_lr = 2 * lr
                else:
                    apply_lr = lr
                if 'mask_weight' in key:
                    wd = 0
                else:
                    wd = args.weight_decay
                params += [{'params': [value], 'lr': apply_lr, 'weight_decay': wd}]
                optimizer_ft = optim.SGD(params, momentum=0.9)
        else:
            params = []
            for key, value in model.named_parameters():
                if not value.requires_grad:
                    continue
                lr = args.lr
                # 2*lr for bias, lr for others
                if 'bias' in key:
                    apply_lr = 2 * lr
                else:
                    apply_lr = lr
                params += [{'params': [value], 'lr': apply_lr}]
            optimizer_ft = optim.SGD(params, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer == 'rmsprop':
        optimizer_ft = optim.RMSprop(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer_ft = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.lr_decay == 'step':
        # Decay LR by a factor of 0.99 every 3 epoch
        lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=args.step_size, gamma=args.gamma)
    elif args.lr_decay == 'cos':
        lr_scheduler = CosineWarmupLR(optimizer=optimizer_ft, epochs=args.num_epochs, iter_in_one_epoch=loaders_len['train'], lr_min=args.lr_min, warmup_epochs=args.warmup_epochs)
    elif args.lr_decay == 'sgdr':
        lr_scheduler = CosineAnnealingWarmRestarts(optimizer=optimizer_ft, T_0=args.T_0, T_mult=args.T_mult, warmup_epochs=args.warmup_epochs, decay_rate=args.decay_rate)

    before3 = torch.zeros(3,3).cuda()
    before5 = torch.zeros(5,5).cuda()
    count3 = 0
    count5 = 0
    for name, param in model.named_parameters():
        if 'mask_weight' in name:
            print(id(param))
            if len(param) == 3:
                before3 += param.data
                count3 += 1
            if len(param) == 5:
                before5 += param.data
                count5 += 1
    if count3 > 0:
        before3 /= count3
    if count5 > 0:
        before5 /= count5

    model = train_model(args=args,
                        model=model,
                        dataloader=dataloaders,
                        loaders_len=loaders_len,
                        criterion=criterion,
                        reg_loss=reg_loss,
                        optimizer=optimizer_ft,
                        scheduler=lr_scheduler,
                        use_gpu=use_gpu)

    '''import torch.nn.functional as F
    for name, param in model.named_parameters():
        if 'mask_weight' in name:
            print(name)
            print(param)
            if args.softmax:
                ks = len(param)
                print(F.softmax(param.view(1, -1)).reshape(ks, ks))
    '''
    ave3 = torch.zeros(3,3).cuda()
    ave5 = torch.zeros(5,5).cuda()
    # clamp_ave = torch.zeros(3,3).cuda()
    count3 = 0
    count5 = 0
    for name, param in model.named_parameters():
        if 'mask_weight' in name:
            print(name)
            print(id(param))
            print(param)
            if len(param) == 3:
                ave3 += param.data
                # clamp_ave += torch.clamp(param.data, 0, 1)
                count3 += 1
                # if count3 % 10 == 0:
                    # print(param.data)
            if len(param) == 5:
                ave5 += param.data
                # clamp_ave += torch.clamp(param.data, 0, 1)
                count5 += 1
                # if count5 % 10 == 0:
                    # print(param.data)
    if count3 > 0:
        ave3 /= count3
        # clamp_ave /= count
        print('before:')
        print(before3)
        print('after:')
        print(ave3)
        # print('clamp_ave:')
        # print(clamp_ave)
    if count5 > 0:
        ave5 /= count5
        # clamp_ave /= count
        print('before:')
        print(before5)
        print('after:')
        print(ave5)