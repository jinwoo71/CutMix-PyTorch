# original code: https://github.com/dyhan0920/PyramidNet-PyTorch/blob/master/train.py

import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import resnet as RN
import pyramidnet as PYRM
import utils
import numpy as np
import torchvision.utils
from torchvision.utils import save_image
import warnings
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from function import calc_mean_std
from function import adaptive_instance_normalization as adain
from copulalib.copulalib import Copula

from torch.utils.tensorboard import SummaryWriter
import net
from function import adaptive_instance_normalization, coral
import torch.nn.functional as F
from IPython import embed
# Because part of the training data is truncated image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from datetime import datetime
from scipy.special import comb
warnings.filterwarnings("ignore")
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Cutmix PyTorch CIFAR-10, CIFAR-100 and ImageNet-1k Training')
parser.add_argument('--net_type', default='resnet', type=str,
                    help='networktype: resnet, and pyamidnet')
parser.add_argument('-j', '--workers', default=40, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run') # 250
parser.add_argument('-b', '--batch_size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=1, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--depth', default=18, type=int,
                    help='depth of the network (default: 32)')
parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                    help='to use basicblock for CIFAR datasets (default: bottleneck)')
parser.add_argument('--dataset', dest='dataset', default='imagenet', type=str,
                    help='dataset (options: cifar10, cifar100, and imagenet)')
parser.add_argument('--no-verbose', dest='verbose', action='store_false',
                    help='to print the status at every iteration')
parser.add_argument('--alpha', default=300, type=float,
                    help='number of new channel increases per depth (default: 300)')
parser.add_argument('--expname', default='ResNet18', type=str,
                    help='name of experiment')
parser.add_argument('--beta', default=1.0, type=float,
                    help='hyperparameter beta')
parser.add_argument('--cutmix_prob', default=1.0, type=float,
                    help='cutmix probability')
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
#parser.add_argument('--decoder', type=str, default='models/decoder_iter_110300.pth.tar')
parser.add_argument('--decoder', type=str, default='models/decoder_iter_151200.pth.tar')
parser.add_argument('--param',default=1.0, type=float, help='temp param')
parser.add_argument('--param2',default=1.0, type=float, help='temp param2')
parser.add_argument('--param3',default=1.0, type=float, help='temp param3')
parser.add_argument('--param4int',default=1, type=int, help='temp param4')
parser.add_argument('--param5',default=0.2, type=float, help='temp param5')
parser.add_argument('--r',default=1.0, type=float, help='augmentation prob')
parser.add_argument('--term',default=100, type=int, help='lr jump term') # 75
#####################################
# Check here before running the code!
# Enter the appropriate method and pretrained model
#####################################
parser.add_argument('--method', type=str, default='a')
parser.add_argument('--pretrained', type=str, default='models/pretrained_100Class.pth.tar')
parser.set_defaults(bottleneck=True)
parser.set_defaults(verbose=True)

writer = SummaryWriter('/home_goya/minui.hong/runs/'+parser.parse_args().expname)
best_err1 = 100
best_err5 = 100


def per_calc_style_loss(input, target):
    assert (input.size() == target.size())
    input_mean, input_std = calc_mean_std(input)
    target_mean, target_std = calc_mean_std(target)
    t1 = F.mse_loss(input_mean, target_mean, reduction='none')
    t2 = F.mse_loss(input_std, target_std, reduction='none')
    return torch.mean(t1, dim=[1,2,3])+torch.mean(t2,dim=[1,2,3])
def linear_combination(x, y, epsilon):
    #epsilon = epsilon.cuda(x.device)
    return epsilon*x + (1-epsilon)*y
def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction=='mean' else loss.sum() if reduction=='sum' else loss
class LSCrossEntropy(nn.Module):
    def __init__(self, epsilon:float=0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
    def forward(self, preds, target, epsilon):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target,  reduction=self.reduction)
        return linear_combination(loss/n, nll, epsilon)


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean=mean
        self.std=std
    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor
unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
def main():

    global args, best_err1, best_err5
    args = parser.parse_args()
    print("args")
    global decoder, vgg, pretrained
    decoder = net.decoder
    vgg = net.vgg
    decoder.eval()
    vgg.eval()
    decoder.load_state_dict(torch.load(args.decoder))
    vgg.load_state_dict(torch.load(args.vgg))
    vgg = nn.Sequential(*list(vgg.children())[:31])
    vgg.cuda()
    decoder.cuda()

    global network_E, network_D, group_info, group_list, numberofclass
    # netowrk code added
    network_E = net.Net_E(vgg)
    network_D = net.Net_D(vgg, decoder)
    network_E.eval()
    network_D.eval()
    network_E = torch.nn.DataParallel(network_E).cuda()
    network_D = torch.nn.DataParallel(network_D).cuda()

    global msenone
    msenone = nn.MSELoss(reduction='none')
    #Copula
    #global frank
    #x = np.random.normal(size=250)
    #y = args.param*x + np.random.normal(size=250)
    #frank = Copula(x,y,family='frank')
    #global group, group_info, group_list, numberofclass

    if args.dataset.startswith('cifar'):
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        if args.dataset == 'cifar100':
            train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR100('../data', train=True, download=True, transform=transform_train),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
            val_loader = torch.utils.data.DataLoader(
                datasets.CIFAR100('../data', train=False, transform=transform_test),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
            numberofclass = 100
        elif args.dataset == 'cifar10':
            train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10('../data', train=True, download=True, transform=transform_train),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
            val_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10('../data', train=False, transform=transform_test),
                batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
            numberofclass = 10
        else:
            raise Exception('unknown dataset: {}'.format(args.dataset))

    elif args.dataset == 'imagenet':
        ##############################################
        # Check here before running the code!
        # You need to check appropriate dataset
        ##############################################
        #traindir = os.path.join('/home_goya/jinwoo.choi/ImageNet/train_100Class_400Each_V2/')
        #traindir = os.path.join('/home_goya/jinwoo.choi/ImageNet/train_20Class_400Each_V2/')
        traindir = os.path.join('/home_goya/jinwoo.choi/ImageNet/train_66Class_300Each_M/')
        #traindir = os.path.join('/home_goya/jinwoo.choi/ImageNet/train/')
        #traindir = os.path.join('/home_goya/jinwoo.choi/ImageNet/train_250Class_400Each_V2/')
        #valdir = os.path.join('/home_goya/jinwoo.choi/ImageNet/val_100Class_50Each_V2/')
        #valdir = os.path.join('/home_goya/jinwoo.choi/ImageNet/val_250Class_50Each_V2/')
        #valdir = os.path.join('/home_goya/jinwoo.choi/ImageNet/val_20Class_50Each_V2/')
        valdir = os.path.join('/home_goya/jinwoo.choi/ImageNet/val_66Class_50Each_M/')
        #valdir = os.path.join('/home_goya/jinwoo.choi/ImageNet/val/')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        jittering = utils.ColorJitter(brightness=0.4, contrast=0.4,
                                      saturation=0.4)
        lighting = utils.Lighting(alphastd=0.1,
                                  eigval=[0.2175, 0.0188, 0.0045],
                                  eigvec=[[-0.5675, 0.7192, 0.4009],
                                          [-0.5808, -0.0045, -0.8140],
                                          [-0.5836, -0.6948, 0.4203]])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                jittering,
                lighting,
                normalize,
            ]))

        train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        ############################################
        # Check here before running the code!
        # You need to check numberofclass
        ############################################
        numberofclass = 66#250

    else:
        raise Exception('unknown dataset: {}'.format(args.dataset))


    c2i = train_dataset.class_to_idx
    group_list_raw = np.genfromtxt('group.txt', delimiter='\n', dtype='str')
    group_list = np.array([np.array(l.split(','), dtype=np.int) for l in group_list_raw])
    group_info = torch.zeros((66,))
    for i in range(group_list.shape[0]):
        for j in range(group_list[i].shape[0]):
            group_info[c2i[str(group_list[i][j])]]=  i
    group_info = group_info.type(torch.LongTensor)
    print("c2i: ",c2i)
    for i in range(66):
        print(i, " : ", group_info[i])


    ###########################################
    # Check here before running the code!
    # There should be a model pretrained with the appropriate number of classes.
    # I recommend 'commenting out' the code below,
    # if you are not using function 'crop10' (for memory)
    ###########################################
    pretrained = RN.ResNet(args.dataset, args.depth, numberofclass, args.bottleneck)
    pretrained.load_state_dict(torch.load(args.pretrained)['state_dict'], strict=False)
    pretrained.cuda()
    ###########################################
    print("=> creating model '{}'".format(args.net_type))
    if args.net_type == 'resnet':
        model = RN.ResNet(args.dataset, args.depth, numberofclass, args.bottleneck)  # for ResNet
    elif args.net_type == 'pyramidnet':
        model = PYRM.PyramidNet(args.dataset, args.depth, args.alpha, numberofclass,
                                args.bottleneck)
    else:
        raise Exception('unknown network architecture: {}'.format(args.net_type))

    model = torch.nn.DataParallel(model).cuda()

    print(model)
    print('the number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss(reduction='none').cuda()
    criterion2 = nn.CrossEntropyLoss().cuda()
    #criterion2 = LSCrossEntropy().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=True)

    cudnn.benchmark = True

    for epoch in range(0, args.epochs):

        adjust_learning_rate(optimizer, epoch, args.term)

        # train for one epoch
        train_loss = train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        err1, err5, val_loss = validate(val_loader, model, criterion2, epoch)

        writer.add_scalar('train_loss', train_loss, epoch+1)
        writer.add_scalar('val_loss', val_loss, epoch+1)
        writer.add_scalar('err1', err1, epoch+1)
        writer.add_scalar('err5', err5, epoch+1)
        # remember best prec@1 and save checkpoint
        is_best = err1 <= best_err1
        best_err1 = min(err1, best_err1)
        if is_best:
            best_err5 = err5

        print('Current best accuracy (top-1 and 5 error):', best_err1, best_err5)
        save_checkpoint({
            'epoch': epoch,
            'arch': args.net_type,
            'state_dict': model.state_dict(),
            'best_err1': best_err1,
            'best_err5': best_err5,
            'optimizer': optimizer.state_dict(),
        }, is_best)

    print('Best accuracy (top-1 and 5 error):', best_err1, best_err5)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    tmp = 0
    tmp2 = 0.
    # switch to train mode
    model.train()

    end = time.time()
    current_LR = get_learning_rate(optimizer)[0]
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda()
        target = target.cuda()
        r = np.random.rand(1)
        if r < args.r:
            rand_index = torch.randperm(input.size()[0]).cuda()
            content = input
            style = input[rand_index]
            target_a = target
            target_b = target[rand_index]
            ########################################
            if args.method == 'a' :
                alpha = np.random.beta(1.0, 1.0)
                beta = np.random.beta(1.0, 1.0)
                gamma_ = np.random.beta(1.0, 1.0)
                gamma = gamma_*torch.ones(content.shape[0])
                #alpha_, beta_ = frank.generate_uv(1)
                #alpha = alpha_[0]
                #beta = beta_[0]

                #class-wise stylemix
                gamma = gamma*torch.ones(content.shape[0])
                same_group = (group_info[target_a] == group_info[target_b])


                if args.param4int == 1:
                    # same group style transfer
                    gamma[~same_group] = args.param*1.0 + (1.-args.param)*gamma_
                elif args.param4int == 2:
                    # all style transfer
                    gamma[:] = args.param*1.0  + (1.-args.param)*gamma
                elif args.param4int == 3:
                    # different group style transfer
                    gamma[same_group] = 1.
                else:
                    exit()

                M = torch.zeros(1,1,224,224).float()
                bbx1, bby1, bbx2, bby2 = rand_bbox(M.size(), 1.-alpha)
                with torch.no_grad():
                    content_feat = network_E(content)
                    """
                    ivicinity = per_calc_style_loss(content_feat[rand_index], content_feat)
                    vicinity = smoothstep(vicinity.cpu(), x_min=20, x_max=200,N=9)
                    gamma = vicinity*gamma
                    if args.param4int == 1:
                        # same group style transfer
                        gamma = args.param*1.0 + (1.-args.param)*gamma
                    elif args.param4int == 2:
                        # diff group style transfer
                        gamma = 1.0 - gamma
                    """
                    mixImage = network_D(content, style, content_feat, content_feat[rand_index], alpha, beta, gamma, bbx1, bby1, bbx2, bby2)
                lam = ((bbx2 - bbx1)*(bby2-bby1)/(224.*224.))
                output  = model(mixImage)

                loss_c = criterion(output, target_a) * (lam) + criterion(output, target_b) * (1. - lam)
                loss_s = criterion(output, target_a) * (beta) + criterion(output, target_b) * (1. - beta)
                gamma = gamma.cuda()
                #sratio = gamma*lam + (1.0 - gamma) * beta
                #loss_s = criterion(output, target_a) * (sratio) + criterion(output, target_b) * (1. - sratio)
                ratio = 0.8

                if torch.is_tensor(gamma):
                    #gamma = gamma.cuda()
                    loss = gamma*loss_c + (1.-gamma)*(ratio*loss_c+(1.-ratio)*loss_s)
                    #loss = ratio*loss_c + (1.- ratio)*loss_s
                    loss = loss.mean()

            elif args.method == 'cutmix' :
                lam = np.random.beta(1.0, 1.0)
                bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
                input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
                output = model(input)
                loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
                loss = loss.mean()
            elif args.method == 'plain':
                output = model(input)
                loss = criterion(output, target)
            elif args.method == 'b':
                alpha = np.random.beta(1.0, 1.0)
                beta = np.random.beta(1.0, 1.0)
                gamma_ = np.random.beta(1.0, 1.0)
                gamma = gamma_*torch.ones(content.shape[0])

                #class-wise stylemix
                gamma = gamma*torch.ones(content.shape[0])
                same_group = (group_info[target_a] == group_info[target_b])
                M = torch.zeros(1,1,224,224).float()
                bbx1, bby1, bbx2, bby2 = rand_bbox(M.size(), 1.-alpha)
                with torch.no_grad():
                    #content_feat = network_E(content)
                    content_feat = network_E(content)
                    s4 = content_feat
                    #adain_cs = adain(content_feat, content_feat[rand_index])
                    #adain_sc = adain(content_feat[rand_index], content_feat)
                    #intensity = torch.mean(msenone(adain_cs, content_feat),dim=[1,2,3])
                    #intensity = torch.mean(msenone(content_feat[rand_index], content_feat),dim=[1,2,3])
                    #i1 = torch.mean(msenone(s1[rand_index], s1),dim=[1,2,3])
                    #i2 = torch.mean(msenone(s2[rand_index], s2),dim=[1,2,3])
                    #i3 = torch.mean(msenone(s3[rand_index], s3),dim=[1,2,3])
                    #i4 = torch.mean(msenone(s4[rand_index], s4),dim=[1,2,3])
                    #i1 = per_calc_style_loss(s1[rand_index], s1)
                    #i2 = per_calc_style_loss(s2[rand_index], s2)
                    #i3 = per_calc_style_loss(s3[rand_index], s3)
                    i4 = per_calc_style_loss(s4[rand_index], s4)
                    #i4 = per_calc_style_loss(adain_cs, s4)
                    #intensity = i1 + i2+ i3+ i4

                    i4 = smoothstep(i4.cpu(), x_min=20, x_max=200,N=3)
                    intensity = i4
                    R = intensity[same_group].mean()/(intensity[same_group].mean()+intensity[~same_group].mean())
                    tmp = tmp + R
                    tmp2 += 1.
                    Avg = tmp / tmp2
                    print(R)
                    print(Avg)
                    #print(intensity[same_group].mean())
                    #print(intensity[~same_group].mean())

                    #embed()
                    #exit()
                    mixImage = network_D(content, style, content_feat, content_feat[rand_index], alpha, beta, gamma, bbx1, bby1, bbx2, bby2)
                lam = ((bbx2 - bbx1)*(bby2-bby1)/(224.*224.))
                output  = model(mixImage)

                loss_c = criterion(output, target_a) * (lam) + criterion(output, target_b) * (1. - lam)
                #loss_s = criterion(output, target_a) * (beta) + criterion(output, target_b) * (1. - beta)
                gamma = gamma.cuda()
                sratio = gamma*lam + (1.0 - gamma) * beta
                loss_s = criterion(output, target_a) * (sratio) + criterion(output, target_b) * (1. - sratio)
                ratio = 0.8

                if torch.is_tensor(gamma):
                    #gamma = gamma.cuda()
                    #loss = gamma*loss_c + (1.-gamma)*(ratio*loss_c+(1.-ratio)*loss_s)
                    loss = ratio*loss_c + (1.- ratio)*loss_s
                    loss = loss.mean()
            else :
                None
            """
            canvas = []
            gamma = 0.0
            for i in range(8):
                beta = i/7.0
                for j in range(8):
                    alpha = j/7.0
                    with torch.no_grad():
                        M = torch.zeros(1,1,224,224).float()
                        bbx1, bby1, bbx2, bby2 = rand_bbox(M.size(), 1.-alpha)
                        M[:,:,bbx1:bbx2,bby1:bby2].fill_(1.)
                        content_feat = network_E(content)
                        mixImage = network_D(content, style, content_feat, content_feat[rand_index], alpha, beta, gamma, bbx1, bby1, bbx2, bby2)
                        mixImage = mixImage[0]
                    canvas.append(mixImage.cpu().squeeze(0))
            grid = torch.stack(canvas, dim = 0)
            img_grid = torchvision.utils.make_grid(
            [unorm(tensor) for tensor in grid])
            grid_name = 'grid_'+str(0)+'.png'
            save_image(img_grid, grid_name)
            embed()
            """
        else:
            output = model(input)
            loss = criterion(output, target)
        # measure accuracy and record loss
        err1, err5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.verbose == True:
            print('Epoch: [{0}/{1}][{2}/{3}]\t'
                  'LR: {LR:.6f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-err {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Top 5-err {top5.val:.4f} ({top5.avg:.4f})'.format(
                epoch, args.epochs, i, len(train_loader), LR=current_LR, batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

    print('* Epoch: [{0}/{1}]\t Top 1-err {top1.avg:.3f}  Top 5-err {top5.avg:.3f}\t Train Loss {loss.avg:.3f}'.format(
        epoch, args.epochs, top1=top1, top5=top5, loss=losses))

    return losses.avg


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()

        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        err1, err5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))

        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.verbose == True:
            print('Test (on val set): [{0}/{1}][{2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-err {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Top 5-err {top5.val:.4f} ({top5.avg:.4f})'.format(
                epoch, args.epochs, i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))

    print('* Epoch: [{0}/{1}]\t Top 1-err {top1.avg:.3f}  Top 5-err {top5.avg:.3f}\t Test Loss {loss.avg:.3f}'.format(
        epoch, args.epochs, top1=top1, top5=top5, loss=losses))
    return top1.avg, top5.avg, losses.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    directory = "runs/%s/" % (args.expname)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)

    if is_best:
        shutil.copyfile(filename, 'runs/%s/' % (args.expname) + 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, term):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if args.dataset.startswith('cifar'):
        lr = args.lr * (0.1 ** (epoch // (args.epochs * 0.5))) * (0.1 ** (epoch // (args.epochs * 0.75)))
    elif args.dataset == ('imagenet'):
        #if args.epochs == 200:
        #    lr = args.lr * (0.1 ** (epoch // 75))
        #elif args.epochs == 300:
        #    lr = args.lr * (0.1 ** (epoch // 75))
        #else:
        lr = args.lr * (0.1 ** (epoch // term))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        wrong_k = batch_size - correct_k
        res.append(wrong_k.mul_(100.0 / batch_size))

    return res

def renormalize(img):
    return (img+1)/2

def imshow(img):
    img = img.cpu().detach()
    #img = (img+1)/2
    img = img.squeeze().numpy()
    img = np.transpose(img, (1, 2, 0))
    plt.imshow((img*255).astype(np.uint8))
    plt.show()

def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def smoothstep(x, x_min = 0, x_max=1, N=1):
    x = np.clip((x-x_min) / (x_max - x_min), 0, 1)
    result = 0
    for n in range(0, N+1):
        result += comb(N+n,n) * comb(2*N+1, N-n) * (-x) **n

    result *= x **(N+1)
    return result

"""
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
"""


if __name__ == '__main__':
    main()
