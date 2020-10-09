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

from torch.utils.tensorboard import SummaryWriter
import net
from function import adaptive_instance_normalization, coral
import torch.nn.functional as F
from IPython import embed
from scipy.optimize import linear_sum_assignment
# Because part of the training data is truncated image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

warnings.filterwarnings("ignore")
# Check
writer = SummaryWriter('/home_goya/jinwoo.choi/runsGroup/group_strange/')
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Cutmix PyTorch CIFAR-10, CIFAR-100 and ImageNet-1k Training')
parser.add_argument('--net_type', default='pyramidnet', type=str,
                    help='networktype: resnet, and pyamidnet')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=1, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--depth', default=32, type=int,
                    help='depth of the network (default: 32)')
parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                    help='to use basicblock for CIFAR datasets (default: bottleneck)')
parser.add_argument('--dataset', dest='dataset', default='imagenet', type=str,
                    help='dataset (options: cifar10, cifar100, and imagenet)')
parser.add_argument('--no-verbose', dest='verbose', action='store_false',
                    help='to print the status at every iteration')
parser.add_argument('--alpha', default=300, type=float,
                    help='number of new channel increases per depth (default: 300)')
parser.add_argument('--expname', default='TEST', type=str,
                    help='name of experiment')
parser.add_argument('--beta', default=0, type=float,
                    help='hyperparameter beta')
parser.add_argument('--cutmix_prob', default=0, type=float,
                    help='cutmix probability')
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
parser.add_argument('--decoder', type=str, default='models/decoder_iter_151200.pth.tar')
#####################################
# Check here before running the code!
# Enter the appropriate method and pretrained model
#####################################
parser.add_argument('--method', type=str, default='style_mixup')
parser.add_argument('--pretrained', type=str, default='models/pretrained_100Class.pth.tar')
parser.set_defaults(bottleneck=True)
parser.set_defaults(verbose=True)

best_err1 = 100
best_err5 = 100

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
    global network, group_info, group_list, numberofclass
    network = net.Net(vgg, decoder)
    network.eval()
    network = torch.nn.DataParallel(network).cuda()


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
        traindir = os.path.join('/home_goya/jinwoo.choi/ImageNet/train_66Class_300Each_M/')
        valdir = os.path.join('/home_goya/jinwoo.choi/ImageNet/val_66Class_50Each_M/')
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
        numberofclass = 66

    else:
        raise Exception('unknown dataset: {}'.format(args.dataset))

    c2i = train_dataset.class_to_idx
    #group_list = np.genfromtxt('list.txt', delimiter='\n')
    group_list_raw = np.genfromtxt('group_strange.txt', delimiter='\n', dtype='str')
    group_list = np.array([np.array(l.split(','), dtype=np.int) for l in group_list_raw])
    group_info = torch.zeros((66,))
    for i in range(group_list.shape[0]):
        for j in range(group_list[i].shape[0]):
            group_info[c2i[str(group_list[i][j])]] = i
    group_info = group_info.type(torch.LongTensor)
    print("c2i : ",c2i)
    for i in range(66):
        print(i, " : ", group_info[i])


    ###########################################
    # Check here before running the code!
    # There should be a model pretrained with the appropriate number of classes.
    # I recommend 'commenting out' the code below,
    # if you are not using function 'crop10' (for memory)
    ###########################################
    #pretrained = RN.ResNet(args.dataset, args.depth, numberofclass, args.bottleneck)
    #pretrained.load_state_dict(torch.load(args.pretrained)['state_dict'], strict=False)
    #pretrained.cuda()
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
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=True)

    cudnn.benchmark = True

    for epoch in range(0, args.epochs):

        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train_loss = train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        err1, err5, val_loss = validate(val_loader, model, criterion, epoch)

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

    # switch to train mode
    model.train()
    start = time.time()
    end = time.time()
    current_LR = get_learning_rate(optimizer)[0]
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda()
        target = target.cuda()
        r = np.random.rand(1)
        # Run style_mixup with a probability of 0.5
        if r < 0.5:
            mixImage = None
            squeeze_image = None
            #rand_index = torch.randperm(input.size()[0]).cuda()
            ########################################
            # Check here before running the code!
            # You need to enter the appropriate method in train.sh.
            ########################################
            if args.method == 'content_style_mixup_loss_labeling' :
                with torch.no_grad():
                    rand_index = makeOrder(target, numberofclass, group_list.shape[0])
                content = input
                style = input[rand_index]
                target_a = target
                target_b = target[rand_index]
                x = np.random.uniform()
                y = np.random.uniform()
                with torch.no_grad():
                    loss_a_s, loss_b_s, mixImage = network(content, style, x, y)
                output = model(mixImage)
                sr = loss_a_s / (loss_a_s + loss_b_s)
                log_preds = F.log_softmax(output, dim=-1) # dimension [batch_size, numberofclass]
                a_loss = -log_preds[torch.arange(output.shape[0]),target_a] # cross-entropy for A
                b_loss = -log_preds[torch.arange(output.shape[0]),target_b] # cross-entropy for B
                cr_loss = a_loss.mean() * (x) + b_loss.mean() * (1.0-x) # scalar
                sr_loss = a_loss * (1-sr) + b_loss * sr # dimension [batch_size]
                ratio = 0.7
                loss = ratio * cr_loss + (1.0-ratio) * sr_loss.mean()
            else :
                None
        else:
            # compute output
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
    print("Time taken for 1 epoch : ",time.time()-start)
    print('* Epoch: [{0}/{1}]\t Top 1-err {top1.avg:.3f}  Top 5-err {top5.avg:.3f}\t Train Loss {loss.avg:.3f}'.format(
        epoch, args.epochs, top1=top1, top5=top5, loss=losses))

    return losses.avg

def makeOrder(target, numberofclass, numberofgroup):
    batch_group_info = group_info[target]
    #print("target : ",target)
    #print(batch_group_info)
    order = torch.zeros((target.shape[0],))
    for group in range(numberofgroup):
        group_element = (batch_group_info == group).nonzero()
        group_element = group_element.view(-1)
        new_order = group_element[torch.randperm(group_element.shape[0])]
        for i in range(group_element.shape[0]):
            order[group_element[i]] = new_order[i]
    order = order.type(torch.LongTensor)
    return order

def mask_transport(cost, eps=0.01):
    '''optimal transport plan'''
    n_iter = 100
    # row and col
    for _ in range(n_iter):
        row_best = cost.min(-1)[1]
        plan = torch.zeros_like(cost).scatter_(-1, row_best.unsqueeze(-1), 1)

        # column resolve
        cost_fight = plan * cost
        col_best = cost_fight.min(-2)[1]
        plan_win = torch.zeros_like(cost).scatter_(-2, col_best.unsqueeze(-2), 1) * plan
        plan_lose = (1-plan_win) * plan

    cost += plan_lose

    return plan_win

def checkRatio(x, y):
    return ((x < 2*y) and (y < 2*x))

def squeezeCoordinate(len):
    while True:
        x = torch.randint(len // 3, (2*len) // 3 , (1,))
        y = torch.randint(1, len-1, (1,))
        c1 = checkRatio(x, y)
        c2 = checkRatio(x, len-y)
        c3 = checkRatio(len-x, y)
        c4 = checkRatio(len-x, len-y)
        if c1 and c2 and c3 and c4 :
            break
    return x, y

def squeezeLam(x, y, s1, s2, s3, s4, len):
    a1 = x*y
    a2 = (len-x)*y
    a3 = x*(len-y)
    a4 = (len-x)*(len-y)
    total = len*len
    a = (a1*s1+a2*s2+a3*s3+a4*s4).float()
    return (a / total)

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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if args.dataset.startswith('cifar'):
        lr = args.lr * (0.1 ** (epoch // (args.epochs * 0.5))) * (0.1 ** (epoch // (args.epochs * 0.75)))
    elif args.dataset == ('imagenet'):
        if args.epochs == 300:
            lr = args.lr * (0.1 ** (epoch // 75))
        else:
            lr = args.lr * (0.1 ** (epoch // 30))

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

def style_transfer(vgg, decoder, content, style, alpha):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    adain_content_style = adaptive_instance_normalization(content_f, style_f)
    feat = adain_content_style * alpha + content_f * (1 - alpha)
    return decoder(feat)

def symmetric_style_transfer(vgg, decoder, content, style, alpha):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    adain_content_style = adaptive_instance_normalization(content_f, style_f)
    adain_style_content = adaptive_instance_normalization(style_f, content_f)
    feat = (1-alpha)*(1-alpha)*content_f + alpha*alpha*style_f + alpha*(1-alpha)*(adain_content_style + adain_style_content)
    return decoder(feat)

def symmetric_style_transfer_v2(vgg, decoder, content, style, alpha, beta):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    adain_content_style = adaptive_instance_normalization(content_f, style_f)
    adain_style_content = adaptive_instance_normalization(style_f, content_f)
    feat = beta * (1-alpha) * content_f + beta * alpha * style_f + (1-beta) * (1-alpha) * adain_content_style + (1-beta) * alpha * adain_style_content
    return decoder(feat)

def style_transfer_no_style(vgg, decoder, content, style, lam):
    #assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    bbx1, bby1, bbx2, bby2 = rand_bbox(content_f.size(), lam)
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (content_f.size()[-1] * content_f.size()[-2]))
    content_f[:,:,bbx1:bbx2,bby1:bby2] = style_f[:,:,bbx1:bbx2,bby1:bby2]
    feat = content_f
    return lam, decoder(feat)

def sty(vgg, decoder, content, style, alpha, gamma):
    assert(0.0 <= alpha <= 1.0)
    assert(0.0 <= gamma <= 1.0)
    x = np.random.beta(alpha, alpha)
    if x >= 1.0 - gamma:
        return content, 1.0, 1.0
    elif x <= gamma:
        return style, 0.0, 0.0
    else:
        y = np.random.beta(1.0, 1.0)
        t = np.random.uniform(max(0, x+y-1), min(x, y), 1)[0]
        content_f = vgg(content)
        style_f = vgg(style)
        adain_content_style = adaptive_instance_normalization(content_f, style_f)
        adain_style_content = adaptive_instance_normalization(style_f, content_f)
        feat = t * content_f + (1.0-x-y+t) * style_f + (x-t) * adain_content_style + (y-t) * adain_style_content
        return decoder(feat), x, y

def crop10(image, model, target, epoch):
    size = image.shape[2]
    h = size // 2
    q = size // 4
    s = size
    upsample = nn.Upsample(size=size, mode='bilinear')
    image_crop = torch.zeros(image.shape[0], 10, image.shape[1], image.shape[2], image.shape[3]).cuda()
    image_crop[:,0,:,:,:] = upsample(image[:,:,0:h,0:h])
    image_crop[:,1,:,:,:] = upsample(image[:,:,0:h,h:s])
    image_crop[:,2,:,:,:] = upsample(image[:,:,h:s,0:h])
    image_crop[:,3,:,:,:] = upsample(image[:,:,h:s,h:s])
    image_crop[:,4,:,:,:] = upsample(image[:,:,q:s-q,q:s-q])
    image_crop[:,5,:,:,:] = upsample(image[:,:,0:s-q,0:s-q])
    image_crop[:,6,:,:,:] = upsample(image[:,:,0:s-q,q:s])
    image_crop[:,7,:,:,:] = upsample(image[:,:,q:s,0:s-q])
    image_crop[:,8,:,:,:] = upsample(image[:,:,q:s,q:s])
    image_crop[:,9,:,:,:] = image
    output = model(image_crop.view(-1, image.shape[1], image.shape[2], image.shape[3]))
    output = nn.Softmax(dim=1)(output)
    output = output.view(image.shape[0], 10, -1)
    output = output[torch.arange(image.shape[0]),:,target]
    maxVal, maxIndex = torch.max(output, 1)
    #cropImage = image_crop[torch.arange(image.shape[0]), maxIndex, :, :, :]
    #for i in range(20):
    #    writer.add_image('image'+str(epoch)+str(i), unorm(image[i]))
    #    writer.add_image('image'+str(epoch)+str(i)+'selected', unorm(cropImage[i]))
    #    writer.add_scalar('image'+str(epoch)+str(i)+'softmax',maxVal[i])
    return image_crop[torch.arange(image.shape[0]), maxIndex, :, :, :]


if __name__ == '__main__':
    main()
