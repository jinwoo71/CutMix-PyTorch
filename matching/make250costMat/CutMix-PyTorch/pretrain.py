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

# Because part of the training data is truncated image
from PIL import ImageFile
torch.manual_seed(10)
ImageFile.LOAD_TRUNCATED_IMAGES = True

warnings.filterwarnings("ignore")
writer = SummaryWriter('runs/naver')
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

res = torch.zeros((3, 250, 250)).cuda()

def main():
    global args, best_err1, best_err5
    args = parser.parse_args()
    print("args")
    global decoder, vgg
    decoder = net.decoder
    vgg = net.vgg
    decoder.eval()
    vgg.eval()
    decoder.load_state_dict(torch.load(args.decoder))
    vgg.load_state_dict(torch.load(args.vgg))
    vgg = nn.Sequential(*list(vgg.children())[:31])
    vgg.cuda()
    decoder.cuda()
    global network
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
            numberofclass = 2
        else:
            raise Exception('unknown dataset: {}'.format(args.dataset))

    elif args.dataset == 'imagenet':
        traindir = os.path.join('/home_goya/jinwoo.choi/ImageNet/train_250Class_400Each_V2')
        valdir = os.path.join('/home_goya/jinwoo.choi/ImageNet/val_250Class_50Each_V2')
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
        numberofclass = 250

    else:
        raise Exception('unknown dataset: {}'.format(args.dataset))

    for epoch in range(0, args.epochs):
        train(train_loader, epoch)

def train(train_loader, epoch):
    start = time.time()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        with torch.no_grad():
            input = input.cuda()
            target = target.cuda()
            rand_index = torch.randperm(input.size()[0]).cuda()
            content = input
            style = input[rand_index]
            target_a = target
            target_b = target[rand_index]

            loss_a_s = network(content, style, 0)
            loss_b_s = network(content, style, 1)

            loss = loss_a_s + loss_b_s
            batch_size = target.shape[0]

            # res = cost matrix
            # res[0, i, j] = total sum of losses in the ith class and jth class
            # res[1, i, j] = count
            # res[2, i, j] = average losses in the ith class and jth class
            res[0, target, target_b] += loss.cuda()
            res[1, target, target_b] += 1
            res[2, target, target_b] = res[0, target, target_b] / res[1, target, target_b]
            # make res symmetric
            res[0, target_b, target] += loss.cuda()
            res[1, target_b, target] += 1
            res[2, target_b, target] = res[0, target_b, target] / res[1, target_b, target]

    if epoch % 1 == 0 :
        torch.save(res, '/home_goya/jinwoo.choi/make250c2/make250c2_'+str(epoch)+'.pt')
        np.savetxt('/home_goya/jinwoo.choi/make250c2/np0_250_'+str(epoch), res.cpu().numpy()[0], fmt='%.18e', delimiter=',')
        np.savetxt('/home_goya/jinwoo.choi/make250c2/np1_250_'+str(epoch), res.cpu().numpy()[1], fmt='%.18e', delimiter=',')
        np.savetxt('/home_goya/jinwoo.choi/make250c2/np2_250_'+str(epoch), res.cpu().numpy()[2], fmt='%.18e', delimiter=',')
    print("Time taken for 1 Epoch : ",time.time()-start)
    return

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

if __name__ == '__main__':
    main()
