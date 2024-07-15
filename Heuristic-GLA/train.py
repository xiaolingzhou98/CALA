import argparse
import os
import shutil
import time
import errno
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import transforms
import networks.resnet
import networks.wideresnet
import networks.resnext
import numpy as np
from NLT_CIFAR import *
from torch.nn.functional import normalize



parser = argparse.ArgumentParser(description='Heuristic-GLA')
parser.add_argument('--dataset', default='', type=str,
                    help='dataset (cifar10 [default] or cifar100)')
parser.add_argument('--class_num', default=None, type=int,
                    help='dataset (cifar10 [default] or cifar100)')
parser.add_argument('--model', default='resnet', type=str,
                    help='deep networks to be trained')

parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')

parser.add_argument('--layers', default=32, type=int,
                    help='total number of layers (have to be explicitly given!)')

parser.add_argument('--droprate', default=0., type=float,
                    help='dropout probability (default: 0.0)')

parser.add_argument('--no-augment', default=False,
                    help='whether to use standard augmentation (default: True)')
parser.set_defaults(augment=True)

parser.add_argument('--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--name', default='cifar-ours', type=str,
                    help='name of experiment')
parser.add_argument('--no', default='1', type=str,
                    help='index of the experiment (for recording convenience)')

parser.add_argument('--nk', type=int, default=20)
parser.add_argument('--tau1', type=float, default=1.2)
parser.add_argument('--tau2', type=float, default=1.0)

parser.add_argument('--smoothalpha', type=float, default=0.05)


# Wide-ResNet
parser.add_argument('--widen-factor', default=10, type=int,
                    help='widen factor for wideresnet (default: 10)')

# ResNeXt
parser.add_argument('--cardinality', default=8, type=int,
                    help='cardinality for resnext (default: 8)')

# Autoaugment
parser.add_argument('--autoaugment', dest='autoaugment', action='store_true',
                    help='whether to use autoaugment')
parser.set_defaults(autoaugment=True)

# cutout
parser.add_argument('--cutout', dest='cutout', action='store_true',
                    help='whether to use cutout')
parser.set_defaults(cutout=False)
parser.add_argument('--n_holes', type=int, default=1,
                    help='number of holes to cut out from image')
parser.add_argument('--length', type=int, default=16,
                    help='length of the holes')

# Cosine learning rate
parser.add_argument('--cos_lr', dest='cos_lr', action='store_true',
                    help='whether to use cosine learning rate')
parser.set_defaults(cos_lr=False)


parser.add_argument('--imb_factor', type=float, default=10)
parser.add_argument('--seed', type=int, default=100, metavar='S',
                    help='random seed (default: 42)')


args = parser.parse_args()


training_configurations = {
    'resnet': {
        'epochs': 200,
        'batch_size': 128,
        'initial_learning_rate': 0.1,
        'changing_lr': [120, 160],
        'lr_decay_rate': 0.1,
        'momentum': 0.9,
        'nesterov': True,
        'weight_decay': 5e-4,
    },
    'wideresnet': {
        'epochs': 240,
        'batch_size': 128,
        'initial_learning_rate': 0.1,
        'changing_lr': [60, 120, 160, 200],
        'lr_decay_rate': 0.2,
        'momentum': 0.9,
        'nesterov': True,
        'weight_decay': 5e-4,
    },
    'resnext': {
        'epochs': 350,
        'batch_size': 128,
        'initial_learning_rate': 0.05,
        'changing_lr': [150, 225, 300],
        'lr_decay_rate': 0.1,
        'momentum': 0.9,
        'nesterov': True,
        'weight_decay': 5e-4,
    },
}



record_path = './GLA test/' + str(args.dataset) \
              + '_' + str(args.model) \
              + '-' + str(args.layers) \
              + (('-' + str(args.widen_factor)) if 'wide' in args.model else '') \
              + (('-' + str(args.cardinality)) if 'resnext' in args.model else '') \
              + '_' + str(args.name) \
              + '/' + 'no_' + str(args.no) \
              + ('_standard-Aug_' if args.augment else '') \
              + ('_dropout_' if args.droprate > 0 else '') \
              + ('_autoaugment_' if args.autoaugment else '') \
              + ('_cutout_' if args.cutout else '') \
              + ('_cos-lr_' if args.cos_lr else '')

record_file = record_path + '/training_process.txt'
accuracy_file = record_path + '/accuracy_epoch.txt'
loss_file = record_path + '/loss_epoch.txt'
check_point = os.path.join(record_path, args.checkpoint)

train_loader, val_loader, test_loader, imbalanced_num_list = build_dataloader(
        seed=args.seed,
        dataset=args.dataset,
        imbalanced_factor=args.imb_factor,
        batch_size=training_configurations["resnet"]["batch_size"],
    )


label_freq_array = np.array(imbalanced_num_list) / np.sum(np.array(imbalanced_num_list))
adjustments = np.log(label_freq_array ** 1.0 + 1e-12)
adjustments = torch.FloatTensor(adjustments).cuda()



# cosine similarity
class CosineSimilarity(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,x1,x2):
        x2 = x2.t()
        x = x1.mm(x2)
        x1_ = x1.norm(dim=1).unsqueeze(0).t()
        x2_ = x2.norm(dim=0).unsqueeze(0)
        x_frobenins = x1_.mm(x2_)
        dist = x.mul(1/x_frobenins)
        return dist


def main():

    global best_prec1
    best_prec1 = 0

    global val_acc
    val_acc = []

    global class_num

    class_num = args.dataset == 'cifar10' and 10 or 100


    kwargs = {'num_workers': 1, 'pin_memory': True}
    assert(args.dataset == 'cifar10' or args.dataset == 'cifar100')

    
    # create model
    if args.model == 'resnet':
        model = eval('networks.resnet.resnet' + str(args.layers) + '_cifar')(dropout_rate=args.droprate)
    elif args.model == 'wideresnet':
        model = networks.wideresnet.WideResNet(args.layers, args.dataset == 'cifar10' and 10 or 100,
                            args.widen_factor, dropRate=args.droprate)
    elif args.model == 'resnext':
        if args.cardinality == 8:
            model = networks.resnext.resnext29_8_64(class_num)
        if args.cardinality == 16:
            model = networks.resnext.resnext29_16_64(class_num)

    if not os.path.isdir(check_point):
        mkdir_p(check_point)

    fc = Full_layer(int(model.feature_num), class_num)

    print('Number of final features: {}'.format(
        int(model.feature_num))
    )

    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])
        + sum([p.data.nelement() for p in fc.parameters()])
    ))

    cudnn.benchmark = True

    ce_criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD([{'params': model.parameters()},
                                {'params': fc.parameters()}],
                                lr=training_configurations[args.model]['initial_learning_rate'],
                                momentum=training_configurations[args.model]['momentum'],
                                nesterov=training_configurations[args.model]['nesterov'],
                                weight_decay=training_configurations[args.model]['weight_decay'])

    model = torch.nn.DataParallel(model).cuda()
    fc = nn.DataParallel(fc).cuda()

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        fc.load_state_dict(checkpoint['fc'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        val_acc = checkpoint['val_acc']
        best_prec1 = checkpoint['best_acc']
        np.savetxt(accuracy_file, np.array(val_acc))
    else:
        start_epoch = 0

    for epoch in range(start_epoch, training_configurations[args.model]['epochs']):

        adjust_learning_rate(optimizer, epoch + 1)

        # train for one epoch
        train(train_loader, model, fc, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, fc, ce_criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'fc': fc.state_dict(),
            'best_acc': best_prec1,
            'optimizer': optimizer.state_dict(),
            'val_acc': val_acc,

        }, is_best, checkpoint=check_point)
        print('Best accuracy: ', best_prec1)
        np.savetxt(accuracy_file, np.array(val_acc))

    print('Best accuracy: ', best_prec1)
    print('Average accuracy', sum(val_acc[len(val_acc) - 10:]) / 10)
    np.savetxt(accuracy_file, np.array(val_acc))


# augment data
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


criterion = nn.CrossEntropyLoss()

def train(train_loader, model, fc, optimizer, epoch):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    train_batches_num = len(train_loader)

    model.train()
    fc.train()
    
    count_iter = 0 
    feature_set = [] 
    prop_set = [] 
    label_set = []
    y1_set = []
    y2_set = []

    end = time.time()
    for i, (_, x, target) in enumerate(train_loader):
        target = target.cuda()
        x = x.cuda()
        input_var = torch.autograd.Variable(x)
        target_var = torch.autograd.Variable(target)
        
        m_x, y1, y2, lam = mixup_data(input_var, target_var)
        
        
        features = model(m_x)

        outputs = fc(features)
        labels_one_hot = F.one_hot(target,num_classes=(args.class_num)).float()
        labels_one_hot1 = F.one_hot(y1,num_classes=(args.class_num)).float()
        labels_one_hot2 = F.one_hot(y2,num_classes=(args.class_num)).float()

        probability = F.softmax(outputs)
        
        
        feature_set.append(features.detach())
        prop_set.append(probability.detach())
        label_set.append(target.detach())
        y1_set.append(y1.detach())
        y2_set.append(y2.detach())
        
        if (count_iter < 10 * args.class_num / 128): 
            neighbor_perturb = torch.ones_like(adjustments)
            count_iter = count_iter + 1 
        else: 
            
            feature_concat = torch.cat(feature_set, dim=0)
            prop_concat = torch.cat(prop_set, dim=0)
            label_concat = torch.cat(label_set, dim=0)
            y1_concat = torch.cat(y1_set, dim=0)
            y2_concat = torch.cat(y1_set, dim=0)
            
            # labels_one_hot_concat = F.one_hot(label_concat,num_classes=(args.class_num)).float()
            # labels_one_hot1_concat = F.one_hot(y1_concat,num_classes=(args.class_num)).float()
            # labels_one_hot2_concat = F.one_hot(y2_concat,num_classes=(args.class_num)).float()
            
            sim_model = CosineSimilarity().cuda()
            
            dist = sim_model(feature_concat, feature_concat) 

            value_total, index_total = torch.topk(dist, args.nk, dim=0) 
            
            index = index_total[:,index_total.shape[1]-128:]
            neighbor_label_1 = y1_concat[index]  
            neighbor_label_2 = y2_concat[index]
            neighbor_prob = prop_concat[index] 
                    
            neighbor_label_t1 = neighbor_label_1.t() 
            neighbor_label_t2 = neighbor_label_2.t()
            neighbor_prob_t = torch.transpose(neighbor_prob,0,1) 
            one_hot_1 = F.one_hot(neighbor_label_t1.long(), args.class_num) 
            one_hot_2 = F.one_hot(neighbor_label_t2.long(), args.class_num)

            class_counts_1 =  torch.sum(one_hot_1, dim=1) * lam
            class_counts_2 =  torch.sum(one_hot_2, dim=1)* (1-lam)
            class_counts = class_counts_1 + class_counts_2 
            
            output_tensor = F.normalize(class_counts.float(), p=1, dim=1)
            
            output_tensor1 = F.normalize(class_counts_1.float(), p=1, dim=1)
            output_tensor2 = F.normalize(class_counts_2.float(), p=1, dim=1)
            
            label_index = output_tensor/neighbor_label_t1.shape[1]
            label_index1 = output_tensor1/neighbor_label_t1.shape[1]
            label_index2 = output_tensor2/neighbor_label_t2.shape[1]
            
            counts = torch.bincount(label_concat, minlength=args.class_num)
            counts1 = torch.bincount(y1, minlength=args.class_num)
            counts2 = torch.bincount(y2, minlength=args.class_num)
            
            logit_sums1 = torch.mm(labels_one_hot1.T, label_index1)
            logit_sums2 = torch.mm(labels_one_hot2.T, label_index2)

            mean_logits1 = torch.div(logit_sums1, counts1.view(-1, 1)+1e-12)
            mean_logits2 = torch.div(logit_sums2, counts2.view(-1, 1)+1e-12)

            element_mean_logits = mean_logits1[y1] + mean_logits2[y2]
            
            label_smooth = args.smoothalpha* element_mean_logits + (1-args.smoothalpha)* label_index

            prob_mean  = torch.mean(neighbor_prob_t,1) 
            
            prop_sums = torch.mm(labels_one_hot.T, prob_mean)

            mean_prop = torch.div(prop_sums, counts.view(-1, 1)+1e-12)

            element_mean_prop = mean_prop[target]

            prop_smooth = args.smoothalpha* element_mean_prop + (1-args.smoothalpha) *  prob_mean

            neighbor_perturb1 = torch.div(prop_smooth, (label_smooth+1e-12))

            neighbor_perturb = torch.nn.functional.normalize(neighbor_perturb1, dim=1)

            feature_set.pop(0)
            label_set.pop(0)
            y1_set.pop(0)
            y2_set.pop(0)
            prop_set.pop(0)
          

        loss  = mixup_criterion(criterion, outputs + args.tau1 * adjustments + args.tau2 * neighbor_perturb, y1, y2, lam)
        
        
        losses.update(loss.data.item(), x.size(0))
       
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()


        if (i+1) % args.print_freq == 0:

            fd = open(record_file, 'a+')
            string = ('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
                      'Loss {loss.value:.4f} ({loss.ave:.4f})\t'.format(
                       epoch, i+1, train_batches_num, batch_time=batch_time,
                       loss=losses))

            print(string)

            fd.write(string + '\n')
            fd.close()

def validate(val_loader, model, fc, criterion, epoch):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    train_batches_num = len(val_loader)


    model.eval()
    fc.eval()

    end = time.time()
    for i, (_, input, target) in enumerate(val_loader):
        target = target.cuda()
        input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        with torch.no_grad():
            features = model(input_var)
            output = fc(features)

        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


        if (i+1) % args.print_freq == 0:
            fd = open(record_file, 'a+')
            string = ('Test: [{0}][{1}/{2}]\t'
                      'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
                      'Loss {loss.value:.4f} ({loss.ave:.4f})\t'
                      'Prec@1 {top1.value:.3f} ({top1.ave:.3f})\t'.format(
                       epoch, (i+1), train_batches_num, batch_time=batch_time,
                       loss=losses, top1=top1))
            print(string)
            fd.write(string + '\n')
            fd.close()

    fd = open(record_file, 'a+')
    string = ('Test: [{0}][{1}/{2}]\t'
              'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
              'Loss {loss.value:.4f} ({loss.ave:.4f})\t'
              'Prec@1 {top1.value:.3f} ({top1.ave:.3f})\t'.format(
        epoch, (i + 1), train_batches_num, batch_time=batch_time,
        loss=losses, top1=top1))
    print(string)
    fd.write(string + '\n')
    fd.close()
    val_acc.append(top1.ave)

    return top1.ave

class Full_layer(torch.nn.Module):
    '''explicitly define the full connected layer'''

    def __init__(self, feature_num, class_num):
        super(Full_layer, self).__init__()
        self.class_num = class_num
        self.fc = nn.Linear(feature_num, class_num)

    def forward(self, x):
        x = self.fc(x)
        return x

def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.ave = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.value = val
        self.sum += val * n
        self.count += n
        self.ave = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate"""
    if not args.cos_lr:
        if epoch in training_configurations[args.model]['changing_lr']:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= training_configurations[args.model]['lr_decay_rate']

    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.5 * training_configurations[args.model]['initial_learning_rate']\
                                * (1 + math.cos(math.pi * epoch / training_configurations[args.model]['epochs']))


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()
