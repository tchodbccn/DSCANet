from __future__ import division

import argparse, json
import random

import numpy as np
import torch
import time
import os

from logopr import mylog
from Utils.AverageMeter import AverageMeter
from Utils import prep_experiment, makedirs, getCurrentTime
from config import cfg
import DataSets
from loss import get_loss
import Network
from optimizer import get_optimizer

# To fix random seeds
def seed_torch(rseed=1024):
    random.seed(rseed)
    os.environ['PYTHONHASHSEED'] = str(rseed)
    np.random.seed(rseed)
    torch.manual_seed(rseed)
    torch.cuda.manual_seed(rseed)
    torch.cuda.manual_seed_all(rseed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

#To calculate accuracy
def accuracy(truelabels, predictlables):
    assert len(truelabels) == len(predictlables)
    sameCount = 0
    for i in range(len(truelabels)):
        if (truelabels[i] == predictlables[i]):
            sameCount += 1

    return sameCount / len(truelabels)

# A training process
def train(args, train_loader, net, optim, curr_epoch, criterion=None):
    net.train()
    train_main_loss = AverageMeter()

    start_time = None

    for i, data in enumerate(train_loader):

        start_time = time.time()
        images, labels = data
        labels = labels - 1
        if args.device == 'gpu':
            images = images.cuda()
            labels = labels.cuda()

        main_loss = None
        if args.arch == 'GoogLeNet.GoogLeNet':
            logits, aux_logits2, aux_logits1 = net(images)
            loss0 = criterion(logits, labels)
            loss1 = criterion(aux_logits1, labels)
            loss2 = criterion(aux_logits2, labels)
            main_loss = loss0 + 0.3 * loss1 + 0.3 * loss2
        else:
            output, _ = net(images)
            main_loss = criterion(output, labels)

        main_loss_clone = main_loss.clone().detach_()
        main_loss_clone = main_loss_clone.mean()
        train_main_loss.update(main_loss_clone.item(), args.batch_size)
        optim.zero_grad()
        main_loss.backward()
        optim.step()

        curr_time = time.time()
        batchtime = (curr_time - start_time)
        msg = ('[epoch {}], [iter {} / {}], [train main loss {:0.6f}],'
               ' [lr {:0.6f}] [batchtime {:0.3g}]')
        msg = msg.format(
            curr_epoch, i + 1, len(train_loader), train_main_loss.avg,
            optim.param_groups[-1]['lr'], batchtime)

        mylog.msg(msg)
        metrics = {'loss': train_main_loss.avg,
                   'lr': optim.param_groups[-1]['lr']}
        curr_iter = curr_epoch * len(train_loader) + i
        mylog.metric('train', metrics, curr_iter)
        del data

# A validation process
def validate(args, val_loader, net, criterion_val, curr_epoch):

    net.eval()
    if args.device == 'gpu':
        torch.cuda.empty_cache()

    val_main_loss = AverageMeter()
    start_time = None

    trueLabels = []
    predictLabels = []

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            start_time = time.time()
            images, labels = data
            labels = labels - 1
            if args.device == 'gpu':
                images = images.cuda()
                labels = labels.cuda()
            output, _ = net(images)

            pLabels = output.argmax(axis=1)
            for index in range(len(pLabels)):
                predictLabels.append(pLabels[index])
            for index in range(len(labels)):
                trueLabels.append(labels[index])

            main_loss = criterion_val(output, labels)
            main_loss_clone = main_loss.clone().detach_()
            main_loss_clone = main_loss_clone.mean()
            val_main_loss.update(main_loss_clone.item(), args.batch_size)

            curr_time = time.time()
            batchtime = (curr_time - start_time)

            msg = ('[iter {} / {}], [validation main loss {:0.6f}], [batchtime {:0.3g}]')
            msg = msg.format(i + 1, len(val_loader), val_main_loss.avg, batchtime)
            mylog.msg(msg)

            metrics = {'val_loss': val_main_loss.avg}
            curr_iter = curr_epoch * len(val_loader) + i
            mylog.metric('validation', metrics, curr_iter)
            del data

    accuracy_rate = accuracy(trueLabels, predictLabels)

# to add arguments
def addArgument():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, help="Set a configuration file for recording parameters")
    parser.add_argument('--netconfig', type=str,help="The directory where the network configuration files are saved")

    parser.add_argument('--optimizer', type=str, help='Optimizer')
    parser.add_argument('--lossfun', type=str, help='Loss function')
    parser.add_argument('--device', type=str, help='device')


    parser.add_argument('--lr', type=float, help='Initial learning rate')
    parser.add_argument('--lr_schedule', type=str, help='Learning rate update strategy')
    parser.add_argument('--lr_fixed', type=bool, help='Is the learning rate fixed')
    parser.add_argument('--poly_exp', type=float, help='polynomial LR exponent')
    parser.add_argument('--poly_step', type=int, help='polynomial epoch step')
    parser.add_argument('--rescale', type=float, help='Warm Restarts new lr ratio compared to original lr')
    parser.add_argument('--momentum', type=float, help='Momentum decay rate')
    parser.add_argument('--max_epoch', type=int, help='max epoch')
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--weight_decay', type=float, help='weight decay')


    parser.add_argument('--dataset', type=str, help='dataset')
    parser.add_argument('--set_dataset_root', type=str,
                        help='The directory where the dataset is saved. This parameter allows you to set the directory where the dataset is located on the command line')
    parser.add_argument('--num_workers', type=int, help='Number of CPUs used when reading a dataset')
    parser.add_argument('--classes_num', type=int, help='Number of categories')


    parser.add_argument('--arch', type=str, help='network')


    parser.add_argument('--start_epoch', type=int, help='start_epoch')
    parser.add_argument('--val_freq', type=int,
                        help='Frequency of running the validation process, run validation once when epoch % val_freq = 0')
    parser.add_argument('--stopWhenConvergence', type=bool, help='Whether to allow stop training when convergence occurs')
    parser.add_argument('--lossNoChangesCount', type=int, help='the count of no significant change in losses')
    parser.add_argument('--errRange', type=float, help='The range of changes in the loss of previous and subsequent batches. If it is less than this value for multiple consecutive times, it indicates that it has converged')

    return parser.parse_args()

# to set arguments
def setArgument(args, config):

    if args.optimizer is None:
        args.optimizer = config["components"]["optimizer"]
    if args.lossfun is None:
        args.lossfun = config["components"]["lossfun"]
    if args.device is None:
        args.device = config["components"]["device"]

    if args.lr is None:
        args.lr = config["Hyperparameter"]["lr"]
    if args.lr_schedule is None:
        args.lr_schedule = config["Hyperparameter"]["lr_schedule"]
    if args.lr_fixed is None:
        args.lr_fixed = config["Hyperparameter"]["lr_fixed"]
    if args.poly_exp is None:
        args.poly_exp = config["Hyperparameter"]["poly_exp"]
    if args.poly_step is None:
        args.poly_step = config["Hyperparameter"]["poly_step"]
    if args.rescale is None:
        args.rescale = config["Hyperparameter"]["rescale"]
    if args.momentum is None:
        args.momentum = config["Hyperparameter"]["momentum"]
    if args.max_epoch is None:
        args.max_epoch = config["Hyperparameter"]["max_epoch"]
    if args.batch_size is None:
        args.batch_size = config["Hyperparameter"]["batch_size"]
    if args.weight_decay is None:
        args.weight_decay = config["Hyperparameter"]["weight_decay"]


    if args.dataset is None:
        args.dataset = config["dataset"]["name"]
    if args.set_dataset_root is None:
        args.set_dataset_root = config["dataset"]["root"]
    if args.num_workers is None:
        args.num_workers = config["dataset"]["num_workers"]
    if args.classes_num is None:
        args.classes_num = config["dataset"]["classes_num"]


    if args.arch is None:
        args.arch = config["arch"]


    if args.start_epoch is None:
        args.start_epoch = config["trainprocess"]["start_epoch"]
    if args.val_freq is None:
        args.val_freq = config["trainprocess"]["val_freq"]
    if args.stopWhenConvergence is None:
        args.stopWhenConvergence = config["trainprocess"]["stopWhenConvergence"]
    if args.lossNoChangesCount is None:
        args.lossNoChangesCount = config["trainprocess"]["lossNoChangesCount"]
    if args.errRange is None:
        args.errRange = config["trainprocess"]["errRange"]


def main():
    args = addArgument()

    # To read Configuration File
    with open(args.config) as f:
        config = json.load(f)

    # To fix random seeds
    rseed = config["seed"]
    seed_torch(rseed)
    mylog.msg("随机种子设置为:{}".format(rseed))

    # Set parameters according to the json file
    setArgument(args, config)

    # Reserved experimental environment configuration operation
    prep_experiment(args)

    # Set the path to save the trained network
    netSavePath = os.getcwd() + cfg.NET.SAVESELFPATH + cfg.STARTTIME
    makedirs(netSavePath)

    # Load the training set, verification set, and test set
    if args.set_dataset_root is not None:
        cfg.DATASET.ROOT = args.set_dataset_root
    train_loader, val_loader, test_loader = DataSets.setup_loaders(args)
    info = "Load the training set, verification set, and test set"
    mylog.msg(info)

    # Load loss function
    criterion = get_loss(args)

    # Load net module
    root = os.getcwd()
    args.netconfig = root + '/configs'
    net = Network.get_net(args)

    # Load Optimizer
    optim, scheduler = get_optimizer(args, net)

    if args.device == 'gpu':
        torch.cuda.empty_cache()
    info = "Device：" + args.device
    mylog.msg((info))

    if args.lr_fixed == True:
        info = "lr：{},optim：{}，batchsize：{}， weight decay：{}，momentum：{}，lr strategy：fixed，max epoch：{}".format(args.lr,
                                                                                                               args.optimizer,
                                                                                                               args.batch_size,
                                                                                                               args.weight_decay,
                                                                                                               args.momentum,
                                                                                                               args.max_epoch)
    else:
        info = "lr：{},optim：{}，batchsize：{}， weight decay：{}，momentum：{}，lr strategy：{}，max epoch：{}".format(args.lr,
                                                                                                             args.optimizer,
                                                                                                             args.batch_size,
                                                                                                             args.weight_decay,
                                                                                                             args.momentum,
                                                                                                             args.lr_schedule,
                                                                                                             args.max_epoch)
    mylog.msg(info)


    currentEpoch = 0

    for epoch in range(args.start_epoch, args.max_epoch):
        currentEpoch += 1
        train(args, train_loader, net, optim, epoch, criterion)
        if (epoch + 1) % args.val_freq == 0:
            validate(args, val_loader, net, criterion, epoch)
        scheduler.step()

    endTime = getCurrentTime()
    info = "All done：{}, {} iterations in total".format(endTime, currentEpoch)
    mylog.msg(info)

    netstate = {'model': net.state_dict(), 'optimizer': optim.state_dict()}
    torch.save(netstate, netSavePath + "/netparam.pth")


if __name__ == '__main__':
    main()
