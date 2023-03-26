from torch import optim
from optimizer.radam import RAdam
import math

from logopr import mylog
from config import cfg


def get_optimizer(args, net):
    '''
    Configuring the optimizer for the network
    '''
    params_group = net.parameters()
    optimizer = None
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(params_group,
                              lr = args.lr,
                              weight_decay = args.weight_decay,
                              momentum = args.momentum,
                              nesterov = False)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(params_group,
                               lr=args.lr,
                               weight_decay = args.weight_decay,
                               amsgrad = args.amsgrad)
    elif args.optimizer == 'radam':
        optimizer = RAdam(params_group,
                          lr=args.lr,
                          weight_decay=args.weight_decay)
    else:
        raise ValueError('Not a valid optimizer')

    info = "optimizer：{}".format(args.optimizer)
    mylog.msg(info)


    def poly_schd(epoch):
        return math.pow(1 - epoch / args.max_epoch, args.poly_exp)

    def poly_fixed(epoch):
        return 1

    def poly2_schd(epoch):
        if epoch < args.poly_step:
            poly_exp = args.poly_exp
        else:
            poly_exp = 2 * args.poly_exp
        return math.pow(1 - epoch / args.max_epoch, poly_exp)

    if args.lr_fixed == True:  # 固定学习率
        scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                                lr_lambda=poly_fixed)
        info = "Learning rate decay strategy：fixed"
    else:
        if args.lr_schedule == 'scl-poly':
            if cfg.REDUCE_BORDER_EPOCH == -1:
                raise ValueError('ERROR Cannot Do Scale Poly')

            rescale_thresh = cfg.REDUCE_BORDER_EPOCH
            scale_value = args.rescale
            lambda1 = lambda epoch: \
                 math.pow(1 - epoch / args.max_epoch,
                          args.poly_exp) if epoch < rescale_thresh else scale_value * math.pow(
                              1 - (epoch - rescale_thresh) / (args.max_epoch - rescale_thresh),
                              args.repoly)
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
        elif args.lr_schedule == 'poly2':
            scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                                    lr_lambda=poly2_schd)
        elif args.lr_schedule == 'poly':
            scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                                    lr_lambda=poly_schd)
        else:
            raise ValueError('unknown lr schedule {}'.format(args.lr_schedule))

        info = "Learning rate decay strategy：{}".format(args.lr_schedule)

    mylog.msg(info)


    return optimizer, scheduler

