
from loss.CrossEntropy2d import CrossEntropyLoss2d
from logopr import mylog
from config import cfg

def get_loss(args):
    '''
    load loss function
    '''
    criterion = None
    if args.lossfun == 'crossentropy':
        if args.device == 'cpu':
            criterion = CrossEntropyLoss2d(ignore_index=cfg.DATASET.IGNORE_LABEL)
        elif args.device == 'gpu':
            criterion = CrossEntropyLoss2d(ignore_index=cfg.DATASET.IGNORE_LABEL).cuda()
    else:
        raise ValueError('Not a valid loss function')

    info = "Load loss function：" + args.lossfun
    mylog.msg(info)


    return criterion