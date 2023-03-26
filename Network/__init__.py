import importlib
from logopr import mylog

def get_model(network, netconfig):
    '''
    Dynamically load a network object class instance based on the network name (string)
    '''
    module = network[:network.rfind('.')]
    model = network[network.rfind('.') + 1 :]
    mod = importlib.import_module(module)
    net_func = getattr(mod, model)
    net = net_func(netconfig)
    return net

def get_net(args):
    net = get_model(network='Network.' + args.arch, netconfig=args.netconfig)
    info = "Load net：{}".format(args.arch)
    mylog.msg(info)


    totalparameters = sum([param.nelement() for param in net.parameters()])
    info = "Number of model parameters: %.2fM" % (totalparameters / 1e6)
    mylog.msg(info)


    if args.device == 'gpu':
        net.cuda()
    return net