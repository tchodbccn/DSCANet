import os
import datetime
import torch
import numpy as np
from torch import nn

def getCurrentTime():
    '''
    current system time
    :return: "year-month-day-hour-minute-second"
    '''
    current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    return current_time


def makedirs(directoryPath):
    folder = os.path.exists(directoryPath)
    if not folder:
        os.makedirs(directoryPath)

def prep_experiment(args):
    """

    """
    args.ngpu = torch.cuda.device_count()


def saveNpyFile(fileanme, data):
    try:
        np.save(fileanme, data)
    except Exception as e:
        print(e)

def readNpyFile(filename):
    data = []
    try:
        data = np.load(filename, allow_pickle=True)
    except Exception as e:
        print(e)
    return data

def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


def split_batch(X, y, devices):
    assert X.shape[0] == y.shape[0]
    return(nn.parallel.scatter(X, devices),
           nn.parallel.scatter(y, devices))

def allreduce(data):
    for i in range(1, len(data)):
        data[0][:] += data[i].to(data[0].device)
    for i in range(1, len(data)):
        data[i][:] = data[0].to(data[i].device)

def get_params(params, device):
    new_params = [p.to(device) for p in params]
    for p in new_params:
        p.requires_grad_()
    return new_params