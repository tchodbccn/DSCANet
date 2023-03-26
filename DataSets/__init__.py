import importlib
import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader

from logopr import mylog

def setup_loaders(args):
    '''
    Load the corresponding training set, verification set, and test set through the dataset module specified by args.dataset
    '''
    mod = importlib.import_module('DataSets.{}'.format(args.dataset))
    dataset_loader = getattr(mod, 'Loader')

    input_transform = standard_transforms.Compose([standard_transforms.ToTensor()])

    train_set = dataset_loader(mode='Train', img_transform=input_transform)
    train_sampler = None
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=True,
                              drop_last=True, sampler=train_sampler)

    val_set = dataset_loader(mode='Validation', img_transform=input_transform)
    val_sampler = None
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=True,
                              drop_last=True, sampler=val_sampler)

    test_set = dataset_loader(mode='Test', img_transform=input_transform)
    test_sampler = None
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            shuffle=True,
                            drop_last=True, sampler=test_sampler)
    return train_loader, val_loader, test_loader