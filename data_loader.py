from importlib import import_module
from torch.utils.data import dataloader
from torch.utils.data import ConcatDataset

class Data:
    def __init__(self, args):
        self.loader_train = None
        for d in args.data_train:
            module_name = d if 