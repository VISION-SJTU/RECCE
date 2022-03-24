from torch.optim import SGD
from torch.optim import Adam
from torch.optim import ASGD
from torch.optim import Adamax
from torch.optim import Adadelta
from torch.optim import Adagrad
from torch.optim import RMSprop

key2opt = {
    'sgd': SGD,
    'adam': Adam,
    'asgd': ASGD,
    'adamax': Adamax,
    'adadelta': Adadelta,
    'adagrad': Adagrad,
    'rmsprop': RMSprop,
}


def get_optimizer(optimizer_name=None):
    if optimizer_name is None:
        print("Using default 'SGD' optimizer")
        return SGD

    else:
        if optimizer_name not in key2opt:
            raise NotImplementedError(f"Optimizer '{optimizer_name}' not implemented")

        print(f"Using optimizer: '{optimizer_name}'")
        return key2opt[optimizer_name]
