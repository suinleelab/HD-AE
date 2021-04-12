import os
import os
import random

import numpy as np
import torch
from torch.autograd import Variable


def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def gram_matrix(x, sigma=1):
    pairwise_distances = x.unsqueeze(1) - x
    return torch.exp(-pairwise_distances.norm(2, dim=2) / (2 * sigma * sigma))

def one_hot(labels, class_size):
    targets = torch.zeros(labels.size(0), class_size)
    for i, label in enumerate(labels):
        targets[i, label] = 1
    return Variable(targets)
