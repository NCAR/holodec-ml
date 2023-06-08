import os
import torch
import random
import numpy as np


def seed_everything(some_seed=42):
    random.seed(some_seed)
    os.environ['PYTHONHASHSEED'] = str(some_seed)
    np.random.seed(some_seed)
    torch.manual_seed(some_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(some_seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True