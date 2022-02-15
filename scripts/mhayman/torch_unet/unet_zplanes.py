"""
Train a UNET to predict the x-y location
of in-focus particles.
"""

data_dir = '/glade/p/cisl/aiml/ai4ess_hackathon/holodec/'


import warnings
warnings.filterwarnings("ignore")

import sys,os

import numpy as np
import matplotlib.pyplot as plt

import glob
import xarray as xr
import datetime

# import yaml
import tqdm
import torch
import pickle
import logging
import random

from typing import List, Dict, Callable, Union, Any, TypeVar, Tuple


import torch
import torch.fft
from torch import nn

import torch.nn.functional as F

# from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

# set this up to point to the libararies directory in holodec-ml
dirP_str = os.path.join(os.environ['HOME'], 
                    'Python', 
                    'holodec-ml',
                    'library')
if dirP_str not in sys.path:
    sys.path.append(dirP_str)

import torch_optics_utils as optics



is_cuda = torch.cuda.is_available()
device = torch.device(torch.cuda.current_device()) if is_cuda else torch.device("cpu")

if is_cuda:
    torch.backends.cudnn.benchmark = True

print(f'Preparing to use device {device}')

dtype = torch.complex64  # fft required data type