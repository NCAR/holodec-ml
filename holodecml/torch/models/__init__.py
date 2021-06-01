import os
import sys
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from holodecml.torch.spectral import SpectralNorm
from holodecml.torch.attention import Self_Attention

from .cnn import *


logger = logging.getLogger(__name__)


def LoadModel(config):
    
    if "type" not in config:
        logger.warning("In order to load a model you must supply the type field.")
        raise OSError("Failed to load model. Exiting")
    
    model_type = config.pop("type")
    logger.info(f"Loading model-type {model_type} with settings")
    
    if model_type == "vae":
        model = CNN_VAE(**config)
    
    elif model_type == "att-vae":
        model = ATTENTION_VAE(**config)
    else:
        logger.info(
            f"Unsupported model type {model_type}. Choose from vae, att-vae. Exiting."
        )
        sys.exit(1)
    
    for key, val in config.items():
        logger.info(f"{key}: {val}")
    
    return model