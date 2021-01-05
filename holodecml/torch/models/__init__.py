import os
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from holodecml.torch.spectral import SpectralNorm
from holodecml.torch.attention import Self_Attention

from .cnn import *
from .dense import *
from .rnn import *

# some pytorch examples - https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py

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
    elif model_type == "encoder-vae":
        model = LatentEncoder(**config)
    elif model_type == "multi-head-dense":
        model = MultiHeadDenseOutput(**config)
    elif model_type == "gru-decoder":
        model = DecoderRNN(**config)
    else:
        logger.info(
            f"Unsupported model type {model_type}. Choose from vae, att-vae, encoder-vae, multi-head-dense, or gru-decoder. Exiting."
        )
        sys.exit(1)
    
    for key, val in config.items():
        logger.info(f"{key}: {val}")
    
    return model