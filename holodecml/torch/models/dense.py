import os
import math
import torch
import logging
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable

from holodecml.torch.models.utils import *


logger = logging.getLogger(__name__)


class MultiHeadDenseOutput(nn.Module):

    def __init__(self,
                 hidden_dims = [100, 50], 
                 dropouts = [0.2, 0.2],
                 tasks = ["x", "y", "z", "d"],
                 batch_norm = False,
                 verbose = True, 
                 weights = False):
        
        super(MultiHeadDenseOutput, self).__init__()
        
        self.hidden_dims = hidden_dims
        self.dropouts = dropouts
        self.tasks = tasks 
        self.batch_norm = batch_norm
        self.verbose = verbose
        
        self.model = None
        self.weights = weights 
        
    def build(self, 
              input_size: int,  
              noise: float = 0.0):
                
        self.model = {}
        for k in self.tasks:
            self.model_list = []
            self.model_list.append(nn.Linear(input_size, self.hidden_dims[0]))
            if noise > 0.0:
                self.model_list.append(GaussianNoise(noise))
            if self.batch_norm:
                self.model_list.append(nn.BatchNorm1d(num_features=self.hidden_dims[0]))
            self.model_list.append(nn.LeakyReLU())
            if len(self.hidden_dims) > 1:
                if self.dropouts[0] > 0.0:
                    self.model_list.append(nn.Dropout(self.dropouts[0]))
                for i in range(len(self.hidden_dims)-1):
                    self.model_list.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i+1]))
                    if noise > 0.0:
                        self.model_list.append(GaussianNoise(noise))
                    if self.batch_norm:
                        self.model_list.append(nn.BatchNorm1d(num_features=self.hidden_dims[i+1]))
                    self.model_list.append(nn.LeakyReLU())
                    if self.dropouts[i+1] > 0.0:
                        self.model_list.append(nn.Dropout(self.dropouts[i+1]))
            self.model_list.append(nn.Linear(self.hidden_dims[-1], 1))
            if noise > 0.0:
                self.model_list.append(GaussianNoise(noise))
            self.model[k] = nn.Sequential(*self.model_list)
            #self.model[k].apply(weights_init)
        self.model = nn.ModuleDict(self.model)
        
        # Weight initialization here
        self.load_weights()
        
            

    def forward(self, 
                x: torch.FloatTensor):
        
        if self.model is None:
            raise OSError(f"You must call DenseNet.build before using the model. Exiting.")
        
        
        x = {k: self.model[k](x) for k in self.tasks}
        
        """
            Put Gaussian noise layer after all linear layers (before batch_norm).
            Use multiplicative noise before the dense layer (noising the weights in subsequent linear)
            See "gaussian dropout" from keras
        
        """            
        return x
    
    
    def load_weights(self):
        
        logger.info(
            f"The model contains {count_parameters(self)} trainable parameters"
        )
        
        # Load weights if supplied
        if os.path.isfile(self.weights):
            logger.info(f"Loading weights from {self.weights}")

            # Load the pretrained weights
            model_dict = torch.load(
                self.weights,
                map_location=lambda storage, loc: storage
            )
            self.load_state_dict(model_dict["model_state_dict"])
            return

        elif self.weights:
            logger.warning(
                f"The weights file {self.weights} does not exist, and so won't be loaded. Is this what you wanted?"
            )
                
        # Initialize the weights of the model layers to be Xavier
        logger.info(
            f"Setting tunable parameter weights according to Xavier's uniform initialization"
        )
                
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
        return