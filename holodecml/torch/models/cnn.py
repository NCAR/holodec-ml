import os
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models

import numpy as np
from holodecml.torch.models.utils import *
from holodecml.torch.spectral import SpectralNorm
from holodecml.torch.attention import Self_Attention

# some pytorch examples - https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py

logger = logging.getLogger(__name__)


class CNN_VAE(nn.Module):

    def __init__(self,
                 image_channels=1,
                 hidden_dims=[8, 16, 32, 64, 128, 256],
                 z_dim=10,
                 out_image_channels = 1, 
                 weights = False):

        super(CNN_VAE, self).__init__()

        self.image_channels = image_channels
        self.hidden_dims = hidden_dims
        self.z_dim = z_dim
        self.out_image_channels = out_image_channels
        
        self.encoder = None
        self.decoder = None
        self.weights = weights
        
    def build(self):

        self.encoder = nn.Sequential(
            # size = B x 1 x 600 x 400
            nn.Conv2d(self.image_channels,
                      self.hidden_dims[0], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.hidden_dims[0]),
            nn.LeakyReLU(),
            # size = B x hidden_dims[0] x 300 x 200
            nn.Conv2d(self.hidden_dims[0], self.hidden_dims[1],
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.hidden_dims[1]),
            nn.LeakyReLU(),
            # size = B x hidden_dims[1] x 150 x 100
            nn.Conv2d(self.hidden_dims[1], self.hidden_dims[2],
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.hidden_dims[2]),
            nn.LeakyReLU(),
            # size = B x hidden_dims[2] x 75 x 50
            nn.Conv2d(self.hidden_dims[2], self.hidden_dims[3], kernel_size=(
                3, 2), stride=(3, 2), padding=0),
            nn.BatchNorm2d(self.hidden_dims[3]),
            nn.LeakyReLU(),
            # size = B x hidden_dims[3] x 25 x 25
            nn.Conv2d(self.hidden_dims[3], self.hidden_dims[4],
                      kernel_size=5, stride=5, padding=0),
            nn.BatchNorm2d(self.hidden_dims[4]),
            nn.LeakyReLU(),
            # size = B x hidden_dims[4] x 5 x 5
            nn.Conv2d(self.hidden_dims[4], self.hidden_dims[5],
                      kernel_size=5, stride=5, padding=0),
            nn.BatchNorm2d(self.hidden_dims[5]),
            nn.LeakyReLU(),
            # size = B x hidden_dims[5] x 1 x 1
        )

        self.fc1 = nn.Linear(self.hidden_dims[-1], self.z_dim)
        self.fc2 = nn.Linear(self.hidden_dims[-1], self.z_dim)
        
        if self.out_image_channels > 1:
            self.hidden_dims = [
                self.out_image_channels * x for x in self.hidden_dims
            ]
        
        self.fc3 = nn.Linear(self.z_dim, self.hidden_dims[-1])

        # 600/5/5/3/2/2/2
        # 400/5/5/2/2/2/2
        
        self.decoder = nn.Sequential(
            # size = B x hidden_dims[5] x 1 x 1
            nn.ConvTranspose2d(
                self.hidden_dims[-1], self.hidden_dims[4], kernel_size=5, stride=5, padding=0),
            nn.BatchNorm2d(self.hidden_dims[4]),
            nn.LeakyReLU(),
            # size = B x hidden_dims[4] x 5 x 5
            nn.ConvTranspose2d(
                self.hidden_dims[4], self.hidden_dims[3], kernel_size=5, stride=5, padding=0),
            nn.BatchNorm2d(self.hidden_dims[3]),
            nn.LeakyReLU(),
            # size = B x hidden_dims[3] x 25 x 25
            nn.ConvTranspose2d(self.hidden_dims[3], self.hidden_dims[2], kernel_size=(
                3, 2), stride=(3, 2), padding=0),
            nn.BatchNorm2d(self.hidden_dims[2]),
            nn.LeakyReLU(),
            # size = B x hidden_dims[2] x 75 x 50
            nn.ConvTranspose2d(
                self.hidden_dims[2], self.hidden_dims[1], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.hidden_dims[1]),
            nn.LeakyReLU(),
            # size = B x hidden_dims[1] x 150 x 100
            nn.ConvTranspose2d(
                self.hidden_dims[1], self.hidden_dims[0], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.hidden_dims[0]),
            nn.LeakyReLU(),
            # size = B x hidden_dims[0] x 300 x 200
            nn.ConvTranspose2d(
                self.hidden_dims[0], self.out_image_channels, kernel_size=4, stride=2, padding=1),
            # size = B x out_image_channels x 600 x 400
        )

        logger.info("Built a CNN-VAE model")

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(std.device)
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)  # flatten
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = z.view(z.size(0), self.hidden_dims[-1], 1, 1)  # flatten/reshape
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        
        if self.out_image_channels > 1:
            z_real = np.sqrt(0.5) * z[:,0,:,:]
            z_imag = z[:,1,:,:]
            z = torch.square(z_real) + torch.square(z_imag)
            z = torch.unsqueeze(z, 1)
        
        return z, mu, logvar
    
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


class ATTENTION_VAE(nn.Module):

    def __init__(self,
                 image_channels=1,
                 hidden_dims=[8, 16, 32, 64, 128, 256],
                 z_dim=10,
                 out_image_channels = 1, 
                 weights = False):

        super(ATTENTION_VAE, self).__init__()

        self.image_channels = image_channels
        self.hidden_dims = hidden_dims
        self.z_dim = z_dim
        self.out_image_channels = out_image_channels
        
        self.encoder = None
        self.decoder = None
        self.weights = weights
        
    def build(self):

        self.encoder_block1 = self.encoder_block(
            self.image_channels, self.hidden_dims[0], 4, 2, 1)
        self.encoder_atten1 = Self_Attention(self.hidden_dims[0])
        self.encoder_block2 = self.encoder_block(
            self.hidden_dims[0], self.hidden_dims[1], 4, 2, 1)
        self.encoder_atten2 = Self_Attention(self.hidden_dims[1])
        self.encoder_block3 = self.encoder_block(
            self.hidden_dims[1], self.hidden_dims[2], 4, 2, 1)
        self.encoder_atten3 = Self_Attention(self.hidden_dims[2])
        self.encoder_block4 = self.encoder_block(
            self.hidden_dims[2], self.hidden_dims[3], (3, 2), (3, 2), 0)
        self.encoder_atten4 = Self_Attention(self.hidden_dims[3])
        self.encoder_block5 = self.encoder_block(
            self.hidden_dims[3], self.hidden_dims[4], 5, 5, 0)
        self.encoder_atten5 = Self_Attention(self.hidden_dims[4])
        self.encoder_block6 = self.encoder_block(
            self.hidden_dims[4], self.hidden_dims[5], 5, 5, 0)

        self.fc1 = nn.Linear(self.hidden_dims[-1], self.z_dim)
        self.fc2 = nn.Linear(self.hidden_dims[-1], self.z_dim)
        
        # Add extra output channel if we are using Matt's physical constraint
        if self.out_image_channels > 1:
            self.hidden_dims = [
                self.out_image_channels * x for x in self.hidden_dims
            ]
        
        self.fc3 = nn.Linear(self.z_dim, self.hidden_dims[-1])

        self.decoder_block1 = self.decoder_block(
            self.hidden_dims[5], self.hidden_dims[4], 5, 5, 0)
        self.decoder_atten1 = Self_Attention(self.hidden_dims[4])
        self.decoder_block2 = self.decoder_block(
            self.hidden_dims[4], self.hidden_dims[3], 5, 5, 0)
        self.decoder_atten2 = Self_Attention(self.hidden_dims[3])
        self.decoder_block3 = self.decoder_block(
            self.hidden_dims[3], self.hidden_dims[2], (3, 2), (3, 2), 0)
        self.decoder_atten3 = Self_Attention(self.hidden_dims[2])
        self.decoder_block4 = self.decoder_block(
            self.hidden_dims[2], self.hidden_dims[1], 4, 2, 1)
        self.decoder_atten4 = Self_Attention(self.hidden_dims[1])
        self.decoder_block5 = self.decoder_block(
            self.hidden_dims[1], self.hidden_dims[0], 4, 2, 1)
        self.decoder_atten5 = Self_Attention(self.hidden_dims[0])
        self.decoder_block6 = self.decoder_block(
            self.hidden_dims[0], self.out_image_channels, 4, 2, 1)

        logger.info("Loaded a self-attentive encoder-decoder VAE model")
        
        self.load_weights()

    def encoder_block(self, dim1, dim2, kernel_size, stride, padding):
        return nn.Sequential(
            SpectralNorm(
                nn.Conv2d(dim1, dim2, kernel_size=kernel_size,
                          stride=stride, padding=padding)
            ),
            nn.BatchNorm2d(dim2),
            nn.LeakyReLU()
        )

    def decoder_block(self, dim1, dim2, kernel_size, stride, padding):
        return nn.Sequential(
            SpectralNorm(
                nn.ConvTranspose2d(
                    dim1, dim2, kernel_size=kernel_size, stride=stride, padding=padding)
            ),
            nn.BatchNorm2d(dim2),
            nn.LeakyReLU()
        )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(std.device)
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder_block1(x)
        #h, att_map1 = self.encoder_atten1(h)
        h = self.encoder_block2(h)
        #h, att_map2 = self.encoder_atten2(h)
        h = self.encoder_block3(h)
        h, att_map3 = self.encoder_atten3(h)
        h = self.encoder_block4(h)
        h, att_map4 = self.encoder_atten4(h)
        h = self.encoder_block5(h)
        h, att_map5 = self.encoder_atten5(h)
        h = self.encoder_block6(h)
        h = h.view(h.size(0), -1)  # flatten
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar, [att_map3, att_map4, att_map5]

    def decode(self, z):
        z = self.fc3(z)
        z = z.view(z.size(0), self.hidden_dims[-1], 1, 1)  # flatten/reshape
        z = self.decoder_block1(z)
        z, att_map1 = self.decoder_atten1(z)
        z = self.decoder_block2(z)
        z, att_map2 = self.decoder_atten2(z)
        z = self.decoder_block3(z)
        z, att_map3 = self.decoder_atten3(z)
        z = self.decoder_block4(z)
        #z, att_map4 = self.decoder_atten4(z)
        z = self.decoder_block5(z)
        #z, att_map5 = self.decoder_atten5(z)
        z = self.decoder_block6(z)
        return z, [att_map1, att_map2, att_map3]

    def forward(self, x):
        z, mu, logvar, encoder_att = self.encode(x)
        z, decoder_att = self.decode(z)
        
        if self.out_image_channels > 1:
            z_real = np.sqrt(0.5) * z[:,0,:,:]
            z_imag = z[:,1,:,:]
            z = torch.square(z_real) + torch.square(z_imag)
            z = torch.unsqueeze(z, 1)
        
        return z, mu, logvar
    
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