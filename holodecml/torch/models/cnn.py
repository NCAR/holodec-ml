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
                 out_image_channels = 1):

        super(CNN_VAE, self).__init__()

        self.image_channels = image_channels
        self.hidden_dims = hidden_dims
        self.z_dim = z_dim
        self.out_image_channels = out_image_channels
        
        self.encoder = None
        self.decoder = None
        
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


class LatentEncoder(ATTENTION_VAE):

    def __init__(self,
                 image_channels=1,
                 out_image_channels = 1,
                 hidden_dims=[8, 16, 32, 64, 128, 256],
                 z_dim=10,
                 dense_hidden_dims=[100, 10],
                 dense_dropouts=[0.0, 0.0],
                 tasks=["x", "y", "z", "d"],
                 num_outputs=100,
                 pretrained_model=None):

        super(LatentEncoder, self).__init__(
            image_channels=image_channels,
            out_image_channels=out_image_channels,
            hidden_dims=hidden_dims,
            z_dim=z_dim)
        
        self.dense_hidden_dims = dense_hidden_dims
        self.dense_dropouts = dense_dropouts
        self.num_outputs = num_outputs
        self.tasks = tasks
        
        # Call inherited base build, then override
        build = super().build() 
        
        if os.path.isfile(pretrained_model):
            # Load params from file
            model_dict = torch.load(
                pretrained_model, map_location=lambda storage, loc: storage)
            self.load_state_dict(model_dict["model_state_dict"])
            # Freeze all parameters from VAE
            for layer in self.modules():
                for param in layer.parameters():
                    param.requires_grad = False

            logging.info(
                f"Loaded VAE weights {pretrained_model} and froze these parameters"
            )
        else:
            logging.info(
                f"Loaded a fresh VAE  with trainable parameters"
            )
            
        # Build the base VAE model from the inherited build method
        self.build()
            
    def build(self):
        # Build the dense network for predicting particle attributes
        self.task_blocks = nn.ModuleDict({
            task: self.build_block(task, self.dense_hidden_dims, self.dense_dropouts)
            for task in self.tasks
        })

    def build_block(self, task, dense_hidden_dims, dense_dropouts):
        blocks = [self.dense_block(
            task, self.z_dim, dense_hidden_dims[0], dense_dropouts[0])]
        N = len(dense_hidden_dims)
        if N > 1:
            for i in range(N-1):
                blocks.append(
                    self.dense_block(
                        task, dense_hidden_dims[i], dense_hidden_dims[i+1], dense_dropouts[i+1])
                )
        blocks.append(self.dense_block(
            task, dense_hidden_dims[-1], self.num_outputs, 0.0, False))
        blocks = [item for sublist in blocks for item in sublist]
        return nn.Sequential(*blocks)

    def dense_block(self, task, input_dim, output_dim, dr=0.0, activation_layer=True):
        block = [nn.Linear(input_dim, output_dim)]
        if dr > 0.0 and activation_layer:
            block.append(nn.Dropout(dr))
        if activation_layer:
            block.append(nn.LeakyReLU())
        elif task == "binary":
            block.append(nn.Sigmoid())
        return block

    def forward(self, x):
        z, mu, logvar, encoder_att = self.encode(x)
        task_dict = {task: self.task_blocks[task](z) for task in self.tasks}
        task_dict["encoder_att"] = encoder_att
        return task_dict
    
    def image_decoder(self, x):
        z, mu, logvar, encoder_att = self.encode(x)
        z, decoder_att = self.decode(z)
        
        if self.out_image_channels > 1:
            z_real = np.sqrt(0.5) * z[:,0,:,:]
            z_imag = z[:,1,:,:]
            z = torch.square(z_real) + torch.square(z_imag)
            z = torch.unsqueeze(z, 1)
        
        return z
    
    
class ResNetVAE(nn.Module):

    def __init__(self,
                 image_channels=1,
                 hidden_dims=[8, 16, 32, 64, 128, 256],
                 padding = [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)],
                 kernel_size = [(8, 11), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2)],
                 stride = [(2, 2), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2)],
                 z_dim=10,
                 out_image_channels = 1, 
                 pretrained = True, 
                 resnet = 18, 
                 weights = None):

        super(ResNetVAE, self).__init__()

        self.image_channels = image_channels
        self.hidden_dims = hidden_dims
        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.z_dim = z_dim
        self.out_image_channels = out_image_channels
        
        self.encoder = None
        self.decoder = None
        self.resnet = None
        
        self.pretrained = pretrained
        self.resnet_model = resnet
        self.weights = weights
        
    def build(self):
        
        if self.resnet_model == 18:
            resnet = models.resnet18(pretrained=self.pretrained)
        elif self.resnet_model == 34:
            resnet = models.resnet34(pretrained=self.pretrained)
        elif self.resnet_model == 50:
            resnet = models.resnet50(pretrained=self.pretrained)
        elif self.resnet_model == 101:
            resnet = models.resnet101(pretrained=self.pretrained)
        elif self.resnet_model == 152:
            resnet = models.resnet152(pretrained=self.pretrained)
        else:
            logger.warning("You must select resnet 18, 34, 50, 101, or 152. Exiting.")
            raise
            
        resnet.conv1 = torch.nn.Conv1d(4, 64, (7, 7), (2, 2), (3, 3), bias=False)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet_output_dim = resnet.fc.in_features
        self.resnet = nn.Sequential(*modules)

        self.fc1 = nn.Linear(self.resnet_output_dim, self.z_dim)
        self.fc2 = nn.Linear(self.resnet_output_dim, self.z_dim)
        
        if self.out_image_channels > 1:
            self.hidden_dims = [
                self.out_image_channels * x for x in self.hidden_dims
            ]
        
        self.fc3 = nn.Linear(self.z_dim, self.hidden_dims[-1])

        # 600/5/5/3/2/2/2
        # 400/5/5/2/2/2/2
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
        
#         self.decoder = nn.Sequential(
#             # size = B x hidden_dims[5] x 1 x 1
#             SpectralNorm(nn.ConvTranspose2d(
#                 self.hidden_dims[-1], self.hidden_dims[4], kernel_size=5, stride=5, padding=0)),
#             nn.BatchNorm2d(self.hidden_dims[4]),
#             nn.LeakyReLU(),
#             # size = B x hidden_dims[4] x 5 x 5
#             SpectralNorm(nn.ConvTranspose2d(
#                 self.hidden_dims[4], self.hidden_dims[3], kernel_size=5, stride=5, padding=0)),
#             nn.BatchNorm2d(self.hidden_dims[3]),
#             nn.LeakyReLU(),
#             # size = B x hidden_dims[3] x 25 x 25
#             SpectralNorm(nn.ConvTranspose2d(self.hidden_dims[3], self.hidden_dims[2], kernel_size=(
#                 3, 2), stride=(3, 2), padding=0)),
#             nn.BatchNorm2d(self.hidden_dims[2]),
#             nn.LeakyReLU(),
#             # size = B x hidden_dims[2] x 75 x 50
#             SpectralNorm(nn.ConvTranspose2d(
#                 self.hidden_dims[2], self.hidden_dims[1], kernel_size=4, stride=2, padding=1)),
#             nn.BatchNorm2d(self.hidden_dims[1]),
#             nn.LeakyReLU(),
#             # size = B x hidden_dims[1] x 150 x 100
#             SpectralNorm(nn.ConvTranspose2d(
#                 self.hidden_dims[1], self.hidden_dims[0], kernel_size=4, stride=2, padding=1)),
#             nn.BatchNorm2d(self.hidden_dims[0]),
#             nn.LeakyReLU(),
#             # size = B x hidden_dims[0] x 300 x 200
#             SpectralNorm(nn.ConvTranspose2d(
#                 self.hidden_dims[0], self.out_image_channels, kernel_size=4, stride=2, padding=1)),
#             # size = B x out_image_channels x 600 x 400
#         )
        
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(
#                 self.hidden_dims[-1], self.hidden_dims[4], kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)),
#             nn.BatchNorm2d(self.hidden_dims[4]),
#             nn.LeakyReLU(),
#             # size = B x hidden_dims[4] x 5 x 5
#             nn.ConvTranspose2d(
#                 self.hidden_dims[4], self.hidden_dims[3], kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)),
#             nn.BatchNorm2d(self.hidden_dims[3]),
#             nn.LeakyReLU(),
#             # size = B x hidden_dims[3] x 25 x 25
#             nn.ConvTranspose2d(
#                 self.hidden_dims[3], self.hidden_dims[2], kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)),
#             nn.BatchNorm2d(self.hidden_dims[2]),
#             nn.LeakyReLU(),
#             # size = B x hidden_dims[2] x 75 x 50
#             nn.ConvTranspose2d(
#                 self.hidden_dims[2], self.hidden_dims[1], kernel_size=(2, 3), stride=(2, 3), padding=(0, 0)),
#             nn.BatchNorm2d(self.hidden_dims[1]),
#             nn.LeakyReLU(),
#             # size = B x hidden_dims[1] x 150 x 100
#             nn.ConvTranspose2d(
#                 self.hidden_dims[1], self.hidden_dims[0], kernel_size=(5, 5), stride=(5, 5), padding=(0, 0)),
#             nn.BatchNorm2d(self.hidden_dims[0]),
#             nn.LeakyReLU(),
#             # size = B x hidden_dims[0] x 300 x 200
#             nn.ConvTranspose2d(
#                 self.hidden_dims[0], self.out_image_channels, kernel_size=(5, 5), stride=(5, 5), padding=(0, 0)),
#             # size = B x out_image_channels x 600 x 400
#         )

        #logger.info("Built a CNN-VAE model")
        self.load_weights()
        
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
        h = self.resnet(x)
        h = h.view(h.size(0), -1)  # flatten
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = z.view(z.size(0), self.hidden_dims[-1], 1, 1)  # flatten/reshape
#         z = self.decoder(z)
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
#         logger.info(
#             f"Setting tunable parameter weights according to Xavier's uniform initialization"
#         )
                
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)
        
        return