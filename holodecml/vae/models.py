import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .spectral import SpectralNorm
from .attention import Self_Attention

# some pytorch examples - https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py


logger = logging.getLogger(__name__)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class CNN_VAE(nn.Module):
    
    def __init__(self,
                 image_channels=1,
                 hidden_dims=[8, 16, 32, 64, 128, 256],
                 z_dim=10):
        
        super(CNN_VAE, self).__init__()
        
        self.image_channels = image_channels
        self.hidden_dims = hidden_dims
        self.z_dim = z_dim
        
        self.encoder = nn.Sequential(
            # size = B x 1 x 600 x 400
            nn.Conv2d(self.image_channels, self.hidden_dims[0], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.hidden_dims[0]),
            nn.LeakyReLU(),
            # size = B x hidden_dims[0] x 300 x 200 
            nn.Conv2d(self.hidden_dims[0], self.hidden_dims[1], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.hidden_dims[1]),
            nn.LeakyReLU(),
            # size = B x hidden_dims[1] x 150 x 100 
            nn.Conv2d(self.hidden_dims[1], self.hidden_dims[2], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.hidden_dims[2]),
            nn.LeakyReLU(),
            # size = B x hidden_dims[2] x 75 x 50 
            nn.Conv2d(self.hidden_dims[2], self.hidden_dims[3], kernel_size=(3,2), stride=(3,2), padding=0),
            nn.BatchNorm2d(self.hidden_dims[3]),
            nn.LeakyReLU(),
            # size = B x hiddem_dims[3] x 25 x 25
            nn.Conv2d(self.hidden_dims[3], self.hidden_dims[4], kernel_size=5, stride=5, padding=0),
            nn.BatchNorm2d(self.hidden_dims[4]),
            nn.LeakyReLU(),
            # size = B x hidden_dims[4] x 5 x 5
            nn.Conv2d(self.hidden_dims[4], self.hidden_dims[5], kernel_size=5, stride=5, padding=0),
            nn.BatchNorm2d(self.hidden_dims[5]),
            nn.LeakyReLU(),
            # size = B x hidden_dims[5] x 1 x 1
        )
        
        self.fc1 = nn.Linear(self.hidden_dims[-1], self.z_dim)
        self.fc2 = nn.Linear(self.hidden_dims[-1], self.z_dim)
        self.fc3 = nn.Linear(self.z_dim, self.hidden_dims[-1])
                
        #600/5/5/3/2/2/2
        #400/5/5/2/2/2/2
        
        self.decoder = nn.Sequential(
            # size = B x hidden_dims[5] x 1 x 1
            nn.ConvTranspose2d(self.hidden_dims[-1], self.hidden_dims[4], kernel_size=5, stride=5, padding=0),
            nn.BatchNorm2d(self.hidden_dims[4]),
            nn.LeakyReLU(),
            # size = B x hidden_dims[4] x 5 x 5
            nn.ConvTranspose2d(self.hidden_dims[4], self.hidden_dims[3], kernel_size=5, stride=5, padding=0),
            nn.BatchNorm2d(self.hidden_dims[3]),
            nn.LeakyReLU(),
            # size = B x hidden_dims[3] x 25 x 25
            nn.ConvTranspose2d(self.hidden_dims[3], self.hidden_dims[2], kernel_size=(3,2), stride=(3,2), padding=0),
            nn.BatchNorm2d(self.hidden_dims[2]),
            nn.LeakyReLU(),
            # size = B x hidden_dims[2] x 75 x 50
            nn.ConvTranspose2d(self.hidden_dims[2], self.hidden_dims[1], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.hidden_dims[1]),
            nn.LeakyReLU(),
            # size = B x hidden_dims[1] x 150 x 100        
            nn.ConvTranspose2d(self.hidden_dims[1], self.hidden_dims[0], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.hidden_dims[0]),
            nn.LeakyReLU(),
            # size = B x hidden_dims[0] x 300 x 200            
            nn.ConvTranspose2d(self.hidden_dims[0], self.image_channels, kernel_size=4, stride=2, padding=1),
            # size = B x 1 x 600 x 400
        )
        
        logger.info("Loaded a CNN-VAE model")
        
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
        h = h.view(h.size(0), -1) # flatten
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = z.view(z.size(0), self.hidden_dims[-1], 1, 1) #flatten/reshape
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar


class ATTENTION_VAE(nn.Module):
    
    def __init__(self,
                 image_channels=1,
                 hidden_dims=[8, 16, 32, 64, 128, 256],
                 z_dim=10):
        
        super(ATTENTION_VAE, self).__init__()
        
        self.image_channels = image_channels
        self.hidden_dims = hidden_dims
        self.z_dim = z_dim
        
        self.encoder_block1 = self.encoder_block(self.image_channels, self.hidden_dims[0], 4, 2, 1)
        self.encoder_atten1 = Self_Attention(self.hidden_dims[0])
        self.encoder_block2 = self.encoder_block(self.hidden_dims[0], self.hidden_dims[1], 4, 2, 1)
        self.encoder_atten2 = Self_Attention(self.hidden_dims[1])
        self.encoder_block3 = self.encoder_block(self.hidden_dims[1], self.hidden_dims[2], 4, 2, 1)
        self.encoder_atten3 = Self_Attention(self.hidden_dims[2])
        self.encoder_block4 = self.encoder_block(self.hidden_dims[2], self.hidden_dims[3], (3,2), (3,2), 0)
        self.encoder_atten4 = Self_Attention(self.hidden_dims[3])
        self.encoder_block5 = self.encoder_block(self.hidden_dims[3], self.hidden_dims[4], 5, 5, 0)
        self.encoder_atten5 = Self_Attention(self.hidden_dims[4])
        self.encoder_block6 = self.encoder_block(self.hidden_dims[4], self.hidden_dims[5], 5, 5, 0)
                
        self.fc1 = nn.Linear(self.hidden_dims[-1], self.z_dim)
        self.fc2 = nn.Linear(self.hidden_dims[-1], self.z_dim)
        self.fc3 = nn.Linear(self.z_dim, self.hidden_dims[-1])
        
        self.decoder_block1 = self.decoder_block(self.hidden_dims[5], self.hidden_dims[4], 5, 5, 0)
        self.decoder_atten1 = Self_Attention(self.hidden_dims[4])
        self.decoder_block2 = self.decoder_block(self.hidden_dims[4], self.hidden_dims[3], 5, 5, 0)
        self.decoder_atten2 = Self_Attention(self.hidden_dims[3])
        self.decoder_block3 = self.decoder_block(self.hidden_dims[3], self.hidden_dims[2], (3,2), (3,2), 0)
        self.decoder_atten3 = Self_Attention(self.hidden_dims[2])
        self.decoder_block4 = self.decoder_block(self.hidden_dims[2], self.hidden_dims[1], 4, 2, 1)
        self.decoder_atten4 = Self_Attention(self.hidden_dims[1])
        self.decoder_block5 = self.decoder_block(self.hidden_dims[1], self.hidden_dims[0], 4, 2, 1)
        self.decoder_atten5 = Self_Attention(self.hidden_dims[0])
        self.decoder_block6 = self.decoder_block(self.hidden_dims[0], self.image_channels, 4, 2, 1)
        
        logger.info("Loaded a self-attentive encoder-decoder VAE model")    
    
    def encoder_block(self, dim1, dim2, kernel_size, stride, padding):
        return nn.Sequential(
            SpectralNorm(
                nn.Conv2d(dim1, dim2, kernel_size=kernel_size, stride=stride, padding=padding)
            ),
            nn.BatchNorm2d(dim2),
            nn.LeakyReLU()
        )
    def decoder_block(self, dim1, dim2, kernel_size, stride, padding):
        return nn.Sequential(
            SpectralNorm(
                nn.ConvTranspose2d(dim1, dim2, kernel_size=kernel_size, stride=stride, padding=padding)
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
        h = h.view(h.size(0), -1) # flatten
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar#, att_map

    def decode(self, z):
        z = self.fc3(z)
        z = z.view(z.size(0), self.hidden_dims[-1], 1, 1) #flatten/reshape
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
        return z#, att_map

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar#, encoder_att_map, decoder_att_map
    

class ConvVAE(nn.Module):
    
    def __init__(self,
                 image_channels = 1,
                 init_kernel = 10,
                 kernel_size = 2,
                 stride = 1,
                 padding = 0):
        
        super(ConvVAE, self).__init__()
 
        # encoder
        self.enc1 = nn.Conv2d(
            in_channels=image_channels, out_channels=init_kernel, kernel_size=kernel_size, 
            stride=stride, padding=padding
        )
        self.ebn1 = nn.BatchNorm2d(init_kernel)
        self.enc2 = nn.Conv2d(
            in_channels=init_kernel, out_channels=init_kernel*2, kernel_size=kernel_size, 
            stride=stride, padding=padding
        )
        self.ebn2 = nn.BatchNorm2d(init_kernel*2)
        self.enc3 = nn.Conv2d(
            in_channels=init_kernel*2, out_channels=init_kernel*4, kernel_size=kernel_size, 
            stride=stride, padding=padding
        )
        self.ebn3 = nn.BatchNorm2d(init_kernel*4)
        self.enc4 = nn.Conv2d(
            in_channels=init_kernel*4, out_channels=init_kernel*8, kernel_size=kernel_size, 
            stride=stride, padding=padding
        )
        self.ebn4 = nn.BatchNorm2d(init_kernel*8)
        self.enc5 = nn.Conv2d(
            in_channels=init_kernel*8, out_channels=init_kernel, kernel_size=kernel_size, 
            stride=stride, padding=padding
        )
        
        # decoder 
        self.dec1 = nn.ConvTranspose2d(
            in_channels=init_kernel, out_channels=init_kernel*8, kernel_size=kernel_size, 
            stride=stride, padding=padding
        )
        self.dbn1 = nn.BatchNorm2d(init_kernel*8)
        self.dec2 = nn.ConvTranspose2d(
            in_channels=init_kernel*8, out_channels=init_kernel*4, kernel_size=kernel_size, 
            stride=stride, padding=padding
        )
        self.dbn2 = nn.BatchNorm2d(init_kernel*4)
        self.dec3 = nn.ConvTranspose2d(
            in_channels=init_kernel*4, out_channels=init_kernel*2, kernel_size=kernel_size, 
            stride=stride, padding=padding
        )
        self.dbn3 = nn.BatchNorm2d(init_kernel*2)
        self.dec4 = nn.ConvTranspose2d(
            in_channels=init_kernel*2, out_channels=init_kernel, kernel_size=kernel_size, 
            stride=stride, padding=padding
        )
        self.dbn4 = nn.BatchNorm2d(init_kernel)
        self.dec5 = nn.ConvTranspose2d(
            in_channels=init_kernel, out_channels=image_channels, kernel_size=kernel_size, 
            stride=stride, padding=padding
        )
        
    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling
        return sample
 
    def forward(self, x):
        # encoding
        
        x = self.enc1(x)
        x = self.ebn1(x)
        x = nn.LeakyReLU()(x)
        x = self.enc2(x)
        x = self.ebn2(x)
        x = nn.LeakyReLU()(x)
        x = self.enc3(x)
        x = self.ebn3(x)
        x = nn.LeakyReLU()(x)
        x = self.enc4(x)
        x = self.ebn4(x)
        x = nn.LeakyReLU()(x)
        x = self.enc5(x)
        
        # get `mu` and `log_var`
        mu = x
        log_var = x
        
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
 
        # decoding
        x = self.dec1(z)
        x = self.dbn1(x)
        x = nn.LeakyReLU()(x)
        x = self.dec2(x)
        x = self.dbn2(x)
        x = nn.LeakyReLU()(x)
        x = self.dec3(x)
        x = self.dbn3(x)
        x = nn.LeakyReLU()(x)
        x = self.dec4(x)
        x = self.dbn4(x)
        x = nn.LeakyReLU()(x)
        x = self.dec5(x)
        
        reconstruction = torch.sigmoid(x)
    
        return reconstruction, mu, log_var



# class Flatten(nn.Module):
#     def forward(self, input):
#         return input.view(input.size(0), -1)
    
# class UnFlatten(nn.Module):
#     def forward(self, input, size=1000):
#         return input.view(input.size(0), size, 1, 1)

    
# class CNN_VAE(nn.Module):
    
#     def __init__(self,
#                  image_channels=1,
#                  hidden_dims=[8, 16, 32, 64, 128],
#                  z_dim=10):
        
#         super(CNN_VAE, self).__init__()
        
#         self.hidden_dims = hidden_dims
#         h_dim = int(216 * self.hidden_dims[-1])
        
#         self.encoder = nn.Sequential(
#             nn.Conv2d(image_channels, self.hidden_dims[0], kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(self.hidden_dims[0]),
#             nn.LeakyReLU(),
#             nn.Conv2d(self.hidden_dims[0], self.hidden_dims[1], kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(self.hidden_dims[1]),
#             nn.LeakyReLU(),
#             nn.Conv2d(self.hidden_dims[1], self.hidden_dims[2], kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(self.hidden_dims[2]),
#             nn.LeakyReLU(),
#             nn.Conv2d(self.hidden_dims[2], self.hidden_dims[3], kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(self.hidden_dims[3]),
#             nn.LeakyReLU(),
#             nn.Conv2d(self.hidden_dims[3], self.hidden_dims[4], kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(self.hidden_dims[4]),
#             nn.LeakyReLU(),
#             Flatten()
#         )
        
#         self.fc1 = nn.Linear(h_dim, z_dim)
#         self.fc2 = nn.Linear(h_dim, z_dim)
#         self.fc3 = nn.Linear(z_dim, 1000)
                
#         #600/5/5/3/2/2/2
#         #400/5/5/2/2/2/2
        
#         self.decoder = nn.Sequential(
#             UnFlatten(),
#             nn.ConvTranspose2d(1000, self.hidden_dims[4], kernel_size=5, stride=5, padding=0),
#             nn.BatchNorm2d(self.hidden_dims[4]),
#             nn.LeakyReLU(),
#             # size = B x 500 x 5 x 5
#             nn.ConvTranspose2d(self.hidden_dims[4], self.hidden_dims[3], kernel_size=5, stride=5, padding=0),
#             nn.BatchNorm2d(self.hidden_dims[3]),
#             nn.LeakyReLU(),
#             # size = B x 250 x 25 x 25
#             nn.ConvTranspose2d(self.hidden_dims[3], self.hidden_dims[2], kernel_size=(3,2), stride=(3,2), padding=0),
#             nn.BatchNorm2d(self.hidden_dims[2]),
#             nn.LeakyReLU(),
#             # size = B x 125 x 75 x 50
#             nn.ConvTranspose2d(self.hidden_dims[2], self.hidden_dims[1], kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(self.hidden_dims[1]),
#             nn.LeakyReLU(),
#             # size = B x 125 x 150 x 100        
#             nn.ConvTranspose2d(self.hidden_dims[1], self.hidden_dims[0], kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(self.hidden_dims[0]),
#             nn.LeakyReLU(),
#             # size = B x 35 x 300 x 200            
#             nn.ConvTranspose2d(self.hidden_dims[0], 1, kernel_size=4, stride=2, padding=1),
#             # size = B x 1 x 600 x 400
#             nn.Sigmoid()
#         )
        
#         logger.info("Loaded a CNN-VAE model")
        
#     def reparameterize(self, mu, logvar):
#         std = logvar.mul(0.5).exp_()
#         # return torch.normal(mu, std)
#         esp = torch.randn(*mu.size()).to(std.device)
#         z = mu + std * esp
#         return z
    
#     def bottleneck(self, h):
#         mu, logvar = self.fc1(h), self.fc2(h)
#         z = self.reparameterize(mu, logvar)
#         return z, mu, logvar

#     def encode(self, x):
#         h = self.encoder(x)
#         z, mu, logvar = self.bottleneck(h)
#         return z, mu, logvar

#     def decode(self, z):
#         z = self.fc3(z)
#         z = self.decoder(z)
#         return z

#     def forward(self, x):
#         z, mu, logvar = self.encode(x)
#         z = self.decode(z)
#         return z, mu, logvar
