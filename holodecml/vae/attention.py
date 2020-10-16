import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
import numpy as np

from .spectral import SpectralNorm
from torch.autograd import Variable


logger = logging.getLogger(__name__)


class Self_Attention(nn.Module):

    """ Self attention Layer
        Based on https://github.com/heykeetae/Self-Attention-GAN/blob/master/sagan_models.py
    """

    def __init__(self, in_dim):

        super(Self_Attention, self).__init__()

        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.query_conv = SpectralNorm(self.query_conv)
        self.key_conv = SpectralNorm(self.key_conv)
        self.value_conv = SpectralNorm(self.value_conv)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps(B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """

        B, C, width, height = x.size()
        proj_query = self.query_conv(x).view(
            B, -1, width*height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(B, -1, width*height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(B, -1, width*height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, width, height)
        out = self.gamma * out + x

        return out, attention
