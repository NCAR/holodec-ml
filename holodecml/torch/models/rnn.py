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


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, pos):
        with torch.no_grad():
            x[:] = x[:] + self.pe[pos, :].to(x.device)
            return self.dropout(x)

    
class DecoderRNN(nn.Module):
    
    def __init__(self, 
                 hidden_size, 
                 output_size, 
                 n_layers = 1, 
                 dropout = 0.0,
                 bidirectional = False, 
                 weights = False):
        
        super(DecoderRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.weights = weights 
        self.dr = nn.Dropout(dropout)
        
        bid = 2 if bidirectional else 1

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(
            hidden_size, 
            hidden_size, 
            n_layers, 
            dropout=dropout, 
            bidirectional=bidirectional
        )
        self.out = nn.Linear(bid * hidden_size, output_size)
        
        if output_size > 20000:
            logger.info(
                "Using an adaptive softmax activation with cutoffs at 2000, 10000, and 20000"
            )
            self.softmax = torch.nn.AdaptiveLogSoftmaxWithLoss(
                    in_features = bid * hidden_size,
                    n_classes = output_size, 
                    cutoffs = [2000, 10000, 20000]
            )
        else:
            logger.info(
                "Using a standard softmax output activation"
            )
            self.softmax = nn.LogSoftmax(dim=1)
                    
    def build(self):
        self.load_weights()

    def forward(self, input, hidden, seq_lens = None):
        output = self.embed(input)
        output = self.dr(output)
        #output = nn.utils.rnn.pack_padded_sequence(output, seq_lens)
        output, hidden = self.gru(output, hidden)
        #output, output_lengths = nn.utils.rnn.pad_packed_sequence(output)
        if isinstance(self.softmax, torch.nn.AdaptiveLogSoftmaxWithLoss):
            output = self.softmax.log_prob(output[0])
        else:
            output = self.softmax(self.out(output[0]))
        return output, hidden
    
    def embed(self, input):
        #print("A", input)
        output = self.embedding(input)
        #print("B", output.size())
        output = output.view(1, input.shape[0], -1)
        output = F.relu(output)
        return output
    
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
            
# class DecoderRNN(nn.Module):
    
#     def __init__(self, 
#                  hidden_size, 
#                  output_size, 
#                  n_layers = 1, 
#                  dropout = 0.0,
#                  positional_dropout = 0.1,
#                  bidirectional = False, 
#                  max_len = 103, 
#                  weights = False):
        
#         super(DecoderRNN, self).__init__()
        
#         self.hidden_size = hidden_size
#         self.n_layers = n_layers
#         self.dropout = dropout
#         self.positional_dropout = positional_dropout
#         self.bidirectional = bidirectional
#         self.weights = weights 
        
#         bid = 2 if bidirectional else 1

#         self.embedding = nn.Embedding(output_size, hidden_size)
#         self.pe = PositionalEncoding(
#             hidden_size,
#             dropout = positional_dropout,
#             max_len = max_len
#         )
#         self.gru = nn.GRU(
#             hidden_size, 
#             hidden_size, 
#             n_layers, 
#             dropout=dropout, 
#             bidirectional=bidirectional
#         )
#         self.out = nn.Linear(bid * hidden_size, output_size)
#         self.softmax = nn.LogSoftmax(dim=1)
                            
#     def build(self):
#         self.load_weights()

#     def forward(self, input, hidden, seq_lens = None, pos = None):
#         output = self.embed(input, pos)
#         output, hidden = self.gru(output, hidden)
#         output = self.softmax(self.out(output[0]))
#         return output, hidden
    
#     def embed(self, input, pos = None):
#         output = self.embedding(input)
#         if pos is not None:
#             output = self.pe(output, pos)
#         output = output.view(1, input.shape[0], -1)
#         output = F.relu(output)
#         return output
    
#     def load_weights(self):
        
#         logger.info(
#             f"The model contains {count_parameters(self)} trainable parameters"
#         )
        
#         # Load weights if supplied
#         if os.path.isfile(self.weights):
#             logger.info(f"Loading weights from {self.weights}")

#             # Load the pretrained weights
#             model_dict = torch.load(
#                 self.weights,
#                 map_location=lambda storage, loc: storage
#             )
#             self.load_state_dict(model_dict["model_state_dict"])
#             return

#         elif self.weights:
#             logger.warning(
#                 f"The weights file {self.weights} does not exist, and so won't be loaded. Is this what you wanted?"
#             )
                
#         # Initialize the weights of the model layers to be Xavier
#         logger.info(
#             f"Setting tunable parameter weights according to Xavier's uniform initialization"
#         )
                
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)
                
#         return