import warnings
warnings.filterwarnings("ignore")

import os
import sys
import glob
import tqdm
import time
import yaml
import torch
import optuna
import pickle
import joblib
import random
import sklearn
import logging
import argparse
import datetime
import torch.fft
import torchvision
import torchvision.models as models

#torch.multiprocessing.set_start_method('spawn')

import numpy as np
import pandas as pd
import xarray as xr
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from collections import defaultdict
from scipy.signal import convolve2d
from torch.optim.lr_scheduler import *
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import List, Dict, Callable, Union, Any, TypeVar, Tuple

from aimlutils.hyper_opt.utils import trial_suggest_loader
from aimlutils.hyper_opt.base_objective import *
from aimlutils.hyper_opt.utils import KerasPruningCallback


logger = logging.getLogger(__name__)

# Set up the GPU device id, or CPU if no GPU available
is_cuda = torch.cuda.is_available()
device = torch.device(torch.cuda.current_device()) if is_cuda else torch.device("cpu")
if is_cuda:
    torch.backends.cudnn.benchmark = True
    
class HologramLoader(Dataset):
    
    def __init__(self, fn, max_buffer_size = 5000, max_images = 40000, shuffle = True):
        self.fn = fn
        self.buffer = []
        self.max_buffer_size = max_buffer_size
        self.shuffle = shuffle
        self.max_images = max_images
            
        self.fid = open(self.fn, "rb")
        self.loaded = 0 
        self.epoch = 0
        
    def __getitem__(self, idx):    
        
        self.on_epoch_end()
        
        try:
            data = joblib.load(self.fid)
            image, label, mask = data
            image = torch.FloatTensor(image.squeeze(0))
            #label = torch.LongTensor([label])
            mask = torch.FloatTensor(mask.squeeze(0))
            data = (image, mask)
            
            self.loaded += 1

            if not self.shuffle:
                return data
            self.buffer.append(data)
            random.shuffle(self.buffer)

            if len(self.buffer) > self.max_buffer_size:
                self.buffer = self.buffer[:self.max_buffer_size]
                
            if self.epoch > 0:
                return self.buffer.pop()
            
            else: # wait until all data has been seen before sampling from the buffer
                return data
            

        except EOFError:
            self.fid = open(self.fn, "rb")
            self.loaded = 0
            return #raise StopIteration

                    
    def __len__(self):
        return self.max_images
    
    def on_epoch_end(self):
        if self.loaded == self.__len__():
            self.fid = open(self.fn, "rb")
            self.loaded = 0
            self.epoch += 1

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size)
    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))
    
class Encoder(nn.Module):
    def __init__(self, chs=(2,64,128,256,512,1024), kns=2):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1], kns) for i in range(len(chs)-1)])
        self.pool       = nn.MaxPool2d(2)
    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs
    
class Decoder(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64), kns=2):
        super().__init__()
        self.chs         = chs
        self.upconvs    = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i+1], kns) for i in range(len(chs)-1)])
    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x        = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x        = torch.cat([x, enc_ftrs], dim=1)
            x        = self.dec_blocks[i](x)
        return x
    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs
    
class UNet(nn.Module):
    def __init__(self,
                 enc_chs=(2, 64, 128, 256, 512, 1024),
                 dec_chs=(1024, 512, 256, 128, 64),
                 kns=2,
                 num_class=1,
                 retain_dim=False,
                 out_sz=(512,512)):
        super().__init__()
        self.encoder     = Encoder(enc_chs)
        self.decoder     = Decoder(dec_chs)
        self.head        = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.retain_dim  = retain_dim
        self.out_sz = out_sz
    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out      = self.head(out)
        if self.retain_dim:
            out = F.interpolate(out, self.out_sz)
        return out
    
            
def objective(trial, config):

    # Get list of hyperparameters from the config
    hyperparameters = config["optuna"]["parameters"]
    trial_hyps = {}
    for param_name in hyperparameters.keys():
        trial_hyps[param_name] = trial_suggest_loader(trial, hyperparameters[param_name]) 
    
    enc_chs = [trial_hyps["chs_start"]]
    for i in range(trial_hyps["chs_length"]):
        trial_hyps["chs_start"] *= 2
        enc_chs.append(x)
    dec_chs = enc_chs[::-1][:-trial_hyps["dec_stop"]]
    
    start = datetime.datetime.now()
    print(f"Loading training data for trial: {trial.number}")    
    train_dataset = HologramLoader(
        config["fn_train"], 
        max_images = 40000, 
        max_buffer_size = 5000, 
        shuffle = True
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=trial_hyps["batch_size"], 
        num_workers=8, # can increase to number of CPUs you asked for in launch script; usually 8
        pin_memory=True,
        shuffle=False) # let the reader do the shuffling
    
    print("Loading testing data")
    test_dataset = HologramLoader(
        config["fn_valid"], 
        max_images = 10000, 
        shuffle = False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=trial_hyps["batch_size"],
        num_workers=8,
        pin_memory=True,
        shuffle=False)
    print(f"Finished loading data took: {datetime.datetime.now() - start}")

    # build model and other components
    unet = UNet(enc_chs=enc_chs,
                dec_chs=dec_chs,
                kns=trial_hyps["kns"],
                retain_dim = config["retain_dim"],
                out_sz = config["mask_size"]
               ).to(device)
    
    optimizer = torch.optim.Adam(unet.parameters(),
                                 lr=trial_hyps["lr"],
                                 weight_decay=weight_decay
                                )
    
    train_criterion = torch.nn.SmoothL1Loss() # Huber (MSE, but once converges, MAE)
    test_criterion = torch.nn.L1Loss() # MAE
    
    lr_scheduler = ReduceLROnPlateau(optimizer,
                                     patience = 1,
                                     min_lr = 1.0e-10,
                                     verbose = True
                                    )
    
    beginning = datetime.datetime.now()
    print(f"BEGINNING model training: {beginning}")
    
    epoch_test_losses = []
    results_dict = defaultdict(list)
    
    for epoch in range(epochs):

        ### Train the model 
        unet.train()

        batch_loss = []

        # set up a custom tqdm
        batch_group_generator = tqdm.tqdm(
            enumerate(train_loader), 
            total=config["batches_per_epoch"],
            leave=True
        )

        # inputs shape is (batch_size x 2 channels x 512 x 512)
        # y shape is (batch_size x 512 x 512)
        for k, (inputs, y) in batch_group_generator:

            # Move data to the GPU, if not there already
            inputs = inputs.to(device).float()
            y = y.to(device).float()

            # Clear gradient
            optimizer.zero_grad()

            # get output from the model, given the inputs
            pred_mask = unet(inputs)

            # get loss for the predicted output
            loss = train_criterion(pred_mask, y)

            # get gradients w.r.t to parameters
            loss.backward()
            batch_loss.append(loss.item())

            # update parameters
            optimizer.step()

            # update tqdm
            to_print = "Epoch {} train_loss: {:.4f}".format(epoch, np.mean(batch_loss))
            to_print += " lr: {:.12f}".format(optimizer.param_groups[0]['lr'])
            batch_group_generator.set_description(to_print)
            batch_group_generator.update()

            # stop the training epoch when train_batches_per_epoch have been used to update 
            # the weights to the model
            if k >= batches_per_epoch and k > 0:
                break

            #lr_scheduler.step(epoch + k / batches_per_epoch)

        # Compuate final performance metrics before doing validation
        train_loss = np.mean(batch_loss)

        # clear the cached memory from the gpu
        torch.cuda.empty_cache()

        ### Test the model 
        unet.eval()
        with torch.no_grad():

            batch_loss = []

            # set up a custom tqdm
            batch_group_generator = tqdm.tqdm(
                enumerate(train_loader),
                leave=True
            )

            for k, (inputs, y) in batch_group_generator:
                # Move data to the GPU, if not there already
                inputs = inputs.to(device).float()
                y = y.to(device).long()
                # get output from the model, given the inputs
                pred_mask = unet(inputs)
                # get loss for the predicted output
                loss = test_criterion(pred_mask, y)
                batch_loss.append(loss.item())
                # update tqdm
                to_print = "Epoch {} test_loss: {:.4f}".format(epoch, np.mean(batch_loss))
                batch_group_generator.set_description(to_print)
                batch_group_generator.update()

        # Use the accuracy as the performance metric to toggle learning rate and early stopping
        test_loss = np.mean(batch_loss)
        epoch_test_losses.append(test_loss)

        # Lower the learning rate if we are not improving
        lr_scheduler.step(test_loss)

        # Save the model if its the best so far.
        if test_loss == min(epoch_test_losses):
            state_dict = {
                'epoch': epoch,
                'model_state_dict': unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': test_loss
            }
            #TODO: add directory
#             torch.save(state_dict, "best_unet.pt")

        # Get the last learning rate
        learning_rate = optimizer.param_groups[0]['lr']

        # Put things into a results dictionary -> dataframe
        results_dict['epoch'].append(epoch)
        results_dict['train_loss'].append(train_loss)
        results_dict['valid_loss'].append(np.mean(batch_loss))
        results_dict["learning_rate"].append(learning_rate)
        df = pd.DataFrame.from_dict(results_dict).reset_index()

        # Save the dataframe to disk
        #TODO: add directory
#         df.to_csv("training_log_unet.csv", index = False)

        # Stop training if we have not improved after X epochs (stopping patience)
        best_epoch = [i for i,j in enumerate(epoch_test_losses) if j == min(epoch_test_losses)][0]
        offset = epoch - best_epoch
        if offset >= stopping_patience:
            break

    print(f"Running entire model took: {datetime.datetime.now() - beginning}")
    
    return test_loss

class Objective(BaseObjective):

    def __init__(self, study, config, metric = "test_loss", device = "gpu"):

        # Initialize the base class
        BaseObjective.__init__(self, study, config, metric, device)

    def train(self, trial, conf):

        result = objective(trial, conf)

        results_dictionary = {
            "test_loss": result
        }
        return results_dictionary

        