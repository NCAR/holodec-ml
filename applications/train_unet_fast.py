import warnings
warnings.filterwarnings("ignore")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys 
sys.path.append("/glade/work/schreck/repos/HOLO/clean/holodec-ml")
from holodecml.data import *
from holodecml.losses import *
from holodecml.models import *
from holodecml.metrics import *
from holodecml.transforms import *

import glob
import tqdm
import time
import yaml
import torch
import psutil
import shutil
import pickle
import joblib
import random
import sklearn
import logging
import datetime

import torch.fft
import torchvision
import torchvision.models as models
import segmentation_models_pytorch as smp

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


available_ncpus = len(psutil.Process().cpu_affinity())


# ### Set seeds for reproducibility
def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        

if __name__ == '__main__':
    
    if len(sys.argv) != 2:
        print("Usage: python train_unet_fast.py config.yml")
        sys.exit()
        
    # ### Set up logger to print stuff
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')

    # Stream output to stdout
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    root.addHandler(ch)
        
    ### Set up torch multiprocessing env
    torch.multiprocessing.set_start_method('spawn', force=True)
        
    # ### Load the configuration and get the relevant variables
    config = sys.argv[1]
    with open(config) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)
       
    ### Set seeds for reproducibility
    seed = 1000 if "seed" not in conf else conf["seed"]
    seed_everything(seed)
    
    tile_size = conf["data"]["tile_size"]
    step_size = conf["data"]["step_size"]
    data_path = conf["data"]["output_path"]
    config_ncpus = int(conf["data"]["cores"])
    
    # Set up number of CPU cores available
    if config_ncpus > available_ncpus:
        ncpus = int(2 * available_ncpus)
    else:
        ncpus = int(2 * config_ncpus)
    logging.info(f"Using {ncpus // 2} CPU cores to run {ncpus} data workers")

    fn_train = f"{data_path}/training_{tile_size}_{step_size}.pkl"
    fn_valid = f"{data_path}/validation_{tile_size}_{step_size}.pkl"

    epochs = conf["trainer"]["epochs"]
    start_epoch = 0 if "start_epoch" not in conf["trainer"] else conf["trainer"]["start_epoch"]
    train_batch_size = conf["trainer"]["train_batch_size"]
    valid_batch_size = conf["trainer"]["valid_batch_size"]
    batches_per_epoch = conf["trainer"]["batches_per_epoch"]
    stopping_patience = conf["trainer"]["stopping_patience"]
    
    model_loc = conf["save_loc"]
    model_name = conf["model"]["name"]
    color_dim = conf["model"]["in_channels"]
    
    training_loss = "dice-bce" if "training_loss" not in conf["trainer"] else conf["trainer"]["training_loss"]
    valid_loss = "dice" if "validation_loss" not in conf["trainer"] else conf["trainer"]["validation_loss"]

    learning_rate = conf["optimizer"]["learning_rate"]
    weight_decay = conf["optimizer"]["weight_decay"]
    
    ### Set up CUDA/CPU devices
    is_cuda = torch.cuda.is_available()
    data_device = torch.device("cpu") if "device" not in conf["data"] else conf["data"]["device"]

    if torch.cuda.device_count() >= 2 and "cuda" in data_device:
        data_device = "cuda:0"
        device = "cuda:1"
        device_ids = list(range(1, torch.cuda.device_count()))
    else:
        data_device = torch.device("cpu")
        device = torch.device(torch.cuda.current_device()) if is_cuda else torch.device("cpu")
        device_ids = list(range(torch.cuda.device_count()))

    logging.info(f"There are {torch.cuda.device_count()} GPUs available")
    logging.info(f"Using device {data_device} to perform wave propagation, and {device_ids} for training the model")
    
    ### Create directories if they do not exist and copy yml file
    os.makedirs(model_loc, exist_ok = True)
    if not os.path.exists(os.path.join(model_loc, "model.yml")):
        shutil.copy(config, os.path.join(model_loc, "model.yml"))

    ### Load the preprocessing transforms
    train_transforms = LoadTransformations(conf["transforms"]["training"])
    valid_transforms = LoadTransformations(conf["transforms"]["validation"])
    
    ### Load the data class for reading and preparing the data as needed to train the u-net
    if conf["data"]["total_positive"] == 5 and conf["data"]["total_negative"] == 5:
        logging.info(f"Reading training data from a cached dataset at {fn_train}")
        train_dataset = PickleReader(
            fn_train, 
            transform = train_transforms,
            max_images = int(0.8 * conf["data"]["total_training"]), 
            max_buffer_size = int(0.1 * conf["data"]["total_training"]), 
            color_dim = color_dim,
            shuffle = True
        )
    else:
        logging.info(f"Preprocessing the training data on the fly with an upsampling generator")
        train_dataset = UpsamplingReader(
            conf,
            transform = train_transforms,
            max_size = 100,
            device = data_device
        )

    test_dataset = PickleReader(
        fn_valid,
        transform = valid_transforms,
        max_images = int(0.1 * conf["data"]["total_training"]),
        max_buffer_size = int(0.1 * conf["data"]["total_training"]),
        color_dim = color_dim,
        shuffle = False
    )

    ### Load the iterators for batching the data
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size, 
        num_workers=ncpus,
        pin_memory=True,
        shuffle=True) 

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=valid_batch_size,
        num_workers=0, # Use only one worker since loading data from pickled file
        pin_memory=True,
        shuffle=False)

    ### Load a u-net model (resnet based on https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch)
    unet = load_model(conf["model"]).to(device)
    
#     for modely in ["unet", "unet++", "manet", "linknet", "fpn", "pspnet", "pan", "deeplabv3", "deeplabv3+"]:
#         for namer in ["resnet18", "densenet121", "xception", "efficientnet-b0", "mobilenet_v2", "dpn68", "vgg11"]:
#             conf["model"]["name"] = modely
#             conf["model"]["encoder_name"] = namer
#             try:
#                 unet = load_model(conf["model"]).to(device)
#                 memory = torch.cuda.max_memory_allocated()
#                 print(f"{modely} {namer} {memory}")
#             except:
#                 print(f"{modely} {namer} 0")
#     raise

#     unet = ResNetUNet(
#         n_class = 1, 
#         color_dim = color_dim
#     )
    
    if start_epoch > 0:
        # Load weights
        logging.info(f"Restarting training starting from epoch {start_epoch}")
        logging.info(f"Loading model weights from {model_loc}")
        checkpoint = torch.load(
            os.path.join(model_loc, "best.pt"),
            map_location=lambda storage, loc: storage
        )
        unet.load_state_dict(checkpoint["model_state_dict"])
        learning_rate = checkpoint["optimizer_state_dict"]["param_groups"][0]["lr"]
        
    unet = unet.to(device)
    total_params = sum(p.numel() for p in unet.parameters())
    trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)

    ### Multi-gpu support
    if len(device_ids) > 1:
        unet = torch.nn.DataParallel(unet, device_ids=device_ids)

    ### Load an optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay
    )
    if start_epoch > 0:
        # Load weights
        logging.info(f"Loading optimizer state from {model_loc}")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    ### Specify the training and validation losses
    train_criterion = load_loss(training_loss) #DiceBCELoss()
    test_criterion = load_loss(valid_loss, split = "validation") #DiceLoss()
        
    ### Load a learning rate scheduler
    lr_scheduler = ReduceLROnPlateau(
        optimizer, 
        patience = 1, 
        min_lr = 1.0e-13,
        verbose = True
    )

    ### Reload the results saved in the training csv if continuing to train
    if start_epoch == 0:
        results_dict = defaultdict(list)
        epoch_test_losses = []
    else:
        results_dict = defaultdict(list)
        saved_results = pd.read_csv(f"{model_loc}/training_log.csv")
        epoch_test_losses = list(saved_results["valid_loss"])
        for key in saved_results.columns:
            if key == "index":
                continue
            results_dict[key] = list(saved_results[key])
        # update the learning rate scheduler
        for valid_loss in epoch_test_losses:
            lr_scheduler.step(valid_loss)

    ### Train a U-net model
    for epoch in range(start_epoch, epochs):

        ### Train the model 
        unet.train()

        batch_loss = []

        # set up a custom tqdm
        batch_group_generator = tqdm.tqdm(
            enumerate(train_loader), 
            total=batches_per_epoch,
            leave=True
        )

        t0 = time.time()

        for k, (inputs, y) in batch_group_generator:
            # Move data to the GPU, if not there already
            inputs = inputs.to(device)
            y = y.to(device)

            # Clear gradient
            optimizer.zero_grad()

            # get output from the model, given the inputs
            pred_mask = unet(inputs)

            # get loss for the predicted output
            loss = train_criterion(pred_mask, y.float())

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

        # Shutdown the progbar
        batch_group_generator.close()

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
                enumerate(test_loader),
                leave=True
            )

            for k, (inputs, y) in batch_group_generator:
                # Move data to the GPU, if not there already
                inputs = inputs.to(device)
                y = y.to(device)
                # get output from the model, given the inputs
                pred_mask = unet(inputs)
                # get loss for the predicted output
                loss = test_criterion(pred_mask, y.float())
                batch_loss.append(loss.item())
                # update tqdm
                to_print = "Epoch {} test_loss: {:.4f}".format(epoch, np.mean(batch_loss))
                batch_group_generator.set_description(to_print)
                batch_group_generator.update()

            # Shutdown the progbar
            batch_group_generator.close()

        # Use the accuracy as the performance metric to toggle learning rate and early stopping
        test_loss = np.mean(batch_loss)
        epoch_test_losses.append(test_loss)
        
        # clear the cached memory from the gpu
        torch.cuda.empty_cache()

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
            torch.save(state_dict, f"{model_loc}/best.pt")

        # Get the last learning rate
        learning_rate = optimizer.param_groups[0]['lr']

        # Put things into a results dictionary -> dataframe
        results_dict['epoch'].append(epoch)
        results_dict['train_loss'].append(train_loss)
        results_dict['valid_loss'].append(np.mean(batch_loss))
        results_dict["learning_rate"].append(learning_rate)
        df = pd.DataFrame.from_dict(results_dict).reset_index()

        # Save the dataframe to disk
        df.to_csv(f"{model_loc}/training_log.csv", index = False)

        # Stop training if we have not improved after X epochs (stopping patience)
        best_epoch = [i for i,j in enumerate(epoch_test_losses) if j == min(epoch_test_losses)][0]
        offset = epoch - best_epoch
        if offset >= stopping_patience:
            break
