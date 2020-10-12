import warnings
warnings.filterwarnings("ignore")

import os
import sys
import yaml
import tqdm
import torch
import pickle
import logging

from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import *
from torch import nn

from typing import List, Dict, Callable, Union, Any, TypeVar, Tuple
from multiprocessing import cpu_count
from shutil import copyfile

# custom
from holodecml.vae.checkpointer import *
from holodecml.vae.data_loader import *
from holodecml.vae.optimizers import *
from holodecml.vae.transforms import *
from holodecml.vae.trainers import *
from holodecml.vae.models import *
from holodecml.vae.visual import *
from holodecml.vae.losses import *


logger = logging.getLogger(__name__)

    
if __name__ == '__main__':
    
    if len(sys.argv) != 2:
        print("Usage: python trainer.py /path/to/config.yml")
        sys.exit()
    
    ############################################################
    
    with open(sys.argv[1]) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    ############################################################
    #
    # Create the save directory if it does not exist
    #
    ############################################################
    
    #try:
    #    os.makedirs(config["log"])
    #except:
    #    pass
    
    #copyfile(sys.argv[1], os.path.join(config["log"], sys.argv[1]))
    
    ############################################################
    #
    # Load a logger
    #
    ############################################################
    
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
    
    # Stream output to stdout
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    root.addHandler(ch)
    
    # Save the log file
    logger_name = os.path.join(config["log"], "log.txt")
    fh = logging.FileHandler(logger_name,
                             mode="w",
                             encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    root.addHandler(fh)


    ############################################################
    #
    # Set the device to a cuda-enabled GPU or the cpu
    #
    ############################################################

    is_cuda = torch.cuda.is_available()
    device = torch.device(torch.cuda.current_device()) if is_cuda else torch.device("cpu")
    
    if is_cuda:
        torch.backends.cudnn.benchmark = True
    
    logging.info(f'Preparing to use device {device}')
    
    ############################################################
    #
    # Load image transformations followed by the data
    #
    ############################################################
    
    # Image transformations
    
    tforms = []
    transform_config = config["transforms"]

    if "RandomVerticalFlip" in transform_config:
        tforms.append(RandVerticalFlip(0.5))
    if "RandomHorizontalFlip" in transform_config:
        tforms.append(RandHorizontalFlip(0.5))
    if "Rescale" in transform_config:
        rescale = transform_config["Rescale"]
        tforms.append(Rescale(rescale))
    if "Normalize" in transform_config:
        mode = transform_config["Normalize"]
        tforms.append(Normalize(mode))
    if "ToTensor" in transform_config:
        tforms.append(ToTensor(device))
    if "RandomCrop" in transform_config:
        tforms.append(RandomCrop())
    if "Standardize" in transform_config:
        tforms.append(Standardize())

    transform = transforms.Compose(tforms)
    
    # Data readers for train/test
    
    train_gen = HologramDataset(
        split = "train", 
        transform = transform,
        **config["data"]
    )

    train_scalers = train_gen.get_transform()

    valid_gen = HologramDataset(
        split = "test",
        transform = transform,
        **config["data"]
    )
    
    # Data iterators using multiprocessing for train/test
    
    logging.info(f"Loading training data iterator using {config['iterator']['num_workers']} workers")
    
    dataloader = DataLoader(
        train_gen,
        **config["iterator"]
    )

    valid_dataloader = DataLoader(
        valid_gen,
        **config["iterator"]
    )
    
    ############################################################
    #
    # Load the model
    #
    ############################################################
    if "load_weights" in config["model"]:
        load_weights = isinstance(config["model"]["load_weights"], str)
        model_save_path = config["model"]["load_weights"]
        del config["model"]["load_weights"]
    else:
        load_weights = False
    
    vae = ATTENTION_VAE(**config["model"]).to(device)
    #vae = CNN_VAE(**config["model"]).to(device)
    
    # Print the total number of model parameters
    logging.info(
        f"The model contains {count_parameters(vae)} parameters"
    )
    
    restart_training = config["trainer"]["start_epoch"] > 0
    
    if load_weights:
        pretrained_model = load_checkpoint(model_save_path)
        vae.load_state_dict(pretrained_model["model_state_dict"])
        logging.info(f"Loaded model weights from {model_save_path}")
      
    ############################################################
    #
    # Load the optimizer (after the model gets mounted onto GPU)
    #
    ############################################################
    optimizer_config = config["optimizer"]
    learning_rate = optimizer_config["lr"]
    optimizer_type = optimizer_config["type"]

    if optimizer_type == "lookahead-diffgrad":
        optimizer = LookaheadDiffGrad(vae.parameters(), lr=learning_rate)
    elif optimizer_type == "diffgrad":
        optimizer = DiffGrad(vae.parameters(), lr=learning_rate)
    elif optimizer_type == "lookahead-radam":
        optimizer = LookaheadRAdam(vae.parameters(), lr=learning_rate)
    elif optimizer_type == "radam":
        optimizer = RAdam(vae.parameters(), lr=learning_rate)
    elif optimizer_type == "adam":
        optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)
    elif optimizer_type == "sgd":
        optimizer = torch.optim.SGD(vae.parameters(), lr=learning_rate)
    else:
        logging.warning(
            f"Optimzer type {optimizer_type} is unknown. Exiting with error."
        )
        sys.exit(1)

    if restart_training and load_weights:
        optimizer.load_state_dict(pretrained_model["optimizer_state_dict"])
        for param_group in optimizer.param_groups:
            learning_rate = param_group["lr"]
            break
        logging.info(f"Loaded the {optimizer_type} optimizer and weights {model_save_path}")
        logging.info(f"... with learning rate {learning_rate}")
        
    else:
        logging.info(
            f"Loaded the {optimizer_type} optimizer with learning rate {learning_rate}"
        )
          
 
    ############################################################
    #
    # Load a Trainer object
    #
    ############################################################
        
    logging.info("Loading trainer object")
        
    trainer = BaseTrainer(
        model = vae,
        optimizer = optimizer,
        train_gen = train_gen,
        valid_gen = valid_gen, 
        dataloader = dataloader, 
        valid_dataloader = valid_dataloader,
        device = device,
        **config["trainer"]
    )
    
    ############################################################
    #
    # Load callbacks
    #
    ############################################################
    
    # Initialize LR annealing scheduler 
    if "ReduceLROnPlateau" in config["callbacks"]:
        schedule_config = config["callbacks"]["ReduceLROnPlateau"]
        scheduler = ReduceLROnPlateau(trainer.optimizer, **schedule_config)
        logging.info(
            f"Loaded ReduceLROnPlateau learning rate annealer with patience {schedule_config['patience']}"
        )
    elif "ExponentialLR" in config["callbacks"]:
        schedule_config = config["callbacks"]["ExponentialLR"]
        scheduler = ExponentialLR(trainer.optimizer, **schedule_config)
        logging.info(
            f"Loaded ExponentialLR learning rate annealer with reduce factor {schedule_config['gamma']}"
        )

    ### No restart_training options yet for checkpoint_config ... 
        
    # Early stopping
    checkpoint_config = config["callbacks"]["EarlyStopping"]
    early_stopping = EarlyStopping(**checkpoint_config)

    # Write metrics to csv each epoch
    
    if restart_training:
        config["callbacks"]["MetricsLogger"]["reload"] = True
    
    metrics_logger = MetricsLogger(**config["callbacks"]["MetricsLogger"])
    
    ############################################################
    #
    # Train the model
    #
    ############################################################
    
    trainer.train(scheduler, early_stopping, metrics_logger)
    
    ############################################################
    #
    # Make a video of the results
    #
    ############################################################

    video = config["trainer"]["path_save"]
    generate_video(video, "generated_hologram.avi") 
