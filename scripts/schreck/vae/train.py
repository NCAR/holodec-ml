import warnings
warnings.filterwarnings("ignore")

import copy
import optuna
import random
import logging
import traceback
import numpy as np

from overrides import overrides
from holodecml.torch.utils import *
from holodecml.torch.losses import *
from holodecml.torch.visual import *
from holodecml.torch.models import *
from holodecml.torch.trainers import *
from holodecml.torch.transforms import *
from holodecml.torch.optimizers import *
from holodecml.torch.data_loader import *
from holodecml.torch.beam_search import *

from aimlutils.torch.checkpoint import *
from aimlutils.hyper_opt.base_objective import *

from torch import nn
from torch.optim.lr_scheduler import *
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Callable, Union, Any, TypeVar, Tuple


def train(conf):   
        
    random.seed(5000)
    
    is_cuda = torch.cuda.is_available()
    device = torch.device(torch.cuda.current_device()) if is_cuda else torch.device("cpu")
    logging.info(f'Preparing to use device {device}')

    ###########################################################
    #
    # Load ML pipeline, train the model, and return the result
    #
    ###########################################################

    # Load image transformations.
    train_transform = LoadTransformations(conf["train_transforms"], device = device)
    valid_transform = LoadTransformations(conf["validation_transforms"], device = device)

    # Load the data readers 
    train_gen = LoadReader(
        transform = train_transform,
        scaler = None,
        config = conf["train_data"]
    )

    valid_gen = LoadReader(
        transform = valid_transform, 
        scaler = train_gen.get_transform(),
        config = conf["validation_data"],
    )

    # Load data iterators from pytorch
    train_dataloader = DataLoader(
        train_gen,
        **conf["iterator"]
    )

    valid_dataloader = DataLoader(
        valid_gen,
        **conf["iterator"]
    )

    # Load a trainer object
    trainer = LoadTrainer(
        train_gen, 
        valid_gen, 
        train_dataloader,
        valid_dataloader,
        device, 
        conf
    )

    # Initialize LR annealing scheduler
    if "ReduceLROnPlateau" in conf["callbacks"]:
        schedule_config = conf["callbacks"]["ReduceLROnPlateau"]
        scheduler = ReduceLROnPlateau(trainer.optimizer, **schedule_config)
        logging.info(
            f"Loaded ReduceLROnPlateau learning rate annealer with patience {schedule_config['patience']}"
        )
    elif "ExponentialLR" in conf["callbacks"]:
        schedule_config = conf["callbacks"]["ExponentialLR"]
        scheduler = ExponentialLR(trainer.optimizer, **schedule_config)
        logging.info(
            f"Loaded ExponentialLR learning rate annealer with reduce factor {schedule_config['gamma']}"
        )

    # Initialize early stopping
    checkpoint_config = conf["callbacks"]["EarlyStopping"]
    early_stopping = EarlyStopping(**checkpoint_config)
    
    # Write metrics to csv each epoch
    metrics_logger = MetricsLogger(**config["callbacks"]["MetricsLogger"])

    # Train the model
    val_loss, val_mse, val_bce = trainer.train(
        scheduler, early_stopping, metrics_logger
    )

    results = {
        "val_loss": val_loss, 
        "val_mse": val_mse, 
        "val_bce": val_bce
    }

    return results


if __name__ == '__main__':
    
    if len(sys.argv) != 2:
        print("Usage: python run.py /path/to/model_config.yml")
        sys.exit()
        
    with open(sys.argv[1]) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    
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
    
    results = train(config)