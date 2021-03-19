import warnings
warnings.filterwarnings("ignore")

import copy
import optuna
import random
import joblib
import logging
import traceback
import numpy as np

from holodecml.torch.utils import *
from holodecml.torch.losses import *
from holodecml.torch.visual import *
from holodecml.torch.models import *
from holodecml.torch.trainers import *
from holodecml.torch.transforms import *
from holodecml.torch.optimizers import *
from holodecml.torch.data_loader import *
from holodecml.torch.beam_search import *

from aimlutils.echo.src.base_objective import *
from aimlutils.torch.checkpoint import *
#from aimlutils.torch.losses import *

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
    scaler_path = os.path.join(conf["trainer"]["path_save"], "scalers.save")
    
    train_gen = LoadReader(
        transform = train_transform,
        scaler = joblib.load(scaler_path) if os.path.isfile(scaler_path) else True,
        config = conf["train_data"]
    )
    
    if not os.path.isfile(scaler_path):
        joblib.dump(train_gen.scaler, scaler_path)

    valid_gen = LoadReader(
        transform = valid_transform, 
        scaler = train_gen.scaler,
        config = conf["validation_data"]
    )

    # Load data iterators from pytorch
    train_dataloader = DataLoader(
        train_gen,
        **conf["train_iterator"]
    )

    valid_dataloader = DataLoader(
        valid_gen,
        **conf["valid_iterator"]
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
        if "decoder" in conf["callbacks"]["ReduceLROnPlateau"]:
            schedule_config1 = conf["callbacks"]["ReduceLROnPlateau"]["decoder"]
            scheduler_rnn = ReduceLROnPlateau(trainer.rnn_optimizer, **schedule_config1)
        if "regressor" in conf["callbacks"]["ReduceLROnPlateau"]:
            schedule_config2 = conf["callbacks"]["ReduceLROnPlateau"]["regressor"]
            scheduler_linear = ReduceLROnPlateau(trainer.particle_optimizer, **schedule_config2)

    if "ExponentialLR" in conf["callbacks"]:
        if "decoder" in conf["callbacks"]["ExponentialLR"]:
            schedule_config1 = conf["callbacks"]["ExponentialLR"]["decoder"]
            scheduler_rnn = ExponentialLR(trainer.rnn_optimizer, **schedule_config1)
        if "regressor" in conf["callbacks"]["ExponentialLR"]:
            schedule_config2 = conf["callbacks"]["ExponentialLR"]["regressor"]
            scheduler_linear = ExponentialLR(trainer.particle_optimizer, **schedule_config2)

    # Early stopping
    early_stopping_rnn = EarlyStopping(**conf["callbacks"]["EarlyStopping"]["decoder"]) 
    early_stopping_linear = EarlyStopping(**conf["callbacks"]["EarlyStopping"]["regressor"])

    # Write metrics to csv each epoch
    metrics_logger = MetricsLogger(**conf["callbacks"]["MetricsLogger"])

    # Train the model
    results = trainer.train(
        scheduler_rnn, 
        scheduler_linear, 
        early_stopping_rnn, 
        early_stopping_linear, 
        metrics_logger
    )
    
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