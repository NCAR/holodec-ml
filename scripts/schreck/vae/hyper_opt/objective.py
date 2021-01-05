import warnings
warnings.filterwarnings("ignore")

import copy
import optuna
import random
import logging
import traceback
import numpy as np

from overrides import overrides
from holodecml.vae.losses import *
from holodecml.vae.visual import *
from holodecml.vae.models import *
from holodecml.vae.trainers import *
from holodecml.vae.transforms import *
from holodecml.vae.optimizers import *
from holodecml.vae.data_loader import *
from holodecml.vae.checkpointer import *
from aimlutils.hyper_opt.base_objective import *

from torch import nn
from torch.optim.lr_scheduler import *
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Callable, Union, Any, TypeVar, Tuple


logger = logging.getLogger(__name__)


def custom_updates(trial, conf):
    
    # Get list of hyperparameters from the config
    hyperparameters = conf["optuna"]["parameters"]
    
    # Now update some via custom rules
    dense1 = trial_suggest_loader(trial, hyperparameters['dense_hidden_dim1'])
    dense2 = trial_suggest_loader(trial, hyperparameters['dense_hidden_dim2'])
    dense3 = trial_suggest_loader(trial, hyperparameters['dense_hidden_dim3'])
    dense4 = trial_suggest_loader(trial, hyperparameters['dense_hidden_dim4'])
    dense5 = trial_suggest_loader(trial, hyperparameters['dense_hidden_dim5'])
    dense6 = trial_suggest_loader(trial, hyperparameters['dense_hidden_dim6'])
    
    # Update the config based on optuna suggestions
    conf["model"]["hidden_dims"] = [dense1, dense2, dense3, dense4, dense5, dense6]        
    
    return conf


class Objective(BaseObjective):
    
    def __init__(self, study, config, metric = "val_loss", device = "cpu"):
        
        BaseObjective.__init__(self, study, config, metric, device)
        
        if self.device != "cpu":
            torch.backends.cudnn.benchmark = True


    def train(self, trial, conf):   
        
        random.seed(5000)
        
        ###########################################################
        #
        # Implement custom changes to config
        #
        ###########################################################
        
        conf = custom_updates(trial, conf)
                
        ###########################################################
        #
        # Load ML pipeline, train the model, and return the result
        #
        ###########################################################
        
        # Load custom option for the VAE/compressor models
        model_type = conf["model"]["type"]

        # Load image transformations.
        train_transform = LoadTransformations(conf["train_transforms"], device = self.device)
        valid_transform = LoadTransformations(conf["validation_transforms"], device = self.device)

        # Load the data readers 
        train_reader_type = conf["train_data"].pop("type")
        train_gen = LoadReader(
            reader_type = train_reader_type,
            transform = train_transform,
            scaler = None,
            config = conf["train_data"]
        )

        valid_reader_type = conf["validation_data"].pop("type")
        valid_gen = LoadReader(
            reader_type = valid_reader_type, 
            transform = valid_transform, 
            scaler = train_gen.get_transform(),
            config = conf["validation_data"],
        )

        # Load data iterators from pytorch
        n_workers = conf['iterator']['num_workers']

        #logging.info(f"Loading training data iterator using {n_workers} workers")

        train_dataloader = DataLoader(
            train_gen,
            **conf["iterator"]
        )

        valid_dataloader = DataLoader(
            valid_gen,
            **conf["iterator"]
        )

        # Load the model 
#         del conf["model"]["type"]
#         model = LoadModel(model_type, conf["model"], self.device)

#         # Load the optimizer
#         optimizer_config = conf["optimizer"]
#         optimizer = LoadOptimizer(
#             optimizer_config["type"], 
#             model.parameters(), 
#             optimizer_config["lr"], 
#             optimizer_config["weight_decay"]
#         )

        # Load the trainer
#         trainer = CustomTrainer(
#             model = model,
#             optimizer = optimizer,
#             train_gen = train_gen,
#             valid_gen = valid_gen,
#             dataloader = dataloader,
#             valid_dataloader = valid_dataloader,
#             device = self.device,
#             **conf["trainer"]
#         )
        
        trainer = CustomTrainer(
            train_gen, 
            valid_gen, 
            train_dataloader, 
            valid_dataloader,
            conf["model"], 
            conf["optimizer"],
            device = self.device,
            **conf["trainer"]
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

        # Train the model
        val_loss, val_mae, val_mse, val_bce = trainer.train(
            trial, scheduler, early_stopping, self.metric
        )
        
        results = {
            "val_loss": val_loss, 
            "val_mae": val_mae,
            "val_mse": val_mse, 
            "val_bce": val_bce
        }
        
        return results #self.save(trial, results)


class CustomTrainer(BaseTrainer):

    def train(self,
              trial,
              scheduler,
              early_stopping, 
              metric = "val_loss"):

        flag = isinstance(
            scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

        val_loss, val_mae, val_mse, val_bce = [], [], [], []
        for epoch in range(self.start_epoch, self.epochs):
            
            try:
                train_loss, train_mse, train_bce = self.train_one_epoch(epoch)
                test_loss, test_mae, test_mse, test_bce = self.test(epoch)
            except Exception as E: # CUDA memory overflow
                if "CUDA" in str(E) or "cublas" in str(E):
                    logger.info(
                        "Failed to train the model due to GPU memory overflow."
                    )
                    raise ValueError(f"{str(E)}") # FAIL the trial, but do not stop the study
                else:
                    raise OSError(f"{str(E)}") # FAIL the trial and stop the study
                    
            test_loss = math.inf if not test_loss else test_loss
            test_loss = math.inf if test_loss == float("nan") else test_loss
                    
            if not isinstance(test_loss, float):
                raise ValueError(f"The test loss was {test_loss} e.g. not a float -- FAILING this trial.")
                
            if not np.isfinite(test_loss):
                logger.info(f"Pruning this trial at epoch {epoch} with loss {test_loss}")
                raise optuna.TrialPruned()
                    
            scheduler.step(test_loss if flag else epoch)
            early_stopping(epoch, test_loss, self.model, self.optimizer)

            val_loss.append(test_loss)
            val_mae.append(test_mae)
            val_mse.append(test_mse)
            val_bce.append(test_bce)
            
            if trial:
                trial.report(test_loss, step=epoch)
                if trial.should_prune():
                    logger.info(f"Pruning this trial at epoch {epoch} with loss {test_loss}")
                    raise optuna.TrialPruned()

            if early_stopping.early_stop:
                logger.info("Stopping early")
                break
                
        # Return the best loss and the other quantities at the same epoch
        temp = min(val_loss) 
        best_idx = [i for i, j in enumerate(val_loss) if j == temp]
        
        if len(best_idx) > 0:    
            return val_loss[best_idx[-1]], val_mae[best_idx[-1]], val_mse[best_idx[-1]], val_bce[best_idx[-1]]
        else:
            return test_loss, test_mae, test_mse, test_bce