import warnings
warnings.filterwarnings("ignore")

import copy
import joblib
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

from torch import nn
from torch.optim.lr_scheduler import *
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Callable, Union, Any, TypeVar, Tuple

from aimlutils.echo.src.base_objective import *
from aimlutils.torch.checkpoint import *
from aimlutils.torch.losses import *
from aimlutils.utils.tqdm import *


logger = logging.getLogger(__name__)


def custom_updates(trial, conf):
    
    # Get list of hyperparameters from the config
    hyperparameters = conf["optuna"]["parameters"]
    
    dense1 = None
    # Now update some via custom rules
    if 'dense_hidden_dim1' in hyperparameters:
        dense1 = trial_suggest_loader(trial, hyperparameters['dense_hidden_dim1'])
        dense2 = trial_suggest_loader(trial, hyperparameters['dense_hidden_dim2'])
    
    dr1 = None
    if 'dr1' in hyperparameters:
        dr1 = trial_suggest_loader(trial, hyperparameters['dr1'])
        dr2 = trial_suggest_loader(trial, hyperparameters['dr2'])
    
    if 'n_layers' in hyperparameters:
        n_layers = trial_suggest_loader(trial, hyperparameters['n_layers'])

    # Update the config based on optuna suggestions
    if dense1 is not None:
        conf["regressor"]["hidden_dims"] = [dense1] + [dense2 for k in range(n_layers)]
    if dr1 is not None:
        conf["regressor"]["dropouts"] = [dr1] + [dr2 for k in range(n_layers)]
    
#     # Update the number of bins in the image
#     if 'bins' in hyperparameters:
#         bins = trial_suggest_loader(trial, hyperparameters['bins'])
#         conf["train_data"]["bins"] = bins
#         conf["validation_data"]["bins"] = bins

    # Update the number of bins in the image
    if 'bins_x' in hyperparameters:
        bins_x = trial_suggest_loader(trial, hyperparameters['bins_x'])
        bins_y = trial_suggest_loader(trial, hyperparameters['bins_y'])
        conf["train_data"]["bins"] = [bins_x, bins_y]
        conf["validation_data"]["bins"] = [bins_x, bins_y]
    
    return conf


class Objective(BaseObjective):
    
    def __init__(self, config, metric = "val_loss", device = "cpu"):
        
        BaseObjective.__init__(self, config, metric, device)
        
        if self.device != "cpu":
            torch.backends.cudnn.benchmark = True


    def train(self, trial, conf):   
        
        random.seed(5000)

        # Implement custom changes to config        
        conf = custom_updates(trial, conf)
        
        # Load image transformations
        train_transform = LoadTransformations(conf["train_transforms"], device = self.device)
        valid_transform = LoadTransformations(conf["validation_transforms"], device = self.device)
        
        # Load the readers
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
        if "type" in conf["trainer"]:
            conf["trainer"].pop("type")
        
        trainer = CustomTrainer(
            train_gen=train_gen,
            valid_gen=valid_gen,
            dataloader=train_dataloader,
            valid_dataloader=valid_dataloader,
            vae_conf=conf["vae"],
            decoder_conf=conf["decoder"],
            regressor_conf=conf["regressor"],
            decoder_optimizer_conf=conf["rnn_optimizer"],
            regressor_optimizer_conf=conf["particle_optimizer"],
            device=self.device,
            **conf["trainer"]
        )
        
        # Initialize LR annealing scheduler 
        if "ReduceLROnPlateau" in conf["callbacks"]:
            schedule_config1 = conf["callbacks"]["ReduceLROnPlateau"]["decoder"]
            schedule_config2 = conf["callbacks"]["ReduceLROnPlateau"]["regressor"]
            scheduler_rnn = ReduceLROnPlateau(trainer.rnn_optimizer, **schedule_config1)
            scheduler_linear = ReduceLROnPlateau(trainer.particle_optimizer, **schedule_config2)

        elif "ExponentialLR" in conf["callbacks"]:
            schedule_config1 = conf["callbacks"]["ExponentialLR"]["decoder"]
            schedule_config2 = conf["callbacks"]["ExponentialLR"]["regressor"]
            scheduler_rnn = ExponentialLR(trainer.rnn_optimizer, **schedule_config1)
            scheduler_linear = ExponentialLR(trainer.particle_optimizer, **schedule_config2)

        # Early stopping
        early_stopping_rnn = EarlyStopping(**conf["callbacks"]["EarlyStopping"]["decoder"]) 
        early_stopping_linear = EarlyStopping(**conf["callbacks"]["EarlyStopping"]["regressor"])
        
        # Train the models
        val_loss, val_ce, val_seq_acc, val_acc, val_stop_acc = trainer.train(
            trial,
            scheduler_rnn, 
            scheduler_linear, 
            early_stopping_rnn, 
            early_stopping_linear
        )
        
        results = {
            "val_loss": val_loss,
            "val_ce": val_ce,
            "val_seq_acc": val_seq_acc,
            "val_acc": val_acc,
            "val_stop_acc": val_stop_acc
        }
        
        return results
    
    
class CustomTrainer(DecoderTrainer):
    
    def train(self, 
              trial,
              scheduler_rnn, 
              scheduler_linear, 
              early_stopping_rnn, 
              early_stopping_linear):

        flag_rnn = isinstance(scheduler_rnn, torch.optim.lr_scheduler.ReduceLROnPlateau)
        flag_linear = isinstance(scheduler_linear, torch.optim.lr_scheduler.ReduceLROnPlateau)
        
        val_loss, val_ce, val_acc, val_seq_acc, val_stop_acc = [], [], [], [], []
        for epoch in range(self.start_epoch, self.epochs):
            
            try:
                tf = 1.0 * (self.forcing) ** epoch 
                train_losses = self.train_one_epoch(epoch, tf)
                test_losses = self.test(epoch)
            
            except Exception as E: # CUDA memory overflow
                if "CUDA" in str(E) or "cublas" in str(E):
                    logger.info(
                        "Failed to train the model due to GPU memory overflow."
                    )
                    raise optuna.TrialPruned()
                    #raise ValueError(f"{str(E)}") # FAIL the trial, but do not stop the study
                else:
                    raise OSError(f"{str(E)}") # FAIL the trial and stop the study
                
            test_loss = np.mean(test_losses["mse"])
            test_ce = np.mean(test_losses["bce"])
            test_acc = np.mean(test_losses["accuracy"])
            test_seq_acc = np.mean(test_losses["seq_acc"])
            test_stop_acc = np.mean(test_losses["stop_accuracy"])
     
            test_loss = math.inf if not test_loss else test_loss
            test_loss = math.inf if test_loss == float("nan") else test_loss
                    
            if not isinstance(test_loss, float):
                raise ValueError(f"The test loss was {test_loss} e.g. not a float -- FAILING this trial.")
                
            if not np.isfinite(test_loss):
                logger.info(f"Pruning this trial at epoch {epoch} with loss {test_loss}")
                raise optuna.TrialPruned()
                
            # Update callbacks
            scheduler_rnn.step(1.0-test_seq_acc if flag_rnn else (1 + epoch))
            scheduler_linear.step(test_loss if flag_linear else (1 + epoch))

            early_stopping_rnn(epoch, 1.0-test_seq_acc, self.decoder, self.rnn_optimizer)
            early_stopping_linear(epoch, test_loss, self.regressor, self.particle_optimizer)
                        
            if early_stopping_linear.early_stop:
                self.train_regressor = False
            if early_stopping_rnn.early_stop:
                self.train_rnn = False
            
            val_loss.append(test_loss)
            val_ce.append(test_ce)
            val_seq_acc.append(test_seq_acc)
            val_acc.append(test_acc)
            val_stop_acc.append(test_stop_acc)
                
            if early_stopping_linear.early_stop: 
                logger.info("Stopping early due to no recent improvement in val_loss")
                if trial:
                    trial.report(test_loss, step=epoch)
                break
                
            if trial:
                trial.report(test_loss, step=epoch)
                if trial.should_prune():
                    logger.info(f"Pruning this trial at epoch {epoch} with loss {test_loss}")
                    raise optuna.TrialPruned()
                    
        # Return the best loss and the other quantities at the same epoch
        temp = min(val_loss) 
        best_idx = [i for i, j in enumerate(val_loss) if j == temp]
        
        if len(best_idx) > 0:    
            return val_loss[best_idx[-1]], val_ce[best_idx[-1]], val_seq_acc[best_idx[-1]], val_acc[best_idx[-1]], val_stop_acc[best_idx[-1]]
        else:
            return test_loss, test_ce, test_seq_acc, test_acc, test_stop_acc