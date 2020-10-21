import warnings
warnings.filterwarnings("ignore")

from overrides import overrides
from holodecml.vae.losses import *
from holodecml.vae.visual import *
from holodecml.vae.models import *
from holodecml.vae.trainers import *
from holodecml.vae.transforms import *
from holodecml.vae.optimizers import *
from holodecml.vae.data_loader import *
from holodecml.vae.checkpointer import *

from shutil import copyfile
from joblib import parallel_backend
from collections import defaultdict
from multiprocessing import cpu_count, Manager
from typing import List, Dict, Callable, Union, Any, TypeVar, Tuple

from torch import nn
from torch.optim.lr_scheduler import *
from torch.utils.data import Dataset, DataLoader

import argparse
import logging
import random
import pickle
import shutil
import optuna
import joblib
import torch
import glob
import copy
import tqdm
import yaml
import sys
import os
import gc

import traceback

from dask.distributed import Client, wait
from dask_cuda import LocalCUDACluster
from dask_cuda.initialize import initialize
import dask_optuna

from aimlutils.gpu import gpu_report

           
class CustomTrainer(BaseEncoderTrainer):

    def train(self,
              trial,
              scheduler,
              early_stopping, 
              metric = "val_loss"):

        flag = isinstance(
            scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

        for epoch in range(self.start_epoch, self.epochs):
            
            try:
                train_loss, train_mse, train_bce, train_accuracy = self.train_one_epoch(epoch)
                test_loss, test_mse, test_bce, test_accuracy = self.test(epoch)
            
                if "val_loss" in metric:
                    metric_val = test_loss
                elif "val_mse_loss" in metric:
                    metric_val = test_mse
                elif "val_bce_loss" in metric:
                    metric_val = test_bce
                elif "val_acc" in metric:
                    metric_val = -test_accuracy
                else:
                    supported = "val_loss, val_mse_loss, val_bce_loss, val_acc"
                    raise ValueError(f"The metric {metric} is not supported. Choose from {supported}")

                trial.report(-metric_val, step=epoch+1)
                scheduler.step(metric_val if flag else epoch)
                early_stopping(epoch, metric_val, self.model, self.optimizer)
                
            except Exception as E: # CUDA memory overflow
                print(traceback.print_exc())
                raise optuna.TrialPruned()
            
            if trial.should_prune():
                raise optuna.TrialPruned()
                
            if early_stopping.early_stop:
                break
                
        return test_loss, test_mse, test_bce, test_accuracy
   

    
class Objective:
    
    def __init__(self, study, device, config, epochs = 100, metric = "val_loss", index = None):
        
        # Shared queue to manage GPU IDs.
        self.study = study
        self.config = config
        self.epochs = epochs
        self.results = defaultdict(list)
        self.results_fn = os.path.join(config["log"], f"hyper_opt_{random.randint(0, 1e5)}.csv")
        

        while os.path.isfile(self.results_fn):
            rand_index = random.randint(0, 1e5)
            self.results_fn = os.path.join(config["log"], f"hyper_opt_{rand_index}.csv")

        self.metric = metric
        
        self.device = f"cuda:{device}" if device != "cpu" else "cpu"
        if self.device != "cpu":
            torch.backends.cudnn.benchmark = True

    def __call__(self, trial):    
        # Copy the config
        conf = copy.deepcopy(self.config)
        save_path = conf["log"]
        model_type = conf["type"]
        
        # Set up the variables that will be used as hyperparameters
        num_dense = trial.suggest_int('num_dense', 0, 10)
        dense1 = trial.suggest_int('dense_hidden_dim1', 10, 10000)
        dense2 = trial.suggest_int('dense_hidden_dim2', 10, 5000)
        dr1 = trial.suggest_float('dr1', 0.0, 0.5)
        dr2 = trial.suggest_float('dr2', 0.0, 0.5)
        alpha = trial.suggest_float('alpha', 0.001, 1.0)
        beta = trial.suggest_float('beta', 0.001, 1.0)
        learning_rate = trial.suggest_loguniform("lr", 1e-7, 1e-2)
        weight_decay = trial.suggest_loguniform("weight_decay", 1e-8, 1e-1)

        # Insert selected parameters from the trial (How to generalize the inputs?)
        conf["model"]["dense_hidden_dims"] = [dense1] + [dense2 for k in range(num_dense)]        
        conf["model"]["dense_dropouts"] = [dr1] + [dr2 for k in range(num_dense)]
        conf["optimizer"]["lr"] = learning_rate
        conf["optimizer"]["weight_decay"] = weight_decay
        conf["trainer"]["epochs"] = self.epochs
        conf["trainer"]["alpha"] = alpha
        conf["trainer"]["beta"] = beta
        
        # Load image transformations.
        transform = LoadTransformations(conf["transforms"], device = self.device)
        
        # Load dataset readers
        train_gen = LoadReader(
            reader_type = model_type,
            split = "train", 
            transform = transform,
            scaler = None,
            config = conf["data"]
        )

        valid_gen = LoadReader(
            reader_type = model_type, 
            split = "test", 
            transform = transform, 
            scaler = train_gen.get_transform(),
            config = conf["data"],
        )
        
        # Load data iterators from pytorch
        n_workers = conf['iterator']['num_workers']

        #logging.info(f"Loading training data iterator using {n_workers} workers")

        dataloader = DataLoader(
            train_gen,
            **conf["iterator"]
        )

        valid_dataloader = DataLoader(
            valid_gen,
            **conf["iterator"]
        )

        # Load the model 
        model = LoadModel(model_type, conf["model"], self.device)
        
        # Load the optimizer
        optimizer_config = conf["optimizer"]
        optimizer = LoadOptimizer(
            optimizer_config["type"], 
            model.parameters(), 
            optimizer_config["lr"], 
            optimizer_config["weight_decay"]
        )
        
        # Load the trainer
        trainer = CustomTrainer(
            model = model,
            optimizer = optimizer,
            train_gen = train_gen,
            valid_gen = valid_gen,
            dataloader = dataloader,
            valid_dataloader = valid_dataloader,
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
        val_loss, val_mse, val_bce, val_acc = trainer.train(
            trial, scheduler, early_stopping, self.metric
        )

        return self.save(trial, val_loss, val_mse, val_bce, val_acc)
    
    def save(self, trial, val_loss, val_mse, val_bce, val_acc):
        # Save results
        self.results["trial"].append(trial.number)
        for param, value in trial.params.items():
            self.results[param].append(value)
        self.results["val_loss"].append(val_loss)
        self.results["val_mse"].append(val_mse)
        self.results["val_bce"].append(val_bce)
        self.results["val_acc"].append(val_acc)
        try:
            self.results[f"best_{self.metric}"].append(self.study.best_value)
            give_back = self.study.best_value
        except:
            if self.metric == "val_loss":
                give_back = -val_loss
            elif self.metric == "val_mse":
                give_back = -val_mse
            elif self.metric == "val_bce":
                give_back = -val_bce
            else:
                give_back = val_acc
            self.results[f"best_{self.metric}"].append(abs(give_back))
        pd.DataFrame.from_dict(self.results).to_csv(self.results_fn)
        return give_back

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', '-e', type = int, default = None,
                        help = 'Number of epochs')
    parser.add_argument('--reload', '-r', type = int, default = 0,
                        help = 'Reload a study')
    parser.add_argument('--metric', '-m', type = str, default = 'val_loss',
                        help = 'Metric to use in study')
    parser.add_argument('--gpu', '-g', type = str, default = False,
                        help = 'GPU ID')
    parser.add_argument('--trials', '-n', type = int, default = 100,
                        help = 'Number of trials.')
    parser.add_argument('--id', '-i', type = int, default = None,
                        help = 'Script ID, for preventing file over-writing.')
    parser.add_argument('--config', '-c', type = str, default = "config.yml",
                        help = 'Path to the configuration file')
    args = parser.parse_args()
    
    if os.path.isfile(args.config):
        with open(args.config) as config_file:
            config = yaml.load(config_file, Loader=yaml.FullLoader)
    else:
        raise OSError(f"Configuration file {args.config} does not exist")
        
    # Get the path to save all the data
    save_path = config["log"]
    
    # If command-line epochs is specified, override whats in the config file
    epochs = args.epochs
    if args.epochs is None:
        epochs = config["trainer"]["epochs"]

    # Get list of devices and initialize the Objective class
    if args.gpu == False:
        gpu_report = sorted(gpu_report().items(), key = lambda x: x[1], reverse = True)
        device = gpu_report[0][0]
    
    try:        
        # Initialize the study object
        reload_study = bool(args.reload)
        cached_study = f"{save_path}/holodec_optimization.db"
        if not os.path.isfile(cached_study):
            load_if_exists = False
        elif not reload_study:
            os.remove(cached_study)
            load_if_exists = reload_study
        else:
            load_if_exists = True

        storage = storage=f"sqlite:///{cached_study}"
        study = optuna.create_study(study_name="holodec_optimization",
                                    storage=storage,
                                    direction="maximize",
                                    load_if_exists=load_if_exists)
        metric = str(args.metric)

        # Initialize objective function
        objective = Objective(study, device, config, 
                              epochs = epochs, metric = metric)

        study.optimize(objective, n_trials=int(args.trials))

    except KeyboardInterrupt:
        pass
    
    # Clean up the data files
    saved_results = glob.glob(os.path.join(save_path, "*.csv"))
    saved_results = pd.concat(
        [pd.read_csv(x) for x in saved_results], sort = True
    ).reset_index(drop=True)
    saved_results = saved_results.drop(
        columns = [x for x in saved_results.columns if "Unnamed" in x]
    )
    saved_results = saved_results.sort_values(["trial"]).reset_index(drop = True)
    best_parameters = saved_results[saved_results[metric]==max(saved_results[metric])]
    
    saved_results.to_csv(os.path.join(save_path, "hyper_opt.csv"))
    best_parameters.to_csv(os.path.join(save_path, "best.csv"))
    