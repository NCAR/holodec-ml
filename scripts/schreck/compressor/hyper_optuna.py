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

import logging
import pickle
import shutil
import optuna
import joblib
import torch
import copy
import tqdm
import yaml
import sys
import os
import gc

from dask.distributed import Client, wait
from dask_cuda import LocalCUDACluster
import dask_optuna

from aimlutils.gpu import gpu_report


import warnings
warnings.filterwarnings("ignore")

import argparse


# single-node multi-GPU usage - https://github.com/optuna/optuna/issues/1365


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
                train_loss, train_accuracy = self.train_one_epoch(epoch) # Catch MemoryError --> prune trial
                test_loss, test_accuracy = self.test(epoch) 
                metric_val = -test_loss if metric == "val_loss" else test_accuracy
                trial.report(metric_val, step=epoch+1)
                scheduler.step(metric_val if flag else epoch)
                early_stopping(epoch, metric_val, self.model, self.optimizer)
                
            except RuntimeError: # CUDA memory overflow
                raise optuna.TrialPruned()
            
            if trial.should_prune():
                raise optuna.TrialPruned()
                
            if early_stopping.early_stop:
                break
                #raise optuna.TrialPruned()
                
        return test_loss, test_accuracy
    
    
class Objective:
    
    def __init__(self, study, gpu_queue, config, epochs = 100, metric = "val_loss"):
        
        # Shared queue to manage GPU IDs.
        self.study = study
        self.gpu_queue = gpu_queue
        self.config = config
        self.epochs = epochs
        self.results = defaultdict(list)
        self.results_fn = os.path.join(config["log"], "hyper_opt.csv")
        self.metric = metric

    def __call__(self, trial):
        
        # Fetch GPU ID for this trial.
        gpu_id = self.gpu_queue.get()
        device = f"cuda:{gpu_id}" if gpu_id != "cpu" else "cpu"
        
        # Copy the config
        conf = copy.deepcopy(self.config)
        save_path = conf["log"]
        model_type = conf["type"]
        
        # Set up the variables that will be used as hyperparameters
        learning_rate = trial.suggest_loguniform("lr", 1e-6, 1e-2)
        dense1 = trial.suggest_int('dense_hidden_dim1', 10, 20000) 
        dense2 = trial.suggest_int('dense_hidden_dim2', 10, 20000) 
        dr1 = trial.suggest_float('dr1', 0.0, 0.5) 
        dr2 = trial.suggest_float('dr2', 0.0, 0.5) 

        # Insert selected parameters from the trial (How to generalize the inputs?)
        conf["model"]["dense_hidden_dims"] = [dense1, dense2]
        conf["model"]["dense_dropouts"] = [dr1, dr2]
        conf["optimizer"]["lr"] = learning_rate
        conf["trainer"]["epochs"] = self.epochs
        #bs = int(config["iterator"]["batch_size"])
        
        # Load image transformations.
        transform = LoadTransformations(conf["transforms"], device = device)
        
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
        logging.info(f"Loading training data iterator using {n_workers} workers")

        dataloader = DataLoader(
            train_gen,
            **conf["iterator"]
        )

        valid_dataloader = DataLoader(
            valid_gen,
            **conf["iterator"]
        )
        
        # Load the model 
        model = LoadModel(model_type, conf["model"], device)
        
        # Load the optimizer
        optimizer_config = conf["optimizer"]
        learning_rate = optimizer_config["lr"]
        optimizer_type = optimizer_config["type"]
        optimizer = LoadOptimizer(optimizer_type, model.parameters(), learning_rate)
        
        # Load the trainer
        trainer = CustomTrainer(
            model = model,
            optimizer = optimizer,
            train_gen = train_gen,
            valid_gen = valid_gen,
            dataloader = dataloader,
            valid_dataloader = valid_dataloader,
            device = device,
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
        val_loss, val_acc = trainer.train(
            trial, scheduler, early_stopping, self.metric
        )

        # Return GPU ID to the queue.
        self.gpu_queue.put(gpu_id)

        return self.save(trial, val_loss, val_acc)
    
    def save(self, trial, val_loss, val_acc):
        # Save results
        self.results["trial"].append(trial.number)
        for param, value in trial.params.items():
            self.results[param].append(value)
        self.results["val_loss"].append(val_loss)
        self.results["val_acc"].append(val_acc)
        try:
            self.results[f"best_{self.metric}"].append(self.study.best_value)
        except:
            self.results[f"best_{self.metric}"].append(val_loss)
        pd.DataFrame.from_dict(self.results).to_csv(self.results_fn)
        return -val_loss

    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--epochs', '-e', type = int, default = None,
                        help = 'Number of epochs')
    parser.add_argument('--reload', '-r', type = int, default = 0,
                        help = 'Reload a study')
    parser.add_argument('--metric', '-m', type = str, default = 'val_loss',
                        help = 'Metric to use in study')
    
    args = parser.parse_args()
    
    
    with open("config.yml") as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    save_path = config["log"]
    
    # If command-line epochs is specified, override whats in the config file
    epochs = args.epochs
    if args.epochs is None:
        epochs = config["trainer"]["epochs"]

    # Get list of devices and initialize the Objective class
    gpu_report = gpu_report()
    gpu_list = [i for i in range(len(gpu_report))] # Can mount more than 1 model per GPU ... 
    
    # Set up dask helpers
    cluster = LocalCUDACluster(n_workers=len(gpu_list),
                               threads_per_worker=1,
                               memory_limit='32GB')
    client = Client(cluster)

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
    
    storage = dask_optuna.DaskStorage(f"sqlite:///{cached_study}")
    #storage = dask_optuna.DaskStorage() In-memory
    study = optuna.create_study(study_name="holodec_optimization",
                                storage=storage,
                                direction="maximize",
                                load_if_exists=load_if_exists)
    
    metric = str(args.metric)
    
    try:
        # Set up python manager object
        with Manager() as manager:

            # Initialize the queue by adding available GPU IDs.
            gpu_queue = manager.Queue()
            for i in gpu_list:
                gpu_queue.put(i)

            # Initialize objective function
            objective = Objective(study, gpu_queue, config, 
                                  epochs = epochs, metric = metric)

            with parallel_backend("dask", n_jobs=len(gpu_list)):
                study.optimize(objective, n_trials=100, n_jobs=len(gpu_list))
                
    except KeyboardInterrupt:
        pass
    
    client.close()
    cluster.close()