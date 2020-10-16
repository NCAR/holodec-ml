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
from multiprocessing import cpu_count
from typing import List, Dict, Callable, Union, Any, TypeVar, Tuple
from torch import nn
from torch.optim.lr_scheduler import *
from torch.utils.data import Dataset, DataLoader
import logging
import pickle
import torch
import tqdm
import yaml
import sys
import os
import gc
import shutil
import sherpa

from sherpa.algorithms import Genetic
import sherpa.algorithms.bayesian_optimization as BayesianOptimization

import warnings
warnings.filterwarnings("ignore")

# single-node multi-GPU usage - https://github.com/optuna/optuna/issues/1365


class CustomTrainer(BaseEncoderTrainer):

    def train(self,
              study,
              trial,
              scheduler,
              early_stopping,
              metrics_logger):

        flag = isinstance(
            scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

        for epoch in range(self.start_epoch, self.epochs):
            train_loss = self.train_one_epoch(epoch)
            test_loss = self.test(epoch)

            study.add_observation(
                trial=trial,
                iteration=epoch,
                objective=-test_loss
            )

            if study.should_trial_stop(trial):
                break

        return study, trial


is_cuda = torch.cuda.is_available()
device = torch.device(torch.cuda.current_device()
                      ) if is_cuda else torch.device("cpu")

if is_cuda:
    torch.backends.cudnn.benchmark = True

print(f'Preparing to use device {device}')

#######################################################################

with open("config.yml") as config_file:
    config = yaml.load(config_file, Loader=yaml.FullLoader)


save_path = config["log"]
model_type = config["type"]

parameters = [
    sherpa.Discrete('hidden1', [1, 100]),
    sherpa.Discrete('hidden2', [1, 200]),
    sherpa.Continuous('lr', [1e-6, 1e-2])
]

#algorithm = Genetic(max_num_trials=100)

algorithm = BayesianOptimization.GPyOpt(
    max_concurrent=1,
    model_type='GP_MCMC',
    acquisition_type='EI_MCMC',
    max_num_trials=100
)

study = sherpa.Study(
    parameters=parameters,
    algorithm=algorithm,
    output_dir=config["log"],
    lower_is_better=False
)

#######################################################################

for trial in study:

    print("Trial {}:\t{}".format(trial.id, trial.parameters))

    # Update the configuration
    config["model"]["dense_hidden_dims"] = [
        int(trial.parameters[f'hidden{k+1}']) for k in range(2)]
    config["optimizer"]["lr"] = float(trial.parameters['lr'])
    config["trainer"]["epochs"] = 5
    bs = int(config["iterator"]["batch_size"])

    while bs >= 2:

        try:

            ############################################################
            #
            # Load image transformations followed by the data
            #
            ############################################################

            # Image transformations
            transform = LoadTransformations(config["transforms"], device = device)

            # Data readers for train/test
            train_gen = LoadReader(
                reader_type = model_type, 
                split = "train", 
                transform = transform,
                scaler = None,
                config = config["data"]
            )

            valid_gen = LoadReader(
                reader_type = model_type, 
                split = "test", 
                transform = transform, 
                scaler = train_gen.get_transform(),
                config = config["data"],
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
            
            model = LoadModel(model_type, config["model"], device)

            ############################################################
            #
            # Load the optimizer (after the model gets mounted onto GPU)
            #
            ############################################################

            optimizer_config = config["optimizer"]
            learning_rate = optimizer_config["lr"] 
            optimizer_type = optimizer_config["type"]

            optimizer = LoadOptimizer(optimizer_type, model.parameters(), learning_rate)

            logging.info(
                f"Loaded the {optimizer_type} optimizer with learning rate {learning_rate}"
            )

            ############################################################
            #
            # Load a Trainer object
            #
            ############################################################

            logging.info("Loading trainer object")

            trainer = CustomTrainer(
                model=model,
                optimizer=optimizer,
                train_gen=train_gen,
                valid_gen=valid_gen,
                dataloader=dataloader,
                valid_dataloader=valid_dataloader,
                device=device,
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

            # Early stopping
            checkpoint_config = config["callbacks"]["EarlyStopping"]
            early_stopping = EarlyStopping(**checkpoint_config)

            # Write metrics to csv each epoch
            metrics_logger = MetricsLogger(**config["callbacks"]["MetricsLogger"])

            ############################################################
            #
            # Train the model
            #
            ############################################################

            study, trial = trainer.train(
                study, trial, scheduler, early_stopping, metrics_logger)
            study.finalize(trial=trial)
            break

        except RuntimeError:
            bs /= 2
            config["batch_size"] = int(bs)

        # torch.cuda.empty_cache()
        # gc.collect()

    
    # Save the parameters and the trial.id
    try:
        shutil.copy(
            os.path.join(
                save_path, f'image_epoch_{config["trainer"]["epochs"]-1}.png'),
            os.path.join(save_path, f'trial_{trial.id}.png')
        )
    except:
        for k in range(5):
            shutil.copy(
                os.path.join(
                    save_path, f'image_epoch_{config["trainer"]["epochs"]-1}_{k}.png'),
                os.path.join(save_path, f'trial_{trial.id}_{k}.png')
            )

    with open(os.path.join(save_path, "stats.txt"), "a+") as fid:
        fid.write(f"{trial.id} {bs} {trial.parameters}\n")

    print(study.get_best_result())
