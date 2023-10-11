from holodecml.scheduler import CosineAnnealingWarmupRestarts

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.distributed import DistributedSampler
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.backends.cudnn as cudnn
import torchvision
import functools
from torchvision import models
import torch.nn as nn

from holodecml.propagation import InferencePropagator
from holodecml.transforms import LoadTransformations
from echo.src.base_objective import BaseObjective

# from holodecml.seed import seed_everything
from holodecml.datasets import LoadHolograms, UpsamplingReader, unpad_images_and_mask
from holodecml.data import XarrayReader
from holodecml.models import load_model
from holodecml.losses import load_loss
from collections import defaultdict
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import numpy as np
import subprocess
import torch.fft
import logging
import shutil
import random
import psutil
import optuna
import time
import tqdm
import gc
import os
import sys
import yaml
import warnings

from torch.cuda.amp import GradScaler, autocast
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)

warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


available_ncpus = len(psutil.Process().cpu_affinity())


class SmallFeatureMapUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(SmallFeatureMapUNet, self).__init__()
        self.encoder_conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.encoder_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder_conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.encoder_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.decoder_upconv1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.decoder_conv1 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        
        self.decoder_upconv2 = nn.ConvTranspose2d(16, out_channels, kernel_size=2, stride=2)
        #self.decoder_conv2 = nn.Conv2d(16 + in_channels, out_channels, kernel_size=3, padding=1)  # Increase in_channels
        
    def forward(self, x):
        enc1 = nn.functional.relu(self.encoder_conv1(x))
        enc1_pool = self.encoder_pool1(enc1)
        
        enc2 = nn.functional.relu(self.encoder_conv2(enc1_pool))
        enc2_pool = self.encoder_pool2(enc2)
        
        dec1_upconv = self.decoder_upconv1(enc2_pool)
        dec1_concat = torch.cat((enc1_pool, dec1_upconv), dim=1)
        dec1 = nn.functional.relu(self.decoder_conv1(dec1_concat))
        
        dec2_upconv = self.decoder_upconv2(dec1)
        return torch.sigmoid(dec2_upconv)
    

def setup(rank, world_size):
    logging.info(f"Running FSDP on rank {rank} with world_size {world_size}.")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True


def launch_pbs_jobs(config, save_path="./"):
    script_path = Path(__file__).absolute()
    script = f"""
    #!/bin/bash -l
    #PBS -N holo-trainer
    #PBS -l select=1:ncpus=32:ngpus=4:mem=480GB
    #PBS -l walltime=12:00:00
    #PBS -A NAML0001
    #PBS -q main
    #PBS -o {os.path.join(save_path, "out")}
    #PBS -e {os.path.join(save_path, "out")}

    source ~/.bashrc
    module load conda
    #conda activate /glade/work/schreck/miniconda3/envs/evidential
    conda activate holodec
    python {script_path} -c {config} -w 4 1> /dev/null
    """
    with open("launcher.sh", "w") as fid:
        fid.write(script)
    jobid = subprocess.Popen(
        "qsub launcher.sh",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).communicate()[0]
    jobid = jobid.decode("utf-8").strip("\n")
    print(jobid)
    os.remove("launcher.sh")


def trainer(rank, world_size, conf, trial=False):
    setup(rank, world_size)
    
    # infer device id from rank
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}") if torch.cuda.is_available() else torch.device("cpu")
    torch.cuda.set_device(rank % torch.cuda.device_count())
    
    seed = 1000 if "seed" not in conf else conf["seed"]
    tile_size = int(conf["data"]["tile_size"])
    step_size = int(conf["data"]["step_size"])
    data_path = conf["data"]["output_path"]
    total_positive = int(conf["data"]["total_positive"])
    total_negative = int(conf["data"]["total_negative"])
    total_examples = int(conf["data"]["total_training"])
    transform_mode = (
        "None"
        if "transform_mode" not in conf["data"]
        else conf["data"]["transform_mode"]
    )
    config_ncpus = int(conf["data"]["cores"])

    # Set up number of CPU cores available
    if config_ncpus > available_ncpus:
        ncpus = available_ncpus
        # ncpus = int(2 * available_ncpus)
    else:
        ncpus = config_ncpus
        # ncpus = int(2 * config_ncpus)

    # Set up training and validation file names. Use the prefix to use style-augmented data sets
    name_tag = f"{tile_size}_{step_size}_{total_positive}_{total_negative}_{total_examples}_{transform_mode}"
    if "prefix" in conf["data"]:
        if conf["data"]["prefix"] != "None":
            data_prefix = conf["data"]["prefix"]
            name_tag = f"{data_prefix}_{name_tag}"
    fn_train = f"{data_path}/train_{name_tag}.nc"
    fn_valid = f"{data_path}/valid_{name_tag}.nc"
    
    epochs = conf["trainer"]["epochs"]
    start_epoch = (
        0 if "start_epoch" not in conf["trainer"] else conf["trainer"]["start_epoch"]
    )
    train_batch_size = conf["trainer"]["train_batch_size"]
    valid_batch_size = conf["trainer"]["valid_batch_size"]
    batches_per_epoch = conf["trainer"]["batches_per_epoch"]
    valid_batches_per_epoch = conf["trainer"]["valid_batches_per_epoch"]
    stopping_patience = conf["trainer"]["stopping_patience"]
    grad_clip = 1.0
    model_loc = conf["save_loc"]

    training_loss = (
        "dice-bce"
        if "training_loss" not in conf["trainer"]
        else conf["trainer"]["training_loss"]
    )
    valid_loss = (
        "dice"
        if "validation_loss" not in conf["trainer"]
        else conf["trainer"]["validation_loss"]
    )

    learning_rate = conf["optimizer"]["learning_rate"]
    weight_decay = conf["optimizer"]["weight_decay"]

    # Load the preprocessing transforms
    if "Normalize" in conf["transforms"]["training"]:
        conf["transforms"]["validation"]["Normalize"]["mode"] = conf["transforms"][
            "training"
        ]["Normalize"]["mode"]
        conf["transforms"]["inference"]["Normalize"]["mode"] = conf["transforms"][
            "training"
        ]["Normalize"]["mode"]

    train_transforms = LoadTransformations(conf["transforms"]["training"])
    valid_transforms = LoadTransformations(conf["transforms"]["validation"])

    # Load the data class for reading and preparing the data as needed to train the u-net
    # train_dataset = XarrayReader(fn_train, train_transforms, mode="mask")

    # train_dataset = LoadHolograms(
    #     "/glade/p/cisl/aiml/ai4ess_hackathon/holodec/synthetic_holograms_500particle_gamma_4872x3248_training.nc", 
    #     shuffle = False, 
    #     device = rank, 
    #     n_bins = int(conf["data"]["n_bins"]), 
    #     transform = train_transforms, 
    #     lookahead = 0, 
    #     tile_size = tile_size, 
    #     step_size = step_size
    # )
    
    # test_dataset = XarrayReader(fn_valid, valid_transforms, mode="mask")

    train_dataset = UpsamplingReader(
        conf, 
        "/glade/p/cisl/aiml/ai4ess_hackathon/holodec/synthetic_holograms_500particle_gamma_4872x3248_training.nc", 
        train_transforms
    )
    
    # test_dataset = LoadHolograms(
    #     "/glade/p/cisl/aiml/ai4ess_hackathon/holodec/synthetic_holograms_500particle_gamma_4872x3248_validation.nc", 
    #     shuffle = False, 
    #     device = rank, 
    #     n_bins = int(conf["data"]["n_bins"]), 
    #     transform = valid_transforms, 
    #     lookahead = 0, 
    #     tile_size = tile_size, 
    #     step_size = step_size
    # )

    test_dataset = UpsamplingReader(
        conf, 
        "/glade/p/cisl/aiml/ai4ess_hackathon/holodec/synthetic_holograms_500particle_gamma_4872x3248_validation.nc",
        valid_transforms
    )

    # setup the distributed sampler
    sampler_tr = DistributedSampler(train_dataset,
                             num_replicas=world_size,
                             rank=rank,
                             shuffle=True,  # May be True
                             seed=seed, 
                             drop_last=True)
    sampler_val = DistributedSampler(test_dataset,
                             num_replicas=world_size,
                             rank=rank,
                             seed=seed, 
                             shuffle=False, 
                             drop_last=True)
    
    # setup the dataloder for this process
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                              batch_size=train_batch_size, 
                              shuffle=False, 
                              sampler=sampler_tr, 
                              pin_memory=True, 
                              num_workers=4,
                              drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, 
                              batch_size=valid_batch_size, 
                              shuffle=False, 
                              sampler=sampler_val, 
                              pin_memory=True, 
                              num_workers=4)

    # Load a segmentation model
    unet = load_model(conf["model"])

    if start_epoch > 0:
        # Load weights
        logging.info(f"Restarting training starting from epoch {start_epoch}")
        logging.info(f"Loading model weights from {model_loc}")
        checkpoint = torch.load(
            os.path.join(model_loc, "best.pt"),
            map_location=lambda storage, loc: storage,
        )
        unet.load_state_dict(checkpoint["model_state_dict"])
        learning_rate = checkpoint["optimizer_state_dict"]["param_groups"][0]["lr"]
        
    # 
    auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=1_000_000
    )

    # have to send the module to the correct device first
    unet.to(device)
    unet = torch.compile(unet)
    
    # will not check that the device is correct here
    model = FSDP(
        unet, 
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=torch.distributed.fsdp.MixedPrecision(
            param_dtype=torch.float32, 
            reduce_dtype=torch.float32, 
            buffer_dtype=torch.float32, 
            cast_forward_inputs=True
        )
    )
    # sharding the optimizer state
    #optimizer = ZeroRedundancyOptimizer(model.parameters(), optimizer_class=torch.optim.AdamW, lr=learning_rate)
    # adam with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Specify the training and validation losses
    train_criterion = load_loss(training_loss)
    test_criterion = load_loss(valid_loss, split="validation")

    # Load a learning rate scheduler
    lr_scheduler = ReduceLROnPlateau(
        optimizer,
        patience=1,
        min_lr=1.0e-13,
        verbose=True
    )
    # lr_scheduler = CosineAnnealingWarmupRestarts(
    #     optimizer,
    #     first_cycle_steps=batches_per_epoch,
    #     cycle_mult=0.5,
    #     max_lr=learning_rate,
    #     min_lr=1e-3 * learning_rate,
    #     warmup_steps=50,
    #     gamma=0.8,
    # )

    # Reload the results saved in the training csv if continuing to train
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

    # Train a U-net model
    manual_loss = []
    scaler = GradScaler(enabled=False)
    for epoch in range(start_epoch, epochs):
        # Train the model
        model.train()

        #batch_loss = []
        train_running_loss = 0.0
        counter = 0

        # set up a custom tqdm
        batch_group_generator = tqdm.tqdm(
            enumerate(train_loader), total=batches_per_epoch, leave=True
        )

        for k, (inputs, y) in batch_group_generator:
            counter += 1
            
            # Move data to the GPU, if not there already
            inputs = inputs.to(device)
            y = y.to(device)
            
            with autocast(enabled=False):
                # get output from the model, given the inputs
                pred_mask = model(inputs)
                
                if isinstance(train_dataset, holodecml.datasets.LoadHolograms):
                    pred_mask, y = outputs, labels = unpad_images_and_mask(pred_mask, y)
                
                # get loss for the predicted output
                loss = train_criterion(pred_mask, y.float())
                
            if not np.isfinite(loss.cpu().item()):
                logging.warning(
                    f"Trial {trial.number} is being pruned due to loss = NaN while training"
                )
                if trial:
                    raise optuna.TrialPruned()
                else:
                    sys.exit(1)

            # get gradients w.r.t to parameters
            # Backpropagation
            optimizer.zero_grad()
            scaler.scale(loss).backward() # mixed precision
            # Update the weights.
            torch.distributed.barrier()
            scaler.step(optimizer) # mixed precision
            scaler.update() # mixed precision

            train_running_loss += loss.item()
            batch_loss = torch.Tensor([train_running_loss / counter]).cuda(device)
            dist.all_reduce(batch_loss, dist.ReduceOp.AVG, async_op=False)

            # update tqdm
            to_print = "Epoch {} train_loss: {:.6f}".format(epoch, batch_loss[0])
            to_print += " lr: {:.12f}".format(optimizer.param_groups[0]["lr"])
            batch_group_generator.set_description(to_print)
            batch_group_generator.update()

            # stop the training epoch when train_batches_per_epoch have been used to update
            # the weights to the model
            if k >= batches_per_epoch and k > 0:
                break

            if isinstance(lr_scheduler, CosineAnnealingWarmupRestarts):
                lr_scheduler.step()  # epoch + k / batches_per_epoch

        # Shutdown the progbar
        batch_group_generator.close()

        # Compuate final performance metrics before doing validation
        batch_loss = torch.Tensor([train_running_loss / counter]).cuda(device)
        dist.all_reduce(batch_loss, dist.ReduceOp.AVG, async_op=False)
        train_loss = batch_loss[0]

        # clear the cached memory from the gpu
        torch.cuda.empty_cache()
        gc.collect()

        # Test the model
        model.eval()
        with torch.no_grad():

            valid_loss = 0.0
            counter = 0

            # set up a custom tqdm
            batch_group_generator = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), leave=True)

            for k, (inputs, y) in batch_group_generator:
                counter += 1
                # Move data to the GPU, if not there already
                inputs = inputs.to(device)
                y = y.to(device)
                # get output from the model, given the inputs
                pred_mask = model(inputs)
                
                if isinstance(train_dataset, holodecml.datasets.LoadHolograms):
                    pred_mask, y = outputs, labels = unpad_images_and_mask(pred_mask, y)
                # get loss for the predicted output
                loss = test_criterion(pred_mask, y.float())
                valid_loss += loss.item()
                # update tqdm
                batch_loss = torch.Tensor([valid_loss / counter]).cuda(device)
                dist.all_reduce(batch_loss, dist.ReduceOp.AVG, async_op=False)
                to_print = "Epoch {} test_loss: {:.6f}".format(
                    epoch, batch_loss[0]
                )
                batch_group_generator.set_description(to_print)
                batch_group_generator.update()

                if k >= valid_batches_per_epoch and k > 0:
                    break

            # Shutdown the progbar
            batch_group_generator.close()

        # Load the manually labeled data
        #man_loss = predict_on_manual(epoch, conf, model, device)  # + np.mean(batch_loss)
        #manual_loss.append(float(man_loss))

        # Use the supplied metric in the config file as the performance metric to toggle learning rate and early stopping
        test_loss = batch_loss[0]

        if trial:
            if not np.isfinite(test_loss):
                raise optuna.TrialPruned()

        epoch_test_losses.append(test_loss)

        # clear the cached memory from the gpu
        torch.cuda.empty_cache()
        gc.collect()

        # Save the model if its the best so far.
        if not trial and (test_loss == min(epoch_test_losses)):
            state_dict = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": test_loss,
            }
            torch.save(state_dict, f"{model_loc}/best.pt")

        # Get the last learning rate
        learning_rate = optimizer.param_groups[0]["lr"]

        # Put things into a results dictionary -> dataframe
        results_dict["epoch"].append(epoch)
        results_dict["train_loss"].append(train_loss.cpu().numpy())
        results_dict["valid_loss"].append(test_loss.cpu().numpy())
        #results_dict["manual_loss"].append(man_loss)
        results_dict["learning_rate"].append(learning_rate)
        df = pd.DataFrame.from_dict(results_dict).reset_index()

        # Save the dataframe to disk
        if trial:
            df.to_csv(
                f"{model_loc}/trial_results/training_log_{trial.number}.csv",
                index=False,
            )
        else:
            df.to_csv(f"{model_loc}/training_log.csv", index=False)

        # Lower the learning rate if we are not improving
        if isinstance(lr_scheduler, ReduceLROnPlateau):
            lr_scheduler.step(test_loss)

        # Report result to the trial
        if trial:
            attempt = 0
            while attempt < 10:
                try:
                    trial.report(test_loss, step=epoch)
                    break
                except Exception as E:
                    logging.warning(
                        f"WARNING failed to update the trial with manual loss {test_loss} at epoch {epoch}. Error {str(E)}"
                    )
                    logging.warning(f"Trying again ... {attempt + 1} / 10")
                    time.sleep(1)
                    attempt += 1

            # Stop training if we have not improved after X epochs (stopping patience)
            best_epoch = [
                i
                for i, j in enumerate(epoch_test_losses)
                if j == min(epoch_test_losses)
            ][0]
            offset = epoch - best_epoch
            if offset >= stopping_patience:
                logging.info(f"Trial {trial.number} is stopping early")
                break

            if len(epoch_test_losses) == 0:
                raise optuna.TrialPruned()

    best_epoch = [
        i for i, j in enumerate(epoch_test_losses) if j == min(epoch_test_losses)
    ][0]

    result = {
        "manual_loss": manual_loss[best_epoch],
        "mask_loss": epoch_test_losses[best_epoch],
    }

    cleanup()

    return result


class Objective(BaseObjective):
    def __init__(self, config, metric="val_loss", device="cpu"):

        # Initialize the base class
        BaseObjective.__init__(self, config, metric, device)

    def train(self, trial, conf):
        try:
            return trainer(conf, trial=trial)

        except Exception as E:
            if "CUDA" in str(E):
                logging.warning(
                    f"Pruning trial {trial.number} due to CUDA memory overflow: {str(E)}."
                )
                raise optuna.TrialPruned()
            elif "dilated" in str(E):
                raise optuna.TrialPruned()
            else:
                logging.warning(f"Trial {trial.number} failed due to error: {str(E)}.")
                raise E


def dice(true, pred, k=1, eps=1e-12):
    true = np.array(true)
    pred = np.array(pred)
    intersection = np.sum(pred[true == k]) * 2.0
    denominator = np.sum(pred) + np.sum(true)
    denominator = np.maximum(denominator, eps)
    dice = intersection / denominator
    return dice


if __name__ == "__main__":

    description = "Train a segmengation model on a hologram data set"
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "-c",
        dest="model_config",
        type=str,
        default=False,
        help="Path to the model configuration (yml) containing your inputs.",
    )
    parser.add_argument(
        "-l",
        dest="launch",
        type=int,
        default=0,
        help="Submit {n_nodes} workers to PBS.",
    )
    parser.add_argument(
        "-w", 
        "--world-size", 
        type=int, 
        default=4, 
        help="Number of processes (world size) for multiprocessing"
    )
    args = parser.parse_args()
    args_dict = vars(args)
    config = args_dict.pop("model_config")
    launch = bool(int(args_dict.pop("launch")))

    # Set up logger to print stuff
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")

    # Stream output to stdout
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    # Load the configuration and get the relevant variables
    with open(config) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    # Create directories if they do not exist and copy yml file
    os.makedirs(conf["save_loc"], exist_ok=True)
    if not os.path.exists(os.path.join(conf["save_loc"], "model.yml")):
        shutil.copy(config, os.path.join(conf["save_loc"], "model.yml"))

    # Launch PBS jobs
    if launch:
        logging.info("Launching to PBS")
        launch_pbs_jobs(config, conf["save_loc"])
        sys.exit()

    seed = 1000 if "seed" not in conf else conf["seed"]
    seed_everything(seed)
  
    try:
        trainer(int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), conf)
    except:
        trainer(0, 1, conf)