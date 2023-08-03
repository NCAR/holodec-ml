from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

#from torch.optim.lr_scheduler import ReduceLROnPlateau
from holodecml.propagation import WavePropagator, InferencePropagator
from holodecml.transforms import LoadTransformations
from holodecml.metrics import DistributedROC
from holodecml.losses import FocalTverskyLoss
from echo.src.base_objective import BaseObjective

# from holodecml.seed import seed_everything
from holodecml.data import XarrayReader
from holodecml.models import load_model
from holodecml.losses import load_loss
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from argparse import ArgumentParser
from pathlib import Path

import xarray as xr
import pandas as pd
import numpy as np
import subprocess
import torch.fft
import logging
import shutil
import random
import psutil
import scipy
import optuna
import torch
import time
import tqdm
import gc
import os
import sys
import yaml
import warnings

from holodecml.metrics import DistributedROC
from scipy.signal import convolve2d
from functools import partial

warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


available_ncpus = len(psutil.Process().cpu_affinity())


# ### Set seeds for reproducibility
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
    #PBS -l select=1:ncpus=8:ngpus=1:mem=128GB
    #PBS -l walltime=24:00:00
    #PBS -l gpu_type=v100
    #PBS -A NAML0001
    #PBS -q casper
    #PBS -o {os.path.join(save_path, "out")}
    #PBS -e {os.path.join(save_path, "out")}

    source ~/.bashrc
    ncar_pylib /glade/work/$USER/py37
    python {script_path} -c {config}
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
    
    
class XarrayReaderBalancer(Dataset):

    def __init__(self,
                 fn,
                 transform=False,
                 mode="mask", 
                 color_dim=1, 
                 upsample = False):

        self.ds = xr.open_dataset(fn)  
        self.transform = transform   
        self.mode = mode
        self.color_dim = color_dim
        self.upsample = upsample
        
        if "x1" in self.ds.dims:
            if self.mode == "mask":
                self.ds = self.ds.rename_dims({"k": "n"})
                self.ds = self.ds.rename_vars({"x": "var_x", "y": "var_y"})
            else:
                self.ds = self.ds.rename_dims({"k": "n"})
                self.ds = self.ds.rename_vars({"x": "var_x", "y": "var_z"})

    def __getitem__(self, idx):

        infocus = False
        if random.random() < self.upsample:
            infocus = True
        
        while True:
            image = self.ds.var_x[idx].values

            if len(image.shape) == 2:
                image = np.expand_dims(image, 0)
            elif len(image.shape) == 3:
                image = image[:self.color_dim, :, :]
            elif len(image.shape) == 4:
                image = image[:, :self.color_dim, :, :]

            if self.mode == "mask":
                label = self.ds.var_y[idx].values
            else: # binary labels 
                label = self.ds.var_z[idx].values 

            if not infocus and label.sum() == 0:
                break
            elif infocus and label.sum() > 0:
                break  
            else:
                idx = random.randint(0, self.__len__() - 1)
            
        im = {
            "image": image,#, np.expand_dims(image, 0),
            "horizontal_flip": False,
            "vertical_flip": False
        }

        if self.transform:
            for image_transform in self.transform:
                im = image_transform(im)
        image = im["image"]
        
        # Update the mask if we flipped the original image
        if self.mode == "mask":
            if im["horizontal_flip"]:
                label = np.flip(label, axis=0)
            if im["vertical_flip"]:
                label = np.flip(label, axis=1)
        
        image = torch.tensor(image, dtype=torch.float)
        label = torch.tensor(label.copy(), dtype=torch.int)
        return (image, label)

    def __len__(self):
        return len(self.ds.n)


class InferencePropagator2(WavePropagator):

    def __init__(self,
                 data_path,
                 n_bins=1000,
                 color_dim=2,
                 tile_size=512,
                 step_size=128,
                 marker_size=10,
                 transform_mode=None,
                 device="cuda",
                 model=None,
                 transforms=None,
                 mode=None,
                 probability_threshold=0.5):

        super(InferencePropagator2, self).__init__(
            data_path,
            n_bins=n_bins,
            tile_size=tile_size,
            step_size=step_size,
            marker_size=marker_size,
            transform_mode=transform_mode,
            device=device
        )

        self.model = model
        self.model.eval()
        self.color_dim = color_dim
        self.transforms = transforms
        self.mode = mode
        self.probability_threshold = probability_threshold
        self.create_mapping()

    def create_mapping(self):

        self.idx2slice = {}
        for row_idx in range(self.Nx//self.step_size):

            if row_idx*self.step_size+self.tile_size > self.Nx:
                image_pixel_x = self.Nx-self.tile_size
                row_slice = slice(-self.tile_size, None)
                row_break = True
            else:
                image_pixel_x = row_idx*self.step_size
                row_slice = slice(row_idx*self.step_size,
                                  row_idx*self.step_size+self.tile_size)
                row_break = False

            for col_idx in range(self.Ny//self.step_size):

                if col_idx*self.step_size+self.tile_size > self.Ny:
                    image_pixel_y = self.Ny-self.tile_size
                    col_slice = slice(-self.tile_size, None)
                    col_break = True
                else:
                    image_pixel_y = col_idx*self.step_size
                    col_slice = slice(col_idx*self.step_size,
                                      col_idx*self.step_size+self.tile_size)
                    col_break = False

                self.idx2slice[row_idx, col_idx] = (row_slice, col_slice)

                if col_break:
                    break

            if row_break:
                break

    def get_sub_images_labeled(self,
                               image_tnsr,
                               z_sub_set,
                               z_counter,
                               xp, yp, zp, dp,
                               infocus_mask,
                               z_part_bin_idx,
                               batch_size=32,
                               return_arrays=False,
                               return_metrics=False,
                               thresholds=None,
                               obs_threshold=None):
        """
        Reconstruct z_sub_set planes from
        the original hologram image and
        split it into tiles of size
        tile_size

        image - 3D tensor on device to reconstruct
        z_sub_set - array of z planes to reconstruct in one batch
        z_counter - counter of how many z images have been reconstructed

        Returns 
            Esub - a list of complex tiled images 
            image_index_lst - tile index of the sub image (x,y,z)
            image_corner_coords - x,y coordinates of the tile corner (starting values)
            z_pos - the z position of the plane in m
        """

        with torch.no_grad():

            # build the torch tensor for reconstruction
            z_plane = torch.tensor(
                z_sub_set*1e-6, device=self.device).unsqueeze(-1).unsqueeze(-1)

            # reconstruct the selected planes
            E_out = self.torch_holo_set(image_tnsr, z_plane)

            if self.color_dim == 2:
                stacked_image = torch.cat([
                    torch.abs(E_out).unsqueeze(1), torch.angle(E_out).unsqueeze(1)], 1)
            elif self.color_dim == 1:
                stacked_image = torch.abs(E_out).unsqueeze(1)
            else:
                raise OSError(f"Unrecognized color dimension {self.color_dim}")
            stacked_image = self.apply_transforms(
                stacked_image.squeeze(0)).unsqueeze(0)

            size = (E_out.shape[1], E_out.shape[2])
            true_output = torch.zeros(size).to(self.device)
            pred_output = torch.zeros(size).to(self.device)
            pred_proba = torch.zeros(size).to(self.device)
            counter = torch.zeros(size).to(self.device)

            chunked = np.array_split(
                list(self.idx2slice.items()),
                int(np.ceil(len(self.idx2slice) / batch_size))
            )

            for z_idx in range(E_out.shape[0]):

                unet_mask = torch.zeros(E_out.shape[1:]).to(
                    self.device)  # initialize the UNET mask
                # locate all particles in this plane
                part_in_plane_idx = np.where(
                    z_part_bin_idx == z_idx+z_counter)[0]

                # build the UNET mask for this z plane
                for part_idx in part_in_plane_idx:
                    unet_mask += torch.from_numpy(
                        (self.y_arr[None, :]*1e6-yp[part_idx])**2 +
                        (self.x_arr[:, None]*1e6-xp[part_idx]
                         )**2 < (dp[part_idx]/2)**2
                    ).float().to(self.device)

                worker = partial(
                    self.collate_masks,
                    image=stacked_image[z_idx, :].float(),
                    mask=unet_mask
                )

                for chunk in chunked:
                    slices, x, true_mask_tile = worker(chunk)
                    pred_proba_tile = self.model(x).squeeze(1)
                    pred_mask_tile = pred_proba_tile > self.probability_threshold

                    for k, ((row_idx, col_idx), (row_slice, col_slice)) in enumerate(slices):
                        counter[row_slice, col_slice] += 1
                        true_output[row_slice,
                                    col_slice] += true_mask_tile[k]
                        pred_output[row_slice,
                                    col_slice] += pred_mask_tile[k]
                        pred_proba[row_slice,
                                   col_slice] += pred_proba_tile[k]

            return_dict = {"z": int(round(z_sub_set[0]))}
                                        
            # Compute the (x,y,d) of predicted masks
            pred_output = pred_output == counter
            true_output = true_output == counter
            
            true_coordinates = []
            if true_output.sum() > 0:
                arr, n = scipy.ndimage.label(true_output.cpu())
                _centroid = scipy.ndimage.find_objects(arr)
                for particle in _centroid:
                    xind = (particle[0].stop + particle[0].start) // 2
                    yind = (particle[1].stop + particle[1].start) // 2
                    dind = max([
                        abs(particle[0].stop - particle[0].start), 
                        abs(particle[1].stop - particle[1].start)
                    ])
                    true_coordinates.append([xind,yind,int(round(z_sub_set[0])),dind])

            pred_coordinates = []
            if pred_output.sum() > 0:
                arr, n = scipy.ndimage.label(pred_output.cpu())
                _centroid = scipy.ndimage.find_objects(arr)
                for particle in _centroid:
                    xind = (particle[0].stop + particle[0].start) // 2
                    yind = (particle[1].stop + particle[1].start) // 2
                    dind = max([
                        abs(particle[0].stop - particle[0].start), 
                        abs(particle[1].stop - particle[1].start)
                    ])
                    pred_coordinates.append([xind,yind,int(round(z_sub_set[0])),dind])
            
            return_dict["pred_output"] = pred_coordinates
            return_dict["true_output"] = true_coordinates
            
            if return_arrays:
                return_dict["pred_array"] = pred_output
                return_dict["pred_proba"] = pred_proba
                return_dict["true_array"] = true_output
                
            if return_metrics:
                pred_output = pred_output.cpu().numpy()
                pred_proba = pred_proba.cpu().numpy()
                true_output = true_output.cpu().numpy()
                roc = DistributedROC(thresholds=thresholds,
                                     obs_threshold=obs_threshold)
                roc.update(pred_proba.ravel(), true_output.ravel())
                return_dict["roc"] = roc
            
        return return_dict

    def collate_labels(self, batch, image=None, label=None):
        x, y = zip(*[
            (image[:, row_slice, col_slice],
             torch.LongTensor([int(label[row_idx, col_idx])]))
            for ((row_idx, col_idx), (row_slice, col_slice)) in batch
        ])
        return batch, torch.stack(x), torch.stack(y)  # / self.image_norm

    def collate_masks(self, batch, image=None, mask=None):
        x, y = zip(*[
            (image[:, row_slice, col_slice], mask[row_slice, col_slice])
            for ((row_idx, col_idx), (row_slice, col_slice)) in batch
        ])
        return batch, torch.stack(x), torch.stack(y)  # / self.image_norm

    def apply_transforms(self, image):
        if self.transforms:
            im = {"image": image}
            for image_transform in self.transforms:
                im = image_transform(im)
            image = im["image"]
        return image

    def get_next_z_planes_labeled(self,
                                  h_idx,
                                  z_planes_lst,
                                  batch_size=32,
                                  return_arrays=False,
                                  return_metrics=False,
                                  thresholds=np.arange(0.0, 1.1, 0.1),
                                  obs_threshold=1.0,
                                  start_z_counter=0):
        """
        Generator that returns reconstructed z patches
        input_image - 2D image array of the original captured hologam 
        z_planes_lst - list containing batchs of arrays of z positions to reconstruct
            create_z_plane_lst() will provide this for a desired batch size and set
            planes

        returns:
            sub_image - list of sub images
            image_index_lst - list of tile indicies to the sub image
            image_coords - x,y corner coordinates of the sub images
            image_z - z location of the sub image in m
        """
        # locate particle information corresponding to this hologram
        particle_idx = np.where(self.h_ds['hid'].values == h_idx+1)
        
        x_part = self.h_ds['x'].values[particle_idx]
        y_part = self.h_ds['y'].values[particle_idx]
        z_part = self.h_ds['z'].values[particle_idx]
        d_part = self.h_ds['d'].values[particle_idx]  # not used but here it is
        
        # create a 3D histogram
        in_data = np.stack((x_part, y_part, z_part)).T
        h_part = np.histogramdd(
            in_data, bins=[self.tile_x_bins, self.tile_y_bins, self.z_bins])[0]
        # specify the z bin locations of the particles
        z_part_bin_idx = np.digitize(z_part, self.z_bins)-1

        # smoothing kernel accounts for overlapping subimages when the
        # subimage is larger than the stride
        if self.step_size < self.tile_size:
            overlap_kernel = np.ones((
                self.tile_size//self.step_size, self.tile_size//self.step_size
            ))
            for z_idx in range(h_part.shape[-1]):
                b = self.tile_size//self.step_size
                h_part[:, :, z_idx] = convolve2d(h_part[:, :, z_idx], overlap_kernel)[
                    b-1:h_part.shape[0]+b-1, b-1:h_part.shape[1]+b-1]

        input_image = self.h_ds['image'].isel(hologram_number=h_idx).values

        z_counter = start_z_counter  # the number of planes reconstructed in this generator
        image_tnsr = torch.tensor(input_image, device=self.device).unsqueeze(0)
        z_planes_lst = self.create_z_plane_lst(z_planes_lst)
        
        for z_sub_set in z_planes_lst:
            yield self.get_sub_images_labeled(
                image_tnsr,
                z_sub_set,
                z_counter,
                x_part, y_part, z_part, d_part, h_part,
                z_part_bin_idx,
                batch_size=batch_size,
                return_arrays=return_arrays,
                return_metrics=return_metrics,
                thresholds=thresholds,
                obs_threshold=obs_threshold
            )
            z_counter += z_sub_set.size

            # clear the cached memory from the gpu
            torch.cuda.empty_cache()

    def create_z_plane_lst(self, z_planes_lst):
        """
        Create a list of z planes according to the requested
        batch size.  This generates the z_planes_lst argument
        needed for gen_next_z_plane()
        """
        z_lst = []
        for z_idx in z_planes_lst:
            z_lst.append(self.z_centers[z_idx:(z_idx+1)])
        return z_lst


def trainer(conf, trial=False):

    seed = 1000 if "seed" not in conf else conf["seed"]
    seed_everything(seed)

    tile_size = int(conf["data"]["tile_size"])
    step_size = int(conf["data"]["step_size"])
    data_path = conf["data"]["output_path"]
    n_bins = conf["data"]["n_bins"]
    color_dim = conf["model"]["in_channels"]
    marker_size = conf["data"]["marker_size"]
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

    # roc threshold
    obs_threshold = 1.0
    if conf["inference"]["data_set"]["name"] == "raw":
        thresholds = np.linspace(0, 1, 100)
    else:
        thresholds = 1.0 - np.logspace(
            -5, 0, num=50, endpoint=True, base=10.0, dtype=None, axis=0
        )
        thresholds = thresholds[::-1]

    inference_mode = conf["inference"]["mode"]

    if "probability_threshold" in conf["inference"]:
        probability_threshold = conf["inference"]["probability_threshold"]
    else:
        probability_threshold = 0.5

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

    # Set up CUDA/CPU devices
    is_cuda = torch.cuda.is_available()
    data_device = (
        torch.device("cpu") if "device" not in conf["data"] else conf["data"]["device"]
    )

    if torch.cuda.device_count() >= 2 and "cuda" in data_device:
        data_device = "cuda:0"
        device = "cuda:1"
        device_ids = list(range(1, torch.cuda.device_count()))
    else:
        data_device = torch.device("cpu")
        device = (
            torch.device(torch.cuda.current_device())
            if is_cuda
            else torch.device("cpu")
        )
        device_ids = list(range(torch.cuda.device_count()))

    logging.info(f"There are {torch.cuda.device_count()} GPUs available")
    logging.info(
        f"Using device {data_device} to perform wave propagation, and {device_ids} for training the model"
    )

    # Create directories if they do not exist and copy yml file
    os.makedirs(model_loc, exist_ok=True)
    model_yml_path = os.path.join(model_loc, "model.yml")
    if not os.path.exists(model_yml_path):
        with open(model_yml_path, "w") as file:
            yaml.dump(conf, file)

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

    # Load the image transformations
    if "inference" in conf["transforms"]:
        if "Normalize" in conf["transforms"]["training"]:
            conf["transforms"]["inference"]["Normalize"]["mode"] = conf["transforms"][
                "training"
            ]["Normalize"]["mode"]
        tile_transforms = LoadTransformations(conf["transforms"]["inference"])
    else:
        tile_transforms = None

    # Load the data class for reading and preparing the data as needed to train the u-net
    train_dataset = XarrayReaderBalancer(fn_train, train_transforms, mode="mask", upsample = float(conf["trainer"]["upsample"]))
    test_dataset = XarrayReader(fn_valid, valid_transforms, mode="mask")

    # Load the iterators for batching the data
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        num_workers=ncpus,
        pin_memory=True,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=valid_batch_size,
        num_workers=ncpus,  # 0 = One worker with the main process
        pin_memory=True,
        shuffle=False,
    )

    # Load a segmentation model
    unet = load_model(conf["model"]).to(device)

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

    unet = unet.to(device)

    # Multi-gpu support
    if len(device_ids) > 1:
        unet = torch.nn.DataParallel(unet, device_ids=device_ids)

    # Load an optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    if start_epoch > 0:
        # Load weights
        logging.info(f"Loading optimizer state from {model_loc}")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Specify the training and validation losses
    train_criterion = load_loss(training_loss)
    
    
    #focal-tyversky loss alpha, beta
    ftl_alpha, ftl_beta = float(conf["trainer"]["ftl_alpha"]), 1 - float(conf["trainer"]["ftl_alpha"])
    test_criterion = load_loss(valid_loss, split="validation")

    #Load a learning rate scheduler
    # lr_scheduler = ReduceLROnPlateau(
    #     optimizer,
    #     patience=1,
    #     min_lr=1.0e-13,
    #     verbose=True
    # )
    lr_scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=batches_per_epoch,
        cycle_mult=1.0,
        max_lr=learning_rate,
        min_lr=1e-3 * learning_rate,
        warmup_steps=50,
        gamma=0.8,
    )

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
    for epoch in range(start_epoch, epochs):
        # Train the model
        unet.train()

        batch_loss = []

        # set up a custom tqdm
        batch_group_generator = tqdm.tqdm(
            enumerate(train_loader), total=batches_per_epoch, leave=True
        )

        for k, (inputs, y) in batch_group_generator:
            # Move data to the GPU, if not there already
            inputs = inputs.to(device)
            y = y.to(device)

            # Clear gradient
            optimizer.zero_grad()

            # get output from the model, given the inputs
            pred_mask = unet(inputs)

            # get loss for the predicted output
            loss = train_criterion(pred_mask, y.float(), alpha = ftl_alpha, beta = ftl_beta)

            if not np.isfinite(loss.cpu().item()):
                logging.warning(
                    f"Trial {trial.number} is being pruned due to loss = NaN while training"
                )
                if trial:
                    raise optuna.TrialPruned()
                else:
                    sys.exit(1)

            # get gradients w.r.t to parameters
            loss.backward()
            batch_loss.append(loss.item())

            # gradient clip
            torch.nn.utils.clip_grad_norm_(unet.parameters(), grad_clip)

            # update parameters
            optimizer.step()

            # update tqdm
            to_print = "Epoch {} train_loss: {:.6f}".format(epoch, np.mean(batch_loss))
            to_print += " lr: {:.12f}".format(optimizer.param_groups[0]["lr"])
            batch_group_generator.set_description(to_print)
            batch_group_generator.update()

            # stop the training epoch when train_batches_per_epoch have been used to update
            # the weights to the model
            if k >= batches_per_epoch and k > 0:
                break

            lr_scheduler.step()  # epoch + k / batches_per_epoch

        # Shutdown the progbar
        batch_group_generator.close()

        # Compuate final performance metrics before doing validation
        train_loss = np.mean(batch_loss)

        # clear the cached memory from the gpu
        torch.cuda.empty_cache()
        gc.collect()

        # Test the model
        unet.eval()

        h_range = [0]
        z_list = np.array(range(170, 220))
        data_set = "/glade/p/cisl/aiml/ai4ess_hackathon/holodec/synthetic_holograms_500particle_gamma_4872x3248_validation.nc"
        with torch.no_grad():
            prop = InferencePropagator2(
                data_set,
                n_bins=n_bins,
                color_dim=color_dim,
                tile_size=tile_size,
                step_size=step_size,
                marker_size=marker_size,
                transform_mode=transform_mode,
                device=device,
                model=unet,
                mode=inference_mode,
                probability_threshold=probability_threshold,
                transforms=tile_transforms,
            )

            # Main loop to call the generator, predict with the model, and aggregate and save the results
            total_roc = DistributedROC(
                thresholds=thresholds, obs_threshold=obs_threshold
            )
            for nc, h_idx in enumerate(h_range):
                inference_generator = prop.get_next_z_planes_labeled(
                    h_idx,
                    z_list,
                    batch_size=valid_batch_size,
                    thresholds=thresholds,
                    obs_threshold=obs_threshold,
                    start_z_counter=z_list[0],
                    return_arrays=True,
                    return_metrics=True,
                )
                t, p = [], []
                for z_idx, inf_results_dict in enumerate(inference_generator):
                    true_coors = inf_results_dict["true_output"]
                    pred_coors = inf_results_dict["pred_output"]
                    total_roc.merge(inf_results_dict["roc"])
                    loss = FocalTverskyLoss()(
                        (inf_results_dict["pred_proba"] > 0.5).float(),
                        inf_results_dict["true_array"].float(),
                    )
                    #print(z_idx, "True", true_coors, "Pred", pred_coors, "Loss", loss)
                    t.append(inf_results_dict["true_array"].float())
                    p.append((inf_results_dict["pred_proba"] > 0.5).float())

            _t = torch.stack([x.unsqueeze(0) for x in t], dim=0)
            _p = torch.stack([x.unsqueeze(0) for x in p], dim=0)
            new_roc = DistributedROC(thresholds=thresholds, obs_threshold=obs_threshold)
            new_roc.update(_t.cpu(), _p.cpu())
            csi = new_roc.max_csi()
            auc = new_roc.auc()
            auc = 0.5 if not np.isfinite(auc) else auc

            # Use the supplied metric in the config file as the performance metric to toggle learning rate and early stopping
            #test_loss = auc

            # set up a custom tqdm
            batch_loss = []
            batch_group_generator = tqdm.tqdm(enumerate(test_loader), leave=True)
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
                to_print = "Epoch {} test_loss: {:.6f}".format(
                    epoch, np.mean(batch_loss)
                )
                batch_group_generator.set_description(to_print)
                batch_group_generator.update()

                if k >= valid_batches_per_epoch and k > 0:
                    break
                    
            test_loss = np.mean(batch_loss)

            # Shutdown the progbar
            batch_group_generator.close()

        # Load the manually labeled data
        man_loss = predict_on_manual(epoch, conf, unet, device)  # + np.mean(batch_loss)
        manual_loss.append(float(man_loss))

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
                "model_state_dict": unet.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": test_loss,
            }
            torch.save(state_dict, f"{model_loc}/best.pt")

        # Get the last learning rate
        learning_rate = optimizer.param_groups[0]["lr"]

        # Put things into a results dictionary -> dataframe
        results_dict["epoch"].append(epoch)
        results_dict["train_loss"].append(train_loss)
        results_dict["valid_loss"].append(test_loss)
        results_dict["manual_loss"].append(man_loss)
        results_dict["auc"].append(auc)
        results_dict["csi"].append(csi)
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
        # lr_scheduler.step(test_loss)

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
        "auc": results_dict["auc"][best_epoch],
        "csi": results_dict["csi"][best_epoch]
    }

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


def predict_on_manual(epoch, conf, model, device, max_cluster_per_image=10000):

    model.eval()

    n_bins = conf["data"]["n_bins"]
    tile_size = conf["data"]["tile_size"]
    step_size = conf["data"]["step_size"]
    marker_size = conf["data"]["marker_size"]
    raw_path = conf["data"]["raw_data"]
    output_path = conf["data"]["output_path"]
    transform_mode = (
        "None"
        if "transform_mode" not in conf["data"]
        else conf["data"]["transform_mode"]
    )
    color_dim = conf["model"]["in_channels"]

    inference_mode = conf["inference"]["mode"]
    probability_threshold = conf["inference"]["probability_threshold"]
    valid_batch_size = conf["trainer"]["valid_batch_size"]

    if "Normalize" in conf["transforms"]["training"]:
        conf["transforms"]["validation"]["Normalize"]["mode"] = conf["transforms"][
            "training"
        ]["Normalize"]["mode"]
        conf["transforms"]["inference"]["Normalize"]["mode"] = conf["transforms"][
            "training"
        ]["Normalize"]["mode"]

    tile_transforms = (
        None
        if "inference" not in conf["transforms"]
        else LoadTransformations(conf["transforms"]["inference"])
    )

    output_path = "/glade/p/cisl/aiml/ai4ess_hackathon/holodec/tiled_synthetic/"
    with torch.no_grad():
        inputs = torch.from_numpy(
            np.load(os.path.join(output_path, f"manual_images_{transform_mode}.npy"))
        ).float()
        labels = torch.from_numpy(
            np.load(os.path.join(output_path, f"manual_labels_{transform_mode}.npy"))
        ).float()

        prop = InferencePropagator(
            raw_path,
            n_bins=n_bins,
            color_dim=color_dim,
            tile_size=tile_size,
            step_size=step_size,
            marker_size=marker_size,
            transform_mode=transform_mode,
            device=device,
            model=model,
            mode=inference_mode,
            probability_threshold=probability_threshold,
            transforms=tile_transforms,
        )
        # apply transforms
        inputs = torch.from_numpy(
            np.expand_dims(
                np.vstack([prop.apply_transforms(x) for x in inputs.numpy()]), 1
            )
        )

        performance = defaultdict(list)
        batched = zip(
            np.array_split(inputs, inputs.shape[0] // valid_batch_size),
            np.array_split(labels, inputs.shape[0] // valid_batch_size),
        )
        my_iter = tqdm.tqdm(
            batched, total=inputs.shape[0] // valid_batch_size, leave=True
        )

        with torch.no_grad():
            for (x, y) in my_iter:
                pred_labels = prop.model(x.to(device)) > probability_threshold
                for pred_label, true_label in zip(pred_labels, y):
                    pred_label = torch.sum(pred_label).float().cpu()
                    pred_label = 1 if pred_label > 0 else 0
                    performance["pred_label"].append(pred_label)
                    performance["true_label"].append(int(true_label[0].item()))
                man_loss = dice(performance["true_label"], performance["pred_label"])
                to_print = "Epoch {} man_loss: {:.6f}".format(epoch, 1.0 - man_loss)
                my_iter.set_description(to_print)
                my_iter.update()

        man_loss = dice(performance["true_label"], performance["pred_label"])

        # Shutdown the progbar
        my_iter.close()

    return 1.0 - man_loss


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
    args_dict = vars(parser.parse_args())
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

    # Launch PBS jobs
    if launch:
        logging.info("Launching to PBS")
        launch_pbs_jobs(config, conf["save_loc"])
        sys.exit()

    result = trainer(conf)
