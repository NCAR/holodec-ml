from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

# from torch.optim.lr_scheduler import ReduceLROnPlateau
from holodecml.propagation import InferencePropagator
from holodecml.transforms import LoadTransformations
from echo.src.base_objective import BaseObjective

# from holodecml.seed import seed_everything
from holodecml.data import XarrayReader
from holodecml.models import load_model
from holodecml.losses import load_loss

from collections import defaultdict
from argparse import ArgumentParser
from pathlib import Path
from torch.utils.data import Dataset


import xarray as xr
import torch.nn.functional as F
import pandas as pd
import numpy as np
import subprocess
import torch.fft
import logging
import shutil
import random
import psutil
import optuna
import torch
import time
import tqdm
import gc
import os
import sys
import itertools
import yaml
import warnings

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
    conda activate holodec
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
    
    
    
def create_mask(prop, h_idx, z_idx, z_ref):
    hid = h_idx + 1
    hid_mask = prop.h_ds["hid"] == hid

    # Filter particles based on h_idx
    x_part = prop.h_ds["x"].values[hid_mask]
    y_part = prop.h_ds["y"].values[hid_mask]
    z_part = prop.h_ds["z"].values[hid_mask]
    d_part = prop.h_ds["d"].values[hid_mask]

    z_indices = np.digitize(z_part, prop.z_bins) - 1
    # Initialize the UNET mask
    unet_mask = np.zeros((prop.x_arr.shape[0], prop.y_arr.shape[0]))
    z_mask = np.zeros((prop.x_arr.shape[0], prop.y_arr.shape[0]))
    num_particles = 0 
    
    if z_idx in z_indices:
        cond = np.where(z_idx == z_indices)
        x_part = x_part[cond]
        y_part = y_part[cond]
        z_part = z_part[cond]
        d_part = d_part[cond]
        
        #print(x_part, y_part, z_part, d_part)
        
        # Build the UNET mask using vectorized operations
        for part_idx in range(len(cond[0])):
            y_diff = (prop.y_arr[None, :] * 1e6 - y_part[part_idx])
            x_diff = (prop.x_arr[:, None] * 1e6 - x_part[part_idx])
            d_squared = (d_part[part_idx] / 2)**2
            unet_mask += ((y_diff**2 + x_diff**2) < d_squared).astype(float)
            z_diff = (z_part - z_ref)
            for particle in z_diff:
                z_mask += ((y_diff**2 + x_diff**2) < d_squared).astype(float) * particle
                num_particles += 1

    return torch.from_numpy(unet_mask).unsqueeze(0), num_particles, torch.from_numpy(z_mask).unsqueeze(0)



# wave propagator class, from propagator.py
class WavePropagator(object):

    def __init__(self,
                 data_path,
                 n_bins=1000,
                 step_size=128,
                 tile_size = 128,
                 marker_size=10,
                 transform_mode=None,
                 device="cpu"):

        self.h_ds = xr.open_dataset(data_path)


        if 'zMin' in self.h_ds.attrs:
            self.zMin = self.h_ds.attrs['zMin']  # minimum z in sample volume
            self.zMax = self.h_ds.attrs['zMax']
        else:  # some of the raw data does not have this parameter
            # should warn the user here through the logger
            self.zMin = 0.014
            self.zMax = 0.158 #15.8 - 1.4 / (1000)

        self.n_bins = n_bins
        self.z_bins = np.linspace(
            self.zMin, self.zMax, n_bins+1)*1e6  # histogram bin edges
        self.z_centers = self.z_bins[:-1] + 0.5 * \
            np.diff(self.z_bins)  # histogram bin center
        
        self.tile_size = step_size
        self.step_size = tile_size
        
        # UNET gaussian marker width (standard deviation) in um
        self.marker_size = marker_size
        self.device = device


        self.dx = self.h_ds.attrs['dx']      # horizontal resolution
        self.dy = self.h_ds.attrs['dy']      # vertical resolution
        self.Nx = int(self.h_ds.attrs['Nx'])  # number of horizontal pixels
        self.Ny = int(self.h_ds.attrs['Ny'])  # number of vertical pixels
        self.lam = self.h_ds.attrs['lambda']  # wavelength
        self.image_norm = 255.0
        self.transform_mode = transform_mode
        self.x_arr = np.arange(-self.Nx//2, self.Nx//2)*self.dx
        self.y_arr = np.arange(-self.Ny//2, self.Ny//2)*self.dy

        self.tile_x_bins = np.arange(-self.Nx//2,
                                     self.Nx//2, self.step_size)*self.dx*1e6
        self.tile_y_bins = np.arange(-self.Ny//2,
                                     self.Ny//2, self.step_size)*self.dy*1e6

        self.fx = torch.fft.fftfreq(
            self.Nx, self.dx, device=self.device).unsqueeze(0).unsqueeze(2)
        self.fy = torch.fft.fftfreq(
            self.Ny, self.dy, device=self.device).unsqueeze(0).unsqueeze(1)

        
        self.create_mapping()
    def torch_holo_set(self,
                       Ein: torch.tensor,
                       z_tnsr: torch.tensor):
        """
        Propagates an electric field a distance z
        Ein complex torch.tensor
        - input electric field

        fx:real torch.tensor
        - x frequency axis (3D, setup to broadcast)

        fy: real torch.tensor
        - y frequency axis (3D, setup to broadcast)

        z_tnsr: torch.tensor
        - tensor of distances to propagate the wave Ein
            expected to have dims (Nz,1,1) where Nz is the number of z
            dimensions

        lam: float
        - wavelength

        returns: complex torch.tensor with dims (Nz,fy,fx)

        Note the torch.fft library uses dtype=torch.complex64
        This may be an issue for GPU implementation

        """

        if self.transform_mode == "standard":
            Ein = Ein.float()
            Ein -= torch.mean(Ein)
            Ein /= torch.std(Ein)

        elif self.transform_mode == "min-max":
            Ein = Ein.float()
            Ein -= torch.min(Ein)
            Ein /= torch.max(Ein)

        Etfft = torch.fft.fft2(Ein)
        Eofft = Etfft*torch.exp(1j*2*np.pi*z_tnsr/self.lam *
                                torch.sqrt(1-self.lam**2*(self.fx**2+self.fy**2)))

        # It might be helpful if we could omit this step.  It would save an inverse fft.
        Eout = torch.fft.ifft2(Eofft)
        return Eout
    
    
    def create_mapping(self):
        """
        Create map from tile coordinates (x,y) to indices to slice in image to extract that tile. 
        """
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
                
                
                
    def collate_image(self, idx_dict, image = None, mask = None):
        x, y = zip(*[(image[:, row_slice, col_slice], mask[row_slice, col_slice]) for ((row_idx, col_idx, row_slice, col_slice)) in idx_dict])


class LoadHolograms(Dataset):
    
    def __init__(self, file_path, n_bins = 1000, shuffle = False, device = "cpu", transform = False, lookahead =0, step_size = 32, tile_size = 32):
        
        # num of waveprop windows
        self.n_bins = n_bins
        # device used
        self.device = device
        # shuffle frames
        self.shuffle = shuffle
        # num of frames to look ahead
        self.lookahead = lookahead
        # wavepropagator object on device
        self.propagator = WavePropagator(file_path, n_bins = n_bins, device = device, step_size = step_size, tile_size = tile_size)
        self.transform = transform
        self.indices =  [(x,y) for x in self.propagator.h_ds.hologram_number for y in range(self.n_bins - self.lookahead)]
        
        self.tile_size = tile_size
        self.idx2slice = self.propagator.idx2slice

    def __len__(self):
        return len(self.indices) * len(self.idx2slice)

    def __getitem__(self, idx):
        
        if self.shuffle:
            idx = random.choice(range(self.__len__()))
            
        #hologram_idx = idx // self.n_bins 
        #plane_idx = idx // len(self.propagator.h_ds.hologram_number)
        hologram_idx, plane_idx = self.indices[(idx) // len(self.idx2slice)]
        z_props = self.propagator.z_centers[plane_idx: plane_idx + self.lookahead + 1]
        z_props -= (z_props[1] - z_props[0]) / 2
        plane_indices = np.arange(plane_idx, plane_idx + self.lookahead + 1)
        # select hologram
        image = self.propagator.h_ds["image"].isel(hologram_number=hologram_idx).values
        
        im = {
            "image": np.expand_dims(image, 0),
            "horizontal_flip": False,
            "vertical_flip": False
        }

        # add transformations here
        if self.transform:
            for image_transform in self.transform:
                im = image_transform(im)
        image = im["image"]
        
        # Update the mask if we flipped the original image
        
        
        # prop
        
        # make tensors of size lookahead + 1, and then add tensors
        prop_synths = []
        prop_phases = []
        masks = []
        z_masks = []
        particles_in_frames = []
        
        #num_frames = len(plane_indices)
        #idx_ran = random.choice(range(num_frames))
        #z_prop = z_props[idx_ran]
        #z_ind = plane_indices[idx_ran]
        
        for k, (z_prop, z_ind) in enumerate(zip(z_props, plane_indices)):
            image_prop = self.propagator.torch_holo_set(
                image.to(self.device),
                torch.FloatTensor([z_prop*1e-6]).to(self.device)
            )
            # ABS (x-input)
            prop_synth = torch.abs(image_prop)
            prop_synths.append(prop_synth)
            # Phase (x-input)
            prop_phase = torch.angle(image_prop)
            prop_phases.append(prop_phase)  
            # Mask (y-label)
            if k == 0:
                mask, num_particles, z_mask = create_mask(self.propagator, hologram_idx, z_ind, z_prop)
                if im["horizontal_flip"]:
                    mask = torch.flip(mask, dims=(1,))
                    z_mask = torch.flip(z_mask, dims= (1,))
                if im["vertical_flip"]:
                    mask = torch.flip(mask, dims=(2,))
                    z_mask = torch.flip(z_mask, dims= (2,))
                masks.append(mask)
                z_masks.append(z_mask)

        
        
        # cat target frames with lookahead context frames, convert to ndarrays
        synth_window = torch.cat(prop_synths, dim = 0)
        phases_window = torch.cat(prop_phases, dim = 0)
        masks_window = torch.cat(masks, dim = 0)
        z_masks_window = torch.cat(z_masks, dim = 0)
        z_masks_window /= (z_props[-1] - z_props[0])
        
        # cat images and masks in color dim (0 since batch not added yet)
        image_stack = torch.cat([synth_window, phases_window], dim = 0)
        masks_stack = torch.cat([masks_window, z_masks_window], dim = 0)
        #masks_stack = torch.cat([masks_window], dim = 0)
        
        #get tiles, slicing along coords in idx2slice array that maps coord position to slice range
        slice_coords = self.idx2slice[list(self.idx2slice)[idx % len(self.idx2slice)]]
        image_stack = image_stack[:, slice_coords[0], slice_coords[1]]
        masks_stack = masks_stack[:, slice_coords[0], slice_coords[1]]
        
        return (image_stack, masks_stack)

def trainer(conf, trial=False):

    seed = 1000 if "seed" not in conf else conf["seed"]
    seed_everything(seed)
    lookahead = int(conf["data"]["lookahead"])
    n_bins = int(conf["data"]["n_bins"])
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
    z_weight = int(conf["trainer"]["z_weight"])
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
    if not os.path.exists(os.path.join(model_loc, "model.yml")):
        shutil.copy(config, os.path.join(model_loc, "model.yml"))

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

    
    # Load data class for reading and preparing data in matched frames
    
    train_dataset = LoadHolograms("/glade/p/cisl/aiml/ai4ess_hackathon/holodec/synthetic_holograms_500particle_gamma_4872x3248_training.nc", shuffle = True, device = data_device, n_bins = n_bins, transform = train_transforms, lookahead = lookahead, tile_size = tile_size, step_size = step_size)
    test_dataset = LoadHolograms("/glade/p/cisl/aiml/ai4ess_hackathon/holodec/synthetic_holograms_500particle_gamma_4872x3248_validation.nc", shuffle = False, device = data_device, n_bins = n_bins, transform = valid_transforms, lookahead = lookahead, tile_size = tile_size, step_size = step_size)
    
    

    # Load the iterators for batching the data
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        num_workers=8,
        #pin_memory=True,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=valid_batch_size,
        num_workers=8,  # 0 = One worker with the main process
        #pin_memory=True,
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
    # if len(device_ids) > 1:
    #     unet = torch.nn.DataParallel(unet, device_ids=device_ids)

    # Load an optimizer
    optimizer = torch.optim.Adam(
        unet.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    if start_epoch > 0:
        # Load weights
        logging.info(f"Loading optimizer state from {model_loc}")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Specify the training and validation losses
    train_criterion = load_loss(training_loss)
    test_criterion = load_loss(valid_loss, split="validation")

    # Load a learning rate scheduler
    #             lr_scheduler = ReduceLROnPlateau(
    #                 optimizer,
    #                 patience=1,
    #                 min_lr=1.0e-13,
    #                 verbose=True
    #             )
    lr_scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=batches_per_epoch,
        cycle_mult=1.0,
        max_lr=learning_rate,
        min_lr=1e-3 * learning_rate,
        warmup_steps=batches_per_epoch / 2,
        # warmup_steps = 50,
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
        mask_losses, z_losses = [],[]
        
        train_niter = 0
        train_npos = 0
        
        
        
        # set up a custom tqdm
        batch_group_generator = tqdm.tqdm(enumerate(train_loader), total = batches_per_epoch, leave = True)
        for k, (inputs, y) in batch_group_generator:
            # Move data to the GPU, if not there already
            #inputs, y is getitem
            
            
            inputs = inputs.to(device)
            y = y.to(device)         
            # Clear gradient
            optimizer.zero_grad()

            # get output from the model, given the inputs
            pred_mask = unet(inputs)
            #print(pred_mask.shape)

            # get loss for the predicted output
            mask, z_mask = pred_mask[:,0:1,:,:].clone().float(), pred_mask[:,1:2,:,:].clone().float()
            #print(mask.shape, z_mask.shape, y.shape)
           
            loss = train_criterion(mask, y.clone()[:,0:1,:,:].float())
            mask_losses.append(loss.detach().cpu().numpy())
            z_pred = y[:,1:2,:,:].float()
            lossfilter = (~torch.isnan(z_pred)) & (~torch.isnan(z_mask)) & (~torch.isinf(z_pred)) & (~torch.isinf(z_mask))
            z_pred = z_pred[lossfilter]
            z_mask = z_mask[lossfilter]
            L1Loss = torch.nn.L1Loss()
            
            zloss = L1Loss(z_pred, z_mask)

            if not np.isfinite(zloss.cpu().item()):
                    print("pred, true", z_mask.shape, y.clone()[:,1:2,:,:].float().shape)
                    logging.warning("nan z-trainloss! dumping file...")
                    y_dumploc = conf["save_loc"] + "/yclone_train.pt"
                    torch.save(y.float(), y_dumploc)
                    x_dumploc = conf["save_loc"] + "/xclone_train.pt"
                    torch.save(inputs.float(), x_dumploc)
                    z_dumploc = conf["save_loc"] + "/zclone_train.pt"
                    torch.save(z_mask.float(), z_dumploc)
                    sys.exit(1)    
            z_losses.append(zloss.detach().cpu().numpy())
            loss += (z_weight * zloss)
            

            # on inference, pred_mask needs to be reconstructed (untiled and unpadded)

            if not np.isfinite(loss.cpu().item()):
                logging.warning(
                    f"Trial {trial.number} is being pruned due to loss = NaN while training")
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


            # ITERATE TO HERE

            # update tqdm
            #to_print = "Epoch {}.{} train_loss: {:.6f} mask_loss: {:.6f}".format(epoch, k, np.mean(batch_loss), np.mean(mask_losses))
            to_print = "Epoch {}.{} train_loss: {:.6f} mask_loss: {:.6f} z_loss: {:.6f}".format(epoch, k, np.mean(batch_loss), np.mean(mask_losses), np.mean(z_losses))
            to_print += " lr: {:.12f}".format(optimizer.param_groups[0]["lr"])
            batch_group_generator.set_description(to_print)
            batch_group_generator.update()
            
            train_niter += 1
            
            if (int(torch.count_nonzero(mask)) != 0):
                 train_npos += 1
            
            if k >= batches_per_epoch and k > 0:
                break

            lr_scheduler.step()  # epoch + k / batches_per_epoch
                    
        # Shutdown the progbar
        batch_group_generator.close()

        # Compuate final performance metrics before doing validation
        train_loss = np.mean(batch_loss)
        mask_loss = np.mean(mask_losses)
        train_posrate = train_npos / train_niter
        z_loss = np.mean(z_losses)
        # clear the cached memory from the gpu
        torch.cuda.empty_cache()
        gc.collect()


        
    
        # Test the model
        unet.eval()
        with torch.no_grad():

            batch_test_loss = []
            mask_test_losses, z_test_losses = [],[]
            
            test_niter = 0
            test_npos = 0
            
            
            # set up a custom tqdm
            batch_group_generator = tqdm.tqdm(enumerate(test_loader), leave=True)
            for k, (inputs, y) in batch_group_generator:
                
                
                    
                # Move data to the GPU, if not there already
                inputs = inputs.to(device)
                y = y.to(device)

                # get output from the model, given the inputs
                pred_mask = unet(inputs)
                # get loss for the predicted output
                mask, z_mask = pred_mask[:,0:1,:,:].clone().float(), pred_mask[:,1:2,:,:].clone().float()

                loss = test_criterion(mask, y.clone()[:,0:1,:,:].float())
                mask_test_losses.append(loss.detach().cpu().numpy())
                L1Loss = torch.nn.L1Loss()
                z_pred = y[:,1:2,:,:].float()
                lossfilter = (~torch.isnan(z_pred)) & (~torch.isnan(z_mask)) & (~torch.isinf(z_pred)) & (~torch.isinf(z_mask))
                z_pred = z_pred[lossfilter]
                z_mask = z_mask[lossfilter]
                zloss = L1Loss(z_pred, z_mask)       

                if not np.isfinite(zloss.cpu().item()):
                    print("pred, true", z_mask.shape, y.clone()[:,1:2,:,:].float().shape)
                    logging.warning("nan z-trainloss! dumping file...")
                    y_dumploc = conf["save_loc"] + "/yclone_test.pt"
                    torch.save(y.float(), y_dumploc)
                    x_dumploc = conf["save_loc"] + "/xclone_test.pt"
                    torch.save(inputs.float(), x_dumploc)
                    z_dumploc = conf["save_loc"] + "/zclone_test.pt"
                    torch.save(z_mask.float(), z_dumploc)
                    sys.exit(1)         

                        
                        
                z_test_losses.append(zloss.detach().cpu().numpy())
                loss += (z_weight * zloss)               

                batch_test_loss.append(loss.item())
                # update tqdm
                to_print = "Epoch {}.{} test_loss: {:.6f} mask_loss: {:.6f} z_loss {:.6f}".format(epoch, k, np.mean(batch_test_loss), np.mean(mask_test_losses), np.mean(z_test_losses))
                #to_print = "Epoch {}.{} test_loss: {:.6f} mask_loss: {:.6f}".format(epoch, k, np.mean(batch_test_loss), np.mean(mask_test_losses))
                batch_group_generator.set_description(to_print)
                batch_group_generator.update()

                test_niter += 1
                
                if (int(torch.count_nonzero(mask)) != 0):
                    test_npos += 1
                
                
                if k >= valid_batches_per_epoch and k > 0:
                    break
            

            # Shutdown the progbar
            batch_group_generator.close()
            
            
        # Load the manually labeled data
        #man_loss = predict_on_manual(epoch, conf, unet, device)  # + np.mean(batch_loss)
        #manual_loss.append(float(man_loss))

        # Use the supplied metric in the config file as the performance metric to toggle learning rate and early stopping
        test_loss = np.mean(batch_test_loss)
        mask_test_loss = np.mean(mask_test_losses)
        test_posrate = test_npos/test_niter
        z_test_loss = np.mean(z_test_losses)

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
        #results_dict["manual_loss"].append(man_loss)
        results_dict["mask_train_loss"].append(mask_loss)
        results_dict["mask_test_loss"].append(mask_test_loss)
        results_dict["z_train_loss"].append(z_loss)
        results_dict["z_test_loss"].append(z_test_loss)
        results_dict["learning_rate"].append(learning_rate)
        results_dict["training_truepr"].append(train_posrate)
        results_dict["test_truepr"].append(test_posrate)
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
        #"manual_loss": manual_loss[best_epoch],
        "mask_loss": epoch_test_losses[best_epoch]
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
    """
    Dice loss function
    """
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

        
    # Edit color channels to be compatible with lookahead
    conf["model"]["in_channels"] = 2 * (conf["data"]["lookahead"] + 1)
    
    
    # Launch PBS jobs
    if launch:
        logging.info("Launching to PBS")
        launch_pbs_jobs(config, conf["save_loc"])
        sys.exit()

    result = trainer(conf)
