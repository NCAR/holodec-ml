import warnings
warnings.filterwarnings("ignore")

import sys 
sys.path.append("/glade/work/schreck/repos/HOLO/clean/holodec-ml")
from holodecml.data import *
from holodecml.losses import *
from holodecml.models import *
from holodecml.metrics import *
from holodecml.transforms import *
from holodecml.propagation import *

import os
import glob
import tqdm
import time
import yaml
import scipy
import pickle
import joblib
import random
import sklearn
import logging
import datetime

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from torch import nn
from collections import defaultdict
from scipy.signal import convolve2d
from sklearn.model_selection import train_test_split
from typing import List, Dict, Callable, Union, Any, TypeVar, Tuple

from colour import Color
import multiprocessing as mp
from functools import partial

######

def main(worker_info = (0, "cuda:0"), conf = None, h_idx = 10, delay = 30):
    
    # unpack the worker and GPU ids 
    this_worker, device = worker_info
    nap_time = delay * this_worker
    
    logging.info(f"Initialized worker {this_worker} on device {device}")
    logging.info(f"Napping for {nap_time} s before mounting the model")
    
    # take a nap before trying to load the model onto the GPU
    time.sleep(nap_time)
    
    # import torch / GPU packages
    import torch
    import torch.fft
    import torch.nn.functional as F
    
    import torchvision
    import torchvision.models as models

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        
    ###################################
    
    n_nodes = conf["inference"]["n_nodes"]
    n_gpus = conf["inference"]["gpus_per_node"]
    threads_per_gpu = conf["inference"]["threads_per_gpu"]
    workers = int(n_nodes * n_gpus * threads_per_gpu)
    
    save_arrays = conf["inference"]["save_arrays"]
    plot = conf["inference"]["plot"]
    
    n_bins = conf["data"]["n_bins"]
    tile_size = conf["data"]["tile_size"]
    step_size = conf["data"]["step_size"]
    marker_size = conf["data"]["marker_size"]
    synthetic_path = conf["data"]["data_path"]
    raw_path = conf["data"]["raw_data"]

    model_loc = conf["trainer"]["output_path"]
    model_name = conf["model"]["name"]
    color_dim = conf["model"]["color_dim"]
    inference_mode = conf["model"]["mode"]

    batch_size = conf["inference"]["batch_size"]
    save_arrays = conf["inference"]["save_arrays"]
    plot = conf["inference"]["plot"]
    data_set = conf["inference"]["data_set"]["path"]
    data_set_name = conf["inference"]["data_set"]["name"]

    prop_data_loc = os.path.join(model_loc, f"{data_set_name}/propagated")
    roc_data_loc = os.path.join(model_loc, f"{data_set_name}/roc")
    image_data_loc = os.path.join(model_loc, f"{data_set_name}/images")

    for directory in [prop_data_loc, roc_data_loc, image_data_loc]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # roc threshold
    obs_threshold = 1.0
    if conf["inference"]["data_set"]["name"] == "raw":
        thresholds = np.linspace(0, 1, 100)
    else:
        thresholds = 1.0 - np.logspace(-5, 0, num=50, endpoint=True, base=10.0, dtype=None, axis=0)
        thresholds = thresholds[::-1]

    ### Load the model 
    logging.info(f"Worker {this_worker}: Loading and moving model to device {device}")
    model = ResNetUNet(
        n_class = 1, 
        color_dim = color_dim
    ).to(device)

    ### Load the weights from the training location
    checkpoint = torch.load(
        os.path.join(model_loc, "best.pt"),
        map_location=lambda storage, loc: storage
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Worker {this_worker}: There are {total_params} total model parameters")
    
    ### Load the image transformations
    logger.info(f"Worker {this_worker}: Loading image transformations")
    inference_transforms = LoadTransformations(conf["transforms"]["inference"])

    ### Load the wave prop interence object
    logger.info(f"Worker {this_worker}: Loading an inference wave-prop generator")
    prop = InferencePropagator(
        data_set, 
        n_bins = n_bins,
        tile_size = tile_size,
        step_size = step_size,
        marker_size = marker_size,
        device = device,
        model = model,
        mode = inference_mode,
        transforms = inference_transforms
    )
    
    ### Create a list of z-values to propagate to
    z_list = prop.create_z_plane_lst(planes_per_call=1)
    z_list = np.array_split(z_list, workers)[this_worker]
    logger.info(f"Worker {this_worker}: Performing inference on subset of of z planes {this_worker} out of {workers}")
    ### Main loop to call the generator, predict with the model, and aggregate and save the results
    
    model.eval()
    with torch.no_grad():

        inference_generator = prop.get_next_z_planes_labeled(
            h_idx, 
            z_list, 
            batch_size = batch_size, 
            thresholds = thresholds,
            obs_threshold = obs_threshold
        )
        
        jiter = tqdm.tqdm(
            enumerate(inference_generator), 
            total = len(z_list),
            leave=True
        )

        roc = DistributedROC(thresholds=thresholds, obs_threshold=obs_threshold)
        holo_acc = []

        unet_particles = 0
        holo_particles = 0
        for z_idx, results_dict in jiter:

            # Get stuff from the results dictionary
            pred_label = results_dict["pred_output"]
            pred_prob = results_dict["pred_proba"]
            true_label = results_dict["true_output"]
            z_plane = int(results_dict["z_plane"])

            if save_arrays:
                # Save the giant matrices as sparse arrays, as most elements are zero
                save_sparse_csr(f"{prop_data_loc}/prob_{h_idx}_{z_plane}.npz", scipy.sparse.csr_matrix(pred_prob))
                save_sparse_csr(f"{prop_data_loc}/pred_{h_idx}_{z_plane}.npz", scipy.sparse.csr_matrix(pred_label))
                save_sparse_csr(f"{prop_data_loc}/true_{h_idx}_{z_plane}.npz", scipy.sparse.csr_matrix(true_label))

            # Merge the ROC result 
            this_roc = results_dict["roc"]
            roc.merge(this_roc)

            # Print some stuff
            #plane = this_roc.binary_metrics()
            #hologram = roc.binary_metrics()
            plane_acc = (pred_label == true_label).mean()
            holo_acc.append(plane_acc)

            unet_plane_particles = np.sum(pred_label == 1)
            holo_plane_particles = np.sum(true_label == 1)
            unet_particles += unet_plane_particles
            holo_particles += holo_plane_particles

            to_print = f"Id {this_worker}: Plane: {z_idx} at z: {z_plane*1e-6}"
            #to_print += f" plane_acc: {plane_acc:.4f}"
            #to_print += f" holo_acc: {np.mean(holo_acc):.4f}"
            to_print += f" plane_csi: {this_roc.max_csi():.4f}"
            to_print += f" holo_csi: {roc.max_csi():.4f}"
            to_print += f" plane_(unet/true): {int(unet_plane_particles)} / {int(holo_plane_particles)}"
            to_print += f" holo_(unet/true): {int(unet_particles)} / {int(holo_particles)}"
            jiter.set_description(to_print)
            jiter.update()
            logging.info(to_print)

            with open(f"{roc_data_loc}/roc_{h_idx}_{z_plane}.pkl", "wb") as fid:
                joblib.dump(results_dict["roc"], fid)

            # Option to plot each result per plane
            if plot:
                fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize = (12, 5))
                p1 = ax0.imshow(pred_prob, vmin = 0,  vmax = 1)
                ax0.set_title("In-focus confidence")
                fig.colorbar(p1, ax=ax0)

                p2 = ax1.imshow(pred_label, vmin = 0,  vmax = 1)
                ax1.set_title("Predicted particles")
                fig.colorbar(p2, ax=ax1)

                p3 = ax2.imshow(true_label, vmin = 0, vmax = 1)
                ax2.set_title("True particles")
                fig.colorbar(p3, ax=ax2)

                plt.tight_layout()
                plt.show()   
        
    # merge rocs that currently exist
    rocs = sorted(glob.glob(f"{roc_data_loc}/roc_{h_idx}_*.pkl"), 
                  key = lambda x: int(x.strip(".pkl").split("_")[-1]))
    
    for k, roc_fn in enumerate(rocs):
        with open(roc_fn, "rb") as fid:
            if k == 0:
                roc = joblib.load(fid)
            else:
                roc.merge(joblib.load(fid))

    with open(f"{roc_data_loc}/roc_{h_idx}.pkl", "wb") as fid:
        joblib.dump(roc, fid)

    roc_curve([roc], [model_name], ["orange"], ["o"], f"{image_data_loc}/roc_comparison_{h_idx}.png")
    performance_diagram([roc], [model_name], ["orange"], ["o"], f"{image_data_loc}/performance_comparison_{h_idx}.png")
    
    return True


if __name__ == '__main__':
    
    with open(sys.argv[1]) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)
    
    n_nodes = conf["inference"]["n_nodes"]
    n_gpus = conf["inference"]["gpus_per_node"]
    threads_per_gpu = conf["inference"]["threads_per_gpu"]
    workers = int(n_nodes * n_gpus * threads_per_gpu)
    n_bins = conf["data"]["n_bins"]

    model_loc = conf["trainer"]["output_path"]
    model_name = conf["model"]["name"]
    data_set = conf["inference"]["data_set"]["path"]
    data_set_name = conf["inference"]["data_set"]["name"]
    prop_data_loc = os.path.join(model_loc, f"{data_set_name}/propagated")
    roc_data_loc = os.path.join(model_loc, f"{data_set_name}/roc")
    image_data_loc = os.path.join(model_loc, f"{data_set_name}/images")

    for directory in [prop_data_loc, roc_data_loc, image_data_loc]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            
    ### Configuration settings for which holograms to process
    h_conf = conf["inference"]["data_set"]["holograms"]
    if isinstance(h_range, dict):
        h_min = conf["inference"]["data_set"]["holograms"]["min"]
        h_max = conf["inference"]["data_set"]["holograms"]["max"]
        h_range = range(h_min, h_max)
    elif isinstance(h_range, list):
        h_range = h_conf
    elif isinstance(h_range, int) or isinstance(h_range, float):
        h_range = [h_conf]
    else:
        raise OSError(f"Unidentified h-range settings {h_conf}")
        
    ############################################################
    # Initialize logger to stream to stdout
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')

    # Stream output to stdout
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    root.addHandler(ch)
    
    # Save the log file
    logger_name = os.path.join(os.path.join(model_loc, f"{data_set_name}/log.txt"))
    fh = logging.FileHandler(logger_name,
                             mode="w",
                             encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    root.addHandler(fh)
    ############################################################    
    
    if n_nodes == 1:
        this_node = 0
    else:
        try:
            this_node = int(sys.argv[2]) 
        except Exception as E:
            raise OSError(f"The number of nodes {n_nodes} is larger than 1; you must specify which split to use (0,...,n_nodes-1).")

    if this_node >= n_nodes:
        raise OSError(f"The id of this worker ({this_node}) exceeded the number of nodes + 1 ({n_nodes + 1}).")
        
    ######
    
    logger.info(f"Performing model inference using {model_name} on {n_bins} reconstructed planes")
    
    total_workers = list(range(workers))
    list_of_devices = [x for gpu in range(n_gpus) for x in [f"cuda:{gpu}"] * threads_per_gpu]
    available_workers = np.array_split(total_workers, int(n_nodes * n_gpus))[this_node]
    gpu_worker_list = list(zip(available_workers, list_of_devices))
        
    # Run models in parallel on the GPU(s)
    with mp.Pool(threads_per_gpu) as p:
        for h_idx in h_range:
            worker = partial(main, conf = conf, h_idx = h_idx)
            results = [r for r in p.imap(worker, gpu_worker_list)]