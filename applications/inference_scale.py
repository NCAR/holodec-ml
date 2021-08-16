#!/usr/bin/env python
# coding: utf-8

from torch.multiprocessing import Pool, set_start_method
try:
     set_start_method('spawn', force=True)
except RuntimeError:
    pass


import warnings
warnings.filterwarnings("ignore")

import os
import sys
import glob
import tqdm
import time
import yaml
import h5py
import scipy
import torch
import pickle
import joblib
import random
import sklearn
import logging
import datetime
import torch.fft
import torchvision
import torchvision.models as models

import numpy as np
import pandas as pd
import xarray as xr
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from collections import defaultdict
from scipy.signal import convolve2d
from torch.optim.lr_scheduler import *
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import List, Dict, Callable, Union, Any, TypeVar, Tuple

from scipy.sparse import csr_matrix
from colour import Color

#import multiprocessing as mp
#mp.set_start_method('spawn')

from functools import partial
from hagelslag.evaluation.ProbabilityMetrics import *
from hagelslag.evaluation.MetricPlotter import *


# is_cuda = torch.cuda.is_available()
# device = torch.device(torch.cuda.current_device()) if is_cuda else torch.device("cpu")

# if is_cuda:
#     torch.backends.cudnn.benchmark = True
#     torch.backends.cudnn.deterministic = True


def save_sparse_csr(filename, array):
    # note that .npz extension is added automatically
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)

def load_sparse_csr(filename):
    # here we need to add .npz extension manually
    loader = np.load(filename + '.npz')
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])


class DistributedROC(DistributedROC):
    
    def from_str(self, in_str):
        """
        Read the DistributedROC string and parse the contingency table values from it.
        Args:
            in_str (str): The string output from the __str__ method
        """
        parts = in_str.split(";")
        for part in parts:
            var_name, value = part.split(":")
            if var_name == "Obs_Threshold":
                self.obs_threshold = float(value)
            elif var_name == "Thresholds":
                self.thresholds = np.array(value.split(), dtype=float)
                self.contingency_tables = pd.DataFrame(columns=self.contingency_tables.columns,
                                                       data=np.zeros((self.thresholds.size,
                                                                     self.contingency_tables.columns.size)))
            elif var_name in self.contingency_tables.columns:
                self.contingency_tables[var_name] = np.array(value.split(), dtype=int)

    def __str__(self):
        """
        Output the information within the DistributedROC object to a string.
        """
        out_str = "Obs_Threshold:{0:0.8f}".format(self.obs_threshold) + ";"
        out_str += "Thresholds:" + " ".join(["{0:0.8f}".format(t) for t in self.thresholds]) + ";"
        for col in self.contingency_tables.columns:
            out_str += col + ":" + " ".join(["{0:d}".format(t) for t in self.contingency_tables[col]]) + ";"
        out_str = out_str.rstrip(";")
        return out_str

    

class ResNet(nn.Module):
    def __init__(self, fcl_layers = [], dr = 0.0, output_size = 1, resnet_model = 18, pretrained = True):
        super(ResNet, self).__init__()
        self.pretrained = pretrained
        self.resnet_model = resnet_model 
        if self.resnet_model == 18:
            resnet = models.resnet18(pretrained=self.pretrained)
        elif self.resnet_model == 34:
            resnet = models.resnet34(pretrained=self.pretrained)
        elif self.resnet_model == 50:
            resnet = models.resnet50(pretrained=self.pretrained)
        elif self.resnet_model == 101:
            resnet = models.resnet101(pretrained=self.pretrained)
        elif self.resnet_model == 152:
            resnet = models.resnet152(pretrained=self.pretrained)
        resnet.conv1 = torch.nn.Conv1d(2, 64, (7, 7), (2, 2), (3, 3), bias=False) # Manually change color dim to match our data
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet_output_dim = resnet.fc.in_features
        self.resnet = nn.Sequential(*modules)
        self.fcn = self.make_fcn(self.resnet_output_dim, output_size, fcl_layers, dr)
        
    def make_fcn(self, input_size, output_size, fcl_layers, dr):
        if len(fcl_layers) > 0:
            fcn = [
                nn.Dropout(dr),
                nn.Linear(input_size, fcl_layers[0]),
                nn.BatchNorm1d(fcl_layers[0]),
                torch.nn.LeakyReLU()
            ]
            if len(fcl_layers) == 1:
                fcn.append(nn.Linear(fcl_layers[0], output_size))
            else:
                for i in range(len(fcl_layers)-1):
                    fcn += [
                        nn.Linear(fcl_layers[i], fcl_layers[i+1]),
                        nn.BatchNorm1d(fcl_layers[i+1]),
                        torch.nn.LeakyReLU(),
                        nn.Dropout(dr)
                    ]
                fcn.append(nn.Linear(fcl_layers[i+1], output_size))
        else:
            fcn = [
                nn.Dropout(dr),
                nn.Linear(input_size, output_size)
            ]
        if output_size > 1:
            fcn.append(torch.nn.LogSoftmax(dim=1))
        return nn.Sequential(*fcn)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fcn(x)
        return x


class InferencePropagator:
    
    def __init__(self, 
                 data_path,
                 n_bins = 1000, 
                 tile_size = 512, 
                 step_size = 128, 
                 marker_size = 10, 
                 model = None,
                 device = "cuda"):
        
        self.h_ds = xr.open_dataset(data_path)
        self.n_bins = n_bins
        self.z_bins = np.linspace(self.h_ds.attrs['zMin'],self.h_ds.attrs['zMax'], n_bins+1)*1e6  # histogram bin edges
        self.z_centers = self.z_bins[:-1] + 0.5*np.diff(self.z_bins)  # histogram bin centers

        self.tile_size = tile_size  # size of tiled images in pixels
        self.step_size = step_size  # amount that we shift the tile to make a new tile
        self.marker_size = marker_size # UNET gaussian marker width (standard deviation) in um
        self.device = device

        # step_size is not allowed be be larger than the tile_size
        assert self.tile_size >= self.step_size

        self.dx = self.h_ds.attrs['dx']      # horizontal resolution
        self.dy = self.h_ds.attrs['dy']      # vertical resolution
        self.Nx = int(self.h_ds.attrs['Nx']) # number of horizontal pixels
        self.Ny = int(self.h_ds.attrs['Ny']) # number of vertical pixels
        self.lam = self.h_ds.attrs['lambda'] # wavelength
        self.zMin = self.h_ds.attrs['zMin']  # minimum z in sample volume
        self.zMax = self.h_ds.attrs['zMax'] 
        self.image_norm = 255.0

        self.x_arr = np.arange(-self.Nx//2, self.Nx//2)*self.dx
        self.y_arr = np.arange(-self.Ny//2, self.Ny//2)*self.dy
        
        self.tile_x_bins = np.arange(-self.Nx//2,self.Nx//2,self.step_size)*self.dx*1e6
        self.tile_y_bins = np.arange(-self.Ny//2,self.Ny//2,self.step_size)*self.dy*1e6
        
        self.fx = None
        self.fy = None
        self.model = model
        self.model.eval()
        
        self.create_mapping()

    
    def compute_f(self):
        self.fx = torch.fft.fftfreq(self.Nx,self.dx,device=self.device).unsqueeze(0).unsqueeze(2)
        self.fy = torch.fft.fftfreq(self.Ny,self.dy,device=self.device).unsqueeze(0).unsqueeze(1)
        
    
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
        if self.fx is None:
            self.compute_f()
        
        Etfft = torch.fft.fft2(Ein)
        Eofft = Etfft*torch.exp(1j*2*np.pi*z_tnsr/self.lam*torch.sqrt(1-self.lam**2*(self.fx**2+self.fy**2)))

        # It might be helpful if we could omit this step.  It would save an inverse fft.
        Eout = torch.fft.ifft2(Eofft)

        return Eout
    

    def create_mapping(self):
        self.idx2slice = {}
        for row_idx in range(self.Nx//self.step_size):

            if row_idx*self.step_size+self.tile_size > self.Nx:
                image_pixel_x = self.Nx-self.tile_size
                row_slice = slice(-self.tile_size,None)
                row_break = True
            else:
                image_pixel_x = row_idx*self.step_size
                row_slice = slice(row_idx*self.step_size,row_idx*self.step_size+self.tile_size)
                row_break = False

            for col_idx in range(self.Ny//self.step_size):

                if col_idx*self.step_size+self.tile_size > self.Ny:
                    image_pixel_y = self.Ny-self.tile_size
                    col_slice = slice(-self.tile_size,None)
                    col_break = True
                else:
                    image_pixel_y = col_idx*self.step_size
                    col_slice = slice(col_idx*self.step_size,col_idx*self.step_size+self.tile_size)
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
                               batch_size = 32, 
                               thresholds = None, 
                               obs_threshold = None):
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
            z_plane = torch.tensor(z_sub_set*1e-6,device=self.device).unsqueeze(-1).unsqueeze(-1)

            # reconstruct the selected planes
            E_out = self.torch_holo_set(image_tnsr,z_plane)

            image = torch.abs(E_out).unsqueeze(1)
            phase = torch.angle(E_out).unsqueeze(1)
            stacked_image = torch.cat([
                torch.abs(E_out).unsqueeze(1), torch.angle(E_out).unsqueeze(1)], 1)

            size = (E_out.shape[1], E_out.shape[2])
            true_label = torch.zeros(size).to(device)
            pred_label = torch.zeros(size).to(device)
            pred_proba = torch.zeros(size).to(device)
            counter = torch.zeros(size).to(device)

            results_dict = defaultdict(list)

            for z_idx in range(E_out.shape[0]):

                chunked = np.array_split(
                    list(self.idx2slice.items()),
                    int(np.ceil(len(self.idx2slice) / batch_size))
                )
                
                input_x = stacked_image[z_idx, :].float()
                true_y = infocus_mask[:, :, z_idx+z_counter]
                
                worker = partial(
                    self.collate, 
                    image = input_x, 
                    label = true_y
                )
                                    
                for chunk in chunked: # Loop over all 512 x 512 tiles
                
                #for ((row_idx, col_idx), (row_slice, col_slice)) in list(self.idx2slice.items()):
            
                    #x = input_x[:, row_slice, col_slice].unsqueeze(0)
                    #true_labels = torch.LongTensor([int(true_y[row_idx, col_idx])])
                    
                    slices, x, true_labels = worker(chunk)

                    true_labels = true_labels.to(device)
                    pred_logits = self.model(x)
                    
                    pred_probs, pred_labels = torch.max(pred_logits, 1)
                    pred_probs = pred_probs.exp()
                    pred_labels = pred_labels.squeeze(-1)
                    
                    pred_cond = pred_labels == 0
                    pred_probs = torch.where(pred_cond, 1.0 - pred_probs, pred_probs)

                    results_dict["acc"] += list(true_labels == pred_labels)

                    for k, ((row_idx, col_idx), (row_slice, col_slice)) in enumerate(slices):
                        counter[row_slice, col_slice] += 1
                        true_label[row_slice, col_slice] += true_labels[k]
                        pred_label[row_slice, col_slice] += pred_labels[k]
                        pred_proba[row_slice, col_slice] += pred_probs[k]
                        
            pred_label = pred_label / counter
            pred_proba = pred_proba / counter
            true_label = true_label / counter    
            
            ave_acc = []
            for x in results_dict["acc"]:
                ave_acc += list(x.detach().cpu().numpy())

            pred_label = pred_label == 1.0
            true_label = true_label == 1.0
            
            pred_label = pred_label.cpu().numpy()
            pred_proba = pred_proba.cpu().numpy()
            true_label = true_label.cpu().numpy()

            roc = DistributedROC(thresholds=thresholds, obs_threshold=obs_threshold)
            roc.update(pred_proba.ravel(), true_label.ravel())
            
            return_dict = {
                "pred_label": pred_label,
                "pred_proba": pred_proba,
                "true_label": true_label,
                #"counter": counter,
                "acc": np.mean(ave_acc),
                "z_plane": z_sub_set[z_idx],
                "roc": roc
            }
                    
        return return_dict
    
    def collate(self, batch, image = None, label = None):
        x, y = zip(*[
            (image[:, row_slice, col_slice], 
            torch.LongTensor([int(label[row_idx, col_idx])]))
            for ((row_idx, col_idx), (row_slice, col_slice)) in batch
        ])
        return batch, torch.stack(x), torch.stack(y)

            
    def get_next_z_planes_labeled(self, 
                                  h_idx, 
                                  z_planes_lst, 
                                  batch_size = 32,
                                  thresholds = np.arange(0.0, 1.1, 0.1),
                                  obs_threshold = 1.0):
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
        particle_idx = np.where(self.h_ds['hid'].values==h_idx+1)

        x_part = self.h_ds['x'].values[particle_idx]
        y_part = self.h_ds['y'].values[particle_idx]
        z_part = self.h_ds['z'].values[particle_idx]
        d_part = self.h_ds['d'].values[particle_idx]  # not used but here it is

        # create a 3D histogram 
        in_data = np.stack((x_part, y_part, z_part)).T
        h_part = np.histogramdd(in_data, bins=[self.tile_x_bins,self.tile_y_bins,self.z_bins])[0]
        z_part_bin_idx = np.digitize(z_part, self.z_bins)-1 # specify the z bin locations of the particles

        # smoothing kernel accounts for overlapping subimages when the 
        # subimage is larger than the stride
        if self.step_size < self.tile_size:
            overlap_kernel = np.ones((
                self.tile_size//self.step_size,self.tile_size//self.step_size
            ))
            for z_idx in range(h_part.shape[-1]):
                b = self.tile_size//self.step_size
                h_part[:,:,z_idx] = convolve2d(h_part[:,:,z_idx],overlap_kernel)[b-1:h_part.shape[0]+b-1,b-1:h_part.shape[1]+b-1]

        input_image = self.h_ds['image'].isel(hologram_number=h_idx).values

        z_counter = 0 # the number of planes reconstructed in this generator
        image_tnsr = torch.tensor(input_image, device=self.device).unsqueeze(0)
        for z_sub_set in z_planes_lst:
            yield self.get_sub_images_labeled(
                image_tnsr, 
                z_sub_set, 
                z_counter, 
                x_part, y_part, z_part, d_part, h_part, 
                z_part_bin_idx, 
                batch_size = batch_size,
                thresholds = thresholds, 
                obs_threshold = obs_threshold
            )
            z_counter+=z_sub_set.size
            
            
    def create_z_plane_lst(self, planes_per_call=1):
        """
        Create a list of z planes according to the requested
        batch size.  This generates the z_planes_lst argument
        needed for gen_next_z_plane()
        """
        z_lst = []
        for z_idx in np.arange(0, self.z_centers.size, planes_per_call):
            z_lst.append(self.z_centers[z_idx:(z_idx+planes_per_call)])
        return z_lst
    
    


if __name__ == '__main__':
    
    is_cuda = torch.cuda.is_available()
    device = torch.device(torch.cuda.current_device()) if is_cuda else torch.device("cpu")

    if is_cuda:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
            
    with open(sys.argv[1]) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)
        
    this_worker = int(sys.argv[2])
    gpu_id = int(sys.argv[3])
    device = f"cuda:{gpu_id}"
    
    workers = conf["inference"]["workers"]
    save_arrays = conf["inference"]["save_arrays"]
    plot = conf["inference"]["plot"]
        
    n_bins = conf["data"]["n_bins"]
    tile_size = conf["data"]["tile_size"]
    step_size = conf["data"]["step_size"]
    marker_size = conf["data"]["marker_size"]
    synthetic_path = conf["data"]["data_path"]

    epochs = conf["trainer"]["epochs"]
    train_batch_size = conf["trainer"]["train_batch_size"]
    valid_batch_size = conf["trainer"]["valid_batch_size"]
    interence_batch_size = conf["inference"]["batch_size"]
    batches_per_epoch = conf["trainer"]["batches_per_epoch"]
    stopping_patience = conf["trainer"]["stopping_patience"]
    model_loc = conf["trainer"]["output_path"]

    model_name = conf["resnet"]["name"]
    fcl_layers = conf["resnet"]["fcl_layers"]
    dropout = conf["resnet"]["dropout"]
    output_size = conf["resnet"]["output_size"]
    resnet_model = conf["resnet"]["resnet_model"]
    pretrained = conf["resnet"]["pretrained"]

    learning_rate = conf["optimizer"]["learning_rate"]
    weight_decay = conf["optimizer"]["weight_decay"]

    # roc threshold
    obs_threshold = 1.0
    thresholds = 1.0 - np.logspace(-5, 0, num=50, endpoint=True, base=10.0, dtype=None, axis=0)
    thresholds = thresholds[::-1]

    ##########################

    model = ResNet(fcl_layers, 
                   dr = dropout, 
                   output_size = output_size, 
                   resnet_model=resnet_model, 
                   pretrained = pretrained).to(device)


    checkpoint = torch.load(
        f"{model_loc}/best.pt",
        map_location=lambda storage, loc: storage
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"There are {total_params} total model parameters")

    prop = InferencePropagator(
        synthetic_path, 
        n_bins = n_bins,
        tile_size = tile_size,
        step_size = step_size,
        marker_size = marker_size,
        model = model,
        device = device
    )

    h_idx = 10
    test_image = prop.h_ds['image'].isel(hologram_number=h_idx)

    # ### Create a list of z-values to propagate to
    z_list = prop.create_z_plane_lst(planes_per_call=1)
    z_list = np.array_split(z_list, workers)[this_worker]
    
    ### Loop over h_idx?

    # ### Main loop to call the generator, predict with the model, and aggregate and save the results

    model.eval()
    with torch.no_grad():

        inference_generator = prop.get_next_z_planes_labeled(
            h_idx, 
            z_list, 
            batch_size = interence_batch_size, 
            thresholds = thresholds,
            obs_threshold = obs_threshold)

        jiter = tqdm.tqdm(
            enumerate(inference_generator), 
            total = len(z_list), 
            leave=True
        )

        total_accuracy = []
        roc = DistributedROC(thresholds=thresholds, obs_threshold=obs_threshold)

        for z_idx, results_dict in jiter:

            # Get stuff from the results dictionary
            pred_label = results_dict["pred_label"]
            pred_prob = results_dict["pred_proba"]
            true_label = results_dict["true_label"]
            acc = results_dict["acc"]
            z_plane = int(results_dict["z_plane"])

            if save_arrays:
                # Save the giant matrices as sparse arrays, as most elements are zero
                save_sparse_csr(f"{model_loc}/data/prob_{h_idx}_{z_plane}.npz", scipy.sparse.csr_matrix(pred_prob))

                pred_label = pred_label == 1.0
                save_sparse_csr(f"{model_loc}/data/pred_{h_idx}_{z_plane}.npz", scipy.sparse.csr_matrix(pred_label))

                true_label = true_label == 1.0
                save_sparse_csr(f"{model_loc}/data/true_{h_idx}_{z_plane}.npz", scipy.sparse.csr_matrix(true_label))

            # Update any metrics you want printed to screen
            total_accuracy.append(acc)
            roc.merge(results_dict["roc"])
            
            with open(f"{model_loc}/data/roc_{h_idx}_{z_plane}.pkl", "wb") as fid:
                joblib.dump(results_dict["roc"], fid)
            
            to_print = f"Plane: {z_idx} at z: {z_plane*1e-6} plane_acc: {acc:.6f} holo_acc: {np.mean(total_accuracy):.6f}"
            to_print += f" holo_csi: {roc.max_csi():.6f}"
            jiter.set_description(to_print)
            jiter.update()

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
    rocs = sorted(glob.glob(f"{model_loc}/data/roc_{h_idx}_*.pkl"), 
                  key = lambda x: int(x.strip(".pkl").split("_")[-1]))
    
    if len(rocs) == n_bins:
        for k, roc_fn in enumerate(rocs):
            with open(roc_fn, "rb") as fid:
                if k == 0:
                    roc = joblib.load(fid)
                else:
                    roc.merge(joblib.load(fid))
                
        with open(f"{model_loc}/roc_{h_idx}.pkl", "wb") as fid:
            joblib.dump(roc, fid)

        roc_curve([roc], [model_name], ["orange"], ["o"], f"{model_loc}/roc_comparison_{h_idx}.png")
        performance_diagram([roc], [model_name], ["orange"], ["o"], f"{model_loc}/performance_comparison_{h_idx}.png")