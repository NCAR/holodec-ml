import warnings
warnings.filterwarnings("ignore")

import os
import sys
import glob
import tqdm
import time
import h5py
import yaml
import pickle
import joblib
import random
import logging
import datetime

import torch
import torch.fft
import torch.nn.functional as F

import numpy as np
import xarray as xr
import pandas as pd

from torch import nn
from functools import partial
from scipy.signal import convolve2d
from collections import defaultdict
from typing import List, Dict, Callable, Union, Any, TypeVar, Tuple

from .metrics import DistributedROC


logger = logging.getLogger(__name__)


class WavePropagator:
    
    def __init__(self, 
                 data_path,
                 n_bins = 1000, 
                 tile_size = 512, 
                 step_size = 128, 
                 marker_size = 10,
                 device = "cpu"):
                
        self.h_ds = xr.open_dataset(data_path)
        
        if 'zMin' in self.h_ds.attrs:
            self.zMin = self.h_ds.attrs['zMin']  # minimum z in sample volume
            self.zMax = self.h_ds.attrs['zMax'] 
        else: # some of the raw data does not have this parameter
            # should warn the user here through the logger
            self.zMin = 0.014
            self.zMax = 0.158
        
        self.n_bins = n_bins
        self.z_bins = np.linspace(self.zMin, self.zMax, n_bins+1)*1e6  # histogram bin edges
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
        self.image_norm = 255.0

        self.x_arr = np.arange(-self.Nx//2, self.Nx//2)*self.dx
        self.y_arr = np.arange(-self.Ny//2, self.Ny//2)*self.dy
        
        self.tile_x_bins = np.arange(-self.Nx//2,self.Nx//2,self.step_size)*self.dx*1e6
        self.tile_y_bins = np.arange(-self.Ny//2,self.Ny//2,self.step_size)*self.dy*1e6
        
        self.fx = torch.fft.fftfreq(self.Nx, self.dx, device=self.device).unsqueeze(0).unsqueeze(2)
        self.fy = torch.fft.fftfreq(self.Ny, self.dy, device=self.device).unsqueeze(0).unsqueeze(1)      

     
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
        
        Etfft = torch.fft.fft2(Ein)
        Eofft = Etfft*torch.exp(1j*2*np.pi*z_tnsr/self.lam*torch.sqrt(1-self.lam**2*(self.fx**2+self.fy**2)))

        # It might be helpful if we could omit this step.  It would save an inverse fft.
        Eout = torch.fft.ifft2(Eofft)

        return Eout
    
    
    
class InferencePropagator(WavePropagator):
    
    def __init__(self, 
                 data_path,
                 n_bins = 1000, 
                 tile_size = 512, 
                 step_size = 128, 
                 marker_size = 10, 
                 device = "cuda", 
                 model = None, 
                 transforms = None,
                 mode = None):
        
        super(InferencePropagator, self).__init__(
            data_path,
            n_bins = n_bins, 
            tile_size = tile_size, 
            step_size = step_size, 
            marker_size = marker_size,
            device = device
        )
        
        self.model = model
        self.model.eval()
        self.transforms = transforms
        self.mode = mode
        self.create_mapping()
        
    
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
                    col_slice = slice(-self.tile_size, None)
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
            stacked_image = self.apply_transforms(stacked_image.squeeze(0)).unsqueeze(0)

            size = (E_out.shape[1], E_out.shape[2])
            true_output = torch.zeros(size).to(self.device)
            pred_output = torch.zeros(size).to(self.device)
            pred_proba = torch.zeros(size).to(self.device)
            counter = torch.zeros(size).to(self.device)
            
            chunked = np.array_split(
                list(self.idx2slice.items()),
                int(np.ceil(len(self.idx2slice) / batch_size))
            )
            
            if self.mode == "mask":
                for z_idx in range(E_out.shape[0]):
                
                    unet_mask = torch.zeros(E_out.shape[1:]).to(self.device)  # initialize the UNET mask
                    part_in_plane_idx = np.where(z_part_bin_idx==z_idx+z_counter)[0]  # locate all particles in this plane
                    
#                     if len(part_in_plane_idx) > 0:
#                         logger.info(f"{z_sub_set[z_idx]} {part_in_plane_idx}")

                    # build the UNET mask for this z plane
                    for part_idx in part_in_plane_idx:
                        unet_mask += torch.from_numpy(
                            (self.y_arr[None,:]*1e6-yp[part_idx])**2 + (self.x_arr[:,None]*1e6-xp[part_idx])**2  < (dp[part_idx]/2)**2
                        ).float().to(self.device)

                    worker = partial(
                        self.collate_masks, 
                        image = stacked_image[z_idx, :].float(), 
                        mask = unet_mask
                    )

                    for chunk in chunked:
                        slices, x, true_mask_tile = worker(chunk)
                        pred_proba_tile = self.model(x).squeeze(1)
                        pred_mask_tile = pred_proba_tile > 0.5

                        for k, ((row_idx, col_idx), (row_slice, col_slice)) in enumerate(slices):
                            counter[row_slice, col_slice] += 1
                            true_output[row_slice, col_slice] += true_mask_tile[k]
                            pred_output[row_slice, col_slice] += pred_mask_tile[k]
                            pred_proba[row_slice, col_slice] += pred_proba_tile[k]
                            
            elif self.mode == "label":
                for z_idx in range(E_out.shape[0]):
                    input_x = stacked_image[z_idx, :].float()
                    true_y = infocus_mask[:, :, z_idx+z_counter]

                    worker = partial(
                        self.collate_labels, 
                        image = input_x, 
                        label = true_y
                    )

                    for chunk in chunked: # Loop over all 512 x 512 tiles
                        slices, x, true_labels = worker(chunk)

                        true_labels = true_labels.to(self.device)
                        pred_logits = self.model(x)

                        pred_probs, pred_labels = torch.max(pred_logits, 1)
                        pred_probs = pred_probs.exp()
                        pred_labels = pred_labels.squeeze(-1)

                        pred_cond = pred_labels == 0
                        pred_probs = torch.where(pred_cond, 1.0 - pred_probs, pred_probs)

                        for k, ((row_idx, col_idx), (row_slice, col_slice)) in enumerate(slices):
                            counter[row_slice, col_slice] += 1
                            true_output[row_slice, col_slice] += true_labels[k]
                            pred_output[row_slice, col_slice] += pred_labels[k]
                            pred_proba[row_slice, col_slice] += pred_probs[k]
                        
            pred_output = pred_output / counter
            pred_proba  = pred_proba / counter
            true_output = true_output / counter    
            
            pred_output = pred_output == 1.0
            true_output = true_output == 1.0
            
            pred_output = pred_output.cpu().numpy()
            pred_proba = pred_proba.cpu().numpy()
            true_output = true_output.cpu().numpy()

            roc = DistributedROC(thresholds=thresholds, obs_threshold=obs_threshold)
            roc.update(pred_proba.ravel(), true_output.ravel())
            
            return_dict = {
                "pred_output": pred_output,
                "pred_proba": pred_proba,
                "true_output": true_output,
                "z_plane": int(z_sub_set[z_idx]),
                "roc": roc
            }
                       
        return return_dict
    
    def collate_labels(self, batch, image = None, label = None):
        x, y = zip(*[
            (image[:, row_slice, col_slice], 
            torch.LongTensor([int(label[row_idx, col_idx])]))
            for ((row_idx, col_idx), (row_slice, col_slice)) in batch
        ])
        return batch, torch.stack(x), torch.stack(y) #  / self.image_norm
    
    
    def collate_masks(self, batch, image = None, mask = None):
        x, y = zip(*[
            (image[:, row_slice, col_slice], mask[row_slice, col_slice])
            for ((row_idx, col_idx), (row_slice, col_slice)) in batch
        ])
        return batch, torch.stack(x), torch.stack(y) # / self.image_norm
    
    
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
                                  batch_size = 32,
                                  thresholds = np.arange(0.0, 1.1, 0.1),
                                  obs_threshold = 1.0, 
                                  start_z_counter = 0):
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

        z_counter = start_z_counter # the number of planes reconstructed in this generator
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
            
            # clear the cached memory from the gpu
            torch.cuda.empty_cache()
            
            
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
    
    
