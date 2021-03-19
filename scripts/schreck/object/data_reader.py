from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from torch.utils.data import DataLoader, Dataset
from collections import Counter, defaultdict
from typing import List, Dict
import xarray as xr
import itertools
import logging
import joblib
import random
import yaml
import sys
import cv2
import os

import pandas as pd
import numpy as np

import logging 

import torchvision, torch
import torchvision.transforms.functional as tvf

from scipy.ndimage.filters import gaussian_filter


logger = logging.getLogger(__name__)


class DetectionDataset(Dataset):

    def __init__(
            self,
            path_data: List[str],
            shuffle: bool = True,
            scaler: Dict[str, str] = True,
            transform = None,
            x_size: int = 600,
            y_size: int = 400,
            seed: int = 5000,
            particle_order: bool = True) -> None:
        
        'Initialization'
        self.ds = {name: xr.open_dataset(name) for name in path_data}
        
        self.hologram_numbers = []
        for name, _ds in sorted(self.ds.items()):
            holo_numbers = list(_ds["hologram_number"].values)
            if isinstance(path_data, dict):
                max_n = path_data[name]
                holo_numbers = random.sample(holo_numbers, max_n)
            for hologram_number in holo_numbers:
                self.hologram_numbers.append([name, hologram_number])
                     
        self.shuffle = shuffle
        self.transform = transform

        self.x_size = x_size
        self.y_size = y_size
        self.particle_order = particle_order
        
        self.z_range = [14000, 158000]
        self.z_dist = np.linspace(17000, 155000, 98, endpoint=False)
        
        self.on_epoch_end()
        self.set_scaler(scaler)
    
        logger.info(
            f"Loaded {path_data} hologram data containing {len(self.hologram_numbers)} images"
        )
        
        random.seed(seed)
        
    def set_scaler(self, scaler = True):
        # z_range = [14000, 158000]
        # diameter_range = [0.00161, 227]
        if scaler == True:
            #logger.info(f"Rescaling the data by subtracting the mean and dividing by sigma")    
            self.scaler = {
                "x": MinMaxScaler((0, self.x_size)),
                "y": MinMaxScaler((0, self.y_size)),
                "z": MinMaxScaler((0, self.y_size)), # Keypoints should be within the range of the image
                "d": MinMaxScaler((0.5, 100)), # true box will never be smaller than a unit area
                #"x_s": StandardScaler(),
                #"y_s": StandardScaler(),
                #"z_s": StandardScaler(),
                #"d_s": StandardScaler()
            }
            for col in ["x", "y", "z", "d"]:
                concat = np.hstack([arr[col].values for arr in self.ds.values()])
                self.scaler[col].fit(concat.reshape(concat.shape[-1], -1))
            #for col in ["x", "y", "z", "d"]:
            #    concat = np.hstack([arr[col].values for arr in self.ds.values()])
            #    self.scaler[f"{col}_s"].fit(concat.reshape(concat.shape[-1], -1))
            
        elif scaler == False:
            self.scaler = False
        else:
            self.scaler = scaler
        logger.info(f"Loaded data scaler transformation {self.scaler}")

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.hologram_numbers)

    def __getitem__(self, idx):
        
        'Generate one data point'
        name, hologram = self.hologram_numbers[idx]
        
        self.processed += 1
        if self.processed == self.__len__():
            self.on_epoch_end()
        
        im = self.ds[name]["image"][hologram].values
        im = {
            "image": np.expand_dims(im, 0), 
            "horizontal_flip": False, 
            "vertical_flip": False
        }
        if self.transform:
            im = self.transform(im)
               
        particles = np.where(self.ds[name]["hid"] == hologram + 1)[0]
        
        bboxes = np.zeros((len(particles), 4), dtype = np.float32)
        area   = np.zeros(len(particles), dtype = np.float32)
        keypoints = np.ones((len(particles), 1, 3), dtype = np.float32)
         # We assume binary: 1 for is_particle, 0 for everything else
        #labels = np.zeros(len(particles), dtype = np.int64)
        labels = np.ones(len(particles), dtype = np.int64)
        iscrowd = np.zeros(bboxes.shape[0], dtype = np.int64)
        
        sorted_idx = np.argsort(self.ds[name]["d"][particles].values)
        particles = particles[sorted_idx]
        particles = particles[::-1] if self.particle_order else particles
        
        rescale = False
        low_high = {"x_low": 0, "x_high": self.x_size, "y_low": 0, "y_high": self.y_size}
        for l, p in enumerate(particles):
            for r, task in enumerate(["d", "z", "x", "y"]):

                val = self.ds[name][task].values[p]
                
                if im["horizontal_flip"] and task == "x":
                    val *= -1
                if im["vertical_flip"] and task == "y":
                    val *= -1
                    
                if task == "x":
                    val =  self.scaler["x"].transform(val.reshape(-1, 1))[0][0]
                    bboxes[l][0] = val - square_shift
                    bboxes[l][2] = val + square_shift
                    keypoints[l][0][0] = val
                    #keypoints[l][1][0] = val
                
                if task == "y":
                    val =  self.scaler["y"].transform(val.reshape(-1, 1))[0][0]
                    bboxes[l][1] = val - square_shift
                    bboxes[l][3] = val + square_shift
                    keypoints[l][0][1] = val
                    
                if task == "z":
                    pass
                    #keypoints[l][1][1] = self.scaler["z"].transform(val.reshape(-1, 1))[0][0]
                    #label = np.digitize(val, self.z_dist, right=True) + 1
                
                if task == "d":
                    ### Assumes we are predicting 2D area with r = d/2
                    #radius = (val / 2.0)
                    #square_shift = np.sqrt(np.pi) * radius / 2.0
                    square_shift = self.scaler["d"].transform(val.reshape(-1, 1))[0][0]
                    
            # Update the label
            #labels[l] = label
                    
            # Compute the area initially, will rescale it later if need to pad the image
            area[l] = (bboxes[l][2] - bboxes[l][0]) * (bboxes[l][3] - bboxes[l][1])
            
            # Adjust bounding boxes that go negative runs off any edge    
            if (bboxes[l][0] < 0.0):
                x_off_low = bboxes[l][0].item()
                low_high["x_low"] = min(low_high["x_low"], x_off_low)
                rescale = True
                
            x_high = 0 
            if (bboxes[l][2] > self.x_size):
                x_off_high = bboxes[l][2].item()
                low_high["x_high"] = max(low_high["x_high"], x_off_high)
                rescale = True
                
            y_low = 0
            if (bboxes[l][1] < 0.0):
                y_off_low = bboxes[l][1].item()
                low_high["y_low"] = min(low_high["y_low"], y_off_low)
                rescale = True
                
            y_high = 0
            if (bboxes[l][3] > self.y_size):
                y_off_high = bboxes[l][3].item()
                low_high["y_high"] = max(low_high["y_high"], y_off_high)
                rescale = True
                    
        # Rescale all images as the largest radius = 50. This will force images to be the same size
        if rescale: 
            x_low = abs(low_high["x_low"]) if low_high["x_low"] < 0 else 0 
            x_high = low_high["x_high"]-self.x_size if low_high["x_high"] > self.x_size else 0 
            y_low = abs(low_high["y_low"]) if low_high["y_low"] < 0 else 0 
            y_high = low_high["y_high"]-self.y_size if low_high["y_high"] > self.y_size else 0 

            padding = [
                (int(np.floor(x_low)), int(np.floor(x_high))),
                (int(np.floor(y_low)), int(np.floor(y_high)))
            ]
 
            # Rescale image so that the bounding boxes fit.       
            original_shape = im["image"].transpose(0, 2, 1).shape[1:]
            im["image"] = np.pad(
                im["image"].squeeze(), 
                padding,
                mode='constant'
            )
            transformed_shape = im["image"].transpose(1,0).shape
            im["image"] = np.expand_dims(im["image"], axis = 0)            
            scale = np.flipud(np.divide(transformed_shape, original_shape))

            for l in range(len(bboxes)):
                top_left_corner = [bboxes[l][0], bboxes[l][3]]
                bottom_right_corner = [bboxes[l][2], bboxes[l][1]]
                new_top_left_corner = np.multiply(top_left_corner, scale)
                new_bottom_right_corner = np.multiply(bottom_right_corner, scale)
                
                new_area = (new_bottom_right_corner[0] - new_top_left_corner[0]) * (new_top_left_corner[1] - new_bottom_right_corner[1])
                preserve_area = np.sqrt(area[l] / new_area)
                
                bboxes[l][0] = new_top_left_corner[0] * preserve_area
                bboxes[l][3] = new_top_left_corner[1] * preserve_area
                bboxes[l][2] = new_bottom_right_corner[0] * preserve_area
                bboxes[l][1] = new_bottom_right_corner[1] * preserve_area

                area[l] = (bboxes[l][2] - bboxes[l][0]) * (bboxes[l][3] - bboxes[l][1])
                
                keypoints[l][0][0] = (bboxes[l][2] - bboxes[l][0]) / 2.0
                keypoints[l][0][1] = (bboxes[l][3] - bboxes[l][1]) / 2.0
                #keypoints[l][1][0] = (bboxes[l][2] - bboxes[l][0]) / 2.0

        # Prepare return dictionary
        target = {}
        target["boxes"] = torch.from_numpy(bboxes)
        target["area"] = torch.from_numpy(area)
        target["keypoints"] = torch.from_numpy(keypoints)
        
        target["labels"] = torch.from_numpy(labels)
        target["image_id"] = torch.tensor([idx])
        target["iscrowd"] = torch.from_numpy(iscrowd)
                     
        ## Convert to torch tensor
        image = torchvision.transforms.ToTensor()(im["image"])        
        image = image.permute(1, 0, 2)
        
        return image, target

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.processed = 0
        if self.shuffle == True:
            random.shuffle(self.hologram_numbers)
            
            

class DetectionDatasetPadded(Dataset):

    def __init__(
            self,
            path_data: List[str],
            shuffle: bool = True,
            scaler: Dict[str, str] = True,
            transform = None,
            x_size: int = 600,
            y_size: int = 400,
            seed: int = 5000,
            particle_order: bool = True) -> None:
        
        'Initialization'
        self.ds = {name: xr.open_dataset(name) for name in path_data}
        
        self.hologram_numbers = []
        for name, _ds in sorted(self.ds.items()):
            holo_numbers = list(_ds["hologram_number"].values)
            if isinstance(path_data, dict):
                max_n = path_data[name]
                holo_numbers = random.sample(holo_numbers, max_n)
            for hologram_number in holo_numbers:
                self.hologram_numbers.append([name, hologram_number])
                     
        self.shuffle = shuffle
        self.transform = transform

        self.x_size = x_size
        self.y_size = y_size
        self.particle_order = particle_order
        
        self.z_range = [14000, 158000]
        self.z_dist = np.linspace(17000, 155000, 98, endpoint=False)
        
        self.on_epoch_end()
        self.set_scaler(scaler)
    
        logger.info(
            f"Loaded {path_data} hologram data containing {len(self.hologram_numbers)} images"
        )
        
        random.seed(seed)
        
    def set_scaler(self, scaler = True):
        # z_range = [14000, 158000]
        # diameter_range = [0.00161, 227]
        if scaler == True:
            #logger.info(f"Rescaling the data by subtracting the mean and dividing by sigma")    
            self.scaler = {
                "x": MinMaxScaler((52, self.x_size + 52)), # Rescale to account for padding / make image divisible by 16
                "y": MinMaxScaler((56, self.y_size + 56)), # Rescale to account for padding / make image divisible by 16
                "z": MinMaxScaler((56, self.y_size + 56)), 
                "d": MinMaxScaler((1.0, 100)), # true box will never be smaller than 2 * unit area
            }
            for col in ["x", "y", "z", "d"]:
                concat = np.hstack([arr[col].values for arr in self.ds.values()])
                self.scaler[col].fit(concat.reshape(concat.shape[-1], -1))
            #for col in ["x", "y", "z", "d"]:
            #    concat = np.hstack([arr[col].values for arr in self.ds.values()])
            #    self.scaler[f"{col}_s"].fit(concat.reshape(concat.shape[-1], -1))
            
        elif scaler == False:
            self.scaler = False
        else:
            self.scaler = scaler
        logger.info(f"Loaded data scaler transformation {self.scaler}")

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.hologram_numbers)

    def __getitem__(self, idx):
        
        'Generate one data point'
        name, hologram = self.hologram_numbers[idx]
        
        self.processed += 1
        if self.processed == self.__len__():
            self.on_epoch_end()
        
        im = self.ds[name]["image"][hologram].values
        im = {
            "image": np.expand_dims(im, 0), 
            "horizontal_flip": False, 
            "vertical_flip": False
        }
        if self.transform:
            im = self.transform(im)
               
        particles = np.where(self.ds[name]["hid"] == hologram + 1)[0]
        
        bboxes = np.zeros((len(particles), 4), dtype = np.float32)
        area   = np.zeros(len(particles), dtype = np.float32)
        keypoints = np.ones((len(particles), 1, 3), dtype = np.float32)
         # We assume binary: 1 for is_particle, 0 for everything else
        #labels = np.zeros(len(particles), dtype = np.int64)
        labels = np.ones(len(particles), dtype = np.int64)
        iscrowd = np.zeros(bboxes.shape[0], dtype = np.int64)
        
        sorted_idx = np.argsort(self.ds[name]["d"][particles].values)
        particles = particles[sorted_idx]
        particles = particles[::-1] if self.particle_order else particles
        
        rescale = False
        low_high = {"x_low": 0, "x_high": self.x_size, "y_low": 0, "y_high": self.y_size}
        for l, p in enumerate(particles):
            for r, task in enumerate(["d", "z", "x", "y"]):

                val = self.ds[name][task].values[p]
                
                if im["horizontal_flip"] and task == "x":
                    val *= -1
                if im["vertical_flip"] and task == "y":
                    val *= -1
                    
                if task == "x":
                    val =  self.scaler["x"].transform(val.reshape(-1, 1))[0][0]
                    bboxes[l][0] = val - square_shift
                    bboxes[l][2] = val + square_shift
                    keypoints[l][0][0] = val
                    #keypoints[l][1][0] = val
                
                if task == "y":
                    val =  self.scaler["y"].transform(val.reshape(-1, 1))[0][0]
                    bboxes[l][1] = val - square_shift
                    bboxes[l][3] = val + square_shift
                    keypoints[l][0][1] = val
                    
                if task == "z":
                    pass
                    #keypoints[l][1][1] = self.scaler["z"].transform(val.reshape(-1, 1))[0][0]
                    #label = np.digitize(val, self.z_dist, right=True) + 1
                
                if task == "d":
                    ### Assumes we are predicting 2D area with r = d/2
                    #radius = (val / 2.0)
                    #square_shift = np.sqrt(np.pi) * radius / 2.0
                    square_shift = self.scaler["d"].transform(val.reshape(-1, 1))[0][0]
                    
            # Compute the area initially, will rescale it later if need to pad the image
            area[l] = (bboxes[l][2] - bboxes[l][0]) * (bboxes[l][3] - bboxes[l][1])
                            
         # Rescale image so that the bounding boxes fit. 
        padding = [
            (52, 52),
            (56, 56)
        ]
        original_shape = im["image"].transpose(0, 2, 1).shape[1:]
        im["image"] = np.pad(
            im["image"].squeeze(), 
            padding,
            mode='constant'
        )
        transformed_shape = im["image"].transpose(1,0).shape
        im["image"] = np.expand_dims(im["image"], axis = 0) 
                    
        # Prepare return dictionary
        target = {}
        target["boxes"] = torch.from_numpy(bboxes)
        target["area"] = torch.from_numpy(area)
        target["keypoints"] = torch.from_numpy(keypoints)
        
        target["labels"] = torch.from_numpy(labels)
        target["image_id"] = torch.tensor([idx])
        target["iscrowd"] = torch.from_numpy(iscrowd)
                     
        ## Convert to torch tensor
        image = torchvision.transforms.ToTensor()(im["image"])        
        image = image.permute(1, 0, 2)
        
        return image, target

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.processed = 0
        if self.shuffle == True:
            random.shuffle(self.hologram_numbers)