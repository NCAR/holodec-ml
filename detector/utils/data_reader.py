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


logger = logging.getLogger(__name__)


class DetectionDatasetPadded(Dataset):

    def __init__(
            self,
            path_data: List[str],
            shuffle: bool = True,
            scaler: Dict[str, str] = True,
            transform = None,
            x_size: int = 600,
            y_size: int = 400,
            seed: int = 5000) -> None:
        
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
                "d": MinMaxScaler((1.0, 100)), # true box will never be smaller than 2
            }
            for col in ["x", "y", "z", "d"]:
                concat = np.hstack([arr[col].values for arr in self.ds.values()])
                self.scaler[col].fit(concat.reshape(concat.shape[-1], -1))
            
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
         # We assume binary: 1 for is_particle, 0 for everything else
        labels = np.ones(len(particles), dtype = np.int64)
        iscrowd = np.zeros(bboxes.shape[0], dtype = np.int64)
        
        sorted_idx = np.argsort(self.ds[name]["d"][particles].values)
        particles = particles[sorted_idx]
        particles = particles[::-1] # Order from largest to smallest by diameter
        
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
                
                if task == "y":
                    val =  self.scaler["y"].transform(val.reshape(-1, 1))[0][0]
                    bboxes[l][1] = val - square_shift
                    bboxes[l][3] = val + square_shift
                    
                if task == "z":
                    pass
                
                if task == "d":
                    ### Assumes we are predicting 2D area with r = d/2
                    #radius = (val / 2.0)
                    #square_shift = np.sqrt(np.pi) * radius / 2.0
                    
                    # Scaled d earlier to be within 1 and 100 as some boxes were too small. 
                    # Consider increasing the floor, possibly to 10
                    square_shift = self.scaler["d"].transform(val.reshape(-1, 1))[0][0]
                    
            # Compute the area initially, will rescale it later if need to pad the image
            area[l] = (bboxes[l][2] - bboxes[l][0]) * (bboxes[l][3] - bboxes[l][1])
                            
        # Rescale image so that the bounding boxes fit and all images are the same size (and divisible by 16)
        # This means we need to pad around the edges
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