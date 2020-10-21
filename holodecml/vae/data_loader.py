from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from torch.utils.data import Dataset
from typing import List, Dict
import xarray as xr
import numpy as np
import logging
import random
import sys
import os


logger = logging.getLogger(__name__)


num_particles_dict = {
    1: '1particle',
    3: '3particle',
    'multi': 'multiparticle',
    '50-100': '50-100particle_gamma'
}


split_dict = {
    'train': 'training',
    'test': 'test',
    'valid': 'validation'
}


def LoadReader(reader_type: str, split: str, transform: str, scaler: str, config: Dict[str, str]):
    logger.info(f"Loading reader-type {reader_type}")
    if reader_type in ["vae", "att-vae"]:
        return HologramDataset(
            split=split,
            transform=transform,
            scaler=scaler if scaler else False,
            **config
        )
    elif reader_type == "encoder-vae":
        return MultiTaskHologramDataset(
            split=split,
            transform=transform,
            scaler=scaler if scaler else False,
            **config
        )
    else:
        logger.info(
            f"Unsupported reader type {reader_type}. Choose from vae, att-vae, or encoder-vae. Exiting.")
        sys.exit(1)


class HologramDataset(Dataset):

    def __init__(
            self,
            path_data: str,
            num_particles: int,
            split: str,
            subset: bool,
            output_cols: List[str],
            shuffle: bool = True,
            maxnum_particles: int = False,
            scaler: Dict[str, str] = False,
            transform=None) -> None:
        'Initialization'
        self.ds = self.open_dataset(path_data, num_particles, split)
        self.output_cols = [x for x in output_cols if x != 'hid']
        self.subset = subset
        self.hologram_numbers = self.ds.hologram_number.values
        self.num_particles = num_particles
        self.xsize = len(self.ds.xsize.values)
        self.ysize = len(self.ds.ysize.values)
        self.shuffle = shuffle
        self.maxnum_particles = maxnum_particles
        self.transform = transform
        self.on_epoch_end()

        if not scaler:
            self.scaler = {col: MinMaxScaler()
                           for col in output_cols}  # StandardScaler()
            for col in output_cols:
                scale = self.ds[col].values
                self.scaler[col].fit(scale.reshape(scale.shape[-1], -1))
        else:
            self.scaler = scaler

        logger.info(
            f"Loaded {split} hologram data containing {len(self.hologram_numbers)} images"
        )

    def get_transform(self):
        return self.scaler

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.hologram_numbers)

    def __getitem__(self, idx):
        'Generate one data point'
        # random.choice(self.hologram_numbers)
        hologram = self.hologram_numbers[idx]
        im = self.ds["image"][hologram].values
        im = {"image": self.reshape(im)}

        if self.transform:
            im = self.transform(im)

        self.processed += 1
        if self.processed == self.__len__():
            self.on_epoch_end()

        return im["image"]

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.processed = 0
        if self.shuffle == True:
            random.shuffle(self.hologram_numbers)

    def reshape(self, X):
        x, y = X.shape
        return X.reshape((1, x, y))

    def open_dataset(self, path_data, num_particles, split):
        """
        Opens a HOLODEC file

        Args: 
            path_data: (str) Path to dataset directory
            num_particles: (int or str) Number of particles per hologram
            split: (str) Dataset split of either 'train', 'valid', or 'test'

        Returns:
            ds: (xarray Dataset) Opened dataset
        """
        path_data = os.path.join(
            path_data, self.dataset_name(num_particles, split))

        if not os.path.isfile(path_data):
            print(f"Data file does not exist at {path_data}. Exiting.")
            raise

        ds = xr.open_dataset(path_data)
        return ds

    def dataset_name(self, num_particles, split, file_extension='nc'):
        """
        Return the dataset filename given user inputs

        Args: 
            num_particles: (int or str) Number of particles per hologram
            split: (str) Dataset split of either 'train', 'valid', or 'test'
            file_extension: (str) Dataset file extension

        Returns:
            ds_name: (str) Dataset name
        """

        valid = [1, 3, 'multi', '50-100']
        if num_particles not in valid:
            raise ValueError(
                "results: num_particles must be one of %r." % valid)
        num_particles = num_particles_dict[num_particles]

        valid = ['train', 'test', 'valid']
        if split not in valid:
            raise ValueError("results: split must be one of %r." % valid)
        split = split_dict[split]
        ds_name = f'synthetic_holograms_{num_particles}_{split}.{file_extension}'

        return ds_name

    def closest(self, lst, K):
        idx = (np.abs(lst - K)).argmin()
        return idx


class MultiTaskHologramDataset(Dataset):

    def __init__(
            self,
            path_data: str,
            num_particles: int,
            split: str,
            subset: bool,
            output_cols: List[str],
            shuffle: bool = True,
            maxnum_particles: int = False,
            scaler: Dict[str, str] = False,
            transform=None,
            cache=False) -> None:
        
        'Initialization'
        self.ds = self.open_dataset(path_data, num_particles, split)
        self.output_cols = [x for x in output_cols if x != 'hid']
        self.subset = subset
        self.hologram_numbers = self.ds.hologram_number.values
        self.num_particles = num_particles
        self.xsize = len(self.ds.xsize.values)
        self.ysize = len(self.ds.ysize.values)
        self.shuffle = shuffle
        self.maxnum_particles = maxnum_particles
        self.transform = transform
        self.cache = {} if cache else False
        self.on_epoch_end()
        
        self.binary = "binary" in self.output_cols
        self.num_outs = len(self.output_cols)-1 if self.binary else len(self.output_cols)

        if not scaler:
            self.scaler = {col: MinMaxScaler() for col in output_cols}
            for col in output_cols:
                if col == "binary": continue
                scale = self.ds[col].values
                self.scaler[col].fit(scale.reshape(scale.shape[-1], -1))
        else:
            self.scaler = scaler

        logger.info(
            f"Loaded {split} hologram data containing {len(self.hologram_numbers)} images"
        )

    def get_transform(self):
        return self.scaler

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.hologram_numbers)

    def __getitem__(self, idx):
        'Generate one data point'
        # random.choice(self.hologram_numbers)
        hologram = self.hologram_numbers[idx]
        
        if isinstance(self.cache, dict):
            if hologram in cache:
                return self.cache[hologram]
        
        im = self.ds["image"][hologram].values
        im = {"image": np.expand_dims(im, 0)}  # reshape
        
        y_out = {}
        for task in self.output_cols:
            y_out[task] = np.zeros((
                self.maxnum_particles if self.maxnum_particles else self.num_particles
            ))
        w_out = np.zeros((
            self.maxnum_particles if self.maxnum_particles else self.num_particles
        ))
        particles = np.where(self.ds["hid"] == hologram + 1)[0]
        for l, p in enumerate(particles):
            for task in self.output_cols:
                if task == "binary":
                    y_out[task][l] = 1
                    continue
                val = self.ds[task].values[p]
                val = self.scaler[task].transform(val.reshape(-1, 1))[0][0]
                y_out[task][l] = val
        
        num_particles = (l + 1)
        w_out[:num_particles] = num_particles / self.maxnum_particles
        w_out[num_particles:] = (self.maxnum_particles - num_particles) / self.maxnum_particles

        if self.transform:
            im = self.transform(im)

        self.processed += 1
        if self.processed == self.__len__():
            self.on_epoch_end()
            
        if isinstance(self.cache, dict):
            if hologram not in self.cache:
                self.cache[hologram] = (im["image"], y_out, w_out)
        
        return im["image"], y_out, w_out

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.processed = 0
        if self.shuffle == True:
            random.shuffle(self.hologram_numbers)

    def open_dataset(self, path_data, num_particles, split):
        """
        Opens a HOLODEC file

        Args: 
            path_data: (str) Path to dataset directory
            num_particles: (int or str) Number of particles per hologram
            split: (str) Dataset split of either 'train', 'valid', or 'test'

        Returns:
            ds: (xarray Dataset) Opened dataset
        """
        path_data = os.path.join(
            path_data, self.dataset_name(num_particles, split))

        if not os.path.isfile(path_data):
            print(f"Data file does not exist at {path_data}. Exiting.")
            raise

        ds = xr.open_dataset(path_data)
        return ds

    def dataset_name(self, num_particles, split, file_extension='nc'):
        """
        Return the dataset filename given user inputs

        Args: 
            num_particles: (int or str) Number of particles per hologram
            split: (str) Dataset split of either 'train', 'valid', or 'test'
            file_extension: (str) Dataset file extension

        Returns:
            ds_name: (str) Dataset name
        """

        valid = [1, 3, 'multi', '50-100']
        if num_particles not in valid:
            raise ValueError(
                "results: num_particles must be one of %r." % valid)
        num_particles = num_particles_dict[num_particles]

        valid = ['train', 'test', 'valid']
        if split not in valid:
            raise ValueError("results: split must be one of %r." % valid)
        split = split_dict[split]
        ds_name = f'synthetic_holograms_{num_particles}_{split}.{file_extension}'

        return ds_name
