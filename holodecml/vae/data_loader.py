from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from torch.utils.data import Dataset
from typing import List, Dict
import xarray as xr
import numpy as np
import sys, os
import logging
import random


logger = logging.getLogger(__name__)


num_particles_dict = {
    1: '1particle',
    3: '3particle',
    'multi': 'multiparticle',
    '50-100': '50-100particle_monodisperse'
}


split_dict = {
    'train' : 'training',
    'test'   : 'test',
    'valid': 'validation'
}


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
        transform = None) -> None:
        
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
            self.scaler = {col: StandardScaler() for col in output_cols}
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
        hologram = self.hologram_numbers[idx] #random.choice(self.hologram_numbers)
        im = self.ds["image"][hologram].values
        im = {"image": self.reshape(im)} 
        
        #particles = np.where(self.ds["hid"] == hologram + 1)[0]
        #for l, p in enumerate(particles):
        #    for m, col in enumerate(self.output_cols):
        #        val = self.ds[col].values[p]
        #        y_out[k, l, m] = val
        
        ##########################################################
        #
        #
        # {"image": image, "outputs": outputs} -- need to transform the outputs if not invariant under transformation
        #
        #
        ##########################################################
        
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
        return X.reshape((1,x,y))
            
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
        path_data = os.path.join(path_data, self.dataset_name(num_particles, split))

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

        valid = [1,3,'multi','50-100']
        if num_particles not in valid:
            raise ValueError("results: num_particles must be one of %r." % valid)
        num_particles = num_particles_dict[num_particles]

        valid = ['train','test','valid']
        if split not in valid:
            raise ValueError("results: split must be one of %r." % valid)
        split = split_dict[split]
        ds_name = f'synthetic_holograms_{num_particles}_{split}.{file_extension}'

        return ds_name
    
    def closest(self, lst, K): 
        idx = (np.abs(lst - K)).argmin() 
        return idx 