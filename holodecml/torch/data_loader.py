from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from torch.utils.data import Dataset
from collections import Counter, defaultdict
from typing import List, Dict
import xarray as xr
import numpy as np
import itertools
import logging
import random
import sys
import os

from holodecml.torch.fourier import *


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


def LoadReader(transform: str, scaler: str, config: Dict[str, str], split: str = None):
    
    if "type" not in config:
        logger.warning("In order to load a model you must supply the type field.")
        raise OSError("Failed to load a data reader. Exiting")
        
    reader_type = config.pop("type")
    logger.info(f"Loading reader-type {reader_type}")

    if reader_type in ["vae", "att-vae"]:
        return HologramDataset(
            split=split,
            transform=transform,
            scaler=scaler if scaler else False,
            **config
        ) 
    if reader_type in ["nshot_kways"]:
        return HologramNShotKWays(
            **config,
            transform=transform,
            scaler=scaler if scaler else False,
        )
    elif reader_type == "encoder-vae":
        return MultiTaskHologramDataset(
            transform=transform,
            scaler=scaler if scaler else False,
            **config
        )
    elif reader_type == "multi":
        return MultiHologramDataset(
            **config,
            transform=transform,
            scaler=scaler if scaler else False,
        )
    else:
        logger.info(
            f"Unsupported reader type {reader_type}. Choose from vae, att-vae, or encoder-vae. Exiting.")
        sys.exit(1)
        
        
        
class MultiHologramDataset(Dataset):

    def __init__(
            self,
            path_data: List[str],
            output_cols: List[str],
            shuffle: bool = True,
            maxnum_particles: int = 100,
            maxnum_holograms: int = 1e10,
            scaler: Dict[str, str] = True,
            transform = None,
            labels: bool = False, 
            cache: Dict[str,tuple] = None,
            max_cached: int = 100000,
            bins: List[int] = [10, 10],
            binned_cols: List[str] = ["x", "y"], 
            fourier: bool = False, 
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
                
        self.output_cols = [x for x in output_cols if x != 'hid']        
        self.shuffle = shuffle
        self.maxnum_particles = maxnum_particles + 3
        self.maxnum_holograms = maxnum_holograms
        self.transform = transform
        self.labels = labels
        
        self.bins = bins
        self.binned_cols = binned_cols
        
        self.on_epoch_end()
        
        self.scaler = None
        if self.labels:
            self.set_scaler(scaler)
        self.set_tokens()
            
        #self.label_weights()
            
        self.cache = {} if cache else False
        self.max_cached = max_cached

        logger.info(
            f"Loaded {path_data} hologram data containing {len(self.hologram_numbers)} images"
        )
        
        self.fourier = RAFT(600, 400) if fourier else False
        self.particle_order = particle_order
        
    def set_scaler(self, scaler = True):
        if scaler == True:
            logger.info(f"Rescaling the data by subtracting the mean and dividing by sigma")
            self.scaler = {col: StandardScaler() for col in self.output_cols}  # StandardScaler() MinMaxScaler()
            for col in self.output_cols:
                concat = []
                for arr in self.ds.values():
                    scale_factor = float(arr.Nx / 600)
                    if col in ["x", "y"]:
                        concat.append(arr[col].values / scale_factor)
                    else:
                        concat.append(arr[col].values)
                scale = np.hstack(concat)
                self.scaler[col].fit(scale.reshape(scale.shape[-1], -1))
                #scale = self.scaler[col].fit_transform(scale.reshape(scale.shape[-1], -1))
                #print(col, np.amax(scale), np.amin(scale))
            #raise
                
        elif scaler == False:
            self.scaler = False
        else:
            self.scaler = scaler
        logger.info(f"Loaded data scaler transformation {self.scaler}")
        print(f"Loaded data scaler transformation {self.scaler}")
        
#     def set_tokens(self):
#         limits = {
#             "x": [-888, 888],
#             "y": [-592, 592],
#             "z": [14000, 158000],
#             "d": [0.00161, 227]
#         }
#         if isinstance(self.bins, int) or isinstance(self.bins, float):
#             self.bins = [self.bins for task in self.binned_cols]
#         bins = {task: np.linspace(limits[task][0], limits[task][1], bin_no, endpoint=False) for task, bin_no in zip(self.binned_cols, self.bins)}
#         self.bins = bins
#         self.token_lookup = {}
#         for k, pair in enumerate(itertools.product(*[range(len(v)+1) for v in self.bins.values()])):
#             self.token_lookup[tuple(pair)] = k + 3
        
    def set_tokens(self):
        if isinstance(self.bins, int) or isinstance(self.bins, float):
            self.bins = [self.bins for task in self.binned_cols]
        bins = {task: np.linspace(-1.7, 1.7, bin_no, endpoint=False) for task, bin_no in zip(self.binned_cols, self.bins)}
        self.bins = bins
        self.token_lookup = {}
        for k, pair in enumerate(itertools.product(*[range(len(v)+1) for v in self.bins.values()])):
            self.token_lookup[tuple(pair)] = k + 3
            
    def label_weights(self):
        logger.info("Creating weights dictionary based on hologram particle number counts")
        weights = Counter([
            (arr.hid.values == a+1).sum() 
            for arr in self.ds.values()
            for a in arr.hologram_number.values 
        ])
        largest = max(list(weights.values()))
        weights = {
            key: largest / value for key, value in weights.items()
        }
        largest = max(list(weights.values()))
        self.weights = {
            key: value / largest for key, value in weights.items()
        }
        
    def get_transform(self):
        return self.scaler

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
        
        raft = None
        if self.fourier is not False:
            raft = self.fourier.radially_averaged_ft(im)
        
        scale_factor = float(im.shape[0] / 600)  
        im = {
            "image": np.expand_dims(im, 0), 
            "horizontal_flip": False, 
            "vertical_flip": False
        }
        
        if self.transform:
            im = self.transform(im)
        
        if not self.labels:
            return im["image"]
        
        output_cols = self.output_cols
        
        y_out = {}
        for task in output_cols:
            y_out[task] = np.zeros((self.maxnum_particles))
        w_out = np.zeros((self.maxnum_particles))
        particles = np.where(self.ds[name]["hid"] == hologram + 1)[0]
        #random.shuffle(particles)
        sorted_idx = np.argsort(self.ds[name]["d"][particles].values)
        particles = particles[sorted_idx]
        
        if self.particle_order:
            # largest-to-smallest
            particles = particles[::-1]  
            
        for l, p in enumerate(particles):
            indices = []
            for task in output_cols:
                if task == "binary":
                    y_out[task][l] = 1
                    continue
                val = self.ds[name][task].values[p]
                
                if im["horizontal_flip"] and task == "x":
                    val *= -1
                if im["vertical_flip"] and task == "y":
                    val *= -1
                    
                if task in ["x", "y"]:
                    val /= scale_factor
                    
                if isinstance(self.scaler, dict):
                    val = self.scaler[task].transform(val.reshape(-1, 1))[0][0]
                    
                if task in self.binned_cols:
                    indices.append(np.digitize(val, self.bins[task], right=True))
                    
                y_out[task][l] = val
            w_out[l] = self.token_lookup[tuple(indices)] #(l+1) # particle token
            
        # Sort by size of particle -- largest first, put fake particles at the end
#         sort_y_out = np.where(y_out["d"] == 0.0, -1e10, y_out["d"])
#         sorted_idx = np.argsort(sort_y_out)[::-1]
#         for task in output_cols:
#             y_out[task] = y_out[task][sorted_idx]
#         w_out = w_out[sorted_idx]
        
        w_out[l+1] = 2 # EOS token
        num_particles = (l + 2)
        w_out[num_particles:] = 0 # PAD token        
        
        if self.fourier:
            return im["image"], y_out, w_out.astype(int), raft
        else:
            return im["image"], y_out, w_out.astype(int)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.processed = 0
        if self.shuffle == True:
            random.shuffle(self.hologram_numbers)


class HologramNShotKWays(Dataset):

    def __init__(
            self,
            path_data: List[str],
            output_cols: List[str],
            shuffle: bool = True,
            maxnum_particles: int = 100,
            maxnum_holograms: int = 1e10,
            scaler: Dict[str, str] = True,
            transform = None,
            labels: bool = False, 
            cache: Dict[str,tuple] = None,
            max_cached: int = 100000,
            bins: List[int] = [10, 10],
            binned_cols: List[str] = ["x", "y"], 
            n_shot: int = 10, 
            k_ways: int = 3, 
            seed: bool = True, 
            particle_order: bool = True) -> None:
        
        'Initialization'
        self.ds = {name: xr.open_dataset(name) for name in path_data}
        self.seed = seed
        
        if isinstance(self.seed, int) or (self.seed == True):
            if self.seed == True:
                seed = random.randint(1, 1000000)
            random.seed(seed)
        
        self.hologram_sets = defaultdict(list)
        self.hologram_numbers = []
        for name, _ds in sorted(self.ds.items()):
            holo_numbers = list(_ds["hologram_number"].values)
            if isinstance(path_data, dict):
                max_n = path_data[name]
                holo_numbers = random.sample(holo_numbers, max_n)
            for hologram_number in holo_numbers:
                self.hologram_sets[name].append(hologram_number)
                self.hologram_numbers.append([name, hologram_number])
        
        # For n-shots, k-ways, need pairs of hologram names
        self.n_shot = n_shot
        self.k_ways = k_ways 
        self.batch_size = n_shot * k_ways
        
        self.hologram_combos = []
        for shot in list(itertools.product(self.ds.keys(), repeat = self.n_shot)):
            if len(set(shot)) != len(shot):
                continue
            self.hologram_combos.append(list(shot))
            
#         [
#             [x, y] for x in self.ds.keys() for y in self.ds.keys() if x != y
#         ]
        
        self.output_cols = [x for x in output_cols if x != 'hid']        
        self.shuffle = shuffle
        self.maxnum_particles = maxnum_particles + 3
        self.maxnum_holograms = maxnum_holograms
        self.transform = transform
        self.labels = labels
        
        self.bins = bins
        self.binned_cols = binned_cols
        
        self.on_epoch_end()
        
        self.scaler = None
        if self.labels:
            self.set_scaler(scaler)
        self.set_tokens()
            
        #self.label_weights()
            
        self.cache = {} if cache else False
        self.max_cached = max_cached

        logger.info(
            f"Loaded {path_data} hologram data containing {len(self.hologram_numbers)} images"
        )
        
        self.particle_order = particle_order
        
    def set_scaler(self, scaler = True):
        if scaler == True:
            logger.info(f"Rescaling the data by subtracting the mean and dividing by sigma")
            self.scaler = {col: StandardScaler() for col in self.output_cols}  # StandardScaler() MinMaxScaler()
            for col in self.output_cols:
                concat = []
                for arr in self.ds.values():
                    scale_factor = float(arr.Nx / 600)
                    if col in ["x", "y"]:
                        concat.append(arr[col].values / scale_factor)
                    else:
                        concat.append(arr[col].values)
                scale = np.hstack(concat)
                self.scaler[col].fit(scale.reshape(scale.shape[-1], -1))
        elif scaler == False:
            self.scaler = False
        else:
            self.scaler = scaler
        logger.info(f"Loaded data scaler transformation {self.scaler}")
        print(f"Loaded data scaler transformation {self.scaler}")
        
    def set_tokens(self):
        if isinstance(self.bins, int) or isinstance(self.bins, float):
            self.bins = [self.bins for task in self.binned_cols]
        bins = {task: np.linspace(-1.7, 1.7, bin_no, endpoint=False) for task, bin_no in zip(self.binned_cols, self.bins)}
        self.bins = bins
        self.token_lookup = {}
        for k, pair in enumerate(itertools.product(*[range(len(v)+1) for v in self.bins.values()])):
            self.token_lookup[tuple(pair)] = k + 3
        
#     def set_tokens(self):
#         limits = {
#             "x": [-888, 888],
#             "y": [-592, 592],
#             "z": [14000, 158000],
#             "d": [0.00161, 227]
#         }
#         if isinstance(self.bins, int) or isinstance(self.bins, float):
#             self.bins = [self.bins for task in self.binned_cols]
#         bins = {task: np.linspace(limits[task][0], limits[task][1], bin_no, endpoint=False) for task, bin_no in zip(self.binned_cols, self.bins)}
#         self.bins = bins
#         self.token_lookup = {}
#         for k, pair in enumerate(itertools.product(*[range(len(v)+1) for v in self.bins.values()])):
#             self.token_lookup[tuple(pair)] = k + 3
            
    def label_weights(self):
        logger.info("Creating weights dictionary based on hologram particle number counts")
        weights = Counter([
            (arr.hid.values == a+1).sum() 
            for arr in self.ds.values()
            for a in arr.hologram_number.values 
        ])
        largest = max(list(weights.values()))
        weights = {
            key: largest / value for key, value in weights.items()
        }
        largest = max(list(weights.values()))
        self.weights = {
            key: value / largest for key, value in weights.items()
        }
        
    def get_transform(self):
        return self.scaler

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(len(self.hologram_numbers) / self.batch_size)

    def __getitem__(self, idx):
        'Generate one data point'
        
        # Needs to be generalized N-shot = number of unique labels in a batch
        # K-ways should be the number of samples each
        
        idx = idx % len(self.hologram_combos)
        #holo1, holo2 = self.hologram_combos[idx]
        #n_shot_selection = random.sample(self.hologram_sets[holo1], self.k_ways)
        #k_ways_selection = random.sample(self.hologram_sets[holo2], self.k_ways)
        
        hologram_names = self.hologram_combos[idx]
        ways = {name: random.sample(self.hologram_sets[name], self.k_ways) for name in hologram_names}
        
        self.processed += 1
        if self.processed == self.__len__():
            self.on_epoch_end()
            
        X = []
        Y = defaultdict(list)
        T = []
        
        for name, hologram_numbers in ways.items():
            for hologram in hologram_numbers:
                x, y, t = self.select(name, hologram)
                X.append(x)
                for task, vals in y.items():
                    Y[task].append(vals)
                T.append(t)
                
        X = np.vstack(X)
        Y = {key: np.vstack(y) for key, y in Y.items()}
        T = np.vstack(T)
        return X, Y, T
        
    def select(self, name, hologram):
        im = self.ds[name]["image"][hologram].values
        
        scale_factor = float(im.shape[0] / 600)  
        im = {
            "image": np.expand_dims(im, 0), 
            "horizontal_flip": False, 
            "vertical_flip": False
        }
        
        if self.transform:
            im = self.transform(im)
        
        output_cols = self.output_cols
        
        y_out = {}
        for task in output_cols:
            y_out[task] = np.zeros((self.maxnum_particles))
        w_out = np.zeros((self.maxnum_particles))
        particles = np.where(self.ds[name]["hid"] == hologram + 1)[0]
        #random.shuffle(particles)
        
        sorted_idx = np.argsort(self.ds[name]["d"][particles].values)
        particles = particles[sorted_idx]
        if self.particle_order:
            # largest-to-smallest
            particles = particles[::-1]        
        
        for l, p in enumerate(particles):
            indices = []
            for task in output_cols:
                if task == "binary":
                    y_out[task][l] = 1
                    continue
                val = self.ds[name][task].values[p]
                
                if im["horizontal_flip"] and task == "x":
                    val *= -1
                if im["vertical_flip"] and task == "y":
                    val *= -1
                    
                if task in ["x", "y"]:
                    val /= scale_factor
                    
                if isinstance(self.scaler, dict):
                    val = self.scaler[task].transform(val.reshape(-1, 1))[0][0]
                    
                if task in self.binned_cols:
                    indices.append(np.digitize(val, self.bins[task], right=True))
                
                y_out[task][l] = val

            w_out[l] = self.token_lookup[tuple(indices)] #(l+1) # particle token
            
        # Sort by size of particle -- largest first, put fake particles at the end
#         sort_y_out = np.where(y_out["d"] == 0.0, -1e10, y_out["d"])
#         sorted_idx = np.argsort(sort_y_out)[::-1]
#         for task in output_cols:
#             y_out[task] = y_out[task][sorted_idx]
#         w_out = w_out[sorted_idx]
        
        w_out[l+1] = 2 # EOS token
        num_particles = (l + 2)
        w_out[num_particles:] = 0 # PAD token        
        
        return im["image"], y_out, w_out.astype(int)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.processed = 0
        if self.shuffle == True:
            if isinstance(self.seed, int) or (self.seed == True):
                if self.seed == True:
                    seed = random.randint(1, 1000000)
                random.seed(seed)
            random.shuffle(self.hologram_numbers)
            
            
# class MultiHologramDataset(Dataset):

#     def __init__(
#             self,
#             path_data: List[str],
#             output_cols: List[str],
#             shuffle: bool = True,
#             maxnum_particles: int = 100,
#             scaler: Dict[str, str] = True,
#             transform = None,
#             labels: bool = False, 
#             cache: Dict[str,tuple] = None,
#             max_cached: int = 100000,
#             null_weight: float = 0.001) -> None:
        
#         'Initialization'
#         self.ds = {name: xr.open_dataset(name) for name in path_data}
        
#         self.hologram_numbers = []
#         for name, _ds in sorted(self.ds.items()):
#             for hologram_number in _ds["hologram_number"].values:
#                 self.hologram_numbers.append([name, hologram_number])
        
#         self.output_cols = [x for x in output_cols if x != 'hid']        
#         self.shuffle = shuffle
#         self.maxnum_particles = maxnum_particles
#         self.transform = transform
#         self.labels = labels
#         self.null_weight = null_weight 
        
#         self.on_epoch_end()
        
#         self.scaler = None
#         if self.labels:
#             self.set_scaler(scaler)
            
#         #self.label_weights()
            
#         self.cache = {} if cache else False
#         self.max_cached = max_cached

#         logger.info(
#             f"Loaded {path_data} hologram data containing {len(self.hologram_numbers)} images"
#         )
        
#     def set_scaler(self, scaler = True):
#         if scaler == True:
#             logger.info(f"Rescaling the data by subtracting the mean and dividing by sigma")
#             self.scaler = {col: StandardScaler() for col in self.output_cols}  # StandardScaler() MinMaxScaler()
#             for col in self.output_cols:
#                 concat = []
#                 for arr in self.ds.values():
#                     scale_factor = float(arr.Nx / 600)
#                     if col in ["x", "y"]:
#                         concat.append(arr[col].values / scale_factor)
#                     else:
#                         concat.append(arr[col].values)
#                 scale = np.hstack(concat)
#                 self.scaler[col].fit(scale.reshape(scale.shape[-1], -1))
#         elif scaler == False:
#             self.scaler = False
#         else:
#             self.scaler = scaler
#         logger.info(f"Loaded data scaler transformation {scaler}")
            
#     def label_weights(self):
#         logger.info("Creating weights dictionary based on hologram particle number counts")
#         weights = Counter([
#             (arr.hid.values == a+1).sum() 
#             for arr in self.ds.values()
#             for a in arr.hologram_number.values 
#         ])
#         largest = max(list(weights.values()))
#         weights = {
#             key: largest / value for key, value in weights.items()
#         }
#         largest = max(list(weights.values()))
#         self.weights = {
#             key: value / largest for key, value in weights.items()
#         }
        
#     def get_transform(self):
#         return self.scaler

#     def __len__(self):
#         'Denotes the number of batches per epoch'
#         return len(self.hologram_numbers)

#     def __getitem__(self, idx):
#         'Generate one data point'
        
#         # random.choice(self.hologram_numbers)
#         name, hologram = self.hologram_numbers[idx]
        
#         if isinstance(self.cache, dict):
#             if (name, hologram) in self.cache:
#                 return self.cache[(name, hologram)]
        
#         self.processed += 1
#         if self.processed == self.__len__():
#             self.on_epoch_end()
        
#         im = self.ds[name]["image"][hologram].values
#         scale_factor = float(im.shape[0] / 600)  
#         im = {"image": np.expand_dims(im, 0)}  # reshape
        
#         if self.transform:
#             im = self.transform(im)
        
#         if not self.labels:
#             return im["image"]
        
#         output_cols = self.output_cols + ["binary"]
        
#         y_out = {}
#         for task in output_cols:
#             y_out[task] = np.zeros((self.maxnum_particles))
#         w_out = np.zeros((self.maxnum_particles))
#         particles = np.where(self.ds[name]["hid"] == hologram + 1)[0]
#         for l, p in enumerate(particles):
#             for task in output_cols:
#                 if task == "binary":
#                     y_out[task][l] = 1
#                     continue
#                 val = self.ds[name][task].values[p]
#                 if task in ["x", "y"]:
#                     val /= scale_factor
#                 if isinstance(self.scaler, dict):
#                     val = self.scaler[task].transform(val.reshape(-1, 1))[0][0]
#                 y_out[task][l] = val
#             w_out[l] = 1 #self.weights[len(particles)]
            
#         # Sort by size of particle -- largest first, put fake particles at the end
#         sort_y_out = np.where(y_out["d"] == 0.0, -1e10, y_out["d"])
#         sorted_idx = np.argsort(sort_y_out)[::-1]
#         for task in output_cols:
#             y_out[task] = y_out[task][sorted_idx]
        
#         num_particles = (l + 1)
#         w_out[num_particles:] = self.null_weight
#         #y_out["d"][num_particles:] = 0.001
#         #w_out /= sum(w_out)
        
#         if isinstance(self.cache, dict):
#             if (name, hologram) not in self.cache:
#                 self.cache[(name, hologram)] = (im["image"], y_out, w_out)
#             if len(self.cache) > self.max_cached:
#                 self.cache.pop(random.choice(self.cache.keys()))

#         return im["image"], y_out, w_out

#     def on_epoch_end(self):
#         'Updates indexes after each epoch'
#         self.processed = 0
#         if self.shuffle == True:
#             random.shuffle(self.hologram_numbers)
            
            

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
            output_cols: List[str],
            shuffle: bool = True,
            maxnum_particles: int = False,
            scaler: Dict[str, str] = False,
            transform=None,
            cache=False,
            null_weight: float = 0.001) -> None:
        
        'Initialization'
        self.ds = self.open_dataset(path_data, num_particles, split)
        self.output_cols = [x for x in output_cols if x != 'hid']
        self.hologram_numbers = self.ds.hologram_number.values
        self.xsize = len(self.ds.xsize.values)
        self.ysize = len(self.ds.ysize.values)
        self.shuffle = shuffle
        self.maxnum_particles = maxnum_particles
        self.transform = transform
        self.cache = {} if cache else False
        self.null_weight = null_weight
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
            y_out[task] = np.zeros((self.maxnum_particles))
        w_out = np.zeros((self.maxnum_particles))
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
        w_out[num_particles:] = self.null_weight
        w_out /= sum(w_out)

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
