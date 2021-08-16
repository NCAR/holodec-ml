import os
import socket

import torch
import random
import joblib
import logging
import numpy as np
import xarray as xr

from scipy.sparse import csr_matrix
from torch.utils.data import Dataset, DataLoader



logger = logging.getLogger(__name__)



num_particles_dict = {
    1 : '1particle',
    3 : '3particle',
    'multi': 'multiparticle',
    'large': '50-100particle_gamma'}

split_dict = {
    'train' : 'training',
    'test'   : 'test',
    'valid': 'validation'}



def save_sparse_csr(filename, array):
    # note that .npz extension is added automatically
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)
    

def load_sparse_csr(filename):
    # here we need to add .npz extension manually
    loader = np.load(filename + '.npz')
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])


def get_dataset_path():
    if 'casper' in socket.gethostname():
        return "/glade/p/cisl/aiml/ai4ess_hackathon/holodec/"
    else:
        return "/Users/ggantos/PycharmProjects/holodec-ml/data/"

def dataset_name(num_particles, split, file_extension='nc'):
    """
    Return the dataset filename given user inputs
    
    Args: 
        num_particles: (int or str) Number of particles per hologram
        split: (str) Dataset split of either 'train', 'valid', or 'test'
        file_extension: (str) Dataset file extension
    
    Returns:
        ds_name: (str) Dataset name
    """
    
    valid = [1,3,'multi','large']
    if num_particles not in valid:
        raise ValueError("results: num_particles must be one of %r." % valid)
    num_particles = num_particles_dict[num_particles]

    valid = ['train','test','valid']
    if split not in valid:
        raise ValueError("results: split must be one of %r." % valid)
    split = split_dict[split]
    ds_name = f'synthetic_holograms_{num_particles}_{split}.{file_extension}'
    
    return ds_name

def open_dataset(path_data, num_particles, split):
    """
    Opens a HOLODEC file
    
    Args: 
        path_data: (str) Path to dataset directory
        num_particles: (int or str) Number of particles per hologram
        split: (str) Dataset split of either 'train', 'valid', or 'test'
    
    Returns:
        ds: (xarray Dataset) Opened dataset
    """
    path_data = os.path.join(path_data, dataset_name(num_particles, split))
    ds = xr.open_dataset(path_data)
    return ds

def load_raw_datasets(path_data, num_particles, split, output_cols, subset):
    """
    Given a path to training or validation datset, the number of particles per
    hologram, and output columns, returns raw inputs and outputs. Can specify
    a subset of the full dataset.
    
    Args: 
        path_data: (str) Path to dataset directory
        num_particles: (int or str) Number of particles per hologram 
        split: (str) Dataset split of either 'train', 'valid', or 'test'
        subset: (float) Fraction of data to be loaded
        output_cols: (list of strings) List of feature columns
        
    Returns:
        inputs: (np array) Input image data
        outputs: (df) Output data specified by output_cols 
    """
    
    ds = open_dataset(path_data, num_particles, split)
    if subset:
        ix = int(subset * ds['image'].shape[0])
        inputs = ds['image'][:ix].values
        outputs = ds[output_cols].to_dataframe()
        outputs = outputs[outputs["hid"] < (ix+1)]
    else:
        inputs = ds["image"].values
        outputs = ds[output_cols].to_dataframe()    
    ds.close()
    return inputs, outputs

def scale_images(images, scaler_in=None):
    """
    Takes in array of images and scales pixel values between 0 and 1
    
    Args: 
        images: (np array) Input image data
        scaler_in: (dict) Image scaler 'max' and 'min' values
        
    Returns:
        images_scaled: (np array) Input image data scaled between 0 and 1
        scaler_in: (dict) Image scaler 'max' and 'min' values
    """
    
    if scaler_in is None:
        scaler_in = {}
        scaler_in["min"] = images.min()
        scaler_in["max"] = images.max()
    images_scaled = (images.astype(np.float32) - scaler_in["min"])
    images_scaled /= (scaler_in["max"] - scaler_in["min"])

    return images_scaled, scaler_in

def calc_z_relative_mass(outputs, num_z_bins=20, z_bins=None):
    """
    Calculate z-relative mass from particle data.
    
    Args: 
        outputs: (df) Output data specified by output_col 
        num_z_bins: (int) Number of bins for z-axis linspace
        z_bins: (np array) Bin linspace along the z-axis
    
    Returns:
        z_mass: (np array) Particle mass distribution by hologram along z-axis
        z_bins: (np array) Bin linspace along the z-axis
    """
    
    if z_bins is None:
        z_bins = np.linspace(outputs["z"].min() - 100,
                             outputs["z"].max() + 100,
                             num_z_bins)
    else:
        num_z_bins = z_bins.size
    holograms = len(outputs["hid"].unique())
    z_mass = np.zeros((holograms, num_z_bins), dtype=np.float32)
    for i in range(outputs.shape[0]):
        z_pos = np.searchsorted(z_bins, outputs.loc[i, "z"], side="right") - 1
        mass = 4 / 3 * np.pi * (outputs.loc[i, "d"]/2)**3
        z_mass[int(outputs.loc[i, "hid"]) - 1, z_pos] += mass
    z_mass /= np.expand_dims(z_mass.sum(axis=1), -1)
    return z_mass, z_bins

def calc_z_dist(outputs, num_z_bins=20, z_bins=None):
    """
    Calculate z distribution
    
    Args: 
        outputs: (df) Output data specified by output_col 
        num_z_bins: (int) Number of bins for z-axis linspace
        z_bins: (np array) Bin linspace along the z-axis
    
    Returns:
        z_dist: (np array) Particle z distribution by hologram along z-axis
        z_bins: (np array) Bin linspace along the z-axis
    """
    
    if z_bins is None:
        z_bins = np.linspace(outputs["z"].min() - 100,
                             outputs["z"].max() + 100,
                             num_z_bins)
    else:
        num_z_bins = z_bins.size
    holograms = len(outputs["hid"].unique())
    z_dist = np.zeros((holograms, num_z_bins), dtype=np.float32)
    for i in range(outputs.shape[0]):
        z_pos = np.searchsorted(z_bins, outputs.loc[i, "z"], side="right") - 1
        z_dist[int(outputs.loc[i, "hid"]) - 1, z_pos] += 1
    z_dist /= np.expand_dims(z_dist.sum(axis=1), -1)
    return z_dist, z_bins

def calc_z_bins(train_outputs, valid_outputs, num_z_bins):
    """
    Calculate z-axis linspace.
    
    Args: 
        train_outputs: (np array) Training output data 
        valid_outputs: (int) Validation output data
        num_z_bins: (int) Bin linspace along the z-axis
    
    Returns:
        z_bins: (np array) Bin linspace along the z-axis
    """
    z_min = np.minimum(train_outputs["z"].min(), valid_outputs["z"].min())
    z_max = np.maximum(train_outputs["z"].max(), valid_outputs["z"].max())
    z_bins = np.linspace(z_min, z_max, num_z_bins)
    return z_bins

# added this because the previous code allowed a different max_particle size
# depending on which split df was opened and the subset
def get_max_particles(path_data, num_particles, output_cols):
    ds = open_dataset(path_data, num_particles, "train")
    outputs = ds[output_cols].to_dataframe()
    max_particles = outputs['hid'].value_counts().max()
    return max_particles

# updated function to create the entire dataset template at one time to
# decrease overhead and eliminate setting random seeds
def make_template(df, num_images, max_particles):
    size = (num_images * max_particles, 1)
    x = np.random.uniform(low=df['x'].min(), high=df['x'].max(), size=size)
    y = np.random.uniform(low=df['y'].min(), high=df['y'].max(), size=size)
    z = np.random.uniform(low=df['z'].min(), high=df['z'].max(), size=size)
    d = np.random.uniform(low=df['d'].min(), high=df['d'].max(), size=size)
    prob = np.zeros(d.shape)
    template = np.hstack((x, y ,z ,d ,prob))
    template = template.reshape((num_images, max_particles, -1))
    return template

def make_random_outputs(ds):
    num_images = ds.shape[0]
    max_particles = ds.shape[1]
    size = (num_images * max_particles, 1)
    x = np.random.uniform(low=np.min(ds[:,:,0:1]), high=np.max(ds[:,:,0:1]), size=size)
    y = np.random.uniform(low=np.min(ds[:,:,1:2]), high=np.max(ds[:,:,1:2]), size=size)
    z = np.random.uniform(low=np.min(ds[:,:,2:3]), high=np.max(ds[:,:,2:3]), size=size)
    d = np.random.uniform(low=np.min(ds[:,:,3:4]), high=np.max(ds[:,:,3:4]), size=size)
    template = np.hstack((x, y, z, d))
    template = template.reshape((num_images, max_particles, -1))
    return template

# cycles through dataset by "hid" to overwrite random data generated in
# make_template with actual data and classification of 1
def outputs_3d(outputs, num_images, max_particles):
    outputs_array = make_template(outputs, num_images, max_particles)
    for hid in outputs["hid"].unique():
        outputs_hid = outputs.loc[outputs['hid'] == hid].to_numpy()
        outputs_hid[:, -1] = 1
        outputs_array[int(hid-1), :outputs_hid.shape[0], :] = outputs_hid
    return outputs_array

def load_scaled_datasets(path_data, num_particles, output_cols,
                         scaler_out=False, subset=False, num_z_bins=False,
                         mass=False):
    """
    Given a path to training or validation datset, the number of particles per
    hologram, and output columns, returns scaled inputs and raw outputs.
    
    Args: 
        path_data: (str) Path to dataset directory
        num_particles: (int or str) Number of particles per hologram
        output_cols: (list of strings) List of feature columns
        scaler_out: (sklearn.preprocessing scaler) Output data scaler
        subset: (float) Fraction of data to be loaded
        num_z_bins: (int) Number of bins along z-axis
        mass: (boolean) If True, calculate particle mass on z-axis
        
    Returns:
        train_inputs: (np array) Train input data scaled between 0 and 1
        train_outputs: (np array) Scaled train output data
        valid_inputs: (np array) Valid input data scaled between 0 and 1
        valid_outputs: (np array) Scaled valid output data
    """
    train_inputs,\
    train_outputs = load_raw_datasets(path_data, num_particles, 'train',
                                      output_cols, subset)
    valid_inputs,\
    valid_outputs = load_raw_datasets(path_data, num_particles, 'valid',
                                      output_cols, subset)
    
    train_inputs, scaler_in = scale_images(train_inputs)
    valid_inputs, _ = scale_images(valid_inputs, scaler_in)
    train_inputs = np.expand_dims(train_inputs, -1)
    valid_inputs = np.expand_dims(valid_inputs, -1)
    
    if num_z_bins:
        z_bins = calc_z_bins(train_outputs, valid_outputs, num_z_bins)
        if mass:
            train_outputs, _ = calc_z_relative_mass(outputs=train_outputs,
                                                    z_bins=z_bins)
            valid_outputs, _ = calc_z_relative_mass(outputs=valid_outputs,
                                                    z_bins=z_bins)
        else:
            train_outputs, _ = calc_z_dist(outputs=train_outputs,
                                           z_bins=z_bins)
            valid_outputs, _ = calc_z_dist(outputs=valid_outputs,
                                           z_bins=z_bins)        
    else:
        if train_inputs.shape[0] != train_outputs.shape[0]:
            col = [c for c in output_cols if c != 'hid']
            max_particles = get_max_particles(path_data, num_particles, output_cols)
            train_outputs[col] = scaler_out.fit_transform(train_outputs[col])
            train_outputs = outputs_3d(train_outputs, train_inputs.shape[0],
                                       max_particles)
            valid_outputs[col] = scaler_out.transform(valid_outputs[col])
            valid_outputs = outputs_3d(valid_outputs, valid_inputs.shape[0],
                                       max_particles)
        else:
            train_outputs.drop(['hid'], axis=1)
            train_outputs = scaler_out.fit_transform(train_outputs)
            valid_outputs.drop(['hid'], axis=1)
            valid_outputs = scaler_out.transform(valid_outputs)
        
    return train_inputs, train_outputs, valid_inputs, valid_outputs



class PickleReader(Dataset):
    
    def __init__(self, 
                 fn, 
                 max_buffer_size = 5000, 
                 max_images = 40000, 
                 shuffle = True, 
                 transform = False,
                 normalize = True):
        
        self.fn = fn
        self.buffer = []
        self.max_buffer_size = max_buffer_size
        self.shuffle = shuffle
        self.max_images = max_images
        self.transform = transform
            
        self.fid = open(self.fn, "rb")
        self.loaded = 0 
        self.epoch = 0
        
        self.normalize = normalize
        self.mean = np.mean([0.485, 0.456, 0.406])
        self.std = np.mean([0.229, 0.224, 0.225])
        
    def __getitem__(self, idx):    
        
        self.on_epoch_end()
        
        while True:
        
            try:
                data = joblib.load(self.fid)
                image, label, mask = data
                
                im = {
                    "image": image, 
                    "horizontal_flip": False, 
                    "vertical_flip": False
                }
                
                if self.transform:
                    for image_transform in self.transform:
                        im = image_transform(im)
                        
                image = im["image"]
                mask = mask.toarray()
                
                # Update the mask if we flipped the original image
                if im["horizontal_flip"]:
                    mask = np.flip(mask, axis=0)
                if im["vertical_flip"]:
                    mask = np.flip(mask, axis=1)
                        
                image = torch.FloatTensor(image)
                #label = torch.LongTensor([label])
                mask = torch.FloatTensor(mask.copy())
                
                
                #######
                
                
                data = (image, mask)

                self.loaded += 1

                if not self.shuffle:
                    return data
                
                self.buffer.append(data)
                random.shuffle(self.buffer)

                if len(self.buffer) > self.max_buffer_size:
                    self.buffer = self.buffer[:self.max_buffer_size]

                if self.epoch > 0:
                    return self.buffer.pop()

                else: # wait until all data has been seen before sampling from the buffer
                    return data


            except EOFError:
                self.fid = open(self.fn, "rb")
                self.loaded = 0
                continue
                    
    def __len__(self):
        return self.max_images
    
    def on_epoch_end(self):
        if self.loaded == self.__len__():
            self.fid = open(self.fn, "rb")
            self.loaded = 0
            self.epoch += 1