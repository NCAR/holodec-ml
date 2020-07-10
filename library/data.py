import os
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime


num_particles_dict = {
    1 : '1particle',
    3 : '3particle',
    'multi': 'multiparticle'}

split_dict = {
    'train' : 'training',
    'test'   : 'test',
    'valid': 'validation'}

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
    
    valid = [1,3,'multi']
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

def load_raw_datasets(path_data, num_particles, split, subset, output_cols):
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
        in_ix = int(subset * ds['image'].shape[0])
        out_ix = int(in_ix * (ds['hid'].shape[0]/ds['image'].shape[0]))
        inputs = ds['image'][:in_ix].values
        outputs = ds[output_cols].sel(particle=slice(0,out_ix)).to_dataframe()
    else:
        inputs = ds["image"].values
        outputs = ds[output_cols].to_dataframe()
    ds.close()
    return inputs, outputs

def scale_images(images, scaler_vals=None):
    """
    Takes in array of images and scales pixel values between 0 and 1
    
    Args: 
        images: (np array) Input image data
        scaler_vals: (dict) Image scaler 'max' and 'min' values
        
    Returns:
        images_scaled: (np array) Input image data scaled between 0 and 1
        scaler_vals: (dict) Image scaler 'max' and 'min' values
    """
    
    if scaler_vals is None:
        scaler_vals = {}
        scaler_vals["min"] = images.min()
        scaler_vals["max"] = images.max()
    images_scaled = (images.astype(np.float32) - scaler_vals["min"])
    images_scaled /= (scaler_vals["max"] - scaler_vals["min"])

    return images_scaled, scaler_vals

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

def load_scaled_datasets(path_data, num_particles, output_cols,
                         output_scaler=False, subset=False, num_z_bins=False):
    """
    Given a path to training or validation datset, the number of particles per
    hologram, and output columns, returns scaled inputs and raw outputs.
    
    Args: 
        path_data: (str) Path to dataset directory
        num_particles: (int or str) Number of particles per hologram
        output_cols: (list of strings) List of feature columns
        output_scaler: (sklearn.preprocessing scaler) Output data scaler
        subset: (float) Fraction of data to be loaded
        num_z_bins: (int) Number of bins along z-axis
        
    Returns:
        train_inputs: (np array) Train input data scaled between 0 and 1
        train_outputs: (np array) Scaled train output data
        valid_inputs: (np array) Valid input data scaled between 0 and 1
        valid_outputs: (np array) Scaled valid output data
    """
    
    train_inputs,\
    train_outputs = load_raw_datasets(path_data, num_particles, 'train',
                                      subset, output_cols)
    valid_inputs,\
    valid_outputs = load_raw_datasets(path_data, num_particles, 'valid',
                                      subset, output_cols)

    train_inputs, scaler_vals = scale_images(train_inputs)
    valid_inputs, _ = scale_images(valid_inputs, scaler_vals)
    train_inputs = np.expand_dims(train_inputs, -1)
    valid_inputs = np.expand_dims(valid_inputs, -1) 
    
    if num_z_bins:
        z_bins = calc_z_bins(train_outputs, valid_outputs, num_z_bins)
        train_outputs, _ = calc_z_relative_mass(outputs=train_outputs,
                                                z_bins=z_bins)
        valid_outputs, _ = calc_z_relative_mass(outputs=valid_outputs,
                                                z_bins=z_bins)
    else:
        train_outputs = output_scaler.fit_transform(train_outputs)
        valid_outputs = output_scaler.transform(valid_outputs)
        
    return train_inputs, train_outputs, valid_inputs, valid_outputs
