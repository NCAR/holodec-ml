import os
import xarray as xr
import numpy as np
import pandas as pd



num_particles_dict = {
    1 : '1particle',
    3 : '3particle',
    'multi': 'multiparticle'}

split_dict = {
    'train' : 'training',
    'test'   : 'test',
    'valid': 'validation'}

def dataset_name(num_particles, split):
    """Return the dataset filename given user inputs"""
    
    valid = [1,3,'multi']
    if num_particles not in valid:
        raise ValueError("results: num_particles must be one of %r." % valid)
    num_particles = num_particles_dict[num_particles]
    
    valid = ['train','test','valid']
    if split not in valid:
        raise ValueError("results: split must be one of %r." % valid)
    split = split_dict[split]
    
    return f'synthetic_holograms_{num_particles}_{split}.nc'

def open_dataset(data_path, num_particles, split):
    """Return xarray dataset given user inputs"""
    data_path = os.path.join(data_path, dataset_name(num_particles, split))
    ds = xr.open_dataset(data_path)
    return ds

def scale_images(images):
    """Return images with pixel values between 0 and 1"""
    return images.astype(np.float16)/255.

def load_scaled_datasets(data_path, num_particles, output_cols, input_scaler):
    """Given the dataset particle numbers, returns scaled training and validation xarrays."""
    
    print("Loading training and validation data")
    xr_train = open_dataset(data_path, num_particles, 'train')
    xr_valid = open_dataset(data_path, num_particles, 'valid')
    
    print("Scaling output data")
    train_outputs = xr_train[output_cols].to_dataframe()
    valid_outputs = xr_valid[output_cols].to_dataframe()
    
    scaled_train_outputs = pd.DataFrame(input_scaler.fit_transform(train_outputs),
                                        index=train_outputs.index, columns=train_outputs.columns)
    
    scaled_valid_outputs = pd.DataFrame(input_scaler.transform(valid_outputs),
                                        index=valid_outputs.index, columns=valid_outputs.columns)
    
    print("Scaling input data")
    scaled_train_inputs = scale_images(xr_train["image"])
    scaled_valid_inputs = scale_images(xr_valid["image"])

    return scaled_train_inputs, scaled_valid_inputs, scaled_train_outputs, scaled_valid_outputs, input_scaler

