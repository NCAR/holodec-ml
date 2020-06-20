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

def flatten_dataset(df):
    columns = ["x1","y1","z1","d1","x2","y2","z2","d2","x3","y3","z3","d3"]
    data = []
    for hid in df['hid'].unique():
        vect = df.loc[df['hid'] == hid].drop(['hid'], axis=1).values
        data.append(vect.flatten())
    data = np.vstack(data)
    df = pd.DataFrame(data=data, columns=columns)
    return df

def load_scaled_datasets(data_path, num_particles, output_cols, input_scaler):
    """Given the dataset particle numbers, returns scaled training and validation xarrays."""
    
    beginning = datetime.now()
    print(f"BEGINNING: {beginning}")
    startTime = datetime.now()
    print("Loading training and validation data")
    xr_train = open_dataset(data_path, num_particles, 'train')
    xr_valid = open_dataset(data_path, num_particles, 'valid')
    print(f"\t- time to load datasets: {datetime.now() - startTime}")
    
    print("\tScaling input data")
    start_time = datetime.now()
    scaled_train_inputs = scale_images(xr_train["image"])
    print(f"\t\tscaled_train_inputs.shape: {scaled_train_inputs.shape}")
    print(f"\t\t- time to scale train input data: {datetime.now() - startTime}")
    start_time = datetime.now()
    scaled_valid_inputs = scale_images(xr_valid["image"])
    print(f"\t\tscaled_valid_inputs.shape: {scaled_valid_inputs.shape}")
    print(f"\t\t- time to scale valid input data: {datetime.now() - startTime}")
    
    print("\tScaling output data")
    start_time = datetime.now()
    train_outputs = xr_train[output_cols].to_dataframe()
    valid_outputs = xr_valid[output_cols].to_dataframe()
    print(f"\t\t- time to slice output data: {datetime.now() - startTime}")
    
    print("\tScaling output data")
    if num_particles == 3:
        train_outputs = flatten_dataset(train_outputs)
        valid_outputs = flatten_dataset(valid_outputs)
    
    scaled_train_outputs = pd.DataFrame(input_scaler.fit_transform(train_outputs),
                                        index=train_outputs.index, columns=train_outputs.columns)
    print(f"\t\tscaled_train_outputs.shape: {scaled_train_outputs.shape}") 
    print(f"\t\t- time to scale train output data: {datetime.now() - startTime}")
    scaled_valid_outputs = pd.DataFrame(input_scaler.transform(valid_outputs),
                                        index=valid_outputs.index, columns=valid_outputs.columns)
    print(f"\t\tscaled_valid_outputs.shape: {scaled_valid_outputs.shape}")
    print(f"\t\t- time to scale valid output data: {datetime.now() - startTime}")
    end = datetime.now()
    print(f"END: {end}\nTIME ELAPSED: {end - beginning}")

    return scaled_train_inputs, scaled_valid_inputs, scaled_train_outputs, scaled_valid_outputs, input_scaler

