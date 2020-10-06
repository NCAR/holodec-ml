import sys
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np
import xarray as xr
import argparse
import random
import yaml
import os
from os.path import join, exists
from datetime import datetime
import tensorflow as tf

sys.path.append('../../')
    
from holodecml.data import load_scaled_datasets
from holodecml.models import Conv2DNeuralNetwork
import holodecml.ml_utils as ml


scalers = {"MinMaxScaler": MinMaxScaler,
           "MaxAbsScaler": MaxAbsScaler,
           "StandardScaler": StandardScaler,
           "RobustScaler": RobustScaler}

metrics = {"mae": mean_absolute_error}


def main():
        
    # parse arguments from config/yaml file
    parser = argparse.ArgumentParser(description='Describe a Conv2D nn')
    parser.add_argument("config", help="Path to config file")
    args = parser.parse_args()
    with open(args.config) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    path_data = config["path_data"]
    path_save = config["path_save"]
    if not os.path.exists(path_save):
        os.makedirs(path_save)    
    seed = config["random_seed"]
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    input_variable = config["input_variable"]
    label_variable = config["label_variable"]
    batch_size = config["conv2d_network"]["batch_size"]
    
    # load data
    load_start = datetime.now()
    fns = [x for x in os.walk(path_data)][0][2]
    fn_train = [x for x in fns if 'training' in x][0]
    fn_valid = [x for x in fns if 'validation' in x][0]
    fn_test = [x for x in fns if 'test' in x][0]
    
    with xr.open_dataset(path_data+fn_train, chunks={'hologram_number': batch_size}) as ds:
        print("Loading TRAINING dataset")

        if len(ds[input_variable].dims) == 4:
            train_inputs = ds[input_variable].transpose('hologram_number','xsize','ysize','input_channels')
        elif len(ds[input_variable].dims) == 3:
            train_inputs = ds[input_variable].transpose('hologram_number','rsize','input_channels')
        input_scaler = ml.MinMaxScalerX(train_inputs)
        train_inputs = input_scaler.fit_transform(train_inputs)
        print(f"\n\ttrain_inputs.shape:{train_inputs.shape}")
        
        train_outputs = ds[label_variable]
        output_scaler = ml.MinMaxScalerX(train_outputs)
        train_outputs = output_scaler.fit_transform(train_outputs)
        print(f"\n\ttrain_outputs.shape:{train_outputs.shape}")

    with xr.open_dataset(path_data+fn_valid, chunks={'hologram_number': batch_size}) as ds:
        print("Loading VALIDATION dataset")

        if len(ds[input_variable].dims) == 4:
            valid_inputs = ds[input_variable].transpose('hologram_number','xsize','ysize','input_channels')
        elif len(ds[input_variable].dims) == 3:
            valid_inputs = ds[input_variable].transpose('hologram_number','rsize','input_channels')
        input_scaler = ml.MinMaxScalerX(valid_inputs)
        valid_inputs = input_scaler.fit_transform(valid_inputs)
        print(f"\n\tvalid_inputs.shape:{valid_inputs.shape}")
        
        valid_outputs = ds[label_variable]
        output_scaler = ml.MinMaxScalerX(valid_outputs)
        valid_outputs = output_scaler.fit_transform(valid_outputs)
        print(f"\n\tvalid_outputs.shape:{valid_outputs.shape}")
    print(f"Loading datasets took {datetime.now() - load_start} time")
    
    # train and save the model
    model_start = datetime.now()
    mod = Conv2DNeuralNetwork(**config["conv2d_network"])
    hist = mod.fit(train_inputs, train_outputs,
                   xv=valid_inputs, yv=valid_outputs)
    print(f"Running model took {datetime.now() - model_start} time")
    
    # predict outputs
    train_outputs_pred = mod.predict(train_inputs)
    valid_outputs_pred = mod.predict(valid_inputs) 
    
    # save results
    print("Saving results and config file..")
    mod.model.save(join(path_save, config["model_name"]+".h5"))
    np.savetxt(join(path_save, "train_outputs_pred.csv"), train_outputs_pred)
    np.savetxt(join(path_save, "valid_outputs_pred.csv"), valid_outputs_pred)   
    for k in hist.keys():
        np.savetxt(join(path_save, k+".csv"), hist[k])
    with open(join(path_save, 'config.yml'), 'w') as f:
        yaml.dump(config, f)
    
    return

if __name__ == "__main__":
    main()
