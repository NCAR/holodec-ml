from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, max_error, mean_squared_error
import pandas as pd
import numpy as np
import argparse
import random
import yaml
import xarray as xr
import os
from os.path import join, exists
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from holodecml.data import load_unet_datasets, load_unet_datasets_xy
from holodecml.models import custom_unet
from holodecml.losses import unet_loss, unet_loss_xy, unet_loss_xy_log
from memory_profiler import profile


scalers = {"MinMaxScaler": MinMaxScaler,
           "MaxAbsScaler": MaxAbsScaler,
           "StandardScaler": StandardScaler,
           "RobustScaler": RobustScaler}

losses = {"unet_loss": unet_loss,
           "unet_loss_xy": unet_loss_xy,
           "unet_loss_xy_log": unet_loss_xy_log}

@profile(precision=4)
def main():
    
    print("Starting script...")
        
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
    num_particles = config["num_particles"]
    output_cols = config["output_cols"]
    seed = config["random_seed"]
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    
    # load data
    print("Loading data...")
    scaler_out = scalers[config["scaler_out"]]()
    train_inputs,\
    train_outputs,\
    valid_inputs,\
    valid_outputs = load_unet_datasets_xy(path_data,
                                         num_particles,
                                         output_cols,
#                                          scaler_out,
                                         config["subset"],
                                         config["bin_factor"])
    
    model_start = datetime.now()
    model = custom_unet(
        np.expand_dims(train_inputs, axis=-1).shape[1:],
        **config["unet"]
    )
    loss = losses[config["loss"]]
    model.compile(optimizer=Adam(lr=config["train"]['learning_rate']), loss=loss)
    hist = model.fit(
        np.expand_dims(train_inputs, axis=-1),
        train_outputs,
        batch_size=config["train"]['batch_size'],
        epochs=config["train"]['epochs'],
        validation_data=(np.expand_dims(valid_inputs, axis=-1), valid_outputs),
        verbose=config["train"]["verbose"]
    )
    print(f"Running model took {datetime.now() - model_start} time")
    
    # predict outputs
    print("Predicting outputs..")
    valid_outputs_pred = model.predict(np.expand_dims(valid_inputs, axis=-1),
                                     batch_size=config['train']["batch_size"]*4)

    valid_outputs_pred_da = xr.DataArray(valid_outputs_pred,
                                         coords={"hid": np.arange(valid_outputs_pred.shape[0]),
                                                 "x": np.arange(valid_outputs_pred.shape[1]),
                                                 "y": np.arange(valid_outputs_pred.shape[2]),
                                                 "output": ["p"]},
#                                                  "output": ["p", "z", "d"]},
                                         dims=("hid", "x", "y", "output"),
                                         name="valid_pred_scaled")
    
    # calculate errors
    print("Calculating errors..")
    
    # save outputs to files
    print("Saving results and config file..")
    model.save(path_save, save_format="tf")
    model.save_weights(path_save + '_weights', save_format='tf')
    valid_outputs_pred_da.to_netcdf(join(path_save, "valid_outputs_pred.nc"))
    for k in hist.history.keys():
        np.savetxt(join(path_save, k+".csv"), hist.history[k])
    with open(join(path_save, 'config.yml'), 'w') as f:
        yaml.dump(config, f)
    print("Finished saving results.")

if __name__ == "__main__":
    main()

