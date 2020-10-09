import sys
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
from tensorflow.keras.losses import binary_crossentropy
from holodecml.data import load_scaled_datasets
from holodecml.models import ParticleAttentionNet
from holodecml.losses import attention_net_loss


scalers = {"MinMaxScaler": MinMaxScaler,
           "MaxAbsScaler": MaxAbsScaler,
           "StandardScaler": StandardScaler,
           "RobustScaler": RobustScaler}



def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def r2(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0, 1] ** 2

metrics = {"mae": mean_absolute_error,
           "rmse": rmse,
           "r2": r2,
           "max_error": max_error}


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
    valid_outputs = load_scaled_datasets(path_data,
                                         num_particles,
                                         output_cols,
                                         scaler_out,
                                         config["subset"],
                                         config["num_z_bins"],
                                         config["mass"])
    
    # add noise to the outputs
    train_outputs_noisy = train_outputs * (1 + np.random.normal(0, config['noisy_sd'], train_outputs.shape))
    valid_outputs_noisy = valid_outputs * (1 + np.random.normal(0, config['noisy_sd'], valid_outputs.shape))
    model_start = datetime.now() 
    net = ParticleAttentionNet(**config["attention_network"])
    print(type(net))
    net.compile(optimizer=Adam(lr=config["train"]['learning_rate']), loss=attention_net_loss)
    hist = net.fit([train_outputs_noisy, train_inputs], train_outputs,
                   validation_data=([valid_outputs_noisy, valid_inputs], valid_outputs),
                   epochs=config["train"]['epochs'],
                   batch_size=config["train"]['batch_size'],
                   verbose=config["train"]['verbose'])
    print(f"Running model took {datetime.now() - model_start} time")
    
    # predict outputs
    print("Predicting outputs..")
    valid_outputs_pred = net.predict([valid_outputs_noisy, valid_inputs],
                                     batch_size=config['train']["batch_size"])
    raw = valid_outputs_pred.reshape(-1, valid_outputs_pred.shape[-1])
    raw = scaler_out.inverse_transform(raw[:,:-1])
    raw = raw.reshape(valid_outputs_pred.shape[0], valid_outputs_pred.shape[1], -1)
    valid_outputs_pred_raw = valid_outputs_pred.copy()
    valid_outputs_pred_raw[:, :, :-1] = raw
    valid_outputs_pred_da = xr.DataArray(valid_outputs_pred,
                                         coords={"hid": np.arange(valid_inputs.shape[0]),
                                                 "particle": np.arange(valid_outputs.shape[1]),
                                                 "output": output_cols},
                                         dims=("hid", "particle", "output"),
                                         name="valid_pred_scaled")
    valid_outputs_pred_raw_da = xr.DataArray(valid_outputs_pred_raw,
                                             coords={"hid": np.arange(valid_inputs.shape[0]),
                                                     "particle": np.arange(valid_outputs.shape[1]),
                                                     "output": output_cols},
                                             dims=("hid", "particle", "output"),
                                             name="valid_pred_raw")
    
    # calculate errors
    print("Calculating errors..")
    scaled_scores = pd.DataFrame(0, index=["mae", "rmse", "r2"], columns=output_cols, dtype=np.float64)
    for metric in scaled_scores.index:
        for c, col in enumerate(output_cols):
            scaled_scores.loc[metric, col] = metrics[metric](valid_outputs[:, c], valid_outputs_pred[:, c])
            print(f"{metric} {col}: {scaled_scores.loc[metric, col]: 0.3f}")
    
    # save outputs to files
    print("Saving results and config file..")
    net.save(path_save, save_format="tf")
    net.save_weights(path_save + '_weights', save_format='tf')
    scaled_scores.to_csv(join(path_save, "scaled_scores_val.csv"), index_label="metric")
    valid_outputs_pred_da.to_netcdf(join(path_save, "valid_outputs_pred.nc"))
    valid_outputs_pred_raw_da.to_netcdf(join(path_save, "valid_outputs_pred_raw.nc"))
    for k in hist.history.keys():
        np.savetxt(join(path_save, k+".csv"), hist.history[k])
    with open(join(path_save, 'config.yml'), 'w') as f:
        yaml.dump(config, f)
    print("Finished saving results.")

if __name__ == "__main__":
    main()

