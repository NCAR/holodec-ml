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


def attention_net_loss(y_true, y_pred):
    # y_true and y_pred will have shape (batch_size x max_num_particles x 5)
    loss_real = tf.reduce_mean(tf.abs(y_true[y_true[:, :, -1] > 0] - y_pred[y_true[:, :, -1] > 0]))
    batch_size = tf.shape(y_true)[0]
    column_count = tf.shape(y_true)[1]
    loss_bce = binary_crossentropy(y_true[:,:,-1], 
                                   y_pred[:,:,-1])
    loss_total = loss_real + loss_bce
    return loss_total

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
    num_particles = config["num_particles"]
    output_cols = config["output_cols"]
    seed = config["random_seed"]
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    
    # load data
    scaler_out = scalers[config["scaler_out"]]()
    load_start = datetime.now()
    #TODO: implement 3D data loading for multiple particles per image
    train_inputs,\
    train_outputs,\
    valid_inputs,\
    valid_outputs = load_scaled_datasets(path_data,
                                         num_particles,
                                         output_cols,
                                         scaler_out,
                                         config["subset"])
    
    # train and save the model
    model_start = datetime.now()
    train_outputs_noisy = train_outputs * (1 + np.random.normal(0, config['noisy_sd'], train_outputs.shape))
    valid_outputs_noisy = valid_outputs * (1 + np.random.normal(0, config['noisy_sd'], valid_outputs.shape))
    
    net = ParticleAttentionNet(**config["attention_network"])
    net.compile(optimizer=Adam(lr=config["train"]['learning_rate']), loss=attention_net_loss)
    net.fit([train_outputs_noisy, train_inputs], train_outputs, epochs=config["train"]['epochs'],
            batch_size=config["train"]['batch_size'], verbose=config["train"]['verbose'])
    print(f"Running model took {datetime.now() - model_start} time")
    val_outputs_pred_scaled = net.predict([valid_outputs_noisy, valid_inputs], batch_size=config["batch_size"] * 4)
    val_outputs_pred_raw = scaler_out.inverse_transform(val_outputs_pred_scaled)
    scaled_scores = pd.DataFrame(0, index=["mae", "rmse", "bias", "r2", "max_error"], columns=output_cols, dtype=np.float64)
    for metric in scaled_scores.index:
        for c, col in enumerate(output_cols):
            scaled_scores.loc[metric, col] = metrics[metric](valid_outputs[:, c], val_outputs_pred_scaled[:, c])
            print(f"{metric} {col}: {scaled_scores.loc[metric, col]: 0.3f}")
    scaled_scores.to_csv(join(path_save, "scaled_scores_val.csv"), index_label="metric")
    valid_pred_scaled_da = xr.DataArray(val_outputs_pred_scaled, coords={"hid": np.arange(valid_inputs.shape[0]),
                                                                        "particle": np.arange(valid_outputs.shape[1]),
                                                                        "output": output_cols},
                                       dims=("hid", "particle", "output"), name="valid_pred_scaled")
    valid_pred_raw_da = xr.DataArray(val_outputs_pred_raw, coords={"hid": np.arange(valid_inputs.shape[0]),
                                                                        "particle": np.arange(valid_outputs.shape[1]),
                                                                        "output": output_cols},
                                       dims=("hid", "particle", "output"), name="valid_pred_raw")
    valid_pred_scaled_da.to_netcdf(join(path_save, "valid_pred_scaled.nc"))
    valid_pred_raw_da.to_netcdf(join(path_save, "valid_pred_raw.nc"))

if __name__ == "__main__":
    main()
    
    
