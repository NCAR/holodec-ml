from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np
import argparse
import logging
import pickle
import random
import yaml
import sys
import os

from os.path import join, exists
from datetime import datetime
import tensorflow as tf

sys.path.append('../../')
    
from library.data import load_scaled_datasets
from library.models import Conv2DNeuralNetwork
from library.callbacks import get_callbacks


scalers = {"MinMaxScaler": MinMaxScaler,
           "MaxAbsScaler": MaxAbsScaler,
           "StandardScaler": StandardScaler,
           "RobustScaler": RobustScaler}


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
    logging.info(f"Loading data")
    scaler_out = scalers[config["scaler_out"]]()
    load_start = datetime.now()
    train_inputs,\
    train_outputs,\
    valid_inputs,\
    valid_outputs = load_scaled_datasets(path_data,
                                         num_particles,
                                         output_cols,
                                         scaler_out,
                                         config["subset"],
                                         config["num_z_bins"])
    logging.info(f"Loading datasets took {datetime.now() - load_start} time")
    
    # load callbacks
    callbacks = get_callbacks(config)
    
    # train and save the model
    model_start = datetime.now()
    mod = Conv2DNeuralNetwork(**config["conv2d_network"])
    hist = mod.fit(train_inputs, train_outputs,
                   xv=valid_inputs, yv=valid_outputs, 
                   callbacks = callbacks)
    logging.info(f"Training the model took {datetime.now() - model_start} time")
    
    # predict outputs - load the best model from the Checkpointer first
    best_model_weights = config["callbacks"]["ModelCheckpoint"]["filepath"]
    mod.load_weights(best_model_weights)
    
    logging.info(f"Predicting on train and validation splits")
    train_outputs_pred = mod.predict_proba(train_inputs)
    valid_outputs_pred = mod.predict_proba(valid_inputs)
    
    # save results
    np.savetxt(join(path_save, "train_outputs_pred.csv"), train_outputs_pred)
    np.savetxt(join(path_save, "valid_outputs_pred.csv"), valid_outputs_pred)   
    
    for k in hist.keys():
        np.savetxt(join(path_save, k + ".csv"), hist[k])
    
    with open(join(path_save, 'config.yml'), 'w') as f:
        yaml.dump(config, f)
    
    return

if __name__ == "__main__":
    
    #########
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')

    # Stream output to stdout
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    root.addHandler(ch)
    #########

    # Save the log file
    fp = os.path.join('log.txt')
    fh = logging.FileHandler(fp,
                             mode='w',
                             encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    root.addHandler(fh)
    #########
    
    main()