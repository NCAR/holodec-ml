import sys
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, max_error
import pandas as pd
import numpy as np
import argparse
import random
import yaml
import os
from os.path import join, exists
from datetime import datetime
import tensorflow as tf
    
from holodecml.data import load_scaled_datasets
from holodecml.models import ParticleAttentionNet


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
    
    net = ParticleAttentionNet()
    net.compile(optimizer=config['optimizer'], loss=config['loss'])
    net.fit([train_outputs_noisy, train_inputs], train_outputs, epochs=config['epochs'],
            batch_size=config['batch_size'], verbose=config['verbose'])
    
    
