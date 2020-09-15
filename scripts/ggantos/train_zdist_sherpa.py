import sherpa
import sys
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import mean_absolute_error
import joblib
import pandas as pd
import numpy as np
import argparse
import random
import yaml
import os
from os.path import join, exists
from datetime import datetime
import tensorflow as tf

sys.path.append('../../')
    
from library.data import load_scaled_datasets
from library.models import Conv2DNeuralNetwork


scalers = {"MinMaxScaler": MinMaxScaler,
           "MaxAbsScaler": MaxAbsScaler,
           "StandardScaler": StandardScaler,
           "RobustScaler": RobustScaler}

metrics = {"mae": mean_absolute_error}


def main():
        
    parser = argparse.ArgumentParser(description='Describe a Conv2D nn')
    parser.add_argument("config", help="Path to config file")
    args = parser.parse_args()
    with open(args.config) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    path_data = config["path_data"]
    path_save = config["path_save"]
    num_particles = config["num_particles"]
    output_cols = config["output_cols"]
    seed = config["random_seed"]
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    
    # load data
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
                                         config["num_z_bins"],
                                         config["mass"])
    print(f"Loading datasets took {datetime.now() - load_start} time")
    
    # set up sherpa variables
    
    max_num_trials = int(config["max_num_trials"])
    lrs = config["conv2d_network"]["lrs"]
    conv_layers = config["conv2d_network"]["conv_layers"]
    kernel_size = config["conv2d_network"]["kernel_size"]
    pool_size = config["conv2d_network"]["pool_size"]
    filter_size = config["conv2d_network"]["filter_size"]
    dense_layers = config["conv2d_network"]["dense_layers"]
    dense_size_0 = config["conv2d_network"]["dense_size_0"]
    
    
    lrs = [1e-5, 1e-4, 1e-3, 3e-5, 3e-4, 3e-3]
    parameters = [sherpa.Choice('lr', lrs),
                  sherpa.Discrete('conv_layers', conv_layers),
                  sherpa.Discrete('kernel_size', kernel_size),
                  sherpa.Discrete('pool_size', pool_size),
                  sherpa.Choice('filter_size', filter_size),
                  sherpa.Discrete('dense_layers', dense_layers),
                  sherpa.Choice('dense_size_0', dense_size_0)]
    
    alg = sherpa.algorithms.bayesian_optimization.GPyOpt(max_num_trials=max_num_trials,
                                                         num_initial_data_points=100)
    study = sherpa.Study(parameters=parameters,
                         algorithm=alg,
                         lower_is_better=True)
    
    for trial in study:
        print(f"TRIAL ID: {trial.id}")
        config_i = config.copy()
        path_save_i = path_save + f"{trial.id}"
        if not exists(path_save_i):
            os.makedirs(path_save_i)

        lr = trial.parameters['lr']
        conv_layers = int(trial.parameters['conv_layers'])
        kernel_size = int(trial.parameters["kernel_size"])
        pool_size = int(trial.parameters["pool_size"])
        filters = []
        for layer in range(conv_layers):
            filters.append(trial.parameters['filter_size'] * (layer + 1))
        kernel_sizes = [kernel_size for i in range(conv_layers)]
        pool_sizes = [pool_size for i in range(conv_layers)] 
        dense_layers = int(trial.parameters["dense_layers"])
        dense_sizes = []
        for dense_layer in range(dense_layers):
            dense_sizes.append(int(trial.parameters['dense_size_0'] / (2 ** dense_layer)))

        config_i["path_save"] = path_save_i
        config_i["conv2d_network"]["lr"] = lr
        config_i["conv2d_network"]["filters"] = filters
        config_i["conv2d_network"]["kernel_sizes"] = kernel_sizes
        config_i["conv2d_network"]["pool_sizes"] = pool_sizes
        config_i["conv2d_network"]["dense_sizes"] = dense_sizes
        config_i["conv2d_network"]["study"] = study
        config_i["conv2d_network"]["trial"] = trial
        
        model_start = datetime.now()
        mod = Conv2DNeuralNetwork(**config_i["conv2d_network"])
        hist = mod.fit(train_inputs, train_outputs,
                       xv=valid_inputs, yv=valid_outputs)
        print(f"Running model took {datetime.now() - model_start} time")

        # predict outputs
        train_outputs_pred = mod.predict(train_inputs)
        valid_outputs_pred = mod.predict(valid_inputs)

        # save results
        print("Saving results and config file..")
        mod.model.save(join(path_save_i, config_i["model_name"]+".h5"))
        np.savetxt(join(path_save_i, "train_outputs_pred.csv"), train_outputs_pred)
        np.savetxt(join(path_save_i, "valid_outputs_pred.csv"), valid_outputs_pred)   
        for k in hist.keys():
            np.savetxt(join(path_save_i, k+".csv"), hist[k])
        print(join(path_save_i, 'config.yml'))
        del config_i["conv2d_network"]["study"]
        del config_i["conv2d_network"]["trial"]
        with open(join(path_save_i, 'config.yml'), 'w') as f:
            yaml.dump(config_i, f)

        study.finalize(trial)
    
    return

if __name__ == "__main__":
    main()