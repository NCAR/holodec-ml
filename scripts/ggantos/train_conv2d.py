import sys
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np
import argparse
import yaml
import os
from os.path import join, exists
from datetime import datetime

sys.path.append('../../')
    
from library.data import load_scaled_datasets, open_dataset, flatten_dataset
from library.models import Conv2DNeuralNetwork


scalers = {"MinMaxScaler": MinMaxScaler,
           "MaxAbsScaler": MaxAbsScaler,
           "StandardScaler": StandardScaler,
           "RobustScaler": RobustScaler}

metrics = {"mae": mean_absolute_error}


def main():
        
    # parse arguments from config/yaml file
    parser = argparse.ArgumentParser(description='Describe a Conv2D neural network for single particle holodec.')
    parser.add_argument("config", help="Path to config file")
    args = parser.parse_args()
    with open(args.config) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    data_path = config["data_path"]
    out_path = config["out_path"]
    num_particles = config["num_particles"]
    output_cols = config["output_cols"]
    np.random.seed(config["random_seed"])
    if not exists(out_path):
        os.makedirs(out_path)
    
    # load data
    input_scaler = scalers[config["input_scaler"]]()

    scaled_train_inputs, \
    scaled_valid_inputs, \
    scaled_train_outputs, \
    scaled_valid_outputs, \
    input_scaler = load_scaled_datasets(data_path,
                                        num_particles,
                                        output_cols,
                                        input_scaler)
    
    # train and save the model
    model_start = datetime.now()
    mod = Conv2DNeuralNetwork(**config["conv2d_network"])
    mod.fit(scaled_train_inputs.values, scaled_train_outputs.values)
    print(f"Running model took {datetime.now() - model_start} time")
    print("Saving the model")
    mod.model.save(join(out_path, config["model_name"]+".h5"))
    
    # predict outputs
    scaled_pred_valid_outputs = pd.DataFrame(mod.predict(scaled_valid_inputs),
                                             index=scaled_valid_outputs.index,
                                             columns=scaled_valid_outputs.columns)
    scaled_pred_train_outputs = pd.DataFrame(mod.predict(scaled_train_inputs),
                                             index=scaled_train_outputs.index,
                                             columns=scaled_train_outputs.columns)
    
    # apply inverse scaler to outputs
    pred_train_outputs = input_scaler.inverse_transform(scaled_pred_train_outputs)        
    pred_valid_outputs = input_scaler.inverse_transform(scaled_pred_valid_outputs)
    
    # calculate error
    train_outputs = open_dataset(data_path, num_particles, "train")[output_cols].to_dataframe()
    valid_outputs = open_dataset(data_path, num_particles, "valid")[output_cols].to_dataframe()
    if len(output_cols) == 5:
        train_outputs = flatten_dataset(train_outputs)
        valid_outputs = flatten_dataset(valid_outputs)
    error = {"train": {}, "valid": {}}
    for i, var in enumerate(train_outputs.columns):
        err = mean_absolute_error(train_outputs[var], pred_train_outputs[:,i])
        error["train"][var] = err
        print (f"Training error in {var}: ", err)
    for i, var in enumerate(valid_outputs.columns):
        err = mean_absolute_error(valid_outputs[var], pred_valid_outputs[:,i])
        error["valid"][var] = err
        print (f"Validation error in {var}: ", err)    
    
    # save results
    print("Saving results and config file..")
    with open(os.path.join(out_path, 'config.yml'), 'w') as f:
        yaml.dump(config, f)
    pd.DataFrame.from_dict(error, orient='index').to_csv(join(out_path, "error.csv"),index=False)
    pd.DataFrame(data=pred_valid_outputs).to_csv(join(out_path, "pred_valid_outputs.csv"), index=False)
    pd.DataFrame(data=pred_train_outputs).to_csv(join(out_path, "pred_train_outputs.csv"), index=False)
    pd.DataFrame(data=scaled_pred_train_outputs).to_csv(join(out_path, "scaled_pred_train_outputs.csv"), index=False)
    pd.DataFrame(data=scaled_pred_valid_outputs).to_csv(join(out_path, "scaled_pred_valid_outputs.csv"), index=False)

    return

if __name__ == "__main__":
    main()