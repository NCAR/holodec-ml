import sys
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np
import argparse
import yaml
import os
from os.path import join, exists

sys.path.append('../../')
    
from library.data import load_scaled_datasets
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

    out_path = config["out_path"]
    np.random.seed(config["random_seed"])
    if not exists(out_path):
        os.makedirs(out_path)
    
    # load data
    input_scaler = scalers[config["input_scaler"]]()

    scaled_train_inputs, \
    scaled_valid_inputs, \
    scaled_train_outputs, \
    scaled_valid_outputs, \
    input_scaler = load_scaled_datasets(config["data_path"],
                                        config["num_particles"],
                                        config["output_cols"],
                                        input_scaler)
    
    # train the model
    mod = Conv2DNeuralNetwork(**config["conv2d_network"])
    mod.fit(scaled_train_inputs.values, scaled_train_outputs.values)
    print("Saving the model")
    mod.model.save(join(out_path, config["model_name"], ".h5"))
    
    # predict outputs
    scaled_pred_valid_outputs = pd.DataFrame(mod.predict(scaled_valid_inputs),
                                             index=scaled_valid_outputs.index,
                                             columns=scaled_valid_outputs.columns)
    scaled_pred_train_outputs = pd.DataFrame(mod.predict(scaled_train_inputs),
                                             index=scaled_train_outputs.index,
                                             columns=scaled_train_outputs.columns)
    
    train_error = metrics[config["metric"]](scaled_train_outputs, scaled_pred_train_outputs)
    valid_error = metrics[config["metric"]](scaled_valid_outputs, scaled_pred_valid_outputs)
    print(f"Training Error of: {train_error}")
    print(f"Validation Error of: {valid_error}")

    pred_valid_outputs = input_scaler.inverse_transform(scaled_pred_valid_outputs)
    pred_train_outputs = input_scaler.inverse_transform(scaled_pred_train_outputs)    
    
    print("Saving the data")
    pd.DataFrame(data={'train_error': train_error}, index=[0]).to_csv(join(out_path, "conv2d_train_error.csv"), index_label="Output")
    pd.DataFrame(data={'valid_error': valid_error}, index=[0]).to_csv(join(out_path, "conv2d_valid_error.csv"), index_label="Output")
    pd.DataFrame(data=pred_valid_outputs).to_csv(join(out_path, "pred_valid_outputs.csv"), index_label="index")
    pd.DataFrame(data=pred_train_outputs).to_csv(join(out_path, "pred_train_outputs.csv"), index_label="index")
    pd.DataFrame(data=scaled_pred_train_outputs).to_csv(join(out_path, "scaled_pred_train_outputs.csv"), index_label="index")
    pd.DataFrame(data=scaled_pred_valid_outputs).to_csv(join(out_path, "scaled_pred_valid_outputs.csv"), index_label="index")
    return

if __name__ == "__main__":
    main()