from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, max_error, mean_squared_error
import pandas as pd
import numpy as np
import argparse
import random
import yaml
import xarray as xr
import os
from os.path import join
from datetime import datetime
import keras
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from holodecml.data import load_scaled_datasets, make_random_valid_outputs
from holodecml.models import ParticleAttentionNet
from holodecml.losses import attention_net_loss, attention_net_validation_loss
from holodecml.callbacks import get_callbacks
import optuna
from aimlutils.hyper_opt.utils import trial_suggest_loader
from aimlutils.hyper_opt.base_objective import *
from aimlutils.hyper_opt.utils import KerasPruningCallback


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

def create_model(trial, config):

    # Get list of hyperparameters from the config
    hyperparameters = config["optuna"]["parameters"]

    # Now update some hyperparameters via custom rules
    attention_neurons = trial_suggest_loader(trial, hyperparameters["attention_neurons"])
    hidden_layers = trial_suggest_loader(trial, hyperparameters["hidden_layers"])
    hidden_neurons = trial_suggest_loader(trial, hyperparameters["hidden_neurons"])
    min_filters = trial_suggest_loader(trial, hyperparameters["min_filters"])
    lr = trial_suggest_loader(trial, hyperparameters['learning_rate'])

    # We define our MLP.
    net = ParticleAttentionNet(attention_neurons=attention_neurons,
                               hidden_layers=hidden_layers,
                               hidden_neurons=hidden_neurons,
                               min_filters=min_filters,
                               **config["attention_network"])

    # We compile our model with a sampled learning rate.
    net.compile(optimizer=Adam(lr=lr), loss=attention_net_loss,
               metrics=[attention_net_validation_loss])
    return net


def objective(trial, config):
    # Clear clutter from previous session graphs.
    keras.backend.clear_session()

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
    train_inputs, \
    train_outputs, \
    valid_inputs, \
    valid_outputs = load_scaled_datasets(path_data,
                                         num_particles,
                                         output_cols,
                                         scaler_out,
                                         config["subset"],
                                         config["num_z_bins"],
                                         config["mass"])

    # add noise to the outputs
    train_outputs_noisy = train_outputs * (1 + np.random.normal(0, config['noisy_sd'], train_outputs.shape))
    valid_outputs_noisy = make_random_valid_outputs(path_data, num_particles,
                                                    valid_inputs.shape[0],
                                                    train_outputs.shape[1])
    valid_outputs_noisy = valid_outputs_noisy * (1 + np.random.normal(0, config['noisy_sd'], valid_outputs_noisy.shape))

    # Generate our trial model.
    model_start = datetime.now()
    net = create_model(trial, config)

    # Fit the model on the training data.
    # The KerasPruningCallback checks for pruning condition every epoch.
    callbacks = get_callbacks(config["callbacks"]) + [KerasPruningCallback(trial, "val_loss")]
    hist = net.fit([train_outputs_noisy[:,:,:-1], train_inputs], train_outputs,
                   validation_data=([valid_outputs_noisy[:,:,:-1], valid_inputs], valid_outputs),
                   epochs=config["train"]['epochs'],
                   batch_size=config["train"]['batch_size'],
                   callbacks=callbacks,
                   verbose=config["train"]['verbose'])
    print(f"Running model took {datetime.now() - model_start} time")

    # predict outputs
    print("Predicting outputs..")
    valid_outputs_pred = net.predict([valid_outputs_noisy[:,:,:-1], valid_inputs],
                                     batch_size=config['train']["batch_size"])
    raw = valid_outputs_pred.reshape(-1, valid_outputs_pred.shape[-1])
    raw = scaler_out.inverse_transform(raw[:, :-1])
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
        np.savetxt(join(path_save, k + ".csv"), hist.history[k])
    with open(join(path_save, 'config.yml'), 'w') as f:
        yaml.dump(config, f)
    print("Finished saving results.")

    # Evaluate the model accuracy on the validation set.
    score = attention_net_validation_loss(valid_outputs, valid_outputs_pred)

    return score

class Objective(BaseObjective):

    def __init__(self, study, config, metric = "val_loss", device = "cpu"):

        # Initialize the base class
        BaseObjective.__init__(self, study, config, metric, device)

    def train(self, trial, conf):

        result = objective(trial, conf)

        results_dictionary = {
            "val_loss": result
        }
        return save(trial, results_dictionary)

if __name__ == "__main__":

    print("Starting script...")

    # parse arguments from config/yaml file
    parser = argparse.ArgumentParser(description='Describe a Conv2D nn')
    parser.add_argument("config", help="Path to config file")
    args = parser.parse_args()
    with open(args.config) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    study = optuna.create_study(direction=config["direction"],
                                pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, config, n_trials=config["n_trials"])
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
