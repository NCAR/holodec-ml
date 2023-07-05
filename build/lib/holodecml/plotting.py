import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import xarray as xr
import os
from os.path import join
import yaml
import argparse

from holodecml.data import load_raw_datasets, load_unet_datasets, load_unet_datasets_xy


def main():
    
    print("Parsing config...")
        
    # parse arguments from config/yaml file
    parser = argparse.ArgumentParser(description='Describe plotting parameters')
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
    bin_factor = config["bin_factor"]
    
    # load data
    print("Loading data...")
    train_inputs,\
    train_outputs,\
    valid_inputs,\
    valid_outputs = load_unet_datasets_xy(path_data,
                                         num_particles,
                                         output_cols,
                                         config["subset"],
                                         bin_factor)

    h = config["h"]
    valid_outputs_pred = xr.open_dataset(join(path_save, "valid_outputs_pred.nc"))
    valid_outputs_pred = valid_outputs_pred.to_array().values[0]

    image_pred = valid_outputs_pred[h, :, :, 0]
    image_true = valid_outputs[h, :, :, 0]

    coords_true = np.where(image_true > 0)

    idx = np.argwhere(np.diff(np.sort(valid_outputs_pred[h, :, :, 0].flatten())) > .0001)+1
    pred_argsort = valid_outputs_pred[h, :, :, 0].flatten().argsort()
    coords_pred = []
    for i in pred_argsort[-idx.shape[0]:][::-1]:
        coord = np.array([c[0] for c in np.where(image_pred == image_pred.flatten()[i])])
        coords_pred.append(coord)
    coords_pred = np.stack(coords_pred)

    print("Plotting...")
    # Plot 1
    fig=plt.figure(figsize=(12, 8))
    plt.pcolormesh(np.log(valid_outputs_pred[h, :, :, 0]).T, cmap="RdBu_r")
    plt.colorbar()
    plt.scatter(np.where(image_true > 0)[0], np.where(image_true > 0)[1], color='blue', s=100, label="True")
    plt.title(f'Log of probability field for validation hologram {h}', fontsize=20)
    plt.legend(fontsize=20)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(join(path_save, "prob_field_log.png"), dpi=200, bbox_inches="tight")
    
    # Plot 2
    plt.figure(figsize=(12, 8))
    x_vals = np.linspace(0, valid_inputs.shape[1]/bin_factor, valid_inputs[h, :, :].shape[0])
    y_vals = np.linspace(0, valid_inputs.shape[2]/bin_factor, valid_inputs[h, :, :].shape[1])
    plt.xticks([])
    plt.yticks([])
    plt.pcolormesh(x_vals, y_vals, valid_inputs[h, :, :].T, cmap="RdBu_r")
    plt.scatter(np.where(image_true > 0)[0], np.where(image_true > 0)[1], color='blue', s=100, label="True", zorder=2)
    plt.scatter(coords_pred[:, 0], coords_pred[:, 1], color='red', s=100, label="Predicted", zorder=1)
    plt.legend(fontsize=20)
    plt.title(f'{int(np.sum(image_true))} True vs Top {idx.shape[0]} Predicted Particles for validation hologram {h}', fontsize=20)
    plt.savefig(join(path_save, "true_vs_pred_diff.png"), dpi=200, bbox_inches="tight")

    # Plot 3
    pred_argsort = valid_outputs_pred[h, :, :, 0].flatten().argsort()
    coords_pred = []
    for i in pred_argsort[-int(np.sum(image_true)):][::-1]:
        coord = np.array([c[0] for c in np.where(image_pred == image_pred.flatten()[i])])
        coords_pred.append(coord)
    coords_pred = np.stack(coords_pred)

    plt.figure(figsize=(12, 8))
    x_vals = np.linspace(0, valid_inputs.shape[1]/bin_factor, valid_inputs[h, :, :].shape[0])
    y_vals = np.linspace(0, valid_inputs.shape[2]/bin_factor, valid_inputs[h, :, :].shape[1])
    plt.xticks([])
    plt.yticks([])
    plt.pcolormesh(x_vals, y_vals, valid_inputs[h, :, :].T, cmap="RdBu_r")
    plt.scatter(np.where(image_true > 0)[0], np.where(image_true > 0)[1], color='blue', s=100, label="True", zorder=2)
    plt.scatter(coords_pred[:, 0], coords_pred[:, 1], color='red', s=100, label="Predicted", zorder=1)
    plt.legend(fontsize=20)
    plt.title(f'{int(np.sum(image_true))} True vs Top {int(np.sum(image_true))} Predicted Particles for validation hologram {h}', fontsize=20)
    plt.savefig(join(path_save, "true_vs_pred_toptrue.png"), dpi=200, bbox_inches="tight")

    # Plot 4
    fig=plt.figure(figsize=(12, 8))
    plt.imshow(valid_outputs[h, :, :, 0].T, interpolation='bilinear', cmap=plt.cm.gray, aspect='auto', vmin=0, vmax=1)
    plt.title(f'True probability field for validation hologram {h}\nSum of non-zero values: {np.sum(valid_outputs[h, :, :, 0]):.2f}\nMax predicted value: {np.max(valid_outputs[h, :, :, 0]):.2f}', fontsize=20)
    plt.savefig(join(path_save, "prob_true.png"), dpi=200, bbox_inches="tight")

    # Plot 5
    fig=plt.figure(figsize=(12, 8))
    plt.imshow(valid_outputs_pred[h, :, :, 0].T, interpolation='bilinear', cmap=plt.cm.gray, aspect='auto', vmin=0, vmax=1)
    plt.title(f'Predicted probability field for validation hologram {h}\nSum of non-zero values: {np.sum(valid_outputs_pred[h, :, :, 0]):.2f}\nMax predicted value: {np.max(valid_outputs_pred[h, :, :, 0]):.2f}', fontsize=20)
    plt.savefig(join(path_save, "prob_pred.png"), dpi=200, bbox_inches="tight")

if __name__ == "__main__":
    main()