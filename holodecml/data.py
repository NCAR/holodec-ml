from holodecml.propagation import UpsamplingPropagator
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix
from scipy.fftpack import fft2
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import xarray as xr
import numpy as np
import logging
import joblib
import random
import torch
import socket
import os
import warnings
warnings.filterwarnings("ignore")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


logger = logging.getLogger(__name__)


num_particles_dict = {
    1: ['1particle_gamma_600x400'],
    2: ['2particle_gamma_600x400'],
    3: ['3particle_gamma_600x400'],
    4: ['4particle_gamma_600x400'],
    5: ['5particle_gamma_600x400'],
    6: ['6particle_gamma_600x400'],
    7: ['7particle_gamma_600x400'],
    8: ['8particle_gamma_600x400'],
    9: ['9particle_gamma_600x400'],
    10: ['10particle_gamma_600x400'],
    '1-3': ['multiparticle'],
    '12-25': ['12-25particle_gamma_600x400'],
    '50-100': ['50-100particle_gamma'],
    'patches': ['10particle_gamma_512x512', 'patches128x128'],
    'real': ['real_holograms_CSET_RF07_20150719_200000-210000_512x512']
}

split_dict = {
    'train': 'training',
    'test': 'test',
    'valid': 'validation'}


def save_sparse_csr(filename, array):
    # note that .npz extension is added automatically
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


def load_sparse_csr(filename):
    # here we need to add .npz extension manually
    loader = np.load(filename + '.npz')
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])


def get_dataset_path():
    if 'casper' in socket.gethostname():
        return "/glade/p/cisl/aiml/ai4ess_hackathon/holodec/"
    else:
        return "/Users/ggantos/PycharmProjects/holodec-ml/data/"


def dataset_name(num_particles, split, file_extension='nc'):
    """
    Return the dataset filename given user inputs

    Args: 
        num_particles: (int or str) Number of particles per hologram
        split: (str) Dataset split of either 'train', 'valid', or 'test'
        file_extension: (str) Dataset file extension

    Returns:
        ds_name: (str) Dataset name
    """

    valid = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, '1-3',
             '12-25', '50-100', 'patches', 'real']
    if num_particles not in valid:
        raise ValueError("results: num_particles must be one of %r." % valid)

    if num_particles == 'real':
        ds_name = f'{num_particles_dict[num_particles][0]}.{file_extension}'
        return ds_name

    num_particles = num_particles_dict[num_particles]

    valid = ['train', 'test', 'valid']
    if split not in valid:
        raise ValueError("results: split must be one of %r." % valid)
    split = split_dict[split]
    if len(num_particles) > 1:
        ds_name = f'synthetic_holograms_{num_particles[0]}_{split}_{num_particles[1]}.{file_extension}'
    else:
        ds_name = f'synthetic_holograms_{num_particles[0]}_{split}.{file_extension}'

    return ds_name


def open_dataset(path_data, num_particles, split):
    """
    Opens a HOLODEC file

    Args: 
        path_data: (str) Path to dataset directory
        num_particles: (int or str) Number of particles per hologram
        split: (str) Dataset split of either 'train', 'valid', or 'test'

    Returns:
        ds: (xarray Dataset) Opened dataset
    """
    path_data = os.path.join(path_data, dataset_name(num_particles, split))
    ds = xr.open_dataset(path_data)
    return ds


def load_raw_datasets(path_data, num_particles, split, output_cols, subset, input_col="image"):
    """
    Given a path to training or validation datset, the number of particles per
    hologram, and output columns, returns raw inputs and outputs. Can specify
    a subset of the full dataset.

    Args: 
        path_data: (str) Path to dataset directory
        num_particles: (int or str) Number of particles per hologram 
        split: (str) Dataset split of either 'train', 'valid', or 'test'
        subset: (float or int) Fraction or int of data to be loaded
        output_cols: (list of strings) List of feature columns

    Returns:
        inputs: (np array) Input image data
        outputs: (df) Output data specified by output_cols 
    """

    ds = open_dataset(path_data, num_particles, split)
    if subset:
        if int(subset) < 1.0:
            ix = int(subset * ds[input_col].shape[0])
        else:
            ix = subset
        outputs = ds[output_cols].to_dataframe()
        outputs = outputs[outputs["hid"] < (ix+1)]
        if input_col == "patch":
            multiplier = int(ds["patch"].shape[0] // ds["image"].shape[0])
            ix *= multiplier
        inputs = ds[input_col][:ix].values
    else:
        inputs = ds[input_col].values
        outputs = ds[output_cols].to_dataframe()
    ds.close()
    return inputs, outputs


def scale_images(images, scaler_in=None):
    """
    Takes in array of images and scales pixel values between 0 and 1

    Args: 
        images: (np array) Input image data
        scaler_in: (dict) Image scaler 'max' and 'min' values

    Returns:
        images_scaled: (np array) Input image data scaled between 0 and 1
        scaler_in: (dict) Image scaler 'max' and 'min' values
    """

    if scaler_in is None:
        scaler_in = {}
        scaler_in["min"] = images.min()
        scaler_in["max"] = images.max()
    images_scaled = (images.astype(np.float32) - scaler_in["min"])
    images_scaled /= (scaler_in["max"] - scaler_in["min"])

    return images_scaled, scaler_in


def get_linspace(input_shape, num_bins=False):
    if num_bins:
        num_bins_x, num_bins_y = num_bins
    else:
        num_bins_x = input_shape[0]
        num_bins_y = input_shape[1]

    if input_shape == (600, 400):
        return np.linspace(-888, 888, num_bins_x), np.linspace(-592, 592, num_bins_y)
    if input_shape == (512, 512):
        return np.linspace(-757, 757, num_bins_x), np.linspace(-757, 757, num_bins_y)
    if input_shape == (1200, 800):
        return np.linspace(-1776, 1776, num_bins_x), np.linspace(-1776, 1776, num_bins_y)


def calc_z_relative_mass(outputs, num_z_bins=20, z_bins=None):
    """
    Calculate z-relative mass from particle data.

    Args: 
        outputs: (df) Output data specified by output_col 
        num_z_bins: (int) Number of bins for z-axis linspace
        z_bins: (np array) Bin linspace along the z-axis

    Returns:
        z_mass: (np array) Particle mass distribution by hologram along z-axis
        z_bins: (np array) Bin linspace along the z-axis
    """

    if z_bins is None:
        z_bins = np.linspace(outputs["z"].min() - 100,
                             outputs["z"].max() + 100,
                             num_z_bins)
    else:
        num_z_bins = z_bins.size
    holograms = len(outputs["hid"].unique())
    z_mass = np.zeros((holograms, num_z_bins), dtype=np.float32)
    for i in range(outputs.shape[0]):
        z_pos = np.searchsorted(z_bins, outputs.loc[i, "z"], side="right") - 1
        mass = 4 / 3 * np.pi * (outputs.loc[i, "d"]/2)**3
        z_mass[int(outputs.loc[i, "hid"]) - 1, z_pos] += mass
    z_mass /= np.expand_dims(z_mass.sum(axis=1), -1)
    return z_mass, z_bins


def calc_z_dist(outputs, num_z_bins=20, z_bins=None):
    """
    Calculate z distribution

    Args: 
        outputs: (df) Output data specified by output_col 
        num_z_bins: (int) Number of bins for z-axis linspace
        z_bins: (np array) Bin linspace along the z-axis

    Returns:
        z_dist: (np array) Particle z distribution by hologram along z-axis
        z_bins: (np array) Bin linspace along the z-axis
    """

    if z_bins is None:
        z_bins = np.linspace(outputs["z"].min() - 100,
                             outputs["z"].max() + 100,
                             num_z_bins)
    else:
        num_z_bins = z_bins.size
    holograms = len(outputs["hid"].unique())
    z_dist = np.zeros((holograms, num_z_bins), dtype=np.float32)
    for i in range(outputs.shape[0]):
        z_pos = np.searchsorted(z_bins, outputs.loc[i, "z"], side="right") - 1
        z_dist[int(outputs.loc[i, "hid"]) - 1, z_pos] += 1
    z_dist /= np.expand_dims(z_dist.sum(axis=1), -1)
    return z_dist, z_bins


def calc_z_bins(train_outputs, valid_outputs, num_z_bins):
    """
    Calculate z-axis linspace.

    Args: 
        train_outputs: (np array) Training output data 
        valid_outputs: (int) Validation output data
        num_z_bins: (int) Bin linspace along the z-axis

    Returns:
        z_bins: (np array) Bin linspace along the z-axis
    """
    z_min = np.minimum(train_outputs["z"].min(), valid_outputs["z"].min())
    z_max = np.maximum(train_outputs["z"].max(), valid_outputs["z"].max())
    z_bins = np.linspace(z_min, z_max, num_z_bins)
    return z_bins

# added this because the previous code allowed a different max_particle size
# depending on which split df was opened and the subset


def get_max_particles(path_data, num_particles, output_cols):
    ds = open_dataset(path_data, num_particles, "train")
    outputs = ds[output_cols].to_dataframe()
    max_particles = outputs['hid'].value_counts().max()
    return max_particles

# updated function to create the entire dataset template at one time to
# decrease overhead and eliminate setting random seeds


def make_template(df, num_images, max_particles):
    size = (num_images * max_particles, 1)
    x = np.random.uniform(low=df['x'].min(), high=df['x'].max(), size=size)
    y = np.random.uniform(low=df['y'].min(), high=df['y'].max(), size=size)
    z = np.random.uniform(low=df['z'].min(), high=df['z'].max(), size=size)
    d = np.random.uniform(low=df['d'].min(), high=df['d'].max(), size=size)
    prob = np.zeros(d.shape)
    template = np.hstack((x, y, z, d, prob))
    template = template.reshape((num_images, max_particles, -1))
    return template


def make_random_outputs(ds):
    num_images = ds.shape[0]
    max_particles = ds.shape[1]
    size = (num_images * max_particles, 1)
    x = np.random.uniform(low=np.min(
        ds[:, :, 0:1]), high=np.max(ds[:, :, 0:1]), size=size)
    y = np.random.uniform(low=np.min(
        ds[:, :, 1:2]), high=np.max(ds[:, :, 1:2]), size=size)
    z = np.random.uniform(low=np.min(
        ds[:, :, 2:3]), high=np.max(ds[:, :, 2:3]), size=size)
    d = np.random.uniform(low=np.min(
        ds[:, :, 3:4]), high=np.max(ds[:, :, 3:4]), size=size)
    template = np.hstack((x, y, z, d))
    template = template.reshape((num_images, max_particles, -1))
    return template

# cycles through dataset by "hid" to overwrite random data generated in
# make_template with actual data and classification of 1


def outputs_3d(outputs, num_images, max_particles):
    outputs_array = make_template(outputs, num_images, max_particles)
    for hid in v:  # the variable v here is undefined
        outputs_hid = outputs.loc[outputs['hid'] == hid].to_numpy()
        outputs_hid[:, -1] = 1
        outputs_array[int(hid-1), :outputs_hid.shape[0], :] = outputs_hid
    return outputs_array


def unet_bin(inputs, outputs, bin_factor):

    if not bin_factor:
        num_bins_x = inputs.shape[1]
        num_bins_y = inputs.shape[2]
    else:
        num_bins_x = inputs.shape[1] // bin_factor
        num_bins_y = inputs.shape[2] // bin_factor

    unet_outputs = []
    for hid in outputs["hid"].unique():
        outputs_hid = outputs.loc[outputs['hid'] == hid]
        x_linspace, y_linspace = get_linspace(inputs[0].shape)
        xs_hid = np.digitize(outputs_hid['x'], x_linspace)
        ys_hid = np.digitize(outputs_hid['y'], y_linspace)
        zs_hid = outputs_hid['z'].values
        ds_hid = outputs_hid['d'].values
        # sort coordinates on first x-axis then y-axis
        coords_hid = np.array(list(zip(xs_hid, ys_hid)))
        coords_hid = coords_hid[np.lexsort(
            (coords_hid[:, 1], coords_hid[:, 0]))]
        unique, unique_idx, unique_counts = np.unique(
            coords_hid, return_index=True, return_counts=True, axis=0)
        # ensure duplicate coordinates use the d and z values closest to the camera (smaller z-value)
        for i in np.argwhere(unique_counts > 1):
            # find indices in original coordinates where there are multiple particles
            idx_equal = np.where((coords_hid == unique[i][0]).all(axis=1))[0]
            # find the index of the particle with the z that is closest to the camera
            idx_max = np.argmin(zs_hid[np.min(idx_equal):np.max(idx_equal)+1])
            unique_idx[i] = idx_equal[idx_max]
        z_hid = zs_hid[unique_idx]
        d_hid = ds_hid[unique_idx]
        # create three images and stack together
        xy = np.zeros((num_bins_x, num_bins_y))
        xy[unique[:, 0], unique[:, 1]] = 1
        z = np.zeros((num_bins_x, num_bins_y))
        z[unique[:, 0], unique[:, 1]] = z_hid
        d = np.zeros((num_bins_x, num_bins_y))
        d[unique[:, 0], unique[:, 1]] = d_hid
        unet_outputs.append(np.stack((xy, z, d), axis=-1))
    unet_outputs = np.stack(unet_outputs, axis=0)
    return unet_outputs


def load_unet_datasets(path_data, num_particles, output_cols,
                       scaler_out=False, subset=False, bin_factor=False,
                       input_col="image"):

    train_inputs,\
        train_outputs = load_raw_datasets(path_data, num_particles, 'train',
                                          output_cols, subset, input_col=input_col)
    valid_inputs,\
        valid_outputs = load_raw_datasets(path_data, num_particles, 'valid',
                                          output_cols, subset, input_col=input_col)

    train_inputs, scaler_in = scale_images(train_inputs)
    valid_inputs, _ = scale_images(valid_inputs, scaler_in)

    if scaler_out:
        train_outputs[["z", "d"]] = scaler_out.fit_transform(
            train_outputs[["z", "d"]])
        valid_outputs[["z", "d"]] = scaler_out.transform(
            valid_outputs[["z", "d"]])

    train_outputs = unet_bin(train_inputs, train_outputs, bin_factor)
    valid_outputs = unet_bin(valid_inputs, valid_outputs, bin_factor)

    return train_inputs, train_outputs, valid_inputs, valid_outputs


def make_gaussian(size, fwhm=3, center=None):
    """ Make a square gaussian kernel.
    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)


def unet_bin_xy(inputs, outputs, bin_factor):

    if not bin_factor:
        num_bins_x = inputs.shape[1]
        num_bins_y = inputs.shape[2]
    else:
        num_bins_x = inputs.shape[1] // bin_factor
        num_bins_y = inputs.shape[2] // bin_factor

    unet_outputs = []
    for hid in outputs["hid"].unique():
        outputs_hid = outputs.loc[outputs['hid'] == hid]
        x_linspace, y_linspace = get_linspace(
            inputs[0].shape, (num_bins_x, num_bins_y))
        xs_hid = np.digitize(outputs_hid['x'], x_linspace)
        ys_hid = np.digitize(outputs_hid['y'], y_linspace)

        # sort coordinates on first x-axis then y-axis, eliminate non-unique
        coords_hid = np.array(list(zip(xs_hid, ys_hid)))
        coords_hid = coords_hid[np.lexsort(
            (coords_hid[:, 1], coords_hid[:, 0]))]
        unique, unique_idx, unique_counts = np.unique(coords_hid,
                                                      return_index=True,
                                                      return_counts=True,
                                                      axis=0)
        xy = np.zeros((num_bins_x, num_bins_y))
        xy[unique[:, 0], unique[:, 1]] = 1
        unet_outputs.append(np.expand_dims(xy, axis=-1))
    unet_outputs = np.stack(unet_outputs, axis=0)
    return unet_outputs


def unet_bin_xy_gauss(inputs, outputs, bin_factor, gauss=False,
                      gauss_size=5, gauss_rad=2):

    if not bin_factor:
        num_bins_x = inputs.shape[1]
        num_bins_y = inputs.shape[2]
    else:
        num_bins_x = inputs.shape[1] // bin_factor
        num_bins_y = inputs.shape[2] // bin_factor

    if gauss in ["z", "d"]:
        scaler = MinMaxScaler((2, 10))
        gauss_rads = outputs[gauss].to_numpy().reshape(-1, 1)
        outputs["gauss_rads"] = scaler.fit_transform(gauss_rads).astype(int)
    if gauss == True:
        gaussarray = make_gaussian(gauss_size, gauss_rad)

    unet_outputs = []
    for hid in outputs["hid"].unique():
        outputs_hid = outputs.loc[outputs['hid'] == hid]
        x_linspace, y_linspace = get_linspace(
            inputs[0].shape, (num_bins_x, num_bins_y))
        xs_hid = np.digitize(outputs_hid['x'], x_linspace)
        ys_hid = np.digitize(outputs_hid['y'], y_linspace)

        # sort coordinates on first x-axis then y-axis
        if gauss in ["z", "d"]:
            coords_hid = np.array(
                list(zip(xs_hid, ys_hid, outputs_hid['gauss_rads'])))
        else:
            coords_hid = np.array(list(zip(xs_hid, ys_hid)))
        coords_hid = coords_hid[np.lexsort(
            (coords_hid[:, 1], coords_hid[:, 0]))]
        unique, unique_idx, unique_counts = np.unique(coords_hid,
                                                      return_index=True,
                                                      return_counts=True,
                                                      axis=0)
        # create three images and stack together
        if gauss in ["z", "d"]:
            max_gauss_rad = max(unique[:, 2])
            xy = np.zeros((num_bins_x + max_gauss_rad*2,
                          num_bins_y + max_gauss_rad*2))
            for x, y, gauss_rad in unique:
                x_max = x + 1 + 2*gauss_rad
                y_max = y + 1 + 2*gauss_rad
                gaussarray = make_gaussian(1+2*gauss_rad, gauss_rad)
                xy[x:x_max, y:y_max] += gaussarray
            xy = xy[max_gauss_rad:xy.shape[0]-max_gauss_rad,
                    max_gauss_rad:xy.shape[1]-max_gauss_rad]
        else:
            xy = np.zeros((num_bins_x + gauss_rad*2, num_bins_y + gauss_rad*2))
            for x, y in unique:
                x_max = x + 1 + 2*gauss_rad
                y_max = y + 1 + 2*gauss_rad
                xy[x:x_max, y:y_max] += gaussarray
            xy = xy[gauss_rad:xy.shape[0]-gauss_rad,
                    gauss_rad:xy.shape[1]-gauss_rad]

        unet_outputs.append(np.expand_dims(xy, axis=-1))
    unet_outputs = np.stack(unet_outputs, axis=0)

    return unet_outputs


def load_unet_datasets_xy(path_data, num_particles, output_cols,
                          subset=False, bin_factor=False, input_col="image",
                          gauss=False):

    train_inputs,\
        train_outputs = load_raw_datasets(path_data, num_particles, 'train',
                                          output_cols, subset, input_col=input_col)
    valid_inputs,\
        valid_outputs = load_raw_datasets(path_data, num_particles, 'valid',
                                          output_cols, subset, input_col=input_col)

    train_inputs, scaler_in = scale_images(train_inputs)
    valid_inputs, _ = scale_images(valid_inputs, scaler_in)

    if gauss:
        train_outputs = unet_bin_xy_gauss(
            train_inputs, train_outputs, bin_factor, gauss)
        valid_outputs = unet_bin_xy_gauss(
            valid_inputs, valid_outputs, bin_factor, gauss)
    else:
        train_outputs = unet_bin_xy(train_inputs, train_outputs, bin_factor)
        valid_outputs = unet_bin_xy(valid_inputs, valid_outputs, bin_factor)

    return train_inputs, train_outputs, valid_inputs, valid_outputs


def load_unet_datasets_xy_1to25(path_data, num_particles, output_cols,
                                subset=False, bin_factor=False, input_col="image"):

    train_inputs_list = []
    train_outputs_list = []
    valid_inputs_list = []
    valid_outputs_list = []
    for num, sub in zip(num_particles, subset):
        train_inputs,\
            train_outputs = load_raw_datasets(path_data, num, 'train',
                                              output_cols, sub)
        valid_inputs,\
            valid_outputs = load_raw_datasets(path_data, num, 'valid',
                                              output_cols, sub//10)

        train_inputs, scaler_in = scale_images(train_inputs)
        valid_inputs, _ = scale_images(valid_inputs, scaler_in)

        train_outputs = unet_bin_xy(train_inputs, train_outputs, bin_factor)
        valid_outputs = unet_bin_xy(valid_inputs, valid_outputs, bin_factor)

        train_inputs_list.append(train_inputs)
        train_outputs_list.append(train_outputs)
        valid_inputs_list.append(valid_inputs)
        valid_outputs_list.append(valid_outputs)

    train_inputs = np.vstack(train_inputs_list)
    train_outputs = np.vstack(train_outputs_list)
    valid_inputs = np.vstack(valid_inputs_list)
    valid_outputs = np.vstack(valid_outputs_list)

    return train_inputs, train_outputs, valid_inputs, valid_outputs


def load_scaled_datasets(path_data, num_particles, output_cols,
                         scaler_out=False, subset=False, num_z_bins=False,
                         mass=False):
    """
    Given a path to training or validation datset, the number of particles per
    hologram, and output columns, returns scaled inputs and raw outputs.

    Args: 
        path_data: (str) Path to dataset directory
        num_particles: (int or str) Number of particles per hologram
        output_cols: (list of strings) List of feature columns
        scaler_out: (sklearn.preprocessing scaler) Output data scaler
        subset: (float) Fraction of data to be loaded
        num_z_bins: (int) Number of bins along z-axis
        mass: (boolean) If True, calculate particle mass on z-axis

    Returns:
        train_inputs: (np array) Train input data scaled between 0 and 1
        train_outputs: (np array) Scaled train output data
        valid_inputs: (np array) Valid input data scaled between 0 and 1
        valid_outputs: (np array) Scaled valid output data
    """
    train_inputs,\
        train_outputs = load_raw_datasets(path_data, num_particles, 'train',
                                          output_cols, subset)
    valid_inputs,\
        valid_outputs = load_raw_datasets(path_data, num_particles, 'valid',
                                          output_cols, subset)

    train_inputs, scaler_in = scale_images(train_inputs)
    valid_inputs, _ = scale_images(valid_inputs, scaler_in)
    train_inputs = np.expand_dims(train_inputs, -1)
    valid_inputs = np.expand_dims(valid_inputs, -1)

    if num_z_bins:
        z_bins = calc_z_bins(train_outputs, valid_outputs, num_z_bins)
        if mass:
            train_outputs, _ = calc_z_relative_mass(outputs=train_outputs,
                                                    z_bins=z_bins)
            valid_outputs, _ = calc_z_relative_mass(outputs=valid_outputs,
                                                    z_bins=z_bins)
        else:
            train_outputs, _ = calc_z_dist(outputs=train_outputs,
                                           z_bins=z_bins)
            valid_outputs, _ = calc_z_dist(outputs=valid_outputs,
                                           z_bins=z_bins)
    else:
        if train_inputs.shape[0] != train_outputs.shape[0]:
            col = [c for c in output_cols if c != 'hid']
            max_particles = get_max_particles(
                path_data, num_particles, output_cols)
            train_outputs[col] = scaler_out.fit_transform(train_outputs[col])
            train_outputs = outputs_3d(train_outputs, train_inputs.shape[0],
                                       max_particles)
            valid_outputs[col] = scaler_out.transform(valid_outputs[col])
            valid_outputs = outputs_3d(valid_outputs, valid_inputs.shape[0],
                                       max_particles)
        else:
            train_outputs = train_outputs.drop(['hid'], axis=1)
            train_outputs = scaler_out.fit_transform(train_outputs)
            valid_outputs = valid_outputs.drop(['hid'], axis=1)
            valid_outputs = scaler_out.transform(valid_outputs)

    return train_inputs, train_outputs, valid_inputs, valid_outputs


def load_train_patches(path_data, num_particles, output_cols,
                       scaler_out=False, subset=False, rad=False,
                       scale_image=True, FFT=False):
    '''Creates hologram patches centered around true particles.'''

    # load raw datasets
    train_inputs,\
        train_outputs = load_raw_datasets(path_data, num_particles, 'train',
                                          output_cols, subset)
    valid_inputs,\
        valid_outputs = load_raw_datasets(path_data, num_particles, 'valid',
                                          output_cols, subset)

    # scale images
    if scale_image:
        train_inputs, scaler_in = scale_images(train_inputs)
        valid_inputs, _ = scale_images(valid_inputs, scaler_in)

    if FFT:
        train_inputs = fft2(train_inputs)
        valid_inputs = fft2(valid_inputs)

    # bin x and y coordinates
    x_linspace, y_linspace = get_linspace(train_inputs[0].shape)
    train_outputs["x_bin"] = np.digitize(train_outputs['x'], x_linspace)
    train_outputs["y_bin"] = np.digitize(train_outputs['y'], y_linspace)
    valid_outputs["x_bin"] = np.digitize(valid_outputs['x'], x_linspace)
    valid_outputs["y_bin"] = np.digitize(valid_outputs['y'], y_linspace)

    # excise patches from zero-padded holograms for training set
    train_patches = []
    for hid in train_outputs["hid"].unique().astype(int):
        input_hid = np.pad(train_inputs[hid-1], rad)
        outputs_hid = train_outputs.loc[train_outputs['hid'] == hid]
        for _, row in outputs_hid.iterrows():
            idx_x, idx_y = int(row["x_bin"])+rad, int(row["y_bin"])+rad
            patch = input_hid[idx_x-rad:idx_x+rad+1, idx_y-rad:idx_y+rad+1]
            train_patches.append(patch)
    train_patches = np.stack(train_patches)

    # excise patches from zero-padded holograms for validation set
    valid_patches = []
    for hid in valid_outputs["hid"].unique().astype(int):
        input_hid = np.pad(valid_inputs[hid-1], rad)
        outputs_hid = valid_outputs.loc[valid_outputs['hid'] == hid]
        for _, row in outputs_hid.iterrows():
            idx_x, idx_y = int(row["x_bin"])+rad, int(row["y_bin"])+rad
            patch = input_hid[idx_x-rad:idx_x+rad+1, idx_y-rad:idx_y+rad+1]
            valid_patches.append(patch)
    valid_patches = np.stack(valid_patches)

    train_hids = train_outputs[['hid', 'x_bin', 'y_bin']].to_numpy()

    train_outputs = train_outputs.drop(
        ['hid', 'x_bin', 'y_bin', 'x', 'y'], axis=1)
    train_outputs = scaler_out.fit_transform(train_outputs)

    valid_hids = valid_outputs[['hid', 'x_bin', 'y_bin']].to_numpy()
    valid_outputs = valid_outputs.drop(
        ['hid', 'x_bin', 'y_bin',  'x', 'y'], axis=1)
    valid_outputs = scaler_out.transform(valid_outputs)

    return train_patches, train_outputs, train_hids, valid_patches, valid_outputs, valid_hids


def load_train_patches_1to25(path_data, num_particles, output_cols,
                             scaler_out=False, subset=False, rad=False,
                             scale_image=True, FFT=False):

    train_inputs_list = []
    train_outputs_list = []
    train_hids_list = []
    valid_inputs_list = []
    valid_outputs_list = []
    valid_hids_list = []
    for num, sub in zip(num_particles, subset):
        train_inputs,\
            train_hids,\
            train_outputs,\
            valid_inputs,\
            valid_hids,\
            valid_outputs = load_train_patches(path_data,
                                               num,
                                               output_cols,
                                               scaler_out,
                                               sub,
                                               rad,
                                               scale_image,
                                               FFT)
        train_inputs_list.append(train_inputs)
        train_outputs_list.append(train_outputs)
        train_hids_list.append(train_hids)
        valid_inputs_list.append(valid_inputs)
        valid_outputs_list.append(valid_outputs)
        valid_hids_list.append(valid_hids)

    train_inputs = np.vstack(train_inputs_list)
    train_outputs = np.vstack(train_outputs_list)
    train_hids = np.vstack(train_hids_list)
    valid_inputs = np.vstack(valid_inputs_list)
    valid_outputs = np.vstack(valid_outputs_list)
    valid_hids = np.vstack(valid_hids_list)

    return train_inputs, train_outputs, train_hids, valid_inputs, valid_outputs, valid_hids


class PickleReader(Dataset):

    def __init__(self,
                 fn,
                 max_buffer_size=5000,
                 max_images=40000,
                 color_dim=2,
                 shuffle=True,
                 transform=False):

        self.fn = fn
        self.buffer = []
        self.max_buffer_size = max_buffer_size
        self.shuffle = shuffle
        self.max_images = max_images
        self.transform = transform
        self.color_dim = color_dim

        self.fid = open(self.fn, "rb")
        self.loaded = 0
        self.epoch = 0

    def __getitem__(self, idx):

        self.on_epoch_end()

        while True:

            try:
                data = joblib.load(self.fid)
                image, label, mask = data
                image = image[:self.color_dim]

                im = {
                    "image": image,
                    "horizontal_flip": False,
                    "vertical_flip": False
                }

                if self.transform:
                    for image_transform in self.transform:
                        im = image_transform(im)

                image = im["image"]
                mask = mask.toarray()

                # Update the mask if we flipped the original image
                if im["horizontal_flip"]:
                    mask = np.flip(mask, axis=0)
                if im["vertical_flip"]:
                    mask = np.flip(mask, axis=1)

                image = torch.tensor(image, dtype=torch.float)
                #image = torch.FloatTensor(image)
                #label = torch.LongTensor([label])
                #mask = torch.FloatTensor(mask.copy())
                mask = torch.tensor(mask.copy(), dtype=torch.int)

                #######

                data = (image, mask)

                self.loaded += 1

                if not self.shuffle:
                    return data

                self.buffer.append(data)
                random.shuffle(self.buffer)

                if len(self.buffer) > self.max_buffer_size:
                    self.buffer = self.buffer[:self.max_buffer_size]

                if self.epoch > 0:
                    return self.buffer.pop()

                else:  # wait until all data has been seen before sampling from the buffer
                    return data

            except EOFError:
                self.fid = open(self.fn, "rb")
                self.loaded = 0
                continue

    def __len__(self):
        return self.max_images

    def on_epoch_end(self):
        if self.loaded == self.__len__():
            self.fid = open(self.fn, "rb")
            self.loaded = 0
            self.epoch += 1


class UpsamplingReader(Dataset):

    def __init__(self, conf=None, transform=None, max_size=10000, device="cpu"):

        config = conf["data"]
        n_bins = config["n_bins"]
        data_path = config["data_path"]
        tile_size = config["tile_size"]  # size of tiled images in pixels
        # amount that we shift the tile to make a new tile
        step_size = config["step_size"]
        # UNET gaussian marker width (standard deviation) in um
        marker_size = config["marker_size"]
        transform_mode = "None" if "transform_mode" not in config else config["transform_mode"]

        self.part_per_holo = config["total_positive"]
        self.empt_per_holo = config["total_negative"]
        self.color_dim = conf["model"]["in_channels"]

        self.prop = UpsamplingPropagator(
            data_path,
            n_bins=n_bins,
            tile_size=tile_size,
            step_size=step_size,
            marker_size=marker_size,
            transform_mode=transform_mode,
            device=device
        )

        self.xy = []
        self.max_size = max_size
        self.transform = transform

    def __getitem__(self, h_idx):

        h_idx = int(self.prop.h_ds["hid"].values[h_idx]) - 1

        if len(self.xy) > 0:
            random.shuffle(self.xy)
            x, y = self.xy.pop()
            return x, y

        data = self.prop.get_reconstructed_sub_images(
            h_idx, self.part_per_holo, self.empt_per_holo
        )
        for idx in range(len(data[0])):
            # result_dict["label"].append(int(data[0][idx]))
            image = np.expand_dims(np.abs(data[1][idx]), 0)
            if self.color_dim == 2:
                phase = np.expand_dims(np.angle(data[1][idx]), 0)
                image = np.vstack([image, phase])
            mask = data[4][idx]
            image, mask = self.apply_transforms(image, mask)
            self.xy.append((image, mask))

        random.shuffle(self.xy)
        x, y = self.xy.pop()
        return x, y

    def apply_transforms(self, image, mask):

        if self.transform == None:

            image = torch.tensor(image, dtype=torch.float)
            mask = torch.tensor(mask, dtype=torch.int)

            return image, mask

        im = {
            "image": image,
            "horizontal_flip": False,
            "vertical_flip": False
        }

        for image_transform in self.transform:
            im = image_transform(im)

        # Update the mask if we flipped the original image
        if im["horizontal_flip"]:
            mask = np.flip(mask, axis=0)
        if im["vertical_flip"]:
            mask = np.flip(mask, axis=1)

        image = torch.tensor(im["image"], dtype=torch.float)
        mask = torch.tensor(mask.copy(), dtype=torch.int)

        return image, mask

    def __len__(self):
        return len(list(self.prop.h_ds["hid"].values))
