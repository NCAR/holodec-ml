from holodecml.propagation import WavePropagator
import gc
import os
import sys
import tqdm
import yaml
import joblib
import random
import logging
import torch
import torch.fft
import xarray as xr
import torch.nn.functional as F
import numpy as np
from scipy.signal import convolve2d
from collections import defaultdict
from functools import partial
from scipy.sparse import csr_matrix
import multiprocessing as mp
import warnings
warnings.filterwarnings("ignore")

#mp.set_start_method('spawn')


# Set the default logger
logger = logging.getLogger(__name__)


class CustomWavePropagator(WavePropagator):

    def get_reconstructed_sub_images(self, h_idx, part_per_holo=None, empt_per_holo=None):
        """
        Reconstruct a hologram at specific planes to provide training data
        with a specified number of sub images containing and not containing
        particles
        """

        with torch.no_grad():

            #### roughly half of the empty cases should be near in focus ####
            empt_near_cnt = empt_per_holo//2
            ####

            # locate particle information corresponding to this hologram
            particle_idx = np.where(self.h_ds['hid'].values == h_idx+1)

            x_part = self.h_ds['x'].values[particle_idx]
            y_part = self.h_ds['y'].values[particle_idx]
            z_part = self.h_ds['z'].values[particle_idx]
            # not used but here it is
            d_part = self.h_ds['d'].values[particle_idx]

            # create a 3D histogram
            in_data = np.stack((x_part, y_part, z_part)).T
            h_part = np.histogramdd(
                in_data, bins=[self.tile_x_bins, self.tile_y_bins, self.z_bins])[0]
            # specify the z bin locations of the particles
            z_part_bin_idx = np.digitize(z_part, self.z_bins)-1

            # smoothing kernel accounts for overlapping subimages when the
            # subimage is larger than the stride
            ratio = self.tile_size//self.step_size
            if self.step_size < self.tile_size:
                overlap_kernel = np.ones((ratio, ratio))
                for z_idx in range(h_part.shape[-1]):
                    h_part[:, :, z_idx] = convolve2d(h_part[:, :, z_idx], overlap_kernel)[
                        ratio-1:h_part.shape[0]+ratio-1, ratio-1:h_part.shape[1]+ratio-1]

            # locate all the cases where particles are and are not present
            # to sample from those cases
            if self.step_size < self.tile_size:
                # note that the last bin is ommitted from each to avoid edge cases where
                # the image is not complete

                edge_idx = ratio-1

                # find the locations where particles are in focus
                loc_idx = np.where(h_part[:-edge_idx, :-edge_idx, :] > 0)
                # find locations where particles are not in focus
                empt_idx = np.where(h_part[:-edge_idx, :-edge_idx, :] == 0)
                #### find locations where particles are nearly in focus  ####
                zdiff = np.diff(h_part[:-edge_idx, :-edge_idx, :], axis=2)
                zero_pad = np.zeros(
                    h_part[:-edge_idx, :-edge_idx, :].shape[0:2]+(1,))
                near_empt_idx = np.where((h_part[:-edge_idx, :-edge_idx, :] == 0) & ((np.concatenate(
                    [zdiff, zero_pad], axis=2) == 1) | (np.concatenate([zero_pad, zdiff], axis=2) == -1)))
                ####
            else:
                # find the locations where particles are in focus
                loc_idx = np.where(h_part > 0)
                # find locations where particles are not in focus
                empt_idx = np.where(h_part == 0)
                #### find locations where particles are nearly in focus ####
                zdiff = np.diff(h_part, axis=2)
                zero_pad = np.zeros(h_part.shape[0:2]+(1,))
                near_empt_idx = np.where((h_part == 0) & ((np.concatenate(
                    [zdiff, zero_pad], axis=2) == 1) | (np.concatenate([zero_pad, zdiff], axis=2) == -1)))
                ####

            # select sub images with particles in them
            if part_per_holo > loc_idx[0].size:
                # pick the entire set
                loc_x_idx = loc_idx[0]
                loc_y_idx = loc_idx[1]
                loc_z_idx = loc_idx[2]
            else:
                # randomly select particles from the set
                sel_part_idx = np.random.choice(
                    np.arange(loc_idx[0].size, dtype=int), size=part_per_holo, replace=False)
                loc_x_idx = loc_idx[0][sel_part_idx]
                loc_y_idx = loc_idx[1][sel_part_idx]
                loc_z_idx = loc_idx[2][sel_part_idx]

            # randomly select empties from the empty set
            #### Add nearly in focus cases to the training data ####
            sel_empt_idx = np.random.choice(np.arange(
                near_empt_idx[0].size, dtype=int), size=empt_near_cnt, replace=False)  # select nearly in focus cases
            ####
            sel_empt_idx = np.concatenate([np.random.choice(np.arange(empt_idx[0].size, dtype=int), size=(
                empt_per_holo-empt_near_cnt), replace=False), sel_empt_idx])  # select random out of focus cases
            empt_x_idx = empt_idx[0][sel_empt_idx]
            empt_y_idx = empt_idx[1][sel_empt_idx]
            empt_z_idx = empt_idx[2][sel_empt_idx]

            # full set of plane indices to reconstruct (empty and with particles)
            z_full_idx = np.unique(np.concatenate((loc_z_idx, empt_z_idx)))

            # build the torch tensor for reconstruction
            z_plane = torch.tensor(
                self.z_centers[z_full_idx]*1e-6, device=self.device).unsqueeze(-1).unsqueeze(-1)

            # create the torch tensor for propagation
            E_input = torch.tensor(self.h_ds['image'].isel(
                hologram_number=h_idx).values).to(self.device).unsqueeze(0)

            # apply any transformations before doing wave-prop

            # reconstruct the selected planes
            E_out = self.torch_holo_set(
                E_input, z_plane).detach().cpu().numpy()

            # grab the sub images corresponding to the selected data points
            particle_in_focus_lst = []  # training labels for if particle is in focus
            particle_unet_labels_lst = []  # training labels for if particle is in focus
            image_lst = []  # sliced reconstructed image
            image_index_lst = []  # indices used to identify the image slice
            image_corner_coords = []  # coordinates of the corner of the image slice

            step_size = self.step_size
            tile_size = self.tile_size

            for sub_idx, z_idx in enumerate(z_full_idx):
                part_set_idx = np.where(loc_z_idx == z_idx)[0]
                empt_set_idx = np.where(empt_z_idx == z_idx)[0]

                # initialize the UNET mask
                unet_mask = np.zeros(E_out.shape[1:])
                part_in_plane_idx = np.where(z_part_bin_idx == z_idx)[
                    0]  # locate all particles in this plane

                # build the UNET mask for this z plane
                for part_idx in part_in_plane_idx:
                    #             unet_mask += np.exp(-(y_arr[None,:]*1e6-y_part[part_idx])**2/(2*marker_size**2) - (x_arr[:,None]*1e6-x_part[part_idx])**2/(2*marker_size**2) )
                    unet_mask += ((self.y_arr[None, :]*1e6-y_part[part_idx])**2 + (
                        self.x_arr[:, None]*1e6-x_part[part_idx])**2 < (d_part[part_idx]/2)**2).astype(float)

                for part_idx in part_set_idx:
                    x_idx = loc_x_idx[part_idx]
                    y_idx = loc_y_idx[part_idx]
                    image_lst.append(E_out[sub_idx, x_idx*step_size:(
                        x_idx*step_size+tile_size), y_idx*step_size:(y_idx*step_size+tile_size)])
                    image_index_lst.append([x_idx, y_idx, z_idx])
                    image_corner_coords.append(
                        [self.x_arr[x_idx*step_size], self.y_arr[y_idx*step_size]])
                    particle_in_focus_lst.append(1)
                    particle_unet_labels_lst.append(
                        unet_mask[x_idx*step_size:(x_idx*step_size+tile_size), y_idx*step_size:(y_idx*step_size+tile_size)])

                for empt_idx in empt_set_idx:
                    x_idx = empt_x_idx[empt_idx]
                    y_idx = empt_y_idx[empt_idx]
                    image_lst.append(E_out[sub_idx, x_idx*step_size:(
                        x_idx*step_size+tile_size), y_idx*step_size:(y_idx*step_size+tile_size)])
                    image_index_lst.append([x_idx, y_idx, z_idx])
                    image_corner_coords.append(
                        [self.x_arr[x_idx*step_size], self.y_arr[y_idx*step_size]])
                    particle_in_focus_lst.append(0)
                    particle_unet_labels_lst.append(
                        unet_mask[x_idx*step_size:(x_idx*step_size+tile_size), y_idx*step_size:(y_idx*step_size+tile_size)])

        return particle_in_focus_lst, image_lst, image_index_lst, image_corner_coords, particle_unet_labels_lst


def worker(h_idx, config=None, part_per_holo=None, empt_per_holo=None):
    n_bins = config["n_bins"]
    data_path = config["data_path"]
    tile_size = config["tile_size"]  # size of tiled images in pixels
    # amount that we shift the tile to make a new tile
    step_size = config["step_size"]
    # UNET gaussian marker width (standard deviation) in um
    marker_size = config["marker_size"]
    transform_mode = config["transform_mode"]
    device = config["device"]

    prop = CustomWavePropagator(
        data_path,
        n_bins=n_bins,
        tile_size=tile_size,
        step_size=step_size,
        marker_size=marker_size,
        transform_mode=transform_mode,
        device=device
    )

    data = prop.get_reconstructed_sub_images(
        h_idx, part_per_holo, empt_per_holo
    )

    result_dict = defaultdict(list)
    for idx in range(len(data[0])):
        result_dict["label"].append(int(data[0][idx]))
        image = np.expand_dims(np.abs(data[1][idx]), 0)
        phase = np.expand_dims(np.angle(data[1][idx]), 0)
        result_dict["stacked_image"].append(np.vstack([image, phase]))
        result_dict["mask"].append(csr_matrix(data[4][idx]))

    # clear the cached memory from the gpu
    torch.cuda.empty_cache()
    gc.collect()

    return result_dict


if __name__ == '__main__':

    config_file = str(sys.argv[1])

    if not os.path.isfile(config_file):
        logger.warning(
            f"The model config does not exist at {config_file}. Failing with error.")
        sys.exit(1)

    with open(config_file) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    config = conf["data"]
    ############################################################

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')

    # Stream output to stdout
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    # Save the log file
    logger_name = os.path.join(config["output_path"], f"log.txt")
    fh = logging.FileHandler(logger_name,
                             mode="w",
                             encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    root.addHandler(fh)

    ############################################################

    n_bins = int(config["n_bins"])
    data_path = config["data_path"]
    output_path = config["output_path"]
    tile_size = int(config["tile_size"])  # size of tiled images in pixels
    # amount that we shift the tile to make a new tile
    step_size = int(config["step_size"])
    # UNET gaussian marker width (standard deviation) in um
    marker_size = config["marker_size"]
    device = config["device"]
    # step_size is not allowed be be larger than the tile_size
    assert tile_size >= step_size

    total_positive = int(config["total_positive"])
    total_negative = int(config["total_negative"])
    total_examples = int(config["total_training"])
    cores = int(config["cores"])

    transform_mode = "None" if "transform_mode" not in config else config["transform_mode"]

    prop = CustomWavePropagator(
        data_path,
        n_bins=n_bins,
        tile_size=tile_size,
        step_size=step_size,
        marker_size=marker_size,
        transform_mode=transform_mode
    )

    number_of_holograms = prop.h_ds.dims['hologram_number']

    # shuffle
    hologram_numbers = list(range(number_of_holograms))

    # split into train/test/val
    training_hologram_numbers = random.sample(
        hologram_numbers, int(0.8 * len(hologram_numbers)))
    leftovers = list(set(hologram_numbers) - set(training_hologram_numbers))
    validation_hologram_numbers = random.sample(leftovers, len(leftovers) // 2)
    test_hologram_numbers = list(
        set(leftovers) - set(validation_hologram_numbers))

    total_training_examples = int(
        0.8 * total_examples / (total_positive + total_negative))
    total_validation_examples = int(
        0.1 * total_examples / (total_positive + total_negative))
    total_testing_examples = int(
        0.1 * total_examples / (total_positive + total_negative))

    training_hologram_numbers = [random.choice(
        training_hologram_numbers) for x in range(total_training_examples)]
    validation_hologram_numbers = [random.choice(
        validation_hologram_numbers) for x in range(total_validation_examples)]
    test_hologram_numbers = [random.choice(
        test_hologram_numbers) for x in range(total_testing_examples)]

    work = partial(
        worker,
        config=config,
        part_per_holo=total_positive,
        empt_per_holo=total_negative
    )

    name_tag = f"{tile_size}_{step_size}_{total_positive}_{total_negative}_{total_examples}_{transform_mode}"

    # Create the training data first
    with mp.Pool(cores) as p:

#         with open(f"{output_path}/training_{name_tag}.pkl", "wb") as fid:
#             for data in tqdm.tqdm(p.imap(work,
#                                          training_hologram_numbers), total=total_training_examples):
#                 for image, label, mask in zip(data["stacked_image"], data["label"], data["mask"]):
#                     joblib.dump((image, label, mask), fid)

#         # clear the cached memory from the gpu
#         torch.cuda.empty_cache()
#         gc.collect()

#         # Create the validation data
#         with open(f"{output_path}/validation_{name_tag}.pkl", "wb") as fid:
#             for data in tqdm.tqdm(p.imap(work,
#                                          validation_hologram_numbers), total=total_validation_examples):
#                 for image, label, mask in zip(data["stacked_image"], data["label"], data["mask"]):
#                     joblib.dump((image, label, mask), fid)

#         # clear the cached memory from the gpu
#         torch.cuda.empty_cache()
#         gc.collect()

#         # Create the test data
#         with open(f"{output_path}/test_{name_tag}.pkl", "wb") as fid:
#             for data in tqdm.tqdm(p.imap(work,
#                                          test_hologram_numbers), total=total_testing_examples):
#                 for image, label, mask in zip(data["stacked_image"], data["label"], data["mask"]):
#                     joblib.dump((image, label, mask), fid)
                    
        # Training split        
        X = np.zeros((total_training_examples, 2, tile_size, tile_size), dtype = np.float32)
        Y1 = np.zeros((total_training_examples, tile_size, tile_size), dtype = np.int)
        Y2 = np.zeros((total_training_examples, 1), dtype = np.int)
        c = 0
        for data in tqdm.tqdm(p.imap(work,
                                     training_hologram_numbers), total=total_training_examples):
            for image, label, mask in zip(data["stacked_image"], data["label"], data["mask"]):
                X[c:c+1] = image
                Y1[c:c+1] = mask.toarray()
                Y2[c:c+1] = np.array([int(label)])
                c += 1

                if c >= total_training_examples:
                    break
            if c >= total_training_examples:
                break
        
        df = xr.Dataset(data_vars=dict(var_x=(['n', 'd', 'x', 'y'], X[:c]),
                                       var_y=(['n', 'x', 'y'], Y1[:c]),
                                       var_z=(['n', 'z'], Y2[:c])))
        df.to_netcdf(f"{output_path}/training_{name_tag}.nc")
        
        # Validation split
        X = np.zeros((total_validation_examples, 2, tile_size, tile_size), dtype = np.float32)
        Y1 = np.zeros((total_validation_examples, tile_size, tile_size), dtype = np.int)
        Y2 = np.zeros((total_validation_examples, 1), dtype = np.int)
        c = 0
        for data in tqdm.tqdm(p.imap(work,
                                     validation_hologram_numbers), total=total_validation_examples):
            for image, label, mask in zip(data["stacked_image"], data["label"], data["mask"]):
                X[c:c+1] = image
                Y1[c:c+1] = mask.toarray()
                Y2[c:c+1] = np.array([int(label)])
                c += 1

                if c >= total_validation_examples:
                    break
            if c >= total_validation_examples:
                break
        
        df = xr.Dataset(data_vars=dict(var_x=(['n', 'd', 'x', 'y'], X[:c]),
                                       var_y=(['n', 'x', 'y'], Y1[:c]),
                                       var_z=(['n', 'z'], Y2[:c])))
        df.to_netcdf(f"{output_path}/validation_{name_tag}.nc")
        
        # Test split
        X = np.zeros((total_testing_examples, 2, tile_size, tile_size), dtype = np.float32)
        Y1 = np.zeros((total_testing_examples, tile_size, tile_size), dtype = np.int)
        Y2 = np.zeros((total_testing_examples, 1), dtype = np.int)
        c = 0
        for data in tqdm.tqdm(p.imap(work,
                                     test_hologram_numbers), total=total_testing_examples):
            for image, label, mask in zip(data["stacked_image"], data["label"], data["mask"]):
                X[c:c+1] = image
                Y1[c:c+1] = mask.toarray()
                Y2[c:c+1] = np.array([int(label)])
                c += 1

                if c >= total_testing_examples:
                    break
            if c >= total_testing_examples:
                break
        
        df = xr.Dataset(data_vars=dict(var_x=(['n', 'd', 'x', 'y'], X[:c]),
                                       var_y=(['n', 'x', 'y'], Y1[:c]),
                                       var_z=(['n', 'z'], Y2[:c])))
        df.to_netcdf(f"{output_path}/test_{name_tag}.nc")