import xarray as xr
import numpy as np
import logging
import random
import torch
import time
import tqdm
import yaml


from holodecml.propagation import InferencePropagationNoLabels
from holodecml.propagation import InferencePropagator
from sklearn.model_selection import train_test_split
from holodecml.seed import seed_everything
from holodecml.models import load_model
from argparse import ArgumentParser
from functools import partial


import warnings

warnings.filterwarnings("ignore")


logger = logging.getLogger(__name__)


is_cuda = torch.cuda.is_available()
if is_cuda:
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
device = torch.device("cpu") if not is_cuda else torch.device("cuda")


class CustomPropagatorNoLabels(InferencePropagationNoLabels):
    """
    Custom prop class for taking random tile samples from holograms.
    A model still needs to be defined and passed for initialization,
    however it is not used, so model details are irrelevant.

    """

    def get_sub_images_labeled(
        self,
        image_tnsr,
        z_sub_set,
        z_counter,
        batch_size=32,
        return_arrays=False,
        return_metrics=False,
        thresholds=None,
        obs_threshold=None,
    ):

        with torch.no_grad():

            # build the torch tensor for reconstruction
            z_plane = (
                torch.tensor(z_sub_set * 1e-6, device=self.device)
                .unsqueeze(-1)
                .unsqueeze(-1)
            )

            # reconstruct the selected planes
            E_out = self.torch_holo_set(image_tnsr, z_plane)

            if self.color_dim == 2:
                stacked_image = torch.cat(
                    [torch.abs(E_out).unsqueeze(1), torch.angle(E_out).unsqueeze(1)], 1
                )
            elif self.color_dim == 1:
                stacked_image = torch.abs(E_out).unsqueeze(1)
            else:
                raise OSError(f"Unrecognized color dimension {self.color_dim}")
            stacked_image = self.apply_transforms(stacked_image.squeeze(0)).unsqueeze(0)

            chunked = np.array_split(
                list(self.idx2slice.items()),
                int(np.ceil(len(self.idx2slice) / batch_size)),
            )

            inputs, masks = [], []
            for z_idx in range(E_out.shape[0]):
                worker = partial(
                    self.collate_masks, image=stacked_image[z_idx, :].float()
                )

                for chunk in chunked:
                    slices, x = worker(chunk)
                    for k in range(len(slices)):
                        inputs.append(x[k].cpu().numpy())
                        masks.append([0.0])

            return_dict = {"inputs": np.vstack(inputs), "masks": np.array(masks)}

        return return_dict


class CustomPropagatorLabels(InferencePropagator):
    """
    Custom prop class for taking random tile samples from holograms.
    A model still needs to be defined and passed for initialization,
    however it is not used, so model details are irrelevant.

    """

    def get_sub_images_labeled(
        self,
        image_tnsr,
        z_sub_set,
        z_counter,
        xp,
        yp,
        zp,
        dp,
        infocus_mask,
        z_part_bin_idx,
        batch_size=32,
        return_arrays=False,
        return_metrics=False,
        thresholds=None,
        obs_threshold=None,
    ):

        with torch.no_grad():

            # build the torch tensor for reconstruction
            z_plane = (
                torch.tensor(z_sub_set * 1e-6, device=self.device)
                .unsqueeze(-1)
                .unsqueeze(-1)
            )

            # reconstruct the selected planes
            E_out = self.torch_holo_set(image_tnsr, z_plane)

            if self.color_dim == 2:
                stacked_image = torch.cat(
                    [torch.abs(E_out).unsqueeze(1), torch.angle(E_out).unsqueeze(1)], 1
                )
            elif self.color_dim == 1:
                stacked_image = torch.abs(E_out).unsqueeze(1)
            else:
                raise OSError(f"Unrecognized color dimension {self.color_dim}")
            stacked_image = self.apply_transforms(stacked_image.squeeze(0)).unsqueeze(0)

            chunked = np.array_split(
                list(self.idx2slice.items()),
                int(np.ceil(len(self.idx2slice) / batch_size)),
            )

            inputs, masks = [], []
            for z_idx in range(E_out.shape[0]):

                unet_mask = torch.zeros(E_out.shape[1:]).to(
                    self.device
                )  # initialize the UNET mask
                # locate all particles in this plane
                part_in_plane_idx = np.where(z_part_bin_idx == z_idx + z_counter)[0]

                # build the UNET mask for this z plane
                for part_idx in part_in_plane_idx:
                    unet_mask += (
                        torch.from_numpy(
                            (self.y_arr[None, :] * 1e6 - yp[part_idx]) ** 2
                            + (self.x_arr[:, None] * 1e6 - xp[part_idx]) ** 2
                            < (dp[part_idx] / 2) ** 2
                        )
                        .float()
                        .to(self.device)
                    )

                worker = partial(
                    self.collate_masks,
                    image=stacked_image[z_idx, :].float(),
                    mask=unet_mask,
                )

                for chunk in chunked:
                    slices, x, true_mask_tile = worker(chunk)
                    for k in range(len(slices)):
                        inputs.append(x[k].cpu().numpy())
                        masks.append(true_mask_tile[k].cpu().unsqueeze(0).numpy().sum())

            return_dict = {"inputs": np.vstack(inputs), "masks": np.array(masks)}

        return return_dict


if __name__ == "__main__":

    description = (
        "Create style training examples by sample tiles from planes in HOLODEC images"
    )

    parser = ArgumentParser(description=description)
    parser.add_argument(
        "-c",
        dest="model_config",
        type=str,
        default=False,
        help="Path to the model configuration (yml) containing your inputs.",
    )

    args_dict = vars(parser.parse_args())
    config_file = args_dict.pop("model_config")

    with open(config_file) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    seed_everything(conf["seed"])
    n_bins = conf["data"]["n_bins"]
    tile_size = conf["data"]["tile_size"]
    step_size = conf["data"]["step_size"]
    marker_size = conf["data"]["marker_size"]
    total_positive = int(conf["data"]["total_positive"])
    total_negative = int(conf["data"]["total_negative"])
    total_examples = int(conf["data"]["total_training"])

    # Do not load the image transformations
    transform_mode = "None"
    tile_transforms = None
    color_dim = conf["model"]["in_channels"]
    inference_mode = conf["inference"]["mode"]

    name_tag = f"{tile_size}_{step_size}_{total_positive}_{total_negative}_{total_examples}_{transform_mode}"

    # Load the model
    model = load_model(conf["model"]).to(device).eval()

    # Set up style configuration parameters
    data_set = conf["style"]["raw"]["path"]
    tiles_per_reconstruction = conf["style"]["raw"]["tiles_per_reconstruction"]
    reconstruction_per_hologram = conf["style"]["raw"]["reconstruction_per_hologram"]
    save_path = conf["style"]["raw"]["save_path"]
    sampler = conf["style"]["raw"]["sampler"]
    name_tag = f"{sampler}_{name_tag}"

    logger.info(f"Using data set {data_set}")

    try:
        prop = CustomPropagatorLabels(
            data_set,
            n_bins=n_bins,
            color_dim=color_dim,
            tile_size=tile_size,
            step_size=step_size,
            marker_size=marker_size,
            transform_mode=transform_mode,
            device=device,
            model=model,
            mode=inference_mode,
            probability_threshold=0.5,
            transforms=tile_transforms,
        )
        prop.h_ds["x"]
    except KeyError:
        prop = CustomPropagatorNoLabels(
            data_set,
            n_bins=n_bins,
            color_dim=color_dim,
            tile_size=tile_size,
            step_size=step_size,
            marker_size=marker_size,
            transform_mode=transform_mode,
            device=device,
            model=model,
            mode=inference_mode,
            probability_threshold=0.5,
            transforms=tile_transforms,
        )

    # Obtain hologram numbers
    h_range = prop.h_ds.hologram_number.values

    # split into train/test/valid
    # remove the "test" set of manually labeled examples from real_holograms_CSET_RF07_20150719_200000-210000.nc
    if "real_holograms_CSET_RF07_20150719_200000-210000.nc" in data_set:
        h_range_prime = list(set(h_range) - set(range(10, 20)))
        h_train, rest_data = train_test_split(h_range_prime, train_size=0.8)
        h_valid, h_test = train_test_split(rest_data, test_size=0.45)
        # Add the labeled test examples to the "test" split
        h_test += list(range(10, 20))
    else:
        h_train, rest_data = train_test_split(h_range, train_size=0.8)
        h_valid, h_test = train_test_split(rest_data, test_size=0.5)

    logger.info(
        f"Total number of train/valid/test holograms: {len(h_train)} {len(h_valid)} {len(h_test)}"
    )

    h_splits = [h_train, h_valid, h_test]
    split_names = ["train", "valid", "test"]

    for split, h_split in zip(split_names, h_splits):

        total = len(h_split) * tiles_per_reconstruction * reconstruction_per_hologram
        X = np.zeros((total, tile_size, tile_size))
        # for now, we only care if there are particles in the tiles, dont need the masks
        Y = np.zeros((total, 1))

        c = 0
        # Main loop to call the generator, predict with the model, and aggregate and save the results
        for nc, h_idx in tqdm.tqdm(enumerate(h_split), total=len(h_split)):

            # Create a list of z-values to propagate to
            z_list = prop.create_z_plane_lst(planes_per_call=1)
            random.shuffle(z_list)
            z_list = z_list[:reconstruction_per_hologram]

            planes_processed = n_bins
            inference_generator = prop.get_next_z_planes_labeled(
                h_idx,
                z_list,
                batch_size=conf["inference"]["batch_size"],
                thresholds=[0.5],
                return_arrays=False,
                return_metrics=False,
                obs_threshold=0.5,
                start_z_counter=planes_processed,
            )

            t0 = time.time()
            for z_idx, results_dict in enumerate(inference_generator):

                if results_dict["masks"].sum() > 0:
                    mins = np.array([p.min() for p in results_dict["inputs"]])
                    idx = np.where(mins == min(mins))[0]
                    print(z_idx, idx, results_dict["masks"].shape)

                idx = random.sample(
                    range(results_dict["inputs"].shape[0]), k=tiles_per_reconstruction
                )
                X[c : c + tiles_per_reconstruction] = results_dict["inputs"][idx]
                Y[c : c + tiles_per_reconstruction] = results_dict["masks"][
                    idx
                ].reshape((tiles_per_reconstruction, 1))
                c += tiles_per_reconstruction

                if c >= X.shape[0]:
                    break

            if c >= X.shape[0]:
                break

        if color_dim == 1:
            X = np.expand_dims(X, axis=1)

        logger.info(f"Completed split {split} with shape {X.shape}")

        df = xr.Dataset(
            data_vars=dict(
                var_x=(["n", "d", "x", "y"], X[:c]), var_y=(["n", "z"], Y[:c])
            )
        )

        logger.info(
            f"Saving the results to netcdf file at {save_path}/{split}_{name_tag}.nc"
        )
        df.to_netcdf(f"{save_path}/{split}_{name_tag}.nc")
