import random
import os
import torch
import numpy as np
from argparse import ArgumentParser
import torchvision
import shutil
import psutil
import yaml
import glob
import tqdm
import sys

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from holodecml.data import XarrayReader, PickleReader
from holodecml.style import (requires_grad, get_input_optimizer, 
                             gram_matrix, ContentLoss, StyleLoss,
                             rename_vgg_layers, get_style_model_and_losses)

import lpips
import xarray as xr
import warnings
warnings.filterwarnings("ignore")


# Set up the GPU, grab number of CPUS
is_cuda = torch.cuda.is_available()
device = torch.device("cpu") if not is_cuda else torch.device("cuda")


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def run_style_transfer(
    cnn,
    content_img,
    style_img,
    input_img,
    content_layers,
    style_layers,
    num_steps=300,
    style_weight=1000000,
    content_weight=1,
    verbose=0,
):
    """Run the style transfer."""
    model, style_losses, content_losses = get_style_model_and_losses(
        cnn, style_img, content_img, content_layers, style_layers
    )

    # We want to optimize the input and not the model parameters so we
    # update all the requires_grad fields accordingly
    input_img.requires_grad_(True)
    model.requires_grad_(False)

    optimizer = get_input_optimizer(input_img)

    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:  # add weights
                style_score += sl.loss
            for cl in content_losses:  # add weights
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 1 == 0 and verbose:
                print("run {}:".format(run))
                print(
                    "Style Loss : {:4f} Content Loss: {:4f}".format(
                        style_score.item(), content_score.item()
                    )
                )
                print()

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img


if __name__ == "__main__":

    description = "Run style transfer using two datasets."

    parser = ArgumentParser(description=description)
    parser.add_argument(
        "-c",
        dest="model_config",
        type=str,
        default=False,
        help="Path to the model configuration (yml) containing your inputs.",
    )
    parser.add_argument(
        "-n",
        dest="nodes",
        type=int,
        default=1,
        help="The total number of nodes being used",
    )
    parser.add_argument(
        "-w",
        dest="worker",
        type=int,
        default=0,
        help="The integer ID of this worker. May range from [0, (nodes - 1)]",
    )
    parser.add_argument(
        "-m",
        dest="merge",
        type=int,
        default=0,
        help="Load and merge all data from workers into a dataset. Set = 1.",
    )

    args_dict = vars(parser.parse_args())
    config_file = args_dict.pop("model_config")
    nodes = int(args_dict.pop("nodes"))
    worker = int(args_dict.pop("worker"))
    merge = bool(int(args_dict.pop("merge")))

    with open(config_file) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    # Set seeds for reproducibility
    seed = 1000 if "seed" not in conf else conf["seed"]
    seed_everything(seed)

    save_loc = conf["save_loc"]
    os.makedirs(save_loc, exist_ok=True)
    if not os.path.isfile(os.path.join(save_loc, "model.yml")):
        shutil.copyfile(config_file, os.path.join(save_loc, "model.yml"))

    tile_size = int(conf["data"]["tile_size"])
    step_size = int(conf["data"]["step_size"])
    data_path = conf["data"]["output_path"]

    total_positive = int(conf["data"]["total_positive"])
    total_negative = int(conf["data"]["total_negative"])
    total_examples = int(conf["data"]["total_training"])

    # Do not set any transformations
    transform_mode = "None"
    train_transforms = None
    valid_transforms = None
    
    name_tag = f"{tile_size}_{step_size}_{total_positive}_{total_negative}_{total_examples}_{transform_mode}"
    fn_train = f"{data_path}/training_{name_tag}.nc"
    fn_valid = f"{data_path}/validation_{name_tag}.nc"
    fn_test = f"{data_path}/test_{name_tag}.nc"

    # HOLODEC tiles to be used as "style" images
    style_data_path = conf["style"]["raw"]["save_path"]
    os.makedirs(style_data_path, exist_ok=True)
    sampler = conf["style"]["raw"]["sampler"]
    fn_train_raw = os.path.join(style_data_path, f"train_{sampler}_{name_tag}.nc")
    fn_valid_raw = os.path.join(style_data_path, f"valid_{sampler}_{name_tag}.nc")
    fn_test_raw = os.path.join(style_data_path, f"test_{sampler}_{name_tag}.nc")

    # Specify locations to save the style-transfered datasets
    aug_data_path = conf["style"]["synthetic"]["save_path"]
    sampler = conf["style"]["synthetic"]["sampler"]
    fn_train_aug = os.path.join(
        aug_data_path, f"train_{sampler}_{name_tag}_{worker}.nc"
    )
    fn_valid_aug = os.path.join(
        aug_data_path, f"valid_{sampler}_{name_tag}_{worker}.nc"
    )
    fn_test_aug = os.path.join(aug_data_path, f"test_{sampler}_{name_tag}_{worker}.nc")
    
    # Check if we are merging only
    if merge:
        for split in ["train", "valid", "test"]:
            shorthand = os.path.join(aug_data_path, f"{split}_{sampler}_{name_tag}_*.nc")
            df = xr.concat(
                [
                    xr.open_dataset(x)
                    for x in tqdm.tqdm(sorted(glob.glob(shorthand)))
                ],
                dim="n",
            )
            df.to_netcdf(
                os.path.join(aug_data_path, f"{split}_{sampler}_{name_tag}.nc")
            )
            for worker_fn in glob.glob(shorthand):
                os.remove(worker_fn)
        sys.exit()

    # Load synthetic and holodec hologram data sets 
    train_synthetic_dataset = XarrayReader(fn_train, train_transforms)
    valid_synthetic_dataset = XarrayReader(fn_valid, valid_transforms)
    test_synthetic_dataset = XarrayReader(fn_test, valid_transforms)

    train_holodec_dataset = XarrayReader(fn_train_raw, train_transforms)
    valid_holodec_dataset = XarrayReader(fn_valid_raw, valid_transforms)
    test_holodec_dataset = XarrayReader(fn_test_raw, valid_transforms)

    # Load VGG19 and rename the layers
    cnn = torchvision.models.vgg19(pretrained=True).features
    cnn = rename_vgg_layers(cnn).to(device).eval()

    # Select content/style layers
    content_layers = ["conv_4"]
    style_layers = ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]
    perceptual_alex = lpips.LPIPS(net="alex").to(device)

    ### Start translation
    filenames = [fn_train_aug, fn_valid_aug, fn_test_aug]
    synthetic = [
        train_synthetic_dataset,
        valid_synthetic_dataset,
        test_synthetic_dataset,
    ]
    holodec = [train_holodec_dataset, valid_holodec_dataset, test_holodec_dataset]

    for fn_name, synth, holo in zip(filenames, synthetic, holodec):

        size = synth.__len__()
        synthetic_idx = list(range(size))
        holodec_idx = list(range(holo.__len__()))
        synthetic_idx = np.array_split(synthetic_idx, nodes)[worker]
        X = np.zeros((len(synthetic_idx), 512, 512), dtype=np.float32)
        Y = np.zeros((len(synthetic_idx), 512, 512), dtype=np.int)

        for i, k in tqdm.tqdm(enumerate(synthetic_idx), total=len(synthetic_idx)):
            x_s, y_s = synth.__getitem__(k)

            # randomly select hologram image
            h_idx = random.sample(holodec_idx, 1)[0]
            x_h, y_h = holo.__getitem__(h_idx)
            content_img = x_s.clone().unsqueeze(0).to(device).float() / 255.0
            style_img = x_h.clone().unsqueeze(0).to(device).float() / 255.0
            input_img = (
                torch.randn_like(x_s.clone()).unsqueeze(0).to(device).float() / 255.0
            )

            output = run_style_transfer(
                cnn,
                content_img,
                style_img,
                input_img,
                content_layers,
                style_layers,
                style_weight=1e9,
                content_weight=2,
                verbose=0,
                num_steps=100,
            )

            X[i] = output.squeeze(0).squeeze(0).detach().cpu().numpy() * 255.0
            Y[i] = y_s.squeeze(0).squeeze(0).cpu().numpy()

        df = xr.Dataset(
            data_vars=dict(
                var_x=(["n", "x", "y"], X[:i]), var_y=(["n", "x", "y"], Y[:i])
            )
        )

        df.to_netcdf(fn_name)