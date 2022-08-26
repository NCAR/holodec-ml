from argparse import ArgumentParser
import xarray as xr
import torchvision
import numpy as np
import subprocess
import logging
import shutil
import random
import torch
import lpips
import yaml
import glob
import tqdm
import sys
import os

from holodecml.seed import seed_everything
from holodecml.data import XarrayReader, PickleReader
from holodecml.style import (requires_grad, get_input_optimizer,
                             gram_matrix, ContentLoss, StyleLoss,
                             rename_vgg_layers, get_style_model_and_losses)

import warnings
warnings.filterwarnings("ignore")


logger = logging.getLogger(__name__)


# Set up the GPU, grab number of CPUS
is_cuda = torch.cuda.is_available()
device = torch.device("cpu") if not is_cuda else torch.device("cuda")


def launch_pbs_jobs(nodes):
    from pathlib import Path 
    script_path = Path(__file__).absolute()
    parent = Path(__file__).parent
    for worker in range(nodes):
        script = f"""
        #!/bin/bash -l
        #PBS -N style-{worker}
        #PBS -l select=1:ncpus=8:ngpus=1:mem=128GB
        #PBS -l walltime=12:00:00
        #PBS -l gpu_type=v100
        #PBS -A NAML0001
        #PBS -q casper
        #PBS -o out
        #PBS -e out

        source ~/.bashrc
        ncar_pylib /glade/work/$USER/py37
        python {script_path} -c {parent}/../config/model_segmentation.yml -n {nodes} -w {worker}
        """
        with open("launcher.sh", "w") as fid:
            fid.write(script)
        jobid = subprocess.Popen("qsub launcher.sh", 
                                 shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()[0]
        jobid = jobid.decode("utf-8").strip("\n")
        print(jobid)
    os.remove("launcher.sh")


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
        verbose=0):

    """Run the style transfer."""

    model, style_losses, content_losses = get_style_model_and_losses(
        cnn, style_img, content_img, content_layers, style_layers
    )

    # We want to optimize the input and not the model parameters, so we
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

    description = "Create hologram images using content/style from two datasets"

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
    parser.add_argument(
        "-l",
        dest="launch",
        type=int,
        default=0,
        help="Submit {nodes} workers to PBS. Run -m option once all workers finish."
    )

    args_dict = vars(parser.parse_args())
    config_file = args_dict.pop("model_config")
    nodes = int(args_dict.pop("nodes"))
    worker = int(args_dict.pop("worker"))
    merge = bool(int(args_dict.pop("merge")))
    launch = bool(int(args_dict.pop("launch")))
    
    if launch:
        logger.info(f"Launching {nodes} workers to PBS")
        launch_pbs_jobs(nodes)
        sys.exit()

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
        logger.info(f"Merging {nodes} dataframes into one common df")
        for split in ["train", "valid", "test"]:
            shorthand = os.path.join(aug_data_path, f"{split}_{sampler}_{name_tag}_*.nc")
            
            if len(shorthand) != nodes:
                logger.warning(
                    "The number of files to be merged does not equal the number of nodes used. Exiting."
                )
                sys.exit(1)
            
            logger.info(f"On split {split}, merging {shorthand}")
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
        
    logger.info("Beginning style augmentation training")

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

    # Start hologram translation
    filenames = [fn_train_aug, fn_valid_aug, fn_test_aug]
    synthetic_datasets = [train_synthetic_dataset, valid_synthetic_dataset, test_synthetic_dataset]
    holodec_datasets = [train_holodec_dataset, valid_holodec_dataset, test_holodec_dataset]

    
    try:
    
        for fn_name, synth, holo in zip(filenames, synthetic_datasets, holodec_datasets):

            size = synth.__len__()
            synthetic_idx = list(range(size))
            holodec_idx = list(range(holo.__len__()))
            synthetic_idx = np.array_split(synthetic_idx, nodes)[worker]
            X = np.zeros((len(synthetic_idx), 512, 512), dtype=np.float32)
            Y = np.zeros((len(synthetic_idx), 512, 512), dtype=np.int)

            last_i = 0
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
                last_i = i

            df = xr.Dataset(data_vars=dict(var_x=(["n", "x", "y"], X[:last_i]), var_y=(["n", "x", "y"], Y[:last_i])))
            df.to_netcdf(fn_name)
            
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
