#from torch.multiprocessing import Pool, set_start_method

import random, os, torch, numpy as np

from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import defaultdict
import pandas as pd
#import numpy as np
import torchvision
import torch.fft
import subprocess
import logging
import matplotlib.pyplot as plt
#import random
import shutil
import psutil
import sklearn
import scipy
#import torch
import copy
import yaml
import time
import tqdm
import sys
import gc

import torch.optim as optim

import segmentation_models_pytorch as smp

from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F

from holodecml.data import PickleReader, UpsamplingReader, XarrayReader, XarrayReaderLabels
from holodecml.propagation import InferencePropagator
from holodecml.transforms import LoadTransformations
from holodecml.models import load_model
from holodecml.losses import load_loss

import warnings
warnings.filterwarnings("ignore")
import lpips

import xarray as xr


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

available_ncpus = len(psutil.Process().cpu_affinity())

# Set up the GPU
is_cuda = torch.cuda.is_available()
device = torch.device("cpu") if not is_cuda else torch.device("cuda")


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

        
class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input
    
    

def rename_vgg_layers(model):
    """Renames VGG model layers to match those in the paper."""
    block, number = 1, 1
    renamed = nn.Sequential()
    for i,layer in enumerate(model.children()):
        if i == 0:
            layer = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            name = f'conv{block}_{number}'
        elif isinstance(layer, nn.Conv2d):
            name = f'conv{block}_{number}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu{block}_{number}'
            # The inplace ReLU version doesn't play nicely with NST.
            layer = nn.ReLU(inplace=False)
            number += 1
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{block}'
            # Average pooling was found to generate images of higher quality than
            # max pooling by Gatys et al.
            layer = nn.AvgPool2d(layer.kernel_size, layer.stride)
            block += 1
            number = 1
        else:
            raise RuntimeError(f'Unrecognized layer "{layer.__class__.__name__}""') 
        renamed.add_module(name, layer)
    return renamed


def get_style_model_and_losses(cnn, 
                               style_img, 
                               content_img,
                               content_layers,
                               style_layers):
    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(*[])

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.AvgPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses


def run_style_transfer(cnn, 
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
        cnn, style_img, content_img, content_layers, style_layers)

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

            for sl in style_losses: # add weights
                style_score += sl.loss
            for cl in content_losses: # add weights 
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 1 == 0 and verbose:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img


def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img], lr = 1)
    #optimizer = optim.Adam([input_img], lr = 0.01)
    return optimizer


if __name__ == "__main__":

    config = "model.yml" #"../config/gan.yml"
    with open(config) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)


    # Set seeds for reproducibility
    seed = 1000 if "seed" not in conf else conf["seed"]
    seed_everything(seed)

    save_loc = conf["save_loc"]
    os.makedirs(save_loc, exist_ok = True)
    os.makedirs(os.path.join(save_loc, "images"), exist_ok = True)
    if not os.path.isfile(os.path.join(save_loc, "model.yml")):
        shutil.copyfile(config, os.path.join(save_loc, "model.yml"))

    tile_size = int(conf["data"]["tile_size"])
    step_size = int(conf["data"]["step_size"])
    data_path = conf["data"]["output_path"]
    data_path_raw = conf["data"]["output_path_raw"]

    total_positive = int(conf["data"]["total_positive"])
    total_negative = int(conf["data"]["total_negative"])
    total_examples = int(conf["data"]["total_training"])

    transform_mode = "None" if "transform_mode" not in conf["data"] else conf["data"]["transform_mode"]
    config_ncpus = int(conf["data"]["cores"])
    use_cached = False if "use_cached" not in conf["data"] else conf["data"]["use_cached"]

    data_path = data_path.replace("style_transfered", "tiled_synthetic")
    name_tag = f"{tile_size}_{step_size}_{total_positive}_{total_negative}_{total_examples}_{transform_mode}"
    fn_train = f"{data_path}/training_{name_tag}.nc"
    fn_valid = f"{data_path}/validation_{name_tag}.nc"
    fn_test = f"{data_path}/test_{name_tag}.pkl"
    #fn_train_raw = data_path_raw
    fn_train_raw = "/glade/p/cisl/aiml/ai4ess_hackathon/holodec/tiled_real/training_512_128_6_6_100000_None.nc"
    fn_valid_raw = "/glade/p/cisl/aiml/ai4ess_hackathon/holodec/tiled_real/validation_512_128_6_6_100000_None.nc"
    fn_test_raw = "/glade/p/cisl/aiml/ai4ess_hackathon/holodec/tiled_real/test_512_128_6_6_100000_None.pkl"

    # Trainer params
    train_batch_size = conf["trainer"]["train_batch_size"]
    valid_batch_size = conf["trainer"]["valid_batch_size"]

    epochs = conf["trainer"]["epochs"]
    batches_per_epoch = conf["trainer"]["batches_per_epoch"]
    Tensor = torch.cuda.FloatTensor if is_cuda else torch.FloatTensor
    adv_loss = conf["trainer"]["adv_loss"]
    lambda_gp = conf["trainer"]["lambda_gp"]
    mask_penalty = conf["trainer"]["mask_penalty"]
    regression_penalty = conf["trainer"]["regression_penalty"]
    train_gen_every = conf["trainer"]["train_gen_every"]
    train_disc_every = conf["trainer"]["train_disc_every"]
    threshold = conf["trainer"]["threshold"]


    # Load the preprocessing transforms
    if "Normalize" in conf["transforms"]["training"]:
        conf["transforms"]["validation"]["Normalize"]["mode"] = conf["transforms"]["training"]["Normalize"]["mode"]
        conf["transforms"]["inference"]["Normalize"]["mode"] = conf["transforms"]["training"]["Normalize"]["mode"]

    train_transforms = None #LoadTransformations(conf["transforms"]["training"])
    valid_transforms = None #LoadTransformations(conf["transforms"]["validation"])

    train_synthetic_dataset = XarrayReader(fn_train, train_transforms)
    valid_synthetic_dataset = XarrayReader(fn_valid, valid_transforms)
    test_synthetic_dataset = PickleReader(
            fn_test,
            transform=valid_transforms,
            max_images=int(0.1 * conf["data"]["total_training"]),
            max_buffer_size=int(0.1 * conf["data"]["total_training"]),
            color_dim=conf["model"]["in_channels"],
            shuffle=False)
    #XarrayReader(fn_test, valid_transforms)

    train_holodec_dataset = XarrayReader(fn_train_raw, train_transforms)
    valid_holodec_dataset = XarrayReader(fn_valid_raw, valid_transforms)
    test_holodec_dataset = PickleReader(
            fn_test_raw,
            transform=valid_transforms,
            max_images=int(0.1 * conf["data"]["total_training"]),
            max_buffer_size=int(0.1 * conf["data"]["total_training"]),
            color_dim=conf["model"]["in_channels"],
            shuffle=False) #XarrayReaderLabels(fn_test_raw, valid_transforms)

    cnn = torchvision.models.vgg19(pretrained=True).features
    cnn = rename_vgg_layers(cnn).to(device).eval()

    content_layers = ['conv_4']
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
    perceptual_alex = lpips.LPIPS(net='alex').to(device)

    ### Start translation 
    
    synthetic = [train_synthetic_dataset, valid_synthetic_dataset, test_synthetic_dataset]
    holodec = [train_holodec_dataset, valid_holodec_dataset, test_holodec_dataset]
    names = ["train", "valid", "test"]
    _id = int(sys.argv[1])
    
    for name, synth, holo in zip(names, synthetic, holodec):
    
        size = synth.__len__()
        synthetic_idx = list(range(size))
        holodec_idx = list(range(holo.__len__()))
        synthetic_idx = np.array_split(synthetic_idx, 30)[_id]
        X = np.zeros((len(synthetic_idx), 512, 512), dtype = np.float32)
        Y = np.zeros((len(synthetic_idx), 512, 512), dtype = np.int)

        for i, k in tqdm.tqdm(enumerate(synthetic_idx), total = len(synthetic_idx)):

            x_s, y_s = synth.__getitem__(k)

            #randomly select hologram image
            h_idx = random.sample(holodec_idx, 1)[0]
            x_h, y_h = holo.__getitem__(h_idx)
            content_img = x_s.clone().unsqueeze(0).to(device).float() / 255.0
            style_img = x_h.clone().unsqueeze(0).to(device).float() / 255.0
            input_img = torch.randn_like(x_s.clone()).unsqueeze(0).to(device).float() / 255.0

            output = run_style_transfer(cnn, 
                                        content_img, 
                                        style_img, 
                                        input_img, 
                                        content_layers, 
                                        style_layers, 
                                        style_weight = 1e9,
                                        content_weight = 2,
                                        verbose = 0,
                                        num_steps = 100)

            X[i] = output.squeeze(0).squeeze(0).detach().cpu().numpy() * 255.0
            Y[i] = y_s.squeeze(0).squeeze(0).cpu().numpy()

        df = xr.Dataset(data_vars=dict(var_x=(['n', 'x', 'y'], X[:i]),
                                       var_y=(['n', 'x', 'y'], Y[:i])))

        df.to_netcdf(f"{name}_style_transfer_{_id}.nc")
        
      
    
    ### Merge section below, needs incorporated for multi-node version
    
#     for name in ["train", "validation", "test"]:
    
#         if name == "validation":
#             _name = "valid"
#         else:
#             _name = name

#         df = xr.concat([xr.open_dataset(x) for x in tqdm.tqdm(sorted(glob.glob(f"{_name}_style_transfer_*nc")))], dim = "n")

#         df.to_netcdf(
#             f"/glade/p/cisl/aiml/ai4ess_hackathon/holodec/style_transfered/{name}_512_128_6_6_100000_None.nc")
