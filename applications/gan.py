from torch.optim.lr_scheduler import ReduceLROnPlateau
from echo.src.base_objective import BaseObjective
from collections import defaultdict
import pandas as pd
import numpy as np
import torch.fft
import subprocess
import logging
import random
import shutil
import psutil
import scipy
import torch
import lpips
import copy
import yaml
import time
import tqdm
import sys
import gc

import segmentation_models_pytorch as smp

from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from holodecml.data import PickleReader, UpsamplingReader, XarrayReader
from holodecml.propagation import InferencePropagator
from holodecml.transforms import LoadTransformations
from holodecml.models import load_model
from holodecml.losses import load_loss

import os
import warnings
warnings.filterwarnings("ignore")


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

available_ncpus = len(psutil.Process().cpu_affinity())

# Set up the GPU
is_cuda = torch.cuda.is_available()
device = torch.device("cpu") if not is_cuda else torch.device("cuda:0")


# ### Set seeds for reproducibility
def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag
        
        
class Objective(BaseObjective):

    def __init__(self, config, metric="val_loss", device="cpu"):

        # Initialize the base class
        BaseObjective.__init__(self, config, metric, device)

    def train(self, trial, conf):
        return trainer(conf, save_images = False, trial = trial)
        
        
def trainer(conf, save_images = True, trial = False):
    
    # Set seeds for reproducibility
    seed = 1000 if "seed" not in conf else conf["seed"]
    seed_everything(seed)
    
    save_loc = conf["save_loc"]
    os.makedirs(save_loc, exist_ok = True)
    os.makedirs(os.path.join(save_loc, "images"), exist_ok = True)
    
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

    name_tag = f"{tile_size}_{step_size}_{total_positive}_{total_negative}_{total_examples}_{transform_mode}"
    fn_train = f"{data_path}/training_{name_tag}.nc"
    fn_valid = f"{data_path}/validation_{name_tag}.nc"
    fn_train_raw = f"{data_path_raw}/training_{name_tag}.nc"
    fn_valid_raw = f"{data_path_raw}/validation_{name_tag}.nc"

    # Trainer params
    train_batch_size = conf["trainer"]["train_batch_size"]
    valid_batch_size = conf["trainer"]["valid_batch_size"]
    
    epochs = conf["trainer"]["epochs"]
    batches_per_epoch = conf["trainer"]["batches_per_epoch"]
    Tensor = torch.cuda.FloatTensor if is_cuda else torch.FloatTensor
    adv_loss = conf["trainer"]["adv_loss"]
    lambda_gp = conf["trainer"]["lambda_gp"]
    train_gen_every = conf["trainer"]["train_gen_every"]
    train_disc_every = conf["trainer"]["train_disc_every"]
    threshold = conf["trainer"]["threshold"]
    mask_penalty  = conf["trainer"]["mask_penalty"]
    regression_penalty = conf["trainer"]["regression_penalty"]
    stopping_patience = conf["trainer"]["stopping_patience"]

    # Load the preprocessing transforms
    if "Normalize" in conf["transforms"]["training"]:
        conf["transforms"]["validation"]["Normalize"]["mode"] = conf["transforms"]["training"]["Normalize"]["mode"]
        conf["transforms"]["inference"]["Normalize"]["mode"] = conf["transforms"]["training"]["Normalize"]["mode"]

    train_transforms = LoadTransformations(conf["transforms"]["training"])
    valid_transforms = LoadTransformations(conf["transforms"]["validation"])


    train_synthetic_dataset = XarrayReader(fn_train, train_transforms)
    test_synthetic_dataset = XarrayReader(fn_valid, valid_transforms)

    train_synthetic_loader = torch.utils.data.DataLoader(
        train_synthetic_dataset,
        batch_size=train_batch_size,
        num_workers=available_ncpus//2,
        pin_memory=True,
        shuffle=True)

    test_synthetic_loader = torch.utils.data.DataLoader(
        test_synthetic_dataset,
        batch_size=valid_batch_size,
        num_workers=0,  # 0 = One worker with the main process
        pin_memory=True,
        shuffle=False)

    train_holodec_dataset = XarrayReader(fn_train_raw, train_transforms)
    test_holodec_dataset = XarrayReader(fn_valid_raw, valid_transforms)

    train_holodec_loader = torch.utils.data.DataLoader(
        train_holodec_dataset,
        batch_size=train_batch_size,
        num_workers=available_ncpus//2,
        pin_memory=True,
        shuffle=True)

    test_holodec_loader = torch.utils.data.DataLoader(
        test_holodec_dataset,
        batch_size=valid_batch_size,
        num_workers=0,  # 0 = One worker with the main process
        pin_memory=True,
        shuffle=False)

    # Load the models
    generator = load_model(conf["generator"]).to(device)
    discriminator = load_model(conf["discriminator"]).to(device)
    model = load_model(conf["discriminator"]).to(device)

    # Load loss function
    adv_loss = conf["trainer"]["adv_loss"]
    if adv_loss == "bce":
        adversarial_loss = torch.nn.BCELoss().to(device)
        
    # Load perceptual-alex-net as the metric
    perceptual_alex = lpips.LPIPS(net='alex').to(device)
    # Load mask loss
    Mask_Loss = load_loss("focal-tyversky")

    # Load the optimizers
    optimizer_G = torch.optim.Adam(
        filter(lambda p: p.requires_grad, generator.parameters()),
        lr = conf["optimizer_G"]["learning_rate"],
        betas = (conf["optimizer_G"]["b0"], conf["optimizer_G"]["b1"]))
    
    optimizer_D = torch.optim.Adam(
        filter(lambda p: p.requires_grad, discriminator.parameters()), 
        lr = conf["optimizer_D"]["learning_rate"], 
        betas = (conf["optimizer_D"]["b0"], conf["optimizer_D"]["b1"]))
    
    model_D = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr = 0.0004,
        betas = (conf["optimizer_D"]["b0"], conf["optimizer_D"]["b1"]))
    
    # Anneal the learning rate
    lr_G_decay = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=30, gamma=0.2)
    lr_D_decay = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=30, gamma=0.2)
    lr_M_decay = torch.optim.lr_scheduler.StepLR(model_D, step_size=30, gamma=0.2)

    results = defaultdict(list)
    for epoch in range(epochs):

        ### Train
        real_images = iter(train_holodec_loader)
        synthethic_images = iter(train_synthetic_loader)
        dual_iter = tqdm.tqdm(
            enumerate(zip(real_images, synthethic_images)),
            total = batches_per_epoch, 
            leave = True)

        train_results = defaultdict(list)
        for i, ((holo_img, holo_label), (synth_img, synth_label)) in dual_iter:

            if holo_img.shape[0] != synth_img.shape[0]:
                continue

            # Adversarial ground truths
            valid = Variable(Tensor(holo_img.size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(holo_img.size(0), 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(holo_img.type(Tensor))
            synthethic_imgs = Variable(synth_img.type(Tensor))
            
            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1.0, holo_img.shape)))
            # C-GAN-like input using the synthethic image as conditional input
            gen_input = torch.cat([synthethic_imgs, z], 1)
            # Generate a batch of images
            gen_imgs = generator(gen_input)
            # Add to the synthetic images
            # Discriminate the fake images
            _, verdict = discriminator(gen_imgs)
            
            # -----------------
            #  Train mask model
            # -----------------

            model_D.zero_grad()
            requires_grad(model, True)
            pred_masks, _ = model(gen_imgs.detach())
            mask_loss = Mask_Loss(pred_masks, synth_label.to(device))
            train_results["mask_loss"].append(mask_loss.item())
            mask_loss.backward()
            model_D.step()
            requires_grad(model, False)

            # -----------------
            #  Train Generator
            # -----------------

            if (i + 1) % train_gen_every == 0 or i == 0:

                optimizer_G.zero_grad()
                requires_grad(generator, True)
                requires_grad(discriminator, False)

                # Loss measures generator's ability to fool the discriminator
                if adv_loss == 'wgan-gp':
                    g_loss = -verdict.mean()
                elif adv_loss == 'hinge':
                    g_loss = -verdict.mean()
                elif adv_loss == 'bce':
                    g_loss = adversarial_loss(verdict, valid)


                # compute L1/L2 reg term
#                 mask = synth_label.to(device).bool()
#                 reg_loss = torch.nn.MSELoss()(
#                     torch.masked_select(gen_imgs, mask),
#                     torch.masked_select(synthethic_imgs, mask)
#                 )
                
                mask = synth_label.to(device)
                reg_loss = torch.nn.MSELoss()(gen_imgs, synthethic_imgs)
            
                train_results["g_reg"].append(reg_loss.item())
                g_loss += regression_penalty * reg_loss

                # compute mask loss reg term
                #pred_masks, _ = model(gen_imgs)
                mask_loss = Mask_Loss(pred_masks.detach(), synth_label.to(device))
                #train_results["fake_mask_loss"].append(mask_loss.item())
                g_loss += mask_penalty * mask_loss
                train_results["g_loss"].append(g_loss.item())

                g_loss.backward()
                optimizer_G.step()

                # compute perception scores
                p_score_syn = perceptual_alex(gen_imgs, synthethic_imgs).mean()
                p_score_real = perceptual_alex(gen_imgs, real_imgs).mean()
                train_results["p_syn"].append(p_score_syn.item())
                train_results["p_real"].append(p_score_real.item())

            # ---------------------
            #  Train Discriminator
            # ---------------------

            if (i + 1) % train_disc_every == 0 or i == 0:

                optimizer_D.zero_grad()
                requires_grad(generator, False)
                requires_grad(discriminator, True)

                # Measure discriminator's ability to classify real from generated samples
                _, disc_real = discriminator(real_imgs)
                _, disc_synth = discriminator(gen_imgs.detach())
                _, disc_synth_true = discriminator(synthethic_imgs)

                train_results["real_acc"].append(((disc_real > threshold) == valid).float().mean().item())
                train_results["syn_acc"].append(((disc_synth > threshold) == fake).float().mean().item())
                train_results["syn_true_acc"].append(((disc_synth_true > threshold) == valid).float().mean().item())

                if adv_loss == 'wgan-gp':
                    real_loss = -torch.mean(disc_real) 
                    fake_loss = disc_synth.mean() 
                elif adv_loss == 'hinge':
                    real_loss = torch.nn.ReLU()(1.0 - disc_real).mean() 
                    fake_loss = torch.nn.ReLU()(1.0 + disc_synth).mean()             
                elif adv_loss == 'bce':
                    real_loss = adversarial_loss(disc_real, valid) 
                    fake_loss = adversarial_loss(disc_synth, fake) 

                d_loss = real_loss + fake_loss 
                train_results["d_loss"].append(d_loss.item())

                if adv_loss == 'wgan-gp':
                    # Compute gradient penalty
                    alpha = torch.rand(real_imgs.size(0), 1, 1, 1).cuda().expand_as(real_imgs)
                    interpolated = Variable(alpha * real_imgs.data + (1 - alpha) * gen_imgs.data, requires_grad=True)
                    out = discriminator(interpolated)[1]

                    grad = torch.autograd.grad(outputs=out,
                                               inputs=interpolated,
                                               grad_outputs=torch.ones(out.size()).cuda(),
                                               retain_graph=True,
                                               create_graph=True,
                                               only_inputs=True)[0]

                    grad = grad.view(grad.size(0), -1)
                    grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
                    d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)
                    d_loss_reg = lambda_gp * d_loss_gp
                    d_loss += d_loss_reg
                    train_results["d_reg"].append(d_loss_reg.item())

                d_loss.backward()
                optimizer_D.step()
                

            print_str =  f'Epoch {epoch}'
            print_str += f' D_loss {np.mean(train_results["d_loss"]):.6f}'
            if adv_loss == 'wgan-gp':
                print_str += f' D_reg {np.mean(train_results["d_reg"]):.6f}'
            print_str += f' G_loss {np.mean(train_results["g_loss"]):6f}'
            print_str += f' G_reg {np.mean(train_results["g_reg"]):6f}'
            print_str += f' mask_loss {np.mean(train_results["mask_loss"]):6f}'
            #print_str += f' fake_mask_loss {np.mean(train_results["fake_mask_loss"]):6f}'
            print_str += f' h_acc {np.mean(train_results["real_acc"]):.4f}'
            print_str += f' f_acc {np.mean(train_results["syn_acc"]):.4f}'
            print_str += f' s_acc {np.mean(train_results["syn_true_acc"]):.4f}'
            print_str += f' p_syn {np.mean(train_results["p_syn"]):.4f}'
            print_str += f' p_real {np.mean(train_results["p_real"]):.4f}'
            dual_iter.set_description(print_str)
            dual_iter.refresh()

            if i == batches_per_epoch and i > 0:
                break

        # Epoch is over. Save some stuff.
        if save_images:
            save_image(synthethic_imgs.data[:16], f'{conf["save_loc"]}/images/synth_{epoch}.png', nrow=4, normalize=True)
            save_image(real_imgs.data[:16], f'{conf["save_loc"]}/images/real_{epoch}.png', nrow=4, normalize=True)
            save_image(gen_imgs.data[:16], f'{conf["save_loc"]}/images/pred_{epoch}.png', nrow=4, normalize=True)
            #save_image(gen_noise.data[:16], f"../results/gan/images/noise_{epoch}.png", nrow=4, normalize=True)

        # Save the dataframe to disk
        results["epoch"].append(epoch)
        results["d_loss"].append(np.mean(train_results["d_loss"]))
        if adv_loss == 'wgan-gp':
            results["d_loss_reg"].append(np.mean(train_results["d_reg"]))
        results["g_loss"].append(np.mean(train_results["g_loss"]))
        results["g_reg"].append(np.mean(train_results["g_reg"]))
        results["mask_loss"].append(np.mean(train_results["mask_loss"]))
        #results["fake_mask_loss"].append(np.mean(train_results["fake_mask_loss"]))
        results["holo_acc"].append(np.mean(train_results["real_acc"]))
        results["fake_acc"].append(np.mean(train_results["syn_acc"]))
        results["synth_acc"].append(np.mean(train_results["syn_true_acc"]))
        results["perception_syn"].append(np.mean(train_results["p_syn"]))
        results["perception_holo"].append(np.mean(train_results["p_real"]))
        
        metric = "custom"
        metric_value = results["mask_loss"][-1] + results["perception_holo"][-1]
        results[metric].append(metric_value)

        if save_images:
            try:
                df = pd.DataFrame.from_dict(results).reset_index()
                df.to_csv(f'{conf["save_loc"]}/training_log.csv', index=False)
            except:
                print(results)
                raise

            # Save the model
            state_dict = {
                'epoch': epoch,
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'generator_state_dict': generator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'model_state_dict': model.state_dict(),
                'model_optimizer_state_dict': model_D.state_dict(),
            }
            torch.save(state_dict, f'{conf["save_loc"]}/best.pt')
            
        # Update optuna trial if using ECHO
        if trial:
            # Report result to the trial
            attempt = 0
            while attempt < 10:
                try:
                    trial.report(metric_value, step=epoch)
                    break
                except Exception as E:
                    logging.warning(
                        f"WARNING failed to update the trial with metric {metric} at epoch {epoch}. Error {str(E)}")
                    logging.warning(f"Trying again ... {attempt + 1} / 10")
                    time.sleep(1)
                    attempt += 1
                    
        # Stop training if we have not improved after X epochs (stopping patience)
        best_epoch = [i for i, j in enumerate(
            results[metric]) if j == min(results[metric])][0]
        offset = epoch - best_epoch
        if offset >= stopping_patience:
            logging.info(f"Trial {trial.number} is stopping early after {stopping_patience} epochs.")
            break
            
        # Anneal learning rates 
        lr_G_decay.step(epoch)
        lr_D_decay.step(epoch)
        lr_M_decay.step(epoch)
            
    if trial:
        trial_id = trial.number
        save_image(synthethic_imgs.data[:8], f'{conf["save_loc"]}/images/synth_{trial_id}.png', nrow=2, normalize=True)
        save_image(real_imgs.data[:8], f'{conf["save_loc"]}/images/real_{trial_id}.png', nrow=2, normalize=True)
        save_image(gen_imgs.data[:8], f'{conf["save_loc"]}/images/pred_{trial_id}.png', nrow=2, normalize=True)
            
    result_dict = {
        "epoch": best_epoch,
        "combined_perception": results["combined_perception"][best_epoch],
        "holodec_perception": results["perception_holo"][best_epoch],
        "synthetic_perception": results["perception_syn"][best_epoch],
        "mask_loss": results["mask_loss"][best_epoch],
        #"fake_mask_loss": results["fake_mask_loss"][best_epoch],
        metric: results[metric][best_epoch]
    }
        
    return result_dict


        
if __name__ == '__main__':

    if len(sys.argv) != 2:
        print("Usage: python gan.py config.yml")
        sys.exit()

    # ### Set up logger to print stuff
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')

    # Stream output to stdout
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    root.addHandler(ch)
    
    # ### Load the configuration and get the relevant variables
    config = sys.argv[1]
    with open(config) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)
        
    save_loc = conf["save_loc"]
    os.makedirs(save_loc, exist_ok = True)
    os.makedirs(os.path.join(save_loc, "images"), exist_ok = True)
    shutil.copyfile(config, os.path.join(save_loc, "model.yml"))
        
    result = trainer(conf, save_images = True)
