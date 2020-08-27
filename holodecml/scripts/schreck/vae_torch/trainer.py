import warnings
warnings.filterwarnings("ignore")

import os
import sys
import yaml
import tqdm
import torch
import pickle
import logging

from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from typing import List, Dict
from multiprocessing import cpu_count

# custom
from src.checkpointer import *
from src.data_loader import *
from src.optimizers import *
from src.transforms import *
from src.models import *
from src.visual import *
from src.losses import *


logger = logging.getLogger(__name__)


class Trainer:
    
    def __init__(self, 
                 model, 
                 optimizer,
                 train_gen, 
                 valid_gen, 
                 dataloader, 
                 valid_dataloader,
                 batch_size,
                 path_save,
                 device,
                 test_image = None):
        
        self.model = model
        self.optimizer = optimizer
        self.train_gen = train_gen
        self.valid_gen = valid_gen
        self.dataloader = dataloader
        self.valid_dataloader = valid_dataloader
        self.batch_size = batch_size
        self.path_save = path_save
        self.device = device
        self.test_image = test_image
        
        
    def train_one_epoch(self, epoch):

        self.model.train()
        batches_per_epoch = int(np.ceil(self.train_gen.__len__() / self.batch_size))
        batch_group_generator = tqdm.tqdm(
            enumerate(self.dataloader),
            total=batches_per_epoch, 
            leave=True
        )

        epoch_losses = {"loss": [], "bce": [], "kld": []}
        for idx, images in batch_group_generator:

            images = images.to(self.device)
            recon_images, mu, logvar = self.model(images)
            loss, bce, kld = loss_fn(recon_images, images, mu, logvar)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_loss = loss.item() / batch_size
            bce_loss = bce.item() / batch_size
            kld_loss = kld.item() / batch_size

            epoch_losses["loss"].append(batch_loss)
            epoch_losses["bce"].append(bce_loss)
            epoch_losses["kld"].append(kld_loss)

            loss = np.mean(epoch_losses["loss"])
            bce = np.mean(epoch_losses["bce"])
            kld = np.mean(epoch_losses["kld"])

            to_print = "loss: {:.3f} bce: {:.3f} kld: {:.3f}".format(loss, bce, kld)
            batch_group_generator.set_description(to_print)
            batch_group_generator.update()

        return loss, bce, kld


    def test(self, epoch):

        self.model.eval()
        batches_per_epoch = int(np.ceil(self.valid_gen.__len__() / self.batch_size))

        with torch.no_grad():

            batch_group_generator = tqdm.tqdm(
                enumerate(self.valid_dataloader),
                total=batches_per_epoch, 
                leave=True
            )

            epoch_losses = {"loss": [], "bce": [], "kld": []}
            for idx, images in batch_group_generator:

                images = images.to(self.device)
                recon_images, mu, logvar = self.model(images)
                loss, bce, kld = loss_fn(recon_images, images, mu, logvar)

                batch_loss = loss.item() / batch_size
                bce_loss = bce.item() / batch_size
                kld_loss = kld.item() / batch_size

                epoch_losses["loss"].append(batch_loss)
                epoch_losses["bce"].append(bce_loss)
                epoch_losses["kld"].append(kld_loss)

                loss = np.mean(epoch_losses["loss"])
                bce = np.mean(epoch_losses["bce"])
                kld = np.mean(epoch_losses["kld"])

                to_print = "val_loss: {:.3f} val_bce: {:.3f} val_kld: {:.3f}".format(loss, bce, kld)
                batch_group_generator.set_description(to_print)
                batch_group_generator.update()

            if os.path.isfile(self.test_image):
                with open(self.test_image, "rb") as fid:
                    pic = pickle.load(fid)
                self.compare(epoch, pic)

        return loss, bce, kld
    
    
    def compare(self, epoch, x):
        x = x.to(self.device)
        recon_x, _, _ = self.model(x)
        compare_x = torch.cat([x, recon_x])
        save_image(compare_x.data.cpu(), f'{self.path_save}/image_epoch_{epoch}.png')


    
if __name__ == '__main__':
    
    if len(sys.argv) != 2:
        print("Usage: python trainer.py /path/to/config.yml")
        sys.exit()
    
    ############################################################
    
    with open(sys.argv[1]) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
        
    ############################################################
    #
    # Create the save directory if it does not exist
    #
    ############################################################
    
    try:
        os.makedirs(config["path_save"])
    except:
        pass
    
    ############################################################
    #
    # Load a logger
    #
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
    logger_name = os.path.join(config["path_save"], "log.txt")
    fh = logging.FileHandler(logger_name,
                             mode="w",
                             encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    root.addHandler(fh)

    ############################################################
    #
    # Read in some of the parameters from the configuration file
    #
    ############################################################

    logging.info(f'Reading parameters from {sys.argv[1]}')
    
    path_data = config["path_data"]
    path_save = config["path_save"]
    num_particles = config["num_particles"]
    maxnum_particles = config["maxnum_particles"]
    output_cols = config["output_cols"]
    subset = config["subset"]
    test_image = config["test_image"]
    
    batch_size = config["batch_size"]
    workers = min(config["workers"], cpu_count())
    epochs = config["epochs"]
    retrain = False if "retrain" not in config else config["retrain"]
    
    model_save_path = os.path.join(f"{path_save}", "checkpoint.pt")
    
    start_epoch = 0
    if retrain:
        saved_model_optimizer = torch.load(model_save_path)
        start_epoch = saved_model_optimizer["epoch"] + 1

    ############################################################
    #
    # Set the device to a cuda-enabled GPU or the cpu
    #
    ############################################################

    is_cuda = torch.cuda.is_available()
    device = torch.device(torch.cuda.current_device()) if is_cuda else torch.device("cpu")
    
    logging.info(f'Preparing to use device {device}')
    
    ############################################################
    #
    # Load image transformations followed by the data
    #
    ############################################################
    
    # Image transformations
    
    tforms = []
    transform_config = config["transforms"]
    
    if "Rescale" in transform_config:
        rescale = transform_config["Rescale"]
        tforms.append(Rescale(rescale))
    if "Normalize" in transform_config:
        tforms.append(Normalize())
    if "ToTensor" in transform_config:
        tforms.append(ToTensor(device))
    if "RandomCrop" in transform_config:
        tforms.append(RandomCrop())
    if "Standardize" in transform_config:
        tforms.append(Standardize())
    
    transform = transforms.Compose(tforms)
    
    # Data readers for train/test
    
    train_gen = HologramDataset(
        path_data, num_particles, "train", subset, 
        output_cols, maxnum_particles = maxnum_particles, 
        transform = transform
    )
    
    train_scalers = train_gen.get_transform()

    valid_gen = HologramDataset(
        path_data, num_particles, "test", subset, 
        output_cols, scaler = train_scalers, 
        maxnum_particles = maxnum_particles,
        transform = transform
    )
    
    # Data iterators using multiprocessing for train/test
    
    logging.info(f"Loading training data iterator using {workers} workers")
    
    dataloader = DataLoader(
        train_gen,
        batch_size = batch_size,
        shuffle = True,
        num_workers = workers
    )
    
    valid_dataloader = DataLoader(
        valid_gen,
        batch_size = batch_size,
        shuffle = False,
        num_workers = workers
    )
    
    ############################################################
    #
    # Load the model
    #
    ############################################################
            
    vae = CNN_VAE(**config["model"]).to(device)
    
    # Print the total number of model parameters
    logging.info(
        f"The model contains {count_parameters(vae)} parameters"
    )
    
    if retrain:
        vae = vae.load_state_dict(
            saved_model_optimizer["model_state_dict"], map_location=device
        )
        logging.info(f"Loaded model weights from {model_save_path}")
        
    
    ############################################################
    #
    # Load the optimizer (after model gets mounted onto GPU)
    #
    ############################################################
    
    optimizer_config = config["optimizer"]
    learning_rate = optimizer_config["lr"] if not retrain else saved_model_optimizer["lr"]
    optimizer_type = optimizer_config["type"]
    
    if optimizer_type == "lookahead-diffgrad":
        optimizer = LookaheadDiffGrad(vae.parameters(), lr=learning_rate)
    elif optimizer_type == "diffgrad":
        optimizer = DiffGrad(vae.parameters(), lr=learning_rate)
    elif optimizer_type == "lookahead-radam":
        optimizer = LookaheadRAdam(vae.parameters(), lr=learning_rate)
    elif optimizer_type == "radam":
        optimizer = RAdam(vae.parameters(), lr=learning_rate)
    elif optimizer_type == "adam":
        optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)
    elif optimizer_type == "sgd":
        optimizer = torch.optim.SGD(vae.parameters(), lr=learning_rate)
    else:
        logging.warning(
            f"Optimzer type {optimizer_type} is unknown. Exiting with error."
        )
        sys.exit(1)
        
    logging.info(
        f"Loaded the {optimizer_type} optimizer with learning rate {learning_rate}"
    )
    
    if retrain:
        optimizer = optimizer.load_state_dict(
            saved_model_optimizer["optimizer_state_dict"], map_location=device
        )
        logging.info(f"Loaded optimizer weights from {model_save_path}")
    
    ############################################################
    #
    # Load callbacks
    #
    ############################################################
    
    # Initialize LR annealing scheduler 
    schedule_config = config["callbacks"]["ReduceLROnPlateau"]
    
    logging.info(
        f"Loaded ReduceLROnPlateau learning rate annealer with patience {schedule_config['patience']}"
    )
        
    scheduler = ReduceLROnPlateau(optimizer,
                                  mode=schedule_config["mode"],
                                  patience=schedule_config["patience"],
                                  factor=schedule_config["factor"],
                                  min_lr=schedule_config["min_lr"],
                                  verbose=schedule_config["verbose"])

    # Early stopping
    checkpoint_config = config["callbacks"]["EarlyStopping"]
    early_stopping = EarlyStopping(path=model_save_path, 
                                   patience=checkpoint_config["patience"], 
                                   verbose=checkpoint_config["verbose"])
    
    # Write metrics to csv each epoch
    metrics_logger = MetricsLogger(path_save, reload = retrain)
    
    ############################################################
    #
    # Load the trainer class
    #
    ############################################################

    logging.info("Loading trainer object")
    
    trainer = Trainer(
        vae,
        optimizer,
        train_gen,
        valid_gen, 
        dataloader, 
        valid_dataloader,
        batch_size,
        path_save,
        device,
        test_image
    )
    
    ############################################################
    #
    # Train the model
    #
    ############################################################
    
    logging.info(
        f"Training the model for up to {epochs} epochs starting at epoch {start_epoch}"
    )
    
    for epoch in range(start_epoch, epochs):
        
        train_loss, train_bce, train_kld = trainer.train_one_epoch(epoch)
        test_loss, test_bce, test_kld = trainer.test(epoch)

        scheduler.step(test_loss)
        early_stopping(epoch, test_loss, trainer.model, trainer.optimizer)

        # Write results to the callback logger 
        result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_bce": train_bce,
            "train_kld": train_kld,
            "valid_loss": test_loss,
            "valid_bce": test_bce,
            "valid_kld": test_kld,
            "lr": early_stopping.print_learning_rate(trainer.optimizer)
        }
        metrics_logger.update(result)

        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    ############################################################
    #
    # Make a video of the progress
    #
    ############################################################

    generate_video(f"{path_save}", "generated_hologram.avi") 