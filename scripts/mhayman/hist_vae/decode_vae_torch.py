"""
Created May 20, 2021
neural net to decode the VAE latent space
of holodec images.

Switching to pytorch based architecture


"""
import warnings
warnings.filterwarnings("ignore")

import os
import sys

import numpy as np

import yaml
import tqdm
import torch
import pickle
import logging

from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from typing import List, Dict, Callable, Union, Any, TypeVar, Tuple
# from multiprocessing import cpu_count

from torch import nn


def loss_fn(y_pred, y_actual,l1=0):
    mse = torch.sum(torch.square(y_pred-y_actual))
    l1reg = l1*torch.sum(torch.abs(y_pred))
    loss_out = mse + l1reg
    return loss_out, mse, l1reg

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
                 test_image = None,
                 l1=0):
        
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

        epoch_losses = {"loss": [], "mse": [], "l1reg": []}
        for idx, images in batch_group_generator:

            images = images.to(self.device)
            recon_images, mu, logvar = self.model(images)
            
            loss,mse,l1reg = loss_fn(y_pred, y_act, l1=l1)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_loss = loss.item() / batch_size
            batch_mse = mse.item() / batch_size
            batch_l1reg = l1reg.item() / batch_size

            epoch_losses["loss"].append(batch_loss)
            epoch_losses["mse"].append(batch_mse)
            epoch_losses["l1reg"].append(batch_l1reg)

            loss = np.mean(epoch_losses["loss"])
            mse = np.mean(epoch_losses["mse"])
            l1reg = np.mean(epoch_losses["l1reg"])

            to_print = "loss: {:.3f} mse: {:.3f} l1reg: {:.3f}".format(loss, mse, l1reg)
            batch_group_generator.set_description(to_print)
            batch_group_generator.update()

        return loss, mse, l1reg


    def test(self, epoch):

        self.model.eval()
        batches_per_epoch = int(np.ceil(self.valid_gen.__len__() / self.batch_size))

        with torch.no_grad():

            batch_group_generator = tqdm.tqdm(
                enumerate(self.valid_dataloader),
                total=batches_per_epoch, 
                leave=True
            )

            epoch_losses = {"loss": [], "mse": [], "l1reg": []}
            for idx, images in batch_group_generator:

                images = images.to(self.device)
                recon_images, mu, logvar = self.model(images)
                loss,mse,l1reg = loss_fn(y_pred, y_act, l1=l1)

                batch_loss = loss.item() / batch_size
                batch_mse = mse.item() / batch_size
                batch_l1reg = l1reg.item() / batch_size

                epoch_losses["loss"].append(batch_loss)
                epoch_losses["mse"].append(batch_mse)
                epoch_losses["l1reg"].append(batch_l1reg)

                loss = np.mean(epoch_losses["loss"])
                mse = np.mean(epoch_losses["mse"])
                l1reg = np.mean(epoch_losses["l1reg"])

                to_print = "loss: {:.3f} mse: {:.3f} l1reg: {:.3f}".format(loss, mse, l1reg)
                batch_group_generator.set_description(to_print)
                batch_group_generator.update()

        return loss
    
    
    def compare(self, epoch, x):
        x = x.to(self.device)
        recon_x, _, _ = self.model(x)
        compare_x = torch.cat([x, recon_x])
        save_image(compare_x.data.cpu(), f'{self.path_save}/image_epoch_{epoch}.png')


### Program Inputs

# number of nodes in each dense layer except the last one
# the size of the last layer is determined by the output
# size
layer_nodes = [1000,500,200]  
epochs = 100
###

### setup the data logger

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

###

"""
with open("config.yml") as config_file:
    config = yaml.load(config_file, Loader=yaml.FullLoader)

logging.info(f'Reading parameters from config.yml')
layer_nodes = config["layer_nodes"]
epochs = config["epochs"]
optimizer_config = config["optimizer"]
learning_rate = optimizer_config["lr"]
optimizer_type = optimizer_config["type"]
"""

### Use gpu if available
is_cuda = torch.cuda.is_available()
device = torch.device(torch.cuda.current_device()) if is_cuda else torch.device("cpu")
    
logging.info(f'Preparing to use device {device}')
### 



input_size = np.prod(scaled_train_input.shape[1:])
output_size = np.prod(scaled_train_labels.shape[1:])

class LatentDecoder(nn.Module):
    def __init__(self,in_size,out_size):
        """
        in_size: int
            size of inputs tensor
        out_size: int
            size of outputs tensor
        """
        super(LatentDecoder, self).__init__()

        # create a list of the NN layers
        self.layer_lst = []
        for idx,node_count in enumerate(layer_nodes):
            if idx == 0:
                self.layer_lst+=[nn.Linear(in_size,node_count)]
            elif idx == len(layer_nodes)-1:
                self.layer_lst+=[nn.Linear(layer_nodes[-1],out_size)]
            else:
                self.layer_lst+=[nn.Linear(layer_nodes[idx-1],node_count)]

 
    def forward(self, x):
        for layer in self.layer_lst[:-1]:
            x = nn.ReLU(layer(x))
        output = nn.Sigmoid(self.layer_lst[-1](x))

        return output


decode_model = LatentDecoder(input_size,output_size)

optimizer_config = config["optimizer"]
learning_rate = optimizer_config["lr"]
optimizer_type = optimizer_config["type"]

# if optimizer_type == "lookahead-diffgrad":
#     optimizer = LookaheadDiffGrad(decode_model.parameters(), lr=learning_rate)
# elif optimizer_type == "diffgrad":
#     optimizer = DiffGrad(decode_model.parameters(), lr=learning_rate)
# elif optimizer_type == "lookahead-radam":
#     optimizer = LookaheadRAdam(decode_model.parameters(), lr=learning_rate)
# elif optimizer_type == "radam":
#     optimizer = RAdam(decode_model.parameters(), lr=learning_rate)
if optimizer_type == "adam":
    optimizer = torch.optim.Adam(decode_model.parameters(), lr=learning_rate)
elif optimizer_type == "sgd":
    optimizer = torch.optim.SGD(decode_model.parameters(), lr=learning_rate)
else:
    logging.warning(
        f"Optimzer type {optimizer_type} is unknown. Exiting with error."
    )
    sys.exit(1)





### Initialize LR annealing scheduler 
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




### Training Loop

logging.info(
    f"Training the model for up to {epochs} epochs"
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