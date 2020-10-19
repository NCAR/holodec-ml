import os
import sys
import yaml
import tqdm
import torch
import pickle
import logging

from torchvision.utils import save_image
from holodecml.vae.losses import *
import numpy as np

logger = logging.getLogger(__name__)


def LoadTrainer(trainer_type, 
                model, 
                optimizer, 
                train_gen, 
                valid_gen, 
                dataloader, 
                valid_dataloader, 
                device, 
                config):
    
    logger.info(f"Loading trainer-type {trainer_type}")
    if trainer_type in ["vae", "att-vae"]:
        return BaseTrainer(
            model=model,
            optimizer=optimizer,
            train_gen=train_gen,
            valid_gen=valid_gen,
            dataloader=dataloader,
            valid_dataloader=valid_dataloader,
            device=device,
            **config
        )
    elif trainer_type == "encoder-vae":
        return BaseEncoderTrainer(
            model=model,
            optimizer=optimizer,
            train_gen=train_gen,
            valid_gen=valid_gen,
            dataloader=dataloader,
            valid_dataloader=valid_dataloader,
            device=device,
            **config
        )
    else:
        logger.info(
            f"Unsupported trainer type {trainer_type}. Choose from vae, att-vae, or encoder-vae. Exiting.")
        sys.exit(1)


class BaseTrainer:

    def __init__(self,
                 model,
                 optimizer,
                 train_gen,
                 valid_gen,
                 dataloader,
                 valid_dataloader,
                 start_epoch=0,
                 epochs=100,
                 device="cpu",
                 clip=1.0,
                 alpha=1.0,
                 beta=1.0,
                 kld_weight=[],
                 path_save="./",
                 test_image=None,
                 save_test_image_every=1):

        self.model = model
        self.optimizer = optimizer
        self.train_gen = train_gen
        self.valid_gen = valid_gen
        self.dataloader = dataloader
        self.valid_dataloader = valid_dataloader
        self.batch_size = dataloader.batch_size
        self.path_save = path_save
        self.device = device

        self.start_epoch = start_epoch
        self.epochs = epochs

        self.alpha = alpha
        self.beta = beta

        self.kld_weight = kld_weight
        if len(kld_weight) == 0:
            self.kld_weight = [
                self.batch_size/self.train_gen.__len__(),
                self.batch_size/self.valid_gen.__len__()
            ]
        self.criterion_train = SymmetricMSE(
            self.alpha, self.beta, self.kld_weight[0]
        )
        self.criterion_test = SymmetricMSE(
            self.alpha, self.beta, self.kld_weight[1]
        )

        self.test_image = test_image
        self.save_test_image_every = save_test_image_every

        # Gradient clipping through hook registration
        for p in self.model.parameters():
            if p.requires_grad:
                p.register_hook(lambda grad: torch.clamp(grad, -clip, clip))
        logger.info(f"Clipping gradients to range [-{clip}, {clip}]")

        # Create the save directory if it does not exist
        try:
            os.makedirs(path_save)
        except:
            pass

    def train_one_epoch(self, epoch):

        self.model.train()
        batches_per_epoch = int(
            np.ceil(self.train_gen.__len__() / self.batch_size))
        batch_group_generator = tqdm.tqdm(
            enumerate(self.dataloader),
            total=batches_per_epoch,
            leave=True
        )

        epoch_losses = {"loss": [], "bce": [], "kld": []}
        for idx, images in batch_group_generator:

            images = images.to(self.device)
            recon_images, mu, logvar = self.model(images)
            loss, bce, kld = self.criterion_train(
                recon_images, images, mu, logvar)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_loss = loss.item()  # / self.batch_size
            bce_loss = bce.item()  # / self.batch_size
            kld_loss = kld.item()  # / self.batch_size

            epoch_losses["loss"].append(batch_loss)
            epoch_losses["bce"].append(bce_loss)
            epoch_losses["kld"].append(kld_loss)

            loss = np.mean(epoch_losses["loss"])
            bce = np.mean(epoch_losses["bce"])
            kld = np.mean(epoch_losses["kld"])

            to_print = "loss: {:.3f} bce: {:.3f} kld: {:.3f}".format(
                loss, bce, kld)
            batch_group_generator.set_description(to_print)
            batch_group_generator.update()

        return loss, bce, kld

    def test(self, epoch):

        self.model.eval()
        batches_per_epoch = int(
            np.ceil(self.valid_gen.__len__() / self.batch_size))

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
                loss, bce, kld = self.criterion_test(
                    recon_images, images, mu, logvar)

                batch_loss = loss.item()  # / self.batch_size
                bce_loss = bce.item()  # / self.batch_size
                kld_loss = kld.item()  # / self.batch_size

                epoch_losses["loss"].append(batch_loss)
                epoch_losses["bce"].append(bce_loss)
                epoch_losses["kld"].append(kld_loss)

                loss = np.mean(epoch_losses["loss"])
                bce = np.mean(epoch_losses["bce"])
                kld = np.mean(epoch_losses["kld"])

                to_print = "val_loss: {:.3f} val_bce: {:.3f} val_kld: {:.3f}".format(
                    loss, bce, kld)
                batch_group_generator.set_description(to_print)
                batch_group_generator.update()

            if os.path.isfile(self.test_image) and (epoch % self.save_test_image_every == 0):
                with open(self.test_image, "rb") as fid:
                    pic = pickle.load(fid)
                self.compare(epoch, pic)

        return loss, bce, kld

    def compare(self, epoch, x):
        x = x.to(self.device)
        recon_x, _, _ = self.model(x)
        if x.shape[0] > 1:
            for k, (_x, _x_recon_x) in enumerate(zip(x[:5], recon_x[:5])):
                _x = torch.unsqueeze(_x, 0)
                _x_recon_x = torch.unsqueeze(_x_recon_x, 0)
                compare_x = torch.cat([_x, _x_recon_x])
                save_image(compare_x.data.cpu(),
                           f'{self.path_save}/image_epoch_{epoch}_{k}.png')
        else:
            compare_x = torch.cat([x, recon_x])
            save_image(compare_x.data.cpu(),
                       f'{self.path_save}/image_epoch_{epoch}.png')

    def train(self,
              scheduler,
              early_stopping,
              metrics_logger):

        logger.info(
            f"Training the model for up to {self.epochs} epochs starting at epoch {self.start_epoch}"
        )

        flag = isinstance(
            scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

        for epoch in range(self.start_epoch, self.epochs):

            train_loss, train_bce, train_kld = self.train_one_epoch(epoch)
            test_loss, test_bce, test_kld = self.test(epoch)

            scheduler.step(test_loss if flag else epoch)
            early_stopping(epoch, test_loss, self.model, self.optimizer)

            # Write results to the callback logger
            result = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_bce": train_bce,
                "train_kld": train_kld,
                "valid_loss": test_loss,
                "valid_bce": test_bce,
                "valid_kld": test_kld,
                "lr": early_stopping.print_learning_rate(self.optimizer)
            }
            metrics_logger.update(result)

            if early_stopping.early_stop:
                logger.info("Early stopping")
                break


class BaseEncoderTrainer:

    def __init__(self,
                 model,
                 optimizer,
                 train_gen,
                 valid_gen,
                 dataloader,
                 valid_dataloader,
                 start_epoch=0,
                 epochs=100,
                 device="cpu",
                 clip=1.0,
                 alpha=1.0,
                 beta=1.0,
                 path_save="./",
                 test_image=None,
                 save_test_image_every=1):

        self.model = model
        self.optimizer = optimizer
        self.train_gen = train_gen
        self.valid_gen = valid_gen
        self.dataloader = dataloader
        self.valid_dataloader = valid_dataloader
        self.batch_size = dataloader.batch_size
        self.path_save = path_save
        self.device = device

        self.start_epoch = start_epoch
        self.epochs = epochs

        self.alpha = alpha
        self.beta = beta
        
        #self.criterion_train = weighted_mse_loss
        #self.criterion_test = nn.MSELoss()
        self.binary = ("binary" in self.train_gen.output_cols)

        self.test_image = test_image
        self.save_test_image_every = save_test_image_every

        # Gradient clipping through hook registration
        for p in self.model.parameters():
            if p.requires_grad:
                p.register_hook(lambda grad: torch.clamp(grad, -clip, clip))
        logger.info(f"Clipping gradients to range [-{clip}, {clip}]")

        # Create the save directory if it does not exist
        try:
            os.makedirs(path_save)
        except:
            pass
        
        
    def wmse(self, input, target, weight = None):
        residuals = (input - target) ** 2
        if weight is not None:
            residuals *= weight
        return residuals.mean()
        
    def criterion(self, input, target, weight = None):
        mse_loss = 0
        accuracy = None
        weight = weight.float() if weight != None else None
        for task in target:
            input[task] = input[task].float()
            target[task] = target[task].float()
            if task != "binary":
                mse_loss += self.wmse(input[task], target[task], weight)
            else:
                mse_loss += nn.BCELoss(weight)(input[task], target[task])
                # Compute BCE accuracy
                predicted = input[task].clone().detach()
                truth = target[task].clone().detach()
                condition = (predicted >= 0.5)
                predicted[~condition] = 0.0
                predicted[condition] = 1.0
                accuracy = (predicted == truth).float().mean()
        return mse_loss, accuracy

    def train_one_epoch(self, epoch):

        self.model.train()
        batches_per_epoch = int(
            np.ceil(self.train_gen.__len__() / self.batch_size))
        batch_group_generator = tqdm.tqdm(
            enumerate(self.dataloader),
            total=batches_per_epoch,
            leave=True
        )

        epoch_losses = {"loss": [], "accuracy": []}
        for idx, (images, y_out, w_out) in batch_group_generator:
            images = images.to(self.device)
            y_out = {task: value.to(self.device) for task, value in y_out.items()}
            w_out = w_out.to(self.device)
            y_pred = self.model(images)
            loss, accuracy = self.criterion(y_pred, y_out, w_out)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_loss = loss.item()
            epoch_losses["loss"].append(batch_loss)
            loss = np.mean(epoch_losses["loss"])
            
            to_print = f"loss: {loss:.3f}"
            if accuracy is not None:
                epoch_losses["accuracy"].append(accuracy.item())
                accuracy = np.mean(epoch_losses["accuracy"])
                to_print += f" accuracy: {accuracy:.3f}"
            
            batch_group_generator.set_description(to_print)
            batch_group_generator.update()

        return loss, accuracy

    def test(self, epoch):

        self.model.eval()
        batches_per_epoch = int(
            np.ceil(self.valid_gen.__len__() / self.batch_size))

        with torch.no_grad():

            batch_group_generator = tqdm.tqdm(
                enumerate(self.valid_dataloader),
                total=batches_per_epoch,
                leave=True
            )

            epoch_losses = {"loss": [], "accuracy": []}
            for idx, (images, y_out, w_out) in batch_group_generator:
                images = images.to(self.device)
                y_out = {task: value.to(self.device) for task, value in y_out.items()}
                y_pred = self.model(images)
                loss, accuracy = self.criterion(y_pred, y_out)

                batch_loss = loss.item()  # / self.batch_size
                epoch_losses["loss"].append(batch_loss)
                loss = np.mean(epoch_losses["loss"])
                
                to_print = f"valid_loss: {loss:.3f}"
                if accuracy is not None:
                    epoch_losses["accuracy"].append(accuracy.item())
                    accuracy = np.mean(epoch_losses["accuracy"])
                    to_print += f" accuracy: {accuracy:.3f}"
                
                batch_group_generator.set_description(to_print)
                batch_group_generator.update()
            
        return loss, accuracy

    def train(self,
              scheduler,
              early_stopping,
              metrics_logger):

        logger.info(
            f"Training the model for up to {self.epochs} epochs starting at epoch {self.start_epoch}"
        )

        flag = isinstance(
            scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

        for epoch in range(self.start_epoch, self.epochs):

            train_loss, train_accuracy = self.train_one_epoch(epoch)
            test_loss, test_accuracy = self.test(epoch)

            scheduler.step(test_loss if flag else epoch)
            early_stopping(epoch, test_loss, self.model, self.optimizer)

            # Write results to the callback logger
            result = {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": test_loss,
                "lr": early_stopping.print_learning_rate(self.optimizer)
            }
            if train_accuracy is not None:
                result["train_acc"] = train_accuracy
                result["valid_acc"] = test_accuracy
            metrics_logger.update(result)

            if early_stopping.early_stop:
                logger.info("Early stopping")
                break
                
        return result