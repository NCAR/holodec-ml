import os
import sys
import copy
import yaml
import tqdm
import torch
import pickle
import random
import logging
import itertools
import numpy as np

from torchvision.utils import save_image
from holodecml.torch.losses import *
from holodecml.torch.models import *
from holodecml.torch.optimizers import *
from holodecml.torch.beam_search import *

# https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer
from ranger import Ranger


logger = logging.getLogger(__name__)


def LoadTrainer(train_gen, 
                valid_gen, 
                dataloader, 
                valid_dataloader, 
                device, 
                config):
    
    if "type" not in config["trainer"]:
        logger.warning("In order to load a model you must supply the type field.")
        raise OSError("Failed to load a trainer. Exiting")
        
    trainer_type = config["trainer"].pop("type")
    logger.info(f"Loading trainer-type {trainer_type}")
    
    if trainer_type in ["vae", "att-vae"]:
        return BaseTrainer(
            train_gen=train_gen,
            valid_gen=valid_gen,
            dataloader=dataloader,
            valid_dataloader=valid_dataloader,
            model_conf = config["model"], 
            optimizer_conf = config["optimizer"],
            device=device,
            **config["trainer"]
        )
    elif trainer_type == "encoder-vae":
        return BaseEncoderTrainer(
            train_gen=train_gen,
            valid_gen=valid_gen,
            dataloader=dataloader,
            valid_dataloader=valid_dataloader,
            model_conf = config["model"], 
            optimizer_conf = config["optimizer"],
            device=device,
            **config["trainer"]
        )
    elif trainer_type == "decoder-vae":
        return DecoderTrainer(
            train_gen=train_gen,
            valid_gen=valid_gen,
            dataloader=dataloader,
            valid_dataloader=valid_dataloader,
            vae_conf=config["vae"],
            decoder_conf=config["decoder"],
            regressor_conf=config["regressor"],
            decoder_optimizer_conf=config["rnn_optimizer"],
            regressor_optimizer_conf=config["particle_optimizer"],
            device=device,
            **config["trainer"]
        )
    else:
        logger.info(
            f"Unsupported trainer type {trainer_type}. Choose from vae, att-vae, encoder-vae, or decoder-vae. Exiting.")
        sys.exit(1)


class BaseTrainer:

    def __init__(self,
                 train_gen,
                 valid_gen,
                 dataloader,
                 valid_dataloader,
                 model_conf,
                 optimizer_conf,
                 start_epoch=0,
                 epochs=100,
                 batches_per_epoch = 1e10,
                 device="cpu",
                 clip=1.0,
                 alpha=1.0,
                 beta=1.0,
                 kld_weight=[],
                 path_save="./",
                 test_image=None,
                 training_loss=None,
                 save_test_image_every=1):
        
        # Initialize and build a model
        self.model = LoadModel(model_conf)
        self.model.build()
        self.model = self.model.to(device)
        
        # Initialize the optimizer
        self.optimizer = LoadOptimizer(
            optimizer_conf, 
            self.model.parameters()
        )
        
        self.train_gen = train_gen
        self.valid_gen = valid_gen
        self.dataloader = dataloader
        self.valid_dataloader = valid_dataloader
        self.batch_size = dataloader.batch_size
        self.batches_per_epoch = batches_per_epoch 
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
            
        logger.info(f"Using mse/kld weights {self.alpha}/{self.beta} during training")
        
        
        if training_loss == "mae":
            self.criterion_train = SymmetricMAE(
                self.alpha, self.beta, self.kld_weight[0]
            )
        elif training_loss == "mse":  
            self.criterion_train = SymmetricMSE(
                self.alpha, self.beta, self.kld_weight[0]
            )
        elif training_loss == "logcosh":
            self.criterion_train = SymmetricLogCosh(
                self.alpha, self.beta, self.kld_weight[0]
            )
        elif training_loss == "xsigmoid":
            self.criterion_train = SymmetricXSigmoid(
                self.alpha, self.beta, self.kld_weight[0]
            )
        elif training_loss == "xtanh":
            self.criterion_train = SymmetricXTanh(
                self.alpha, self.beta, self.kld_weight[0]
            )
        else:
            self.criterion_train = SymmetricMSE(
                self.alpha, self.beta, self.kld_weight[0]
            )
            
        self.criterion_test = SymmetricMSE(
            1.0, 1.0, self.kld_weight[1]
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
        
        if batches_per_epoch > self.batches_per_epoch:
            batches_per_epoch = self.batches_per_epoch
        
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

            to_print = "train_mse: {:.3f} train_bce: {:.3f} train_kld: {:.3f}".format(
                loss, bce, kld)
            batch_group_generator.set_description(to_print)
            batch_group_generator.update()
            
            if idx % batches_per_epoch == 0 and idx > 0:
                break
                
            if not np.isfinite(loss):
                logger.warning(f"Ending training early due to an exploding loss {loss}")
                break

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

            epoch_losses = {"loss": [], "bce": [], "kld": [], "mae": []}
            for idx, images in batch_group_generator:
                images = images.to(self.device)
                recon_images, mu, logvar = self.model(images)
                loss, bce, kld = self.criterion_test(
                    recon_images, images, mu, logvar)
                mae = nn.L1Loss()(recon_images, images)

                batch_loss = loss.item()  # / self.batch_size
                bce_loss = bce.item()  # / self.batch_size
                kld_loss = kld.item()  # / self.batch_size
                mae = mae.item()

                epoch_losses["loss"].append(batch_loss)
                epoch_losses["bce"].append(bce_loss)
                epoch_losses["kld"].append(kld_loss)
                epoch_losses["mae"].append(mae)
                

                loss = np.mean(epoch_losses["loss"])
                bce = np.mean(epoch_losses["bce"])
                kld = np.mean(epoch_losses["kld"])
                mae = np.mean(epoch_losses["mae"])

                to_print = "val_mse: {:.3f} val_mae: {:.3f} val_bce: {:.3f} val_kld: {:.3f}".format(
                    loss, mae, bce, kld)
                batch_group_generator.set_description(to_print)
                batch_group_generator.update()
                
                if not np.isfinite(loss):
                    logger.warning(f"Ending validation early due to an exploding loss {loss}")
                    break

            if os.path.isfile(self.test_image) and (epoch % self.save_test_image_every == 0):
                with open(self.test_image, "rb") as fid:
                    pic = pickle.load(fid)
                self.compare(epoch, pic)

        return loss, mae, bce, kld

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
            test_loss, test_mae, test_bce, test_kld = self.test(epoch)

            scheduler.step(test_loss if flag else epoch)
            early_stopping(epoch, test_loss, self.model, self.optimizer)

            # Write results to the callback logger
            result = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_bce": train_bce,
                "train_kld": train_kld,
                "valid_loss": test_loss,
                "valid_mae": test_mae,
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
                 train_gen,
                 valid_gen,
                 dataloader,
                 valid_dataloader,
                 model_conf,
                 optimizer_conf,
                 start_epoch=0,
                 epochs=100,
                 batches_per_epoch=100000000,
                 device="cpu",
                 clip=1.0,
                 alpha=1.0,
                 beta=1.0,
                 path_save="./",
                 test_image=None,
                 save_test_image_every=1):

        # Initialize and build a model
        model_type = model_conf.pop("type")
        self.model = LoadModel(model_type, model_conf)
        self.model.build()
        self.model = self.model.to(device)
        
        # Initialize the optimizer
        optimizer_type = optimizer_conf.pop("type")
        self.optimizer = LoadOptimizer(
            optimizer_type, 
            self.model.parameters(), 
            optimizer_conf["lr"], 
            optimizer_conf["weight_decay"]
        )
        
        self.train_gen = train_gen
        self.valid_gen = valid_gen
        self.dataloader = dataloader
        self.valid_dataloader = valid_dataloader
        self.batch_size = dataloader.batch_size
        self.batches_per_epoch = batches_per_epoch 
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
        
        if self.model.training:
            loss = nn.SmoothL1Loss()(input, target)
            #loss = XSigmoidLoss()(input, target, weight)
        else:
            loss = nn.L1Loss()(input, target)
            #loss = XSigmoidLoss()(input, target, weight)
        return loss
        
    def criterion(self, input, target, weight = None):
        mse_loss = 0
        bce_loss = 0
        accuracy = None
        weight = weight.float() if weight != None else None
        
        if self.model.training:
            idx = (target["binary"] == 1)
        else:
            #predicted = input["binary"].clone().detach()
            #idx = (predicted >= 0.5)
            idx = (target["binary"] == 1)
        
        for task in target:
            input[task] = input[task].float()
            target[task] = target[task].float()
            if task != "binary":
                if self.model.training:
                    mse_loss += self.wmse(input[task], target[task], weight)
                else:
                    mse_loss += nn.L1Loss()(input[task][idx], target[task][idx])
            else:
                bce_loss += nn.BCELoss()(input[task], target[task])
                # Compute BCE accuracy
                predicted = input[task][idx].clone().detach()
                truth = target[task][idx].clone().detach()
                condition = (predicted >= 0.5)
                predicted[~condition] = 0.0
                predicted[condition] = 1.0
                accuracy = (predicted == truth).float().mean()
        return mse_loss, bce_loss, accuracy

    def train_one_epoch(self, epoch):

        self.model.train()
        batches_per_epoch = int(
            np.ceil(self.train_gen.__len__() / self.batch_size))
        
        bce_weight = 1.0 #/ batches_per_epoch
        
        if batches_per_epoch > self.batches_per_epoch:
            batches_per_epoch = self.batches_per_epoch
        
        batch_group_generator = tqdm.tqdm(
            enumerate(self.dataloader),
            total=batches_per_epoch,
            leave=True
        )
        
        epoch_losses = {"loss": [], "mse": [], "bce": [], "accuracy": []}
        for idx, (images, y_out, w_out) in batch_group_generator:
            
            images = images.to(self.device)
            
            y_out = {task: value.to(self.device) for task, value in y_out.items()}
            w_out = w_out.to(self.device)
            y_pred = self.model(images)
            
            #mse_loss, bce_loss, accuracy = self.criterion(y_pred, y_out, w_out)
            #loss = self.alpha * mse_loss + self.beta * bce_weight * bce_loss
            #loss /= (self.alpha + self.beta + 1e-8)
            
            mse_loss = 0 
            for task in ["x", "y", "z", "d"]:
                #mse_loss += torch.sum(w_out * torch.abs(y_pred[task] - y_out[task])) / torch.sum(w_out)
                mse_loss += nn.SmoothL1Loss()(w_out * y_pred[task].float(), w_out * y_out[task].float()) / torch.sum(w_out)
            bce_loss = nn.BCELoss(w_out)(y_pred["binary"].float(), y_out["binary"].float())

            predicted = y_pred["binary"].clone().detach()
            truth = y_out["binary"].clone().detach()
            condition = (predicted >= 0.5)
            predicted[~condition] = 0.0
            predicted[condition] = 1.0
            accuracy = (predicted == truth).float().mean()
            
            loss = self.alpha * mse_loss + self.beta * bce_weight * bce_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_losses["mse"].append(mse_loss.item())
            epoch_losses["bce"].append(bce_loss.item())
            epoch_losses["loss"].append(loss.item())
            mse = np.mean(epoch_losses["mse"])
            bce = np.mean(epoch_losses["bce"])
            loss = np.mean(epoch_losses["loss"])
            
            to_print = f"Epoch: {epoch} loss: {loss:.5f} mse: {mse:.5f} bce: {bce:.5f}"
            if accuracy is not None:
                epoch_losses["accuracy"].append(accuracy.item())
                accuracy = np.mean(epoch_losses["accuracy"])
                to_print += f" acc: {accuracy:.5f}"
            
            batch_group_generator.set_description(to_print)
            batch_group_generator.update()
            
            if idx % batches_per_epoch == 0 and idx > 0:
                break
            
        return loss, mse, bce, accuracy

    def test(self, epoch):
        
        self.model.eval()
        batches_per_epoch = int(
            np.ceil(self.valid_gen.__len__() / self.batch_size))
        
        bce_weight = 1.0 #/ batches_per_epoch

        with torch.no_grad():

            batch_group_generator = tqdm.tqdm(
                enumerate(self.valid_dataloader),
                total=batches_per_epoch,
                leave=True
            )

            epoch_losses = {"loss": [], "mae": [], "bce": [], "accuracy": []}
            for idx, (images, y_out, w_out) in batch_group_generator:
                images = images.to(self.device)
                
                y_out = {task: value.to(self.device) for task, value in y_out.items()}
                w_out = w_out.to(self.device)
                y_pred = self.model(images)
                
                mae_loss = 0.0
                for task in ["x", "y", "z", "d"]:
                    y_pred[task] = y_pred[task].float()
                    y_out[task] = y_out[task].float()
                    mae_loss += torch.sum(w_out * torch.abs(y_pred[task] - y_out[task])) / torch.sum(w_out)
                bce_loss = nn.BCELoss(w_out)(y_pred["binary"].float(), y_out["binary"].float())
                
                predicted = y_pred["binary"].clone().detach()
                truth = y_out["binary"].clone().detach()
                condition = (predicted >= 0.5)
                predicted[~condition] = 0.0
                predicted[condition] = 1.0
                accuracy = (predicted == truth).float().mean()

                loss = mae_loss + bce_weight * bce_loss

                epoch_losses["mae"].append(mae_loss.item())
                epoch_losses["bce"].append(bce_loss.item())
                epoch_losses["loss"].append(loss.item())
                mae = np.mean(epoch_losses["mae"])
                bce = np.mean(epoch_losses["bce"])
                loss = np.mean(epoch_losses["loss"])
                
                to_print = f"Epoch: {epoch} val_loss: {loss:.5f} val_mae: {mae:.5f} val_bce: {bce:.5f}"
                if accuracy is not None:
                    epoch_losses["accuracy"].append(accuracy.item())
                    accuracy = np.mean(epoch_losses["accuracy"])
                    to_print += f" val_acc: {accuracy:.5f}"
                
                batch_group_generator.set_description(to_print)
                batch_group_generator.update()
            
        return loss, mae, bce, accuracy

    def train(self,
              scheduler,
              early_stopping,
              metrics_logger,
              metric = "val_loss"):

        logger.info(
            f"Training the model for up to {self.epochs} epochs starting at epoch {self.start_epoch}"
        )

        flag = isinstance(
            scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

        for epoch in range(self.start_epoch, self.epochs):

            train_loss, train_mse, train_bce, train_accuracy = self.train_one_epoch(epoch)
            test_loss, test_mae, test_bce, test_accuracy = self.test(epoch)
            
            if "val_loss" in metric:
                metric_val = test_loss
            elif "val_mae_loss" in metric:
                metric_val = test_mae
            elif "val_bce_loss" in metric:
                metric_val = test_bce
            elif "val_acc" in metric:
                metric_val = -test_accuracy
            else:
                supported = "val_loss, val_mae_loss, val_bce_loss, val_acc"
                raise ValueError(f"The metric {metric} is not supported. Choose from {supported}")

            scheduler.step(metric_val if flag else epoch)
            early_stopping(epoch, metric_val, self.model, self.optimizer)

            # Write results to the callback logger
            result = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_mse": train_mse,
                "train_bce": train_bce,
                "valid_loss": test_loss,
                "valid_mae": test_mae,
                "valid_bce": test_bce,
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



class DecoderTrainer:
        
    def __init__(self, 
                 train_gen,
                 valid_gen,
                 dataloader,
                 valid_dataloader,
                 vae_conf,
                 decoder_conf,
                 regressor_conf,
                 decoder_optimizer_conf,
                 regressor_optimizer_conf,
                 start_epoch=0,
                 epochs=100,
                 batches_per_epoch=100000000,
                 device="cpu",
                 regressor_loss="mae",
                 clip=2.0,
                 max_grad_norm=2.0,
                 alpha=1.0,
                 beta=1.0,
                 path_save="./",
                 forcing = 0.0,
                 label_smoothing = 0.0,
                 focal_gamma = 0.0,
                 beam_size = 10, 
                 PAD_token = 0,
                 SOS_token = 1,
                 EOS_token = 2):
        
        self.train_gen = train_gen
        self.valid_gen = valid_gen
        self.dataloader = dataloader
        self.valid_dataloader = valid_dataloader
        
        vae_conf = copy.deepcopy(vae_conf)
        decoder_conf = copy.deepcopy(decoder_conf)
        regressor_conf = copy.deepcopy(regressor_conf)
        
        # Build vae        
        vae = LoadModel(vae_conf)
        vae.build()
        self.vae = vae.to(device)
        
        # Build decoder
        decoder_conf["output_size"] = len(train_gen.token_lookup) + 3
        logger.info(
            f"Updating the output size of the RNN decoder to {decoder_conf['output_size']}"
        )
        self.decoder = LoadModel(decoder_conf).to(device)
        self.decoder.build()
        
        # Build regressor
        self.regressor = LoadModel(regressor_conf)
        self.regressor.build(vae_conf["z_dim"] + 2 * decoder_conf["hidden_size"] + 1250)
        self.regressor = self.regressor.to(device)
        self.tasks = self.regressor.tasks
        
        # Load regressor loss 
        self.regressor_loss = LoadLoss(regressor_loss)
        
        # Load RNN optimizer
        self.rnn_optimizer = LoadOptimizer(
            decoder_optimizer_conf,
            [p for p in self.decoder.parameters() if p.requires_grad]
        )
        self.particle_optimizer = LoadOptimizer(
            regressor_optimizer_conf,
            [p for p in self.regressor.parameters() if p.requires_grad]
        )
                
        # Load other attributes
        self.batch_size = dataloader.batch_size
        self.batches_per_epoch = batches_per_epoch 
        self.path_save = path_save
        self.device = device

        self.start_epoch = start_epoch
        self.epochs = epochs
        self.alpha = alpha
        self.beta = beta
        
        self.forcing = forcing
        self.label_smoothing = label_smoothing
        self.focal_gamma = focal_gamma
        
        # Tokenization, beam search and bleu
        self.PAD_token = PAD_token
        self.SOS_token = SOS_token
        self.EOS_token = EOS_token
        
        max_steps = self.valid_gen.maxnum_particles
        self.beam_search = BeamSearch(
            end_index = EOS_token, 
            max_steps = max_steps, 
            beam_size = beam_size
        )
        #self._bleu = BLEU(exclude_indices={PAD_token, EOS_token, SOS_token})
        
        self.max_grad_norm = max_grad_norm
        
        self.train_rnn = True

    def train_one_epoch(self, epoch, use_teacher_forcing):

        self.vae.eval()
        self.decoder.train()
        self.regressor.train()

        batch_size = self.dataloader.batch_size
        batches_per_epoch = int(np.ceil(self.train_gen.__len__() / batch_size))

        if self.batches_per_epoch < batches_per_epoch:
                batches_per_epoch = self.batches_per_epoch

        batch_group_generator = tqdm.tqdm(
            enumerate(self.dataloader), 
            total=batches_per_epoch, 
            leave=True
        )
        
        criterion = WeightedCrossEntropyLoss(
            label_smoothing = self.label_smoothing,
            gamma = self.focal_gamma
        )
        
        diameter_loss = RMSLELoss()

        epoch_losses = {"mse": [], "bce": [], "accuracy": [], 
                        "stop_accuracy": [], "frac": [], "seq_acc": []}
        
        for idx, (images, y_out, w_out) in batch_group_generator:

            images = images.to(self.device)
            y_out = {task: value.to(self.device) for task, value in y_out.items()}
            w_out = w_out.to(self.device)
            
            if hasattr(self.train_gen, 'n_shot'): # Support for n-shot, k-ways
                images = images.transpose(1, 0)
                y_out = {task: value.squeeze(0) for task, value in y_out.items()}
                w_out = w_out.squeeze(0)

            with torch.no_grad():
                # 1. Predict the latent vector and image reconstruction
                z, mu, logvar, encoder_att = self.vae.encode(images)
                image_pred, decoder_att = self.vae.decode(z)

                combined_att = torch.cat([
                    encoder_att[2].flatten(start_dim = 1),
                    decoder_att[0].flatten(start_dim = 1)
                ], 1)
                combined_att = combined_att.clone()

                if self.vae.out_image_channels > 1:
                    z_real = np.sqrt(0.5) * image_pred[:,0,:,:]
                    z_imag = image_pred[:,1,:,:]
                    image_pred = torch.square(z_real) + torch.square(z_imag)
                    image_pred = torch.unsqueeze(image_pred, 1)

            # 2. Predict the number of particles
            decoder_input = torch.LongTensor([self.SOS_token] * w_out.shape[0]).to(self.device)
            encoded_image = z.to(self.device)
            decoder_hidden = encoded_image.clone().reshape((1, w_out.shape[0], encoded_image.shape[-1]))
            
            n_dims = 2 if self.decoder.bidirectional else 1
            n_dims *= self.decoder.n_layers
            if n_dims > 1:
                decoder_hidden = torch.cat([decoder_hidden for k in range(n_dims)])

            target_tensor = w_out.long()
            target_length = w_out.shape[1]
            seq_lens = w_out.max(axis = 1)[0] + 1
            class_weights = torch.ones(w_out.shape).to(self.device)
            
            # Use beam search to get predictions
            predictions, probabilities = self.beam_search.search(
                decoder_input, decoder_hidden, self.decoder
            )

            # Validate on top-1 most likely sequence
            top_preds = predictions[:, 0, :]

            # Compute bleu metric for each sequence in the batch
            for pred, true in zip(top_preds, target_tensor):
                epoch_losses["frac"].append(frac_overlap(pred, true))

            # Reshape the predicted tensor to match with the target_tensor
            ## This will work only if limit the beam search = target size
            B, T = target_tensor.size()
            _, t = top_preds.size()
            if t < T:
                reshaped_preds = torch.zeros(B, T)
                reshaped_preds[:, :t] = top_preds
                reshaped_preds = reshaped_preds.long().to(self.device)
            else:
                reshaped_preds = top_preds
                
            
            # Decode again but force answers from the beam search
            hidden_vectors = []
            accuracy, stop_accuracy, rnn_loss = [], [], []
            
            for di in range(target_length + 1):    
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, seq_lens)
                topv, topi = decoder_output.topk(1)
                
                force = (random.uniform(0, 1) < use_teacher_forcing)
                if force:
                    decoder_input = target_tensor[:, di]
                else:
                    decoder_input = reshaped_preds[:, di].detach()
                    
                c1 = (target_tensor[:, di] != self.PAD_token)
                c2 = (target_tensor[:, di] != self.EOS_token)
                condition = c1 & c2
                real_plus_stop = torch.where(c1)
                real_particles = torch.where(condition)
                stop_token = torch.where(~c2)

                if real_plus_stop[0].size(0) == 0:
                    break
                    
                rnn_loss.append(
                    criterion(
                        decoder_output[real_plus_stop], 
                        target_tensor[:, di][real_plus_stop],
                        class_weights[:, di][real_plus_stop]
                    )
                )
                
                accuracy += [
                    int(i.item()==j.item())
                    for i, j in zip(topi[real_particles], target_tensor[:, di][real_particles])
                ]

                if stop_token[0].size(0) > 0:
                    stop_accuracy += [
                        int(i.item()==j.item()) 
                        for i, j in zip(topi[stop_token], target_tensor[:, di][stop_token])
                    ]
                    
                if real_particles[0].size(0) > 0:
                    token_input = target_tensor[:, di].squeeze() # topi.squeeze()
                    embedding = self.decoder.embed(token_input).squeeze(0)
                    
                    embedding = torch.cat([embedding, torch.mean(decoder_hidden, dim=0)], -1)
                    
                    hidden_vectors.append([real_particles, embedding])

            # Compute error and accuracy after finding closest particles 
            accuracy = np.mean(accuracy)
            epoch_losses["accuracy"].append(accuracy)
            epoch_losses["stop_accuracy"].append(np.mean(stop_accuracy))

            rnn_loss = torch.mean(torch.stack(rnn_loss))
            epoch_losses["bce"].append(rnn_loss.item())    

            if self.train_rnn:
                
                # Normalize the accumulated gradient
                if self.max_grad_norm is not None and self.max_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        self.decoder.parameters(), 
                        self.max_grad_norm
                    )
                
                self.rnn_optimizer.zero_grad()
                rnn_loss.backward()
                self.rnn_optimizer.step()

            if len(hidden_vectors) == 0:
                continue

            # 3. Use particle embeddings to predict (x,y,z,d)
            regressor_loss = []
            true_part, pred_part = [], []
            for di in range(len(hidden_vectors)):
                real_particles, h_vecs = hidden_vectors[di]    
                x_input = torch.cat([h_vecs.detach(), encoded_image, combined_att], axis = 1)            
                particle_attributes = self.regressor(x_input[real_particles])
                loss = []
                for task in self.tasks:
                    _loss = self.regressor_loss(
                         particle_attributes[task].squeeze(1),
                         y_out[task][:, di][real_particles].float()
                    )
                    loss.append(_loss)
                regressor_loss.append(torch.mean(torch.stack(loss)))
            regressor_loss = torch.mean(torch.stack(regressor_loss))
            
            # Compute "order-less" accuracy 
            seq_acc = []
            for (true, pred) in zip(target_tensor, reshaped_preds):
                cond = (true > 2)
                frac = orderless_acc(true[cond], pred[cond])
                seq_acc.append(frac)
            seq_acc = np.mean(seq_acc)
            
            epoch_losses["mse"].append(regressor_loss.item())
            epoch_losses["seq_acc"].append(seq_acc)
            
            
            # Normalize the accumulated gradient
            if self.max_grad_norm is not None and self.max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    self.regressor.parameters(), 
                    self.max_grad_norm
                )

            # Backprop on the regressor model
            self.particle_optimizer.zero_grad()
            regressor_loss.backward()
            self.particle_optimizer.step()

            to_print = "Epoch {} train_bce: {:.3f} train_mse: {:.3f} train_acc: {:.3f} train_stop_acc: {:.3f} train_seq_acc: {:.3f}".format(
                epoch, 
                np.mean(epoch_losses["bce"]), 
                np.mean(epoch_losses["mse"]), 
                np.mean(epoch_losses["accuracy"]), 
                np.mean(epoch_losses["stop_accuracy"]),
                np.mean(epoch_losses["seq_acc"])
            )
            batch_group_generator.set_description(to_print)
            batch_group_generator.update()

            if idx % batches_per_epoch == 0 and idx > 0:
                break

        return epoch_losses
    
    
    def test(self, epoch):
    
        self.vae.eval()
        self.decoder.eval()
        self.regressor.eval()
        
        with torch.no_grad():

            batch_size = self.valid_dataloader.batch_size
            batches_per_epoch = int(np.ceil(self.valid_gen.__len__() / batch_size))

            batch_group_generator = tqdm.tqdm(
                enumerate(self.valid_dataloader), 
                total=batches_per_epoch, 
                leave=True
            )
            
            criterion = WeightedCrossEntropyLoss()
            
            epoch_losses = {"mse": [], "bce": [], "frac": [], 
                            "accuracy": [], "stop_accuracy": [], "seq_acc": []}
            
            for idx, (images, y_out, w_out) in batch_group_generator:
                images = images.to(self.device)
                y_out = {task: value.to(self.device) for task, value in y_out.items()}
                w_out = w_out.to(self.device)
                
                if hasattr(self.valid_gen, 'n_shot'): # Support for n-shot, k-ways
                    images = images.transpose(1, 0)
                    y_out = {task: value.squeeze(0) for task, value in y_out.items()}
                    w_out = w_out.squeeze(0)

                # 1. Predict the latent vector and image reconstruction
                z, mu, logvar, encoder_att = self.vae.encode(images)
                image_pred, decoder_att = self.vae.decode(z)
                
                combined_att = torch.cat([
                    encoder_att[2].flatten(start_dim = 1),
                    decoder_att[0].flatten(start_dim = 1)
                ], 1)
                combined_att = combined_att.clone()

                if self.vae.out_image_channels > 1:
                    z_real = np.sqrt(0.5) * image_pred[:,0,:,:]
                    z_imag = image_pred[:,1,:,:]
                    image_pred = torch.square(z_real) + torch.square(z_imag)
                    image_pred = torch.unsqueeze(image_pred, 1)

                # 2. Predict the number of particles
                decoder_input = torch.LongTensor([self.SOS_token] * w_out.shape[0]).to(self.device)
                encoded_image = z.to(self.device)
                decoder_hidden = encoded_image.clone().reshape((1, w_out.shape[0], encoded_image.shape[-1]))
                
                n_dims = 2 if self.decoder.bidirectional else 1
                n_dims *= self.decoder.n_layers
                if n_dims > 1:
                    decoder_hidden = torch.cat([decoder_hidden for k in range(n_dims)])

                target_tensor = w_out.long()
                target_length = w_out.shape[1]
                seq_lens = w_out.max(axis = 1)[0] + 1
                class_weights = torch.ones(w_out.shape).to(self.device)

                #
                hidden_vectors = []
                bleu, accuracy, stop_accuracy, rnn_loss = [], [], [], []
                
                # Use beam search to get predictions
                predictions, probabilities = self.beam_search.search(
                    decoder_input, decoder_hidden, self.decoder
                )

                # Validate on top-1 most likely sequence
                top_preds = predictions[:, 0, :]
                
                # Compute bleu metric for each sequence in the batch
                for pred, true in zip(top_preds, target_tensor):
                    epoch_losses["frac"].append(frac_overlap(pred, true))
                
                # Reshape the predicted tensor to match with the target_tensor
                ## This will work only if limit the beam search = target size
                B, T = target_tensor.size()
                _, t = top_preds.size()
                if t < T:
                    reshaped_preds = torch.zeros(B, T)
                    reshaped_preds[:, :t] = top_preds
                    reshaped_preds = reshaped_preds.long().to(self.device)
                else:
                    reshaped_preds = top_preds
    
                # Use greedy evaluation to get the loss
                for di in range(target_length + 1):
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                    topv, topi = decoder_output.topk(1)
                    decoder_input = reshaped_preds[:, di].detach()
                    c1 = (target_tensor[:, di] != self.PAD_token)
                    c2 = (target_tensor[:, di] != self.EOS_token)
                    condition = c1 & c2
                    real_plus_stop = torch.where(c1)
                    real_particles = torch.where(condition)
                    stop_token = torch.where(~c2)

                    if real_plus_stop[0].size(0) == 0:
                        break
                                          
                    rnn_loss.append(
                        criterion(
                            decoder_output[real_plus_stop], 
                            target_tensor[:, di][real_plus_stop],
                            class_weights[:, di][real_plus_stop]
                        )
                    )
                    accuracy += [
                        int(i.item()==j.item())
                        for i, j in zip(reshaped_preds[:, di][real_particles], 
                                        target_tensor[:, di][real_particles])
                    ]

                    if stop_token[0].size(0) > 0:
                        stop_accuracy += [
                            int(i.item()==j.item()) 
                            for i, j in zip(reshaped_preds[:, di][stop_token], 
                                            target_tensor[:, di][stop_token])
                        ]

                    if real_particles[0].size(0) > 0:
                        token_input = reshaped_preds[:, di] # topi.squeeze()
                        embedding = self.decoder.embed(token_input).squeeze(0)
                        
                        embedding = torch.cat([embedding, torch.mean(decoder_hidden, dim=0)], -1)
                        
                        hidden_vectors.append([real_particles, embedding])
                        
                epoch_losses["accuracy"].append(np.mean(accuracy))
                epoch_losses["stop_accuracy"].append(np.mean(stop_accuracy))
                rnn_loss = torch.mean(torch.stack(rnn_loss))
                epoch_losses["bce"].append(rnn_loss.item())

                if len(hidden_vectors) == 0:
                    continue

                #3. Use particle embeddings to predict (x,y,z,d)
                regressor_loss = []
                true_part, pred_part, real_part = [], [], []
                for di in range(len(hidden_vectors)):
                    real_particles, h_vecs = hidden_vectors[di]
                    x_input = torch.cat([h_vecs.detach(), encoded_image, combined_att], axis = 1)
                    particle_attributes = self.regressor(x_input[real_particles])
                    batch_true, batch_pred, loss = [], [], []
                    for task in self.tasks:
                        batch_true.append(y_out[task][:, di][real_particles].float())
                        batch_pred.append(particle_attributes[task].squeeze(1))
                        
                        _loss = self.regressor_loss(
                             particle_attributes[task].squeeze(1),
                             y_out[task][:, di][real_particles].float()
                        )
                        loss.append(_loss)
                    regressor_loss.append(torch.mean(torch.stack(loss)))
                    true_part.append(batch_true)
                    pred_part.append(batch_pred)
                    real_part.append(real_particles[0].cpu().numpy())
                
                try: # sometimes this fails
                    regressor_loss = distance_sorted_loss(true_part, pred_part, real_part)
                except:
                    regressor_loss = torch.mean(torch.stack(regressor_loss))
                
                seq_acc = []
                for (true, pred) in zip(target_tensor, reshaped_preds):
                    cond = (true > 2)
                    frac = orderless_acc(true[cond], pred[cond])
                    seq_acc.append(frac)
                seq_acc = np.mean(seq_acc)

                epoch_losses["mse"].append(regressor_loss.item())
                epoch_losses["seq_acc"].append(seq_acc)

                to_print = "Epoch {} val_bce: {:.3f} val_mae: {:.3f} val_acc: {:.3f} val_stop_acc: {:.3f} val_seq_acc: {:.3f}".format(
                    epoch, 
                    np.mean(epoch_losses["bce"]), 
                    np.mean(epoch_losses["mse"]), 
                    np.mean(epoch_losses["accuracy"]),
                    np.mean(epoch_losses["stop_accuracy"]),
                    np.mean(epoch_losses["seq_acc"])
                )

                batch_group_generator.set_description(to_print)
                batch_group_generator.update()

        return epoch_losses
    
    def train(self, 
              scheduler_rnn, 
              scheduler_linear, 
              early_stopping_rnn, 
              early_stopping_linear, 
              metrics_logger):

        flag_rnn = isinstance(scheduler_rnn, torch.optim.lr_scheduler.ReduceLROnPlateau)
        flag_linear = isinstance(scheduler_linear, torch.optim.lr_scheduler.ReduceLROnPlateau)

        for epoch in range(self.start_epoch, self.epochs):    
            tf = 1.0 * (self.forcing) ** epoch
            
            train_losses = self.train_one_epoch(epoch, tf)
            test_losses = self.test(epoch)

            # Write results to the callback logger 
            result = {
                "epoch": epoch,
                "train_bce": np.mean(train_losses["bce"]),
                "valid_bce": np.mean(test_losses["bce"]),
                "train_mse": np.mean(train_losses["mse"]),
                "valid_mae": np.mean(test_losses["mse"]),
                "train_acc": np.mean(train_losses["accuracy"]),
                "valid_acc": np.mean(test_losses["accuracy"]),
                "train_stop_acc": np.mean(train_losses["stop_accuracy"]),
                "valid_stop_acc": np.mean(test_losses["stop_accuracy"]),                
                "train_seq_acc": np.mean(train_losses["seq_acc"]),
                "valid_seq_acc": np.mean(test_losses["seq_acc"]),
                "train_frac_overlap": np.mean(train_losses["frac"]),
                "valid_frac_overlap": np.mean(test_losses["frac"]),
                "lr_rnn": early_stopping_rnn.print_learning_rate(self.rnn_optimizer),
                "lr_linear": early_stopping_linear.print_learning_rate(self.particle_optimizer),
                "forcing_value": tf
            }
            metrics_logger.update(result)

            if early_stopping_rnn.early_stop:# and early_stopping_linear.early_stop:
                self.train_rnn = False
            if early_stopping_linear.early_stop:
                logger.info("Early stopping")
                break

            scheduler_rnn.step(1.0-result["valid_seq_acc"] if flag_rnn else (1 + epoch))
            scheduler_linear.step(result["valid_mae"] if flag_linear else (1 + epoch))
            
            early_stopping_rnn(epoch, 1.0-result["valid_seq_acc"], self.decoder, self.rnn_optimizer)
            early_stopping_linear(epoch, result["valid_mae"], self.regressor, self.particle_optimizer)
            
# class DecoderTrainer:
    
#     def __init__(self, 
#                  train_gen,
#                  valid_gen,
#                  dataloader,
#                  valid_dataloader,
#                  vae_conf,
#                  decoder_conf,
#                  regressor_conf,
#                  decoder_optimizer_conf,
#                  regressor_optimizer_conf,
#                  start_epoch=0,
#                  epochs=100,
#                  batches_per_epoch=100000000,
#                  device="cpu",
#                  regressor_loss="mae",
#                  clip=2.0,
#                  max_grad_norm=2.0,
#                  alpha=1.0,
#                  beta=1.0,
#                  path_save="./",
#                  forcing = 0.0,
#                  label_smoothing = 0.0,
#                  focal_gamma = 0.0,
#                  beam_size = 10, 
#                  PAD_token = 0,
#                  SOS_token = 1,
#                  EOS_token = 2):
        
#         self.train_gen = train_gen
#         self.valid_gen = valid_gen
#         self.dataloader = dataloader
#         self.valid_dataloader = valid_dataloader
        
#         vae_conf = copy.deepcopy(vae_conf)
#         decoder_conf = copy.deepcopy(decoder_conf)
#         regressor_conf = copy.deepcopy(regressor_conf)
        
#         # Build vae        
#         vae = LoadModel(vae_conf)
#         vae.build()
#         self.vae = vae.to(device)
        
#         # Build decoder
#         decoder_conf["output_size"] = len(train_gen.token_lookup) + 3
#         logger.info(
#             f"Updating the output size of the RNN decoder to {decoder_conf['output_size']}"
#         )
#         self.decoder = LoadModel(decoder_conf).to(device)
#         self.decoder.build()
#         #self.decoder = DecoderRNN(**decoder_conf).to(device)
        
#         # Build regressor
#         #self.regressor = DenseNet2(**regressor_conf)
#         self.regressor = LoadModel(regressor_conf)
#         self.regressor.build(vae_conf["z_dim"] + decoder_conf["hidden_size"] + 1250)
#         self.regressor = self.regressor.to(device)
#         self.tasks = self.regressor.tasks
        
#         # Load regressor loss 
#         self.regressor_loss = LoadLoss(regressor_loss)
        
#         # Load RNN optimizer
#         self.rnn_optimizer = LoadOptimizer(
#             decoder_optimizer_conf,
#             self.decoder.parameters()
#         )
#         self.particle_optimizer = LoadOptimizer(
#             regressor_optimizer_conf,
#             self.regressor.parameters()
#         )

# #         self.rnn_optimizer = LookaheadRAdam(
# #             self.decoder.parameters(), 
# #             **decoder_optimizer_conf
# #         )
# #         self.particle_optimizer = LookaheadRAdam(
# #             self.regressor.parameters(), 
# #             **regressor_optimizer_conf
# #         )
        
#         # Load other attributes
#         self.batch_size = dataloader.batch_size
#         self.batches_per_epoch = batches_per_epoch 
#         self.path_save = path_save
#         self.device = device

#         self.start_epoch = start_epoch
#         self.epochs = epochs
#         self.alpha = alpha
#         self.beta = beta
        
#         self.forcing = forcing
#         self.label_smoothing = label_smoothing
#         self.focal_gamma = focal_gamma
        
#         # Tokenization, beam search and bleu
#         self.PAD_token = PAD_token
#         self.SOS_token = SOS_token
#         self.EOS_token = EOS_token
        
#         max_steps = self.valid_gen.maxnum_particles
#         self.beam_search = BeamSearch(
#             end_index = EOS_token, 
#             max_steps = max_steps, 
#             beam_size = beam_size
#         )
#         #self._bleu = BLEU(exclude_indices={PAD_token, EOS_token, SOS_token})
        
#         self.max_grad_norm = max_grad_norm
        
#         self.train_rnn = True

#     def train_one_epoch(self, epoch, use_teacher_forcing):

#         self.vae.eval()
#         self.decoder.train()
#         self.regressor.train()

#         batch_size = self.dataloader.batch_size
#         batches_per_epoch = int(np.ceil(self.train_gen.__len__() / batch_size))

#         if self.batches_per_epoch < batches_per_epoch:
#                 batches_per_epoch = self.batches_per_epoch

#         batch_group_generator = tqdm.tqdm(
#             enumerate(self.dataloader), 
#             total=batches_per_epoch, 
#             leave=True
#         )
        
#         criterion = WeightedCrossEntropyLoss(
#             label_smoothing = self.label_smoothing,
#             gamma = self.focal_gamma
#         )

#         epoch_losses = {"mse": [], "bce": [], "accuracy": [], 
#                         "stop_accuracy": [], "frac": [], "seq_acc": []}
        
#         for idx, (images, y_out, w_out) in batch_group_generator:

#             images = images.to(self.device)
#             y_out = {task: value.to(self.device) for task, value in y_out.items()}
#             w_out = w_out.to(self.device)
            
#             if hasattr(self.train_gen, 'n_shot'): # Support for n-shot, k-ways
#                 images = images.transpose(1, 0)
#                 y_out = {task: value.squeeze(0) for task, value in y_out.items()}
#                 w_out = w_out.squeeze(0)

#             with torch.no_grad():
#                 # 1. Predict the latent vector and image reconstruction
#                 z, mu, logvar, encoder_att = self.vae.encode(images)
#                 image_pred, decoder_att = self.vae.decode(z)

#                 combined_att = torch.cat([
#                     encoder_att[2].flatten(start_dim = 1),
#                     decoder_att[0].flatten(start_dim = 1)
#                 ], 1)
#                 combined_att = combined_att.clone()

#                 if self.vae.out_image_channels > 1:
#                     z_real = np.sqrt(0.5) * image_pred[:,0,:,:]
#                     z_imag = image_pred[:,1,:,:]
#                     image_pred = torch.square(z_real) + torch.square(z_imag)
#                     image_pred = torch.unsqueeze(image_pred, 1)

#             # 2. Predict the number of particles
#             decoder_input = torch.LongTensor([self.SOS_token] * w_out.shape[0]).to(self.device)
#             encoded_image = z.to(self.device)
#             decoder_hidden = encoded_image.clone().reshape((1, w_out.shape[0], encoded_image.shape[-1]))
            
#             n_dims = 2 if self.decoder.bidirectional else 1
#             n_dims *= self.decoder.n_layers
#             if n_dims > 1:
#                 decoder_hidden = torch.cat([decoder_hidden for k in range(n_dims)])

#             target_tensor = w_out.long()
#             target_length = w_out.shape[1]
#             seq_lens = w_out.max(axis = 1)[0] + 1
#             class_weights = torch.ones(w_out.shape).to(self.device)
            
#             # Use beam search to get predictions
#             predictions, probabilities = self.beam_search.search(
#                 decoder_input, decoder_hidden, self.decoder
#             )

#             # Validate on top-1 most likely sequence
#             top_preds = predictions[:, 0, :]

#             # Compute bleu metric for each sequence in the batch
#             for pred, true in zip(top_preds, target_tensor):
#                 epoch_losses["frac"].append(frac_overlap(pred, true))

#             # Reshape the predicted tensor to match with the target_tensor
#             ## This will work only if limit the beam search = target size
#             B, T = target_tensor.size()
#             _, t = top_preds.size()
#             if t < T:
#                 reshaped_preds = torch.zeros(B, T)
#                 reshaped_preds[:, :t] = top_preds
#                 reshaped_preds = reshaped_preds.long().to(self.device)
#             else:
#                 reshaped_preds = top_preds
                
            
#             # Decode again but force answers from the beam search
#             hidden_vectors = []
#             accuracy, stop_accuracy, rnn_loss = [], [], []
            
#             for di in range(target_length + 1):    
#                 decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, seq_lens)
#                 topv, topi = decoder_output.topk(1)
                
#                 if random.uniform(0, 1) < use_teacher_forcing:
#                     decoder_input = target_tensor[:, di]
#                 else:
#                     decoder_input = reshaped_preds[:, di].detach()
                    
#                 c1 = (target_tensor[:, di] != self.PAD_token)
#                 c2 = (target_tensor[:, di] != self.EOS_token)
#                 condition = c1 & c2
#                 real_plus_stop = torch.where(c1)
#                 real_particles = torch.where(condition)
#                 stop_token = torch.where(~c2)

#                 if real_plus_stop[0].size(0) == 0:
#                     break
                    
#                 rnn_loss.append(
#                     criterion(
#                         decoder_output[real_plus_stop], 
#                         target_tensor[:, di][real_plus_stop],
#                         class_weights[:, di][real_plus_stop]
#                     )
#                 )
                
#                 accuracy += [
#                     int(i.item()==j.item())
#                     for i, j in zip(topi[real_particles], target_tensor[:, di][real_particles])
#                 ]

#                 if stop_token[0].size(0) > 0:
#                     stop_accuracy += [
#                         int(i.item()==j.item()) 
#                         for i, j in zip(topi[stop_token], target_tensor[:, di][stop_token])
#                     ]
                    
#                 if real_particles[0].size(0) > 0:
#                     token_input = target_tensor[:, di].squeeze() # topi.squeeze()
#                     embedding = self.decoder.embed(token_input).squeeze(0)
#                     hidden_vectors.append([real_particles, embedding])
                    
#             # Compute error and accuracy after finding closest particles 
#             accuracy = np.mean(accuracy)
#             epoch_losses["accuracy"].append(accuracy)
#             epoch_losses["stop_accuracy"].append(np.mean(stop_accuracy))

#             rnn_loss = torch.mean(torch.stack(rnn_loss))
#             epoch_losses["bce"].append(rnn_loss.item())    

#             if self.train_rnn:
                
#                 # Normalize the accumulated gradient
#                 if self.max_grad_norm is not None and self.max_grad_norm > 0.0:
#                     params = itertools.chain.from_iterable([group['params'] for group in self.rnn_optimizer.param_groups])
#                     torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
                
#                 self.rnn_optimizer.zero_grad()
#                 rnn_loss.backward()
#                 self.rnn_optimizer.step()

#             if len(hidden_vectors) == 0:
#                 continue

#             # 3. Use particle embeddings to predict (x,y,z,d)
#             regressor_loss = []
#             true_part, pred_part = [], []
#             for di in range(len(hidden_vectors)):
#                 real_particles, h_vecs = hidden_vectors[di]    
#                 x_input = torch.cat([h_vecs.detach(), encoded_image, combined_att], axis = 1)            
#                 particle_attributes = self.regressor(x_input[real_particles])
#                 loss = []
#                 for task in self.tasks:
#                     _loss = self.regressor_loss(
#                          particle_attributes[task].squeeze(1),
#                          y_out[task][:, di][real_particles].float()
#                     ) # XSigmoidLoss()
#                     loss.append(_loss)
#                 regressor_loss.append(torch.mean(torch.stack(loss)))
#             regressor_loss = torch.mean(torch.stack(regressor_loss))
            
#             # Compute "order-less" accuracy 
#             seq_acc = []
#             for (true, pred) in zip(target_tensor, reshaped_preds):
#                 cond = (true > 2)
#                 frac = orderless_acc(true[cond], pred[cond])
#                 seq_acc.append(frac)
#             seq_acc = np.mean(seq_acc)
            
#             epoch_losses["mse"].append(regressor_loss.item())
#             epoch_losses["seq_acc"].append(seq_acc)
            
            
#             # Normalize the accumulated gradient
#             if self.max_grad_norm is not None and self.max_grad_norm > 0.0:
#                 params = itertools.chain.from_iterable([group['params'] for group in self.particle_optimizer.param_groups])
#                 torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)

#             # Backprop on the regressor model
#             self.particle_optimizer.zero_grad()
#             regressor_loss.backward()
#             self.particle_optimizer.step()

#             to_print = "Epoch {} train_bce: {:.3f} train_mse: {:.3f} train_acc: {:.3f} train_stop_acc: {:.3f} train_seq_acc: {:.3f}".format(
#                 epoch, 
#                 np.mean(epoch_losses["bce"]), 
#                 np.mean(epoch_losses["mse"]), 
#                 np.mean(epoch_losses["accuracy"]), 
#                 np.mean(epoch_losses["stop_accuracy"]),
#                 np.mean(epoch_losses["seq_acc"])
#             )
#             batch_group_generator.set_description(to_print)
#             batch_group_generator.update()

#             if idx % batches_per_epoch == 0 and idx > 0:
#                 break

#         return epoch_losses
    
    
#     def test(self, epoch):
    
#         self.vae.eval()
#         self.decoder.eval()
#         self.regressor.eval()
        
#         with torch.no_grad():

#             batch_size = self.valid_dataloader.batch_size
#             batches_per_epoch = int(np.ceil(self.valid_gen.__len__() / batch_size))

#             batch_group_generator = tqdm.tqdm(
#                 enumerate(self.valid_dataloader), 
#                 total=batches_per_epoch, 
#                 leave=True
#             )
            
#             criterion = WeightedCrossEntropyLoss()
            
#             epoch_losses = {"mse": [], "bce": [], "frac": [], 
#                             "accuracy": [], "stop_accuracy": [], "seq_acc": []}
            
#             for idx, (images, y_out, w_out) in batch_group_generator:
#                 images = images.to(self.device)
#                 y_out = {task: value.to(self.device) for task, value in y_out.items()}
#                 w_out = w_out.to(self.device)
                
#                 if hasattr(self.valid_gen, 'n_shot'): # Support for n-shot, k-ways
#                     images = images.transpose(1, 0)
#                     y_out = {task: value.squeeze(0) for task, value in y_out.items()}
#                     w_out = w_out.squeeze(0)

#                 # 1. Predict the latent vector and image reconstruction
#                 z, mu, logvar, encoder_att = self.vae.encode(images)
#                 image_pred, decoder_att = self.vae.decode(z)
                
#                 combined_att = torch.cat([
#                     encoder_att[2].flatten(start_dim = 1),
#                     decoder_att[0].flatten(start_dim = 1)
#                 ], 1)
#                 combined_att = combined_att.clone()

#                 if self.vae.out_image_channels > 1:
#                     z_real = np.sqrt(0.5) * image_pred[:,0,:,:]
#                     z_imag = image_pred[:,1,:,:]
#                     image_pred = torch.square(z_real) + torch.square(z_imag)
#                     image_pred = torch.unsqueeze(image_pred, 1)

#                 # 2. Predict the number of particles
#                 decoder_input = torch.LongTensor([self.SOS_token] * w_out.shape[0]).to(self.device)
#                 encoded_image = z.to(self.device)
#                 decoder_hidden = encoded_image.clone().reshape((1, w_out.shape[0], encoded_image.shape[-1]))
                
#                 n_dims = 2 if self.decoder.bidirectional else 1
#                 n_dims *= self.decoder.n_layers
#                 if n_dims > 1:
#                     decoder_hidden = torch.cat([decoder_hidden for k in range(n_dims)])

#                 target_tensor = w_out.long()
#                 target_length = w_out.shape[1]
#                 seq_lens = w_out.max(axis = 1)[0] + 1
#                 class_weights = torch.ones(w_out.shape).to(self.device)

#                 #
#                 hidden_vectors = []
#                 bleu, accuracy, stop_accuracy, rnn_loss = [], [], [], []
                
#                 # Use beam search to get predictions
#                 predictions, probabilities = self.beam_search.search(
#                     decoder_input, decoder_hidden, self.decoder
#                 )

#                 # Validate on top-1 most likely sequence
#                 top_preds = predictions[:, 0, :]
                
#                 # Compute bleu metric for each sequence in the batch
#                 for pred, true in zip(top_preds, target_tensor):
#                     epoch_losses["frac"].append(frac_overlap(pred, true))
                
#                 # Reshape the predicted tensor to match with the target_tensor
#                 ## This will work only if limit the beam search = target size
#                 B, T = target_tensor.size()
#                 _, t = top_preds.size()
#                 if t < T:
#                     reshaped_preds = torch.zeros(B, T)
#                     reshaped_preds[:, :t] = top_preds
#                     reshaped_preds = reshaped_preds.long().to(self.device)
#                 else:
#                     reshaped_preds = top_preds
    
#                 # Use greedy evaluation to get the loss
#                 for di in range(target_length + 1):
#                     decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
#                     topv, topi = decoder_output.topk(1)
#                     decoder_input = reshaped_preds[:, di].detach()
#                     c1 = (target_tensor[:, di] != self.PAD_token)
#                     c2 = (target_tensor[:, di] != self.EOS_token)
#                     condition = c1 & c2
#                     real_plus_stop = torch.where(c1)
#                     real_particles = torch.where(condition)
#                     stop_token = torch.where(~c2)

#                     if real_plus_stop[0].size(0) == 0:
#                         break
                                          
#                     rnn_loss.append(
#                         criterion(
#                             decoder_output[real_plus_stop], 
#                             target_tensor[:, di][real_plus_stop],
#                             class_weights[:, di][real_plus_stop]
#                         )
#                     )
#                     accuracy += [
#                         int(i.item()==j.item())
#                         for i, j in zip(reshaped_preds[:, di][real_particles], 
#                                         target_tensor[:, di][real_particles])
#                     ]

#                     if stop_token[0].size(0) > 0:
#                         stop_accuracy += [
#                             int(i.item()==j.item()) 
#                             for i, j in zip(reshaped_preds[:, di][stop_token], 
#                                             target_tensor[:, di][stop_token])
#                         ]

#                     if real_particles[0].size(0) > 0:
#                         token_input = reshaped_preds[:, di] # topi.squeeze()
#                         embedding = self.decoder.embed(token_input).squeeze(0)
#                         hidden_vectors.append([real_particles, embedding])
                        
#                 epoch_losses["accuracy"].append(np.mean(accuracy))
#                 epoch_losses["stop_accuracy"].append(np.mean(stop_accuracy))
#                 rnn_loss = torch.mean(torch.stack(rnn_loss))
#                 epoch_losses["bce"].append(rnn_loss.item())

#                 if len(hidden_vectors) == 0:
#                     continue

#                 #3. Use particle embeddings to predict (x,y,z,d)
#                 regressor_loss = []
#                 true_part, pred_part, real_part = [], [], []
#                 for di in range(len(hidden_vectors)):
#                     real_particles, h_vecs = hidden_vectors[di]
#                     x_input = torch.cat([h_vecs.detach(), encoded_image, combined_att], axis = 1)
#                     particle_attributes = self.regressor(x_input[real_particles])
#                     batch_true, batch_pred = [], []
#                     for task in self.tasks:
#                         batch_true.append(y_out[task][:, di][real_particles].float())
#                         batch_pred.append(particle_attributes[task].squeeze(1))
#                     true_part.append(batch_true)
#                     pred_part.append(batch_pred)
#                     real_part.append(real_particles[0].cpu().numpy())
#                 regressor_loss = distance_sorted_loss(true_part, pred_part, real_part)
                
#                 seq_acc = []
#                 for (true, pred) in zip(target_tensor, reshaped_preds):
#                     cond = (true > 2)
#                     frac = orderless_acc(true[cond], pred[cond])
#                     seq_acc.append(frac)
#                 seq_acc = np.mean(seq_acc)

#                 epoch_losses["mse"].append(regressor_loss.item())
#                 epoch_losses["seq_acc"].append(seq_acc)

#                 to_print = "Epoch {} val_bce: {:.3f} val_mae: {:.3f} val_acc: {:.3f} val_stop_acc: {:.3f} val_seq_acc: {:.3f}".format(
#                     epoch, 
#                     np.mean(epoch_losses["bce"]), 
#                     np.mean(epoch_losses["mse"]), 
#                     np.mean(epoch_losses["accuracy"]),
#                     np.mean(epoch_losses["stop_accuracy"]),
#                     np.mean(epoch_losses["seq_acc"])
#                 )

#                 batch_group_generator.set_description(to_print)
#                 batch_group_generator.update()

#         return epoch_losses
    
#     def train(self, 
#               scheduler_rnn, 
#               scheduler_linear, 
#               early_stopping_rnn, 
#               early_stopping_linear, 
#               metrics_logger):

#         flag_rnn = isinstance(scheduler_rnn, torch.optim.lr_scheduler.ReduceLROnPlateau)
#         flag_linear = isinstance(scheduler_linear, torch.optim.lr_scheduler.ReduceLROnPlateau)

#         for epoch in range(self.start_epoch, self.epochs):    
#             tf = 1.0 * (self.forcing) ** epoch
            
#             train_losses = self.train_one_epoch(epoch, tf)
#             test_losses = self.test(epoch)

#             # Write results to the callback logger 
#             result = {
#                 "epoch": epoch,
#                 "train_bce": np.mean(train_losses["bce"]),
#                 "valid_bce": np.mean(test_losses["bce"]),
#                 "train_mse": np.mean(train_losses["mse"]),
#                 "valid_mae": np.mean(test_losses["mse"]),
#                 "train_acc": np.mean(train_losses["accuracy"]),
#                 "valid_acc": np.mean(test_losses["accuracy"]),
#                 "train_stop_acc": np.mean(train_losses["stop_accuracy"]),
#                 "valid_stop_acc": np.mean(test_losses["stop_accuracy"]),                
#                 "train_seq_acc": np.mean(train_losses["seq_acc"]),
#                 "valid_seq_acc": np.mean(test_losses["seq_acc"]),
#                 "train_frac_overlap": np.mean(train_losses["frac"]),
#                 "valid_frac_overlap": np.mean(test_losses["frac"]),
#                 "lr_rnn": early_stopping_rnn.print_learning_rate(self.rnn_optimizer),
#                 "lr_linear": early_stopping_linear.print_learning_rate(self.particle_optimizer),
#                 "forcing_value": tf
#             }
#             metrics_logger.update(result)

#             if early_stopping_rnn.early_stop:# and early_stopping_linear.early_stop:
#                 self.train_rnn = False
#             if early_stopping_linear.early_stop:
#                 logger.info("Early stopping")
#                 break

#             scheduler_rnn.step(1.0-result["valid_seq_acc"] if flag_rnn else (1 + epoch))
#             scheduler_linear.step(result["valid_mae"] if flag_linear else (1 + epoch))
            
#             early_stopping_rnn(epoch, 1.0-result["valid_seq_acc"], self.decoder, self.rnn_optimizer)
#             early_stopping_linear(epoch, result["valid_mae"], self.regressor, self.particle_optimizer)