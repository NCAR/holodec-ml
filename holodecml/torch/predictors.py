import warnings
warnings.filterwarnings("ignore")

import copy
import optuna
import random
import joblib
import logging
import traceback
import numpy as np

from holodecml.torch.utils import *
from holodecml.torch.losses import *
from holodecml.torch.visual import *
from holodecml.torch.models import *
from holodecml.torch.trainers import *
from holodecml.torch.transforms import *
from holodecml.torch.optimizers import *
from holodecml.torch.data_loader import *
from holodecml.torch.beam_search import *

from aimlutils.echo.src.base_objective import *
from aimlutils.torch.checkpoint import *


from torch import nn
from torch.optim.lr_scheduler import *
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Callable, Union, Any, TypeVar, Tuple

from collections import defaultdict


logger = logging.getLogger(__name__)


def LoadPredictor(train_gen,
                  valid_gen,
                  dataloader,
                  valid_dataloader, 
                  device, 
                  config):
    
    if "type" not in config["trainer"]:
        logger.warning("In order to load a predictor you must supply the trainer type field.")
        raise OSError("Failed to load a predictor trainer. Exiting")
        
    trainer_type = config["trainer"].pop("type")
    logger.info(f"Loading trainer-type {trainer_type}")
    
#     if trainer_type in ["vae", "att-vae"]:
#         return BaseTrainer(
#             train_gen=train_gen,
#             valid_gen=valid_gen,
#             dataloader=dataloader,
#             valid_dataloader=valid_dataloader,
#             model_conf = config["model"], 
#             optimizer_conf = config["optimizer"],
#             device=device,
#             **config["trainer"]
#         )
#     elif trainer_type == "encoder-vae":
#         return BaseEncoderTrainer(
#             train_gen=train_gen,
#             valid_gen=valid_gen,
#             dataloader=dataloader,
#             valid_dataloader=valid_dataloader,
#             model_conf = config["model"], 
#             optimizer_conf = config["optimizer"],
#             device=device,
#             **config["trainer"]
#         )
    if trainer_type == "decoder-vae":
        return DecoderPredictor(
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
            f"Unsupported trainer type {trainer_type}. Choose from decoder-vae. Exiting.")
        sys.exit(1)
        
        
        
class DecoderPredictor(DecoderTrainer):
    
    def load_weights(self, 
                     vae_weights = None, 
                     rnn_weights = None, 
                     regressor_weights = None):
        
        if vae_weights:
            if os.path.isfile(vae_weights):
                self.vae.weights = vae_weights
                self.vae.load_weights()
            else:
                logger.warning(f"The vae weight file {vae_weights} does not exist.")
            
        if rnn_weights:
            if os.path.isfile(rnn_weights):
                self.decoder.weights = rnn_weights
                self.decoder.load_weights()
            else:
                logger.warning(f"The rnn weight file {rnn_weights} does not exist.")
            
        if regressor_weights:
            if os.path.isfile(regressor_weights):
                self.regressor.weights = regressor_weights 
                self.regressor.load_weights()
            else:
                logger.warning(f"The fully-connected weight file {regressor_weights} does not exist.")
                
    
    def predict(self, return_only_losses = False):
    
        self.vae.eval()
        self.decoder.eval()
        self.regressor.eval()

        with torch.no_grad():

            batch_size = self.valid_dataloader.batch_size
            batches_per_epoch = int(np.ceil(self.valid_gen.__len__() / batch_size))
            
            criterion = WeightedCrossEntropyLoss()
            
            epoch_losses = {"mse": [], "bce": [], "frac": [], 
                            "accuracy": [], "stop_accuracy": [], "seq_acc": []}
            
            for idx, (images, y_out, w_out) in enumerate(self.valid_dataloader):
                images = images.to(self.device)
                y_out = {task: value.to(self.device) for task, value in y_out.items()}
                w_out = w_out.to(self.device)

                with torch.no_grad():
                    # 1. Predict the latent vector and image reconstruction
                    z, mu, logvar, encoder_att = self.vae.encode(images)
                    image_pred, decoder_att = self.vae.decode(z)

                    combined_att = torch.cat([
                        encoder_att[2].flatten(start_dim = 1),
                        decoder_att[0].flatten(start_dim = 1)
                    ], 1)
                    combined_att = combined_att.clone()

                    encoder_att = [x.detach().cpu().numpy() for x in encoder_att]
                    decoder_att = [x.detach().cpu().numpy() for x in decoder_att]

                    if self.vae.out_image_channels > 1:
                        z_real = np.sqrt(0.5) * image_pred[:,0,:,:]
                        z_imag = image_pred[:,1,:,:]
                        image_pred = torch.square(z_real) + torch.square(z_imag)
                        image_pred = torch.unsqueeze(image_pred, 1)

                # 2. Predict the number of particles
                decoder_input = torch.LongTensor([self.SOS_token] * w_out.shape[0]).to(self.device)
                encoded_image = z.to(self.device)
                decoder_hidden = encoded_image.clone().reshape((1, w_out.shape[0], z.size(-1)))

                n_dims = 2 if self.decoder.bidirectional else 1
                n_dims *= self.decoder.n_layers
                if n_dims > 1:
                    decoder_hidden = torch.cat([decoder_hidden for k in range(n_dims)])

                target_tensor = w_out.long()
                target_length = w_out.shape[1]
                seq_lens = w_out.max(axis = 1)[0] + 1
                class_weights = torch.ones(w_out.shape).to(self.device)

                hidden_vectors = []
                bleu, accuracy, stop_accuracy, rnn_loss = [], [], [], []

                # Use beam search to get predictions
                predictions, probabilities = self.beam_search.search(
                    decoder_input, decoder_hidden, self.decoder
                )

                # Validate on top-1 most likely sequence
                top_preds = predictions[:, 0, :]

                # Compute fractional accuracy metric for each sequence in the batch
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
                di = 0
                while (di < 101):
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, seq_lens)
                    topv, topi = decoder_output.topk(1)
                    decoder_input = reshaped_preds[:, di].detach()
                    c1 = (reshaped_preds[:, di] != self.PAD_token)
                    c2 = (reshaped_preds[:, di] != self.EOS_token)
                    condition = c1 & c2
                    real_plus_stop = torch.where(c1)
                    real_particles = torch.where(condition)
                    stop_token = torch.where(~c2)
                    
                    if real_plus_stop[0].size(0) == 0:
                        break

                    rnn_loss.append(
                        nn.NLLLoss()(
                            decoder_output[real_plus_stop], 
                            target_tensor[:, di][real_plus_stop]
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
                        hidden_vectors.append([real_particles, embedding])

                    di += 1

                accuracy = np.mean(accuracy)
                epoch_losses["accuracy"].append(accuracy)
                epoch_losses["stop_accuracy"].append(np.mean(stop_accuracy))

                rnn_loss = torch.mean(torch.stack(rnn_loss))
                epoch_losses["bce"].append(rnn_loss.item())
                
                # Do another for loop over the real particles to get real_particle_true correctly
                real_parts_true = []
                particle_true = {"x": [], "y": [], "z": [], "d": []}
                
                di = 0
                for di in range(target_length):
                    c1 = (target_tensor[:, di] != self.PAD_token)
                    c2 = (target_tensor[:, di] != self.EOS_token)
                    condition = c1 & c2
                    real_plus_stop = torch.where(c1)
                    real_particles_true = torch.where(condition)
                    
                    
                    if real_particles_true[0].size(0) == 0:
                        break
                    
                    real_parts_true.append(real_particles_true[0].cpu().numpy())
                    for task in self.tasks:
                        particle_true[task].append(y_out[task][:, di][real_particles_true].float().cpu().numpy())

                if len(hidden_vectors) == 0:
                    continue

                    
                #3. Use particle embeddings to predict (x,y,z,d)
                particle_results = {"x": [], "y": [], "z": [], "d": [], "tokens": []}
                regressor_loss = []
                real_parts, true_part, pred_part = [], [], []
                for di in range(len(hidden_vectors)):
                    real_particles, h_vecs = hidden_vectors[di]
                    x_input = torch.cat([h_vecs.detach(), encoded_image, combined_att], axis = 1)
                    particle_attributes = self.regressor(x_input[real_particles])
                    batch_pred, batch_true = [], []
                    for task in self.tasks:
                        batch_pred.append(particle_attributes[task].squeeze(1))
                        batch_true.append(y_out[task][:, di][real_particles].float())
                        particle_results[task].append(particle_attributes[task].cpu().numpy().squeeze(1))
                        #particle_true[task].append(y_out[task][:, di][real_particles].float().cpu().numpy())
                    pred_part.append(batch_pred)
                    true_part.append(batch_true)
                    real_parts.append(real_particles[0].cpu().numpy())
                    particle_results["tokens"].append(real_particles)

                regressor_loss, coordinate_losses = distance_sorted_loss(
                    true_part, pred_part, real_parts, return_coordinate_errors = True
                )
                
                seq_acc = []
                for (true, pred) in zip(target_tensor, reshaped_preds):
                    cond = (true > 2)
                    frac = orderless_acc(true[cond], pred[cond])
                    seq_acc.append(frac)
                seq_acc = np.mean(seq_acc)

                epoch_losses["mse"].append(regressor_loss.item())
                epoch_losses["seq_acc"].append(seq_acc)

                if return_only_losses:
                    result = {
                        "mae_coordinates": coordinate_losses,
                        "sequence_accuracy": seq_acc
                    }
                
                else:
                    result = {
                            "image_true": images.cpu().numpy(),
                            "image_pred": image_pred.cpu().numpy(),
                            "particle_true": particle_true,
                            "particle_pred": particle_results,
                            "accuracy": accuracy,
                            "sequence_accuracy": seq_acc,
                            "stop_accuracy": np.mean(stop_accuracy),
                            "wce": rnn_loss.item(),
                            "mae": regressor_loss.item(),
                            "mae_coordinates": coordinate_losses,
                            "real_particles": real_parts,
                            "real_particles_true": real_parts_true,
                            "encoder_att": encoder_att,
                            "decoder_att": decoder_att,
                            #"target_tokens": target_tensor,
                            #"target_coors": y_out
                        }

                yield self.post_process(result, batch_size = target_tensor.shape[0])
    
    def post_process(self, result, batch_size = None):
        
        if batch_size == None:
            batch_size = result["image_pred"].shape[0]
            
        
        for coordinate_batch in result["mae_coordinates"]:
            for task in self.valid_gen.output_cols:
                self.train_gen.scaler[task].inverse_transform(coordinate_batch[task])

        batched_coors = defaultdict(list)
        coordinates = result["mae_coordinates"]
        for task in self.valid_gen.output_cols:
            for coordinate_batch in coordinates:
                batch_errors = coordinate_batch[task]
                batch_errors = self.train_gen.scaler[task].inverse_transform(batch_errors)
                coordinate_batch[task] = abs(batch_errors[:,0] - batch_errors[:,1])
                batched_coors[task].append(coordinate_batch[task])
        #batched_coors = {key: value for key, value in batched_coors.items()}
        result["mae_coordinates"] = batched_coors
      
        for result_key in ["particle_true", "particle_pred", "target_coors"]:
            if result_key in result:
                result[result_key] = {
                    key: self.train_gen.scaler[key].inverse_transform(values) 
                    for key, values in result[result_key].items() if key in self.train_gen.output_cols
                }
        return result
    
#     def predict(self, return_only_losses = False):
    
#         self.vae.eval()
#         self.decoder.eval()
#         self.regressor.eval()

#         with torch.no_grad():

#             batch_size = self.valid_dataloader.batch_size
#             batches_per_epoch = int(np.ceil(self.valid_gen.__len__() / batch_size))
            
#             criterion = WeightedCrossEntropyLoss()
            
#             epoch_losses = {"mse": [], "bce": [], "frac": [], 
#                             "accuracy": [], "stop_accuracy": [], "seq_acc": []}
            
#             for idx, (images, y_out, w_out) in enumerate(self.valid_dataloader):
#                 images = images.to(self.device)
#                 y_out = {task: value.to(self.device) for task, value in y_out.items()}
#                 w_out = w_out.to(self.device)

#                 with torch.no_grad():
#                     # 1. Predict the latent vector and image reconstruction
#                     z, mu, logvar, encoder_att = self.vae.encode(images)
#                     image_pred, decoder_att = self.vae.decode(z)

#                     combined_att = torch.cat([
#                         encoder_att[2].flatten(start_dim = 1),
#                         decoder_att[0].flatten(start_dim = 1)
#                     ], 1)
#                     combined_att = combined_att.clone()

#                     encoder_att = [x.detach().cpu().numpy() for x in encoder_att]
#                     decoder_att = [x.detach().cpu().numpy() for x in decoder_att]

#                     if self.vae.out_image_channels > 1:
#                         z_real = np.sqrt(0.5) * image_pred[:,0,:,:]
#                         z_imag = image_pred[:,1,:,:]
#                         image_pred = torch.square(z_real) + torch.square(z_imag)
#                         image_pred = torch.unsqueeze(image_pred, 1)

#                 # 2. Predict the number of particles
#                 decoder_input = torch.LongTensor([self.SOS_token] * w_out.shape[0]).to(self.device)
#                 encoded_image = z.to(self.device)
#                 decoder_hidden = encoded_image.clone().reshape((1, w_out.shape[0], z.size(-1)))

#                 n_dims = 2 if self.decoder.bidirectional else 1
#                 n_dims *= self.decoder.n_layers
#                 if n_dims > 1:
#                     decoder_hidden = torch.cat([decoder_hidden for k in range(n_dims)])

#                 target_tensor = w_out.long()
#                 target_length = w_out.shape[1]
#                 seq_lens = w_out.max(axis = 1)[0] + 1
#                 class_weights = torch.ones(w_out.shape).to(self.device)

#                 hidden_vectors = []
#                 bleu, accuracy, stop_accuracy, rnn_loss = [], [], [], []

#                 # Use beam search to get predictions
#                 predictions, probabilities = self.beam_search.search(
#                     decoder_input, decoder_hidden, self.decoder
#                 )

#                 # Validate on top-1 most likely sequence
#                 top_preds = predictions[:, 0, :]

#                 # Compute fractional accuracy metric for each sequence in the batch
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
#                 di = 0
#                 while (di < 101):
#                 #for di in range(target_length + 1):
#                     decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, seq_lens)
#                     topv, topi = decoder_output.topk(1)
#                     decoder_input = reshaped_preds[:, di].detach()
#                     c1 = (reshaped_preds[:, di] != self.PAD_token)
#                     c2 = (reshaped_preds[:, di] != self.EOS_token)
#                     condition = c1 & c2
#                     real_plus_stop = torch.where(c1)
#                     real_particles = torch.where(condition)
#                     stop_token = torch.where(~c2)

#                     if real_plus_stop[0].size(0) == 0:
#                         break

#                     rnn_loss.append(
#                         nn.NLLLoss()(
#                             decoder_output[real_plus_stop], 
#                             target_tensor[:, di][real_plus_stop]
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

#                     di += 1

#                 accuracy = np.mean(accuracy)
#                 epoch_losses["accuracy"].append(accuracy)
#                 epoch_losses["stop_accuracy"].append(np.mean(stop_accuracy))

#                 rnn_loss = torch.mean(torch.stack(rnn_loss))
#                 epoch_losses["bce"].append(rnn_loss.item())

#                 if len(hidden_vectors) == 0:
#                     continue

#                 particle_results = {"x": [], "y": [], "z": [], "d": [], "tokens": []}
#                 particle_true = {"x": [], "y": [], "z": [], "d": []}
                
#                 #3. Use particle embeddings to predict (x,y,z,d)
#                 regressor_loss = []
#                 real_parts, true_part, pred_part = [], [], []
#                 for di in range(len(hidden_vectors)):
#                     real_particles, h_vecs = hidden_vectors[di]
#                     x_input = torch.cat([h_vecs.detach(), encoded_image, combined_att], axis = 1)
#                     particle_attributes = self.regressor(x_input[real_particles])
#                     batch_pred, batch_true = [], []
#                     for task in self.tasks:
#                         batch_pred.append(particle_attributes[task].squeeze(1))
#                         batch_true.append(y_out[task][:, di][real_particles].float())
#                         particle_results[task].append(particle_attributes[task].cpu().numpy().squeeze(1))
#                         particle_true[task].append(y_out[task][:, di][real_particles].float().cpu().numpy())
#                     pred_part.append(batch_pred)
#                     true_part.append(batch_true)
#                     real_parts.append(real_particles[0].cpu().numpy())
#                     particle_results["tokens"].append(real_particles)

#                 regressor_loss, coordinate_losses = distance_sorted_loss(
#                     true_part, pred_part, real_parts, return_coordinate_errors = True
#                 )
                
#                 seq_acc = []
#                 for (true, pred) in zip(target_tensor, reshaped_preds):
#                     cond = (true > 2)
#                     frac = orderless_acc(true[cond], pred[cond])
#                     seq_acc.append(frac)
#                 seq_acc = np.mean(seq_acc)

#                 epoch_losses["mse"].append(regressor_loss.item())
#                 epoch_losses["seq_acc"].append(seq_acc)

#                 if return_only_losses:
#                     result = {
#                         "mae_coordinates": coordinate_losses,
#                         "sequence_accuracy": seq_acc
#                     }
                
#                 else:
#                     result = {
#                             "image_true": images.cpu().numpy(),
#                             "image_pred": image_pred.cpu().numpy(),
#                             "particle_true": particle_true,
#                             "particle_pred": particle_results,
#                             "accuracy": accuracy,
#                             "sequence_accuracy": seq_acc,
#                             "stop_accuracy": np.mean(stop_accuracy),
#                             "wce": rnn_loss.item(),
#                             "mae": regressor_loss.item(),
#                             "mae_coordinates": coordinate_losses,
#                             "real_particles": real_parts,
#                             "encoder_att": encoder_att,
#                             "decoder_att": decoder_att
#                         }

#                 yield self.post_process(result, batch_size = target_tensor.shape[0])
                
#     def post_process(self, result, batch_size = None):
        
#         if batch_size == None:
#             batch_size = result["image_pred"].shape[0]
            
#         batched_coors = defaultdict(list)
#         coordinates = result["mae_coordinates"]
#         for task in self.valid_gen.output_cols:
#             for coordinate_batch in coordinates:
#                 batch_errors = np.array(coordinate_batch[task])
#                 batch_errors = self.train_gen.scaler[task].inverse_transform(batch_errors)
#                 coordinate_batch[task] = abs(batch_errors[:,0] - batch_errors[:,1])
#                 batched_coors[task].append(coordinate_batch[task])
#         #batched_coors = {key: np.stack(value) for key, value in batched_coors.items()}
#         result["mae_coordinates"] = batched_coors
      
#         for result_key in ["particle_true", "particle_pred"]:
#             if result_key in result:
#                 result[result_key] = {
#                     key: self.train_gen.scaler[key].inverse_transform(values) 
#                     for key, values in result[result_key].items() if key in self.train_gen.output_cols
#                 }
#         return result
    
    def compute_total_loss(self):
        
        batch_size = self.valid_dataloader.batch_size
        batches_per_epoch = int(np.ceil(self.valid_gen.__len__() / batch_size))

        batch_group_generator = tqdm.tqdm(
            enumerate(self.predict(return_only_losses = True)), 
            total=batches_per_epoch, 
            leave=True
        )
        
        errors = {"x": [], "y": [], "z": [], "d": [], "seq_acc": []}
        for idx, result in batch_group_generator:
            errors["seq_acc"].append(result["sequence_accuracy"])
            for task in self.train_gen.output_cols:
                for value in result["mae_coordinates"][task]:
                    errors[task].append(value)
            to_print = "x: {:.3f} y: {:.3f} z: {:.3f} d: {:.3f} seq_acc: {:.3f}".format( 
                np.mean(np.concatenate(errors["x"])), 
                np.mean(np.concatenate(errors["y"])), 
                np.mean(np.concatenate(errors["z"])),
                np.mean(np.concatenate(errors["d"])),
                np.mean(errors["seq_acc"])
            )
            batch_group_generator.set_description(to_print)
            batch_group_generator.update()

        return errors