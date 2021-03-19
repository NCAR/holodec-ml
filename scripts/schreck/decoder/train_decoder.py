#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings

import copy
import yaml
import torch
import scipy
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import *

import nltk
import itertools
import pickle
import random
import joblib
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

from aimlutils.hyper_opt.base_objective import *
from aimlutils.torch.checkpoint import *
#from aimlutils.torch.losses import *
from aimlutils.utils.tqdm import *

from typing import List, Callable, Tuple, Dict, Union


# In[2]:


root = logging.getLogger()
root.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')

# Stream output to stdout
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
root.addHandler(ch)


# In[3]:


is_cuda = torch.cuda.is_available()
device = torch.device(torch.cuda.current_device()) if is_cuda else torch.device("cpu")


# In[4]:


with open("results/multi_particle_redo/model.yml") as config_file:
    conf = yaml.load(config_file, Loader=yaml.FullLoader)


# ### Load data readers

# In[5]:


# Load the image/(x,y,z,d) transformations
train_transform = LoadTransformations(conf["train_transforms"], device = device)
valid_transform = LoadTransformations(conf["validation_transforms"], device = device)


# In[6]:


# Load the readers
scaler_path = os.path.join(conf["trainer"]["path_save"], "scalers.save") 


# In[7]:


train_gen = LoadReader( 
    transform = train_transform, 
    scaler = joblib.load(scaler_path) if os.path.isfile(scaler_path) else True,
    config = conf["train_data"]
)

if not os.path.isfile(scaler_path):
    joblib.dump(train_gen.scaler, scaler_path)


# In[8]:


valid_gen = LoadReader(
    transform = valid_transform, 
    scaler = train_gen.scaler,
    config = conf["validation_data"]
)


# ### Load Torch's iterator class

# In[9]:


# Load data iterators from pytorch
train_dataloader = DataLoader(
    train_gen,
    **conf["train_iterator"]
)

valid_dataloader = DataLoader(
    valid_gen,
    **conf["valid_iterator"]
)


# ### Load trainer

# In[10]:


trainer = LoadTrainer(
    train_gen, 
    valid_gen, 
    train_dataloader,
    valid_dataloader,
    device, 
    conf
)


# In[11]:


# class DTrainer:
    
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
        
#         # Build regressor
#         self.regressor = LoadModel(regressor_conf)
#         self.regressor.build(vae_conf["z_dim"] + decoder_conf["hidden_size"] + 1250 + 4)
#         self.regressor = self.regressor.to(device)
#         self.tasks = self.regressor.tasks
        
#         # Load RNN optimizer
#         self.rnn_optimizer = LookaheadRAdam(
#             self.decoder.parameters(), 
#             **decoder_optimizer_conf
#         )
#         self.particle_optimizer = LookaheadRAdam(
#             self.regressor.parameters(), 
#             **regressor_optimizer_conf
#         )
        
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

# #         # Gradient clipping through hook registration
# #         for model in [self.regressor, self.decoder]:
# #             for p in model.parameters():
# #                 if p.requires_grad:
# #                     p.register_hook(lambda grad: torch.clamp(grad, -clip, clip))
# #         logger.info(f"Clipping gradients to range [-{clip}, {clip}]")
        
#         self.max_grad_norm = max_grad_norm
        
#         self.train_rnn = True
        
#         self.order_embed = nn.Embedding(max_steps, 4).to(self.device)

#     def train_one_epoch(self, epoch, use_teacher_forcing):

#         self.vae.eval()
#         self.decoder.train()
#         self.regressor.train()

#         batch_size = self.dataloader.batch_size
#         batches_per_epoch = int(np.ceil(self.train_gen.__len__() / batch_size))

#         if self.batches_per_epoch < batches_per_epoch:
#                 batches_per_epoch = self.batches_per_epoch

#         batch_group_generator = tqdm(
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
#                     torch.nn.utils.clip_grad_norm_(
#                         self.decoder.parameters(), 
#                         self.max_grad_norm
#                     )
                
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

#                 last = torch.LongTensor([di] * w_out.shape[0]).to(self.device)
#                 last = self.order_embed(last)
#                 _h_vecs = torch.cat([h_vecs.detach(), last], 1) 
                
#                 x_input = torch.cat([_h_vecs, encoded_image, combined_att], axis = 1)            
#                 particle_attributes = self.regressor(x_input[real_particles])
#                 loss = []
#                 for task in self.tasks:
#                     _loss = nn.L1Loss()(
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
#                 torch.nn.utils.clip_grad_norm_(
#                     self.regressor.parameters(), 
#                     self.max_grad_norm
#                 )

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

#             batch_group_generator = tqdm(
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
#                 #last = torch.LongTensor([self.SOS_token] * w_out.shape[0]).to(self.device)
#                 #last = self.decoder.embed(token_input).squeeze(0)
                
#                 regressor_loss, true_part, pred_part, real_part = [], [], [], []
#                 for di in range(len(hidden_vectors)):
#                     real_particles, h_vecs = hidden_vectors[di]
                    
#                     last = torch.LongTensor([di] * w_out.shape[0]).to(self.device)
#                     last = self.order_embed(last)
#                     _h_vecs = torch.cat([h_vecs.detach(), last], 1)    
                    
#                     x_input = torch.cat([_h_vecs, encoded_image, combined_att], axis = 1)
#                     particle_attributes = self.regressor(x_input[real_particles])
                    
#                     # Compute loss by trying to decode the in-order
#                     loss = []
#                     for task in self.tasks:
#                         _loss = nn.L1Loss()(
#                              particle_attributes[task].squeeze(1),
#                              y_out[task][:, di][real_particles].float()
#                         ) # XSigmoidLoss()
#                         loss.append(_loss)
#                     regressor_loss.append(torch.mean(torch.stack(loss)))
                    
#                     # Compute loss again but pair off the particles by distance
#                     batch_true, batch_pred = [], []
#                     for task in self.tasks:
#                         batch_true.append(y_out[task][:, di][real_particles].float())
#                         batch_pred.append(particle_attributes[task].squeeze(1))
#                     true_part.append(batch_true)
#                     pred_part.append(batch_pred)
#                     real_part.append(real_particles[0].cpu().numpy())
                         
#                 regressor_loss = torch.mean(torch.stack(regressor_loss))
#                 sorted_regressor_loss = distance_sorted_loss(true_part, pred_part, real_part)
                
#                 # Now take the smallest loss
#                 # Experimenting with this b/c sorted loss seems poor initially for large particles, but the accuracy looks good
#                 regressor_loss = min(regressor_loss, sorted_regressor_loss)
                
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


# # In[12]:


# if "type" in conf["trainer"]:
#     conf["trainer"].pop("type")
    
    
# trainer = DTrainer(
#     train_gen=train_gen,
#     valid_gen=valid_gen,
#     dataloader=train_dataloader,
#     valid_dataloader=valid_dataloader,
#     vae_conf=conf["vae"],
#     decoder_conf=conf["decoder"],
#     regressor_conf=conf["regressor"],
#     decoder_optimizer_conf=conf["rnn_optimizer"],
#     regressor_optimizer_conf=conf["particle_optimizer"],
#     device=device,
#     **conf["trainer"]
# )


# ### Load metrics and callbacks

# In[13]:


# Initialize LR annealing scheduler 
if "ReduceLROnPlateau" in conf["callbacks"]:
    if "decoder" in conf["callbacks"]["ReduceLROnPlateau"]:
        schedule_config1 = conf["callbacks"]["ReduceLROnPlateau"]["decoder"]
        scheduler_rnn = ReduceLROnPlateau(trainer.rnn_optimizer, **schedule_config1)
    if "regressor" in conf["callbacks"]["ReduceLROnPlateau"]:
        schedule_config2 = conf["callbacks"]["ReduceLROnPlateau"]["regressor"]
        scheduler_linear = ReduceLROnPlateau(trainer.particle_optimizer, **schedule_config2)

if "ExponentialLR" in conf["callbacks"]:
    if "decoder" in conf["callbacks"]["ExponentialLR"]:
        schedule_config1 = conf["callbacks"]["ExponentialLR"]["decoder"]
        scheduler_rnn = ExponentialLR(trainer.rnn_optimizer, **schedule_config1)
    if "regressor" in conf["callbacks"]["ExponentialLR"]:
        schedule_config2 = conf["callbacks"]["ExponentialLR"]["regressor"]
        scheduler_linear = ExponentialLR(trainer.particle_optimizer, **schedule_config2)

# Early stopping
early_stopping_rnn = EarlyStopping(**conf["callbacks"]["EarlyStopping"]["decoder"])
early_stopping_linear = EarlyStopping(**conf["callbacks"]["EarlyStopping"]["regressor"])

# Write metrics to csv each epoch
metrics_logger = MetricsLogger(**conf["callbacks"]["MetricsLogger"])


# ### Train the model

# In[ ]:


results = trainer.train(scheduler_rnn, scheduler_linear, early_stopping_rnn, early_stopping_linear, metrics_logger)


# In[ ]:




