import warnings
warnings.filterwarnings("ignore")

import copy
import joblib
import optuna
import random
import logging
import traceback
import numpy as np

from overrides import overrides
from holodecml.vae.losses import *
from holodecml.vae.visual import *
from holodecml.vae.models import *
from holodecml.vae.trainers import *
from holodecml.vae.transforms import *
from holodecml.vae.optimizers import *
from holodecml.vae.data_loader import *
from holodecml.vae.checkpointer import *
from aimlutils.hyper_opt.base_objective import *

from torch import nn
from torch.optim.lr_scheduler import *
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Callable, Union, Any, TypeVar, Tuple

from aimlutils.hyper_opt.base_objective import *
from aimlutils.torch.checkpoint import *
from aimlutils.torch.losses import *
from aimlutils.utils.tqdm import *

import sys
sys.path.append("/glade/work/schreck/repos/holodec-ml/scripts/schreck/decoder")

from beam_search import *
from data_reader import *
from losses import *
from models import *


logger = logging.getLogger(__name__)


PAD_token = 0
SOS_token = 1
EOS_token = 2


def custom_updates(trial, conf):
    
    # Get list of hyperparameters from the config
    hyperparameters = conf["optuna"]["parameters"]
    
    # Now update some via custom rules
    dense1 = trial_suggest_loader(trial, hyperparameters['dense_hidden_dim1'])
    dense2 = trial_suggest_loader(trial, hyperparameters['dense_hidden_dim2'])
    dr1 = trial_suggest_loader(trial, hyperparameters['dr1'])
    dr2 = trial_suggest_loader(trial, hyperparameters['dr2'])
    n_layers = trial_suggest_loader(trial, hyperparameters['n_layers'])

    # Update the config based on optuna suggestions
    conf["regressor"]["hidden_dims"] = [dense1] + [dense2 for k in range(n_layers)]
    conf["regressor"]["dropouts"] = [dr1] + [dr2 for k in range(n_layers)]
    
    # Update the number of bins in the image
    bins = trial_suggest_loader(trial, hyperparameters['bins'])
    conf["train_data"]["bins"] = bins
    conf["validation_data"]["bins"] = bins
    
    return conf


class Objective(BaseObjective):
    
    def __init__(self, study, config, metric = "val_loss", device = "cpu"):
        
        BaseObjective.__init__(self, study, config, metric, device)
        
        if self.device != "cpu":
            torch.backends.cudnn.benchmark = True


    def train(self, trial, conf):   
        
        random.seed(5000)
        
        ###########################################################
        #
        # Implement custom changes to config
        #
        ###########################################################
        
        conf = custom_updates(trial, conf)
                
        ###########################################################
        #
        # Load ML pipeline, train the model, and return the result
        #
        ###########################################################
        # Load the readers
        train_transform = LoadTransformations(conf["train_transforms"], device = self.device)
        scaler_path = os.path.join(conf["trainer"]["path_save"], "scalers.save")

        train_reader_type = conf["train_data"].pop("type")
        train_gen = MultiHologramDataset(
            **conf["train_data"],
            transform=train_transform,
            scaler=joblib.load(scaler_path) if os.path.isfile(scaler_path) else True,
        )

        if not os.path.isfile(scaler_path):
            joblib.dump(train_gen.scaler, scaler_path)
            
        valid_transform = LoadTransformations(conf["validation_transforms"], device = self.device)

        valid_reader_type = conf["validation_data"].pop("type")
        valid_gen = MultiHologramDataset(
            **conf["validation_data"],
            transform=valid_transform,
            scaler = train_gen.scaler
        )
        
        # Load data iterators from pytorch
        train_dataloader = DataLoader(
            train_gen,
            **conf["train_iterator"]
        )

        valid_dataloader = DataLoader(
            valid_gen,
            **conf["valid_iterator"]
        )
        
        # Load the models confs
        vae_conf = conf["vae"]
        decoder_conf = conf["decoder"]
        regressor_conf = conf["regressor"]
        
        # Load the optimizer confs
        decoder_optimizer_conf = conf["rnn_optimizer"]
        regressor_optimizer_conf = conf["particle_optimizer"]

        trainer = Trainer(
            train_gen,
            valid_gen,
            train_dataloader,
            valid_dataloader,
            vae_conf,
            decoder_conf,
            regressor_conf,
            decoder_optimizer_conf,
            regressor_optimizer_conf,
            device = self.device,
            **conf["trainer"]
        )

        # Initialize LR annealing scheduler 
        if "ReduceLROnPlateau" in conf["callbacks"]:
            schedule_config1 = conf["callbacks"]["ReduceLROnPlateau"]["decoder"]
            schedule_config2 = conf["callbacks"]["ReduceLROnPlateau"]["regressor"]
            scheduler_rnn = ReduceLROnPlateau(trainer.rnn_optimizer, **schedule_config1)
            scheduler_linear = ReduceLROnPlateau(trainer.particle_optimizer, **schedule_config2)

        elif "ExponentialLR" in conf["callbacks"]:
            schedule_config1 = conf["callbacks"]["ExponentialLR"]["decoder"]
            schedule_config2 = conf["callbacks"]["ExponentialLR"]["regressor"]
            scheduler_rnn = ExponentialLR(trainer.rnn_optimizer, **schedule_config1)
            scheduler_linear = ExponentialLR(trainer.particle_optimizer, **schedule_config2)

        # Early stopping
        early_stopping_rnn = EarlyStopping(**conf["callbacks"]["EarlyStopping"]["decoder"]) 
        early_stopping_linear = EarlyStopping(**conf["callbacks"]["EarlyStopping"]["regressor"])
        
        # Train the models
        val_loss, val_ce, val_bleu, val_acc, val_stop_acc = trainer.train(
            trial,
            scheduler_rnn, 
            scheduler_linear, 
            early_stopping_rnn, 
            early_stopping_linear
        )
        
        results = {
            "val_loss": val_loss,
            "val_ce": val_ce,
            "val_bleu": val_bleu,
            "val_acc": val_acc,
            "val_stop_acc": val_stop_acc
        }
        
        return results
    
    
class Trainer:
    
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
                 clip=1.0,
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
        model_type = vae_conf.pop("type")
        vae_model_weights = vae_conf.pop("weights")
        vae = LoadModel(model_type, vae_conf)
        vae.build()
        self.vae = vae.to(device)
        # Load the pretrained weights
        model_dict = torch.load(
            vae_model_weights,
            map_location=lambda storage, loc: storage
        )
        self.vae.load_state_dict(model_dict["model_state_dict"])
        
        # Build decoder
        decoder_conf["output_size"] = len(train_gen.token_lookup) + 3
        self.decoder = DecoderRNN(**decoder_conf).to(device)
        
        # Build regressor
        self.regressor = DenseNet2(**regressor_conf)
        self.regressor.build(vae_conf["z_dim"] + decoder_conf["hidden_size"] + 1250)
        self.regressor = self.regressor.to(device)
        
        # Load RNN optimizer
        self.rnn_optimizer = optim.Adam(self.decoder.parameters(), **decoder_optimizer_conf)
        self.particle_optimizer = optim.Adam(self.regressor.parameters(), **regressor_optimizer_conf)
        
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
        self._bleu = BLEU(exclude_indices={PAD_token, EOS_token, SOS_token})

        # Gradient clipping through hook registration
        for model in [self.regressor, self.decoder]:
            for p in model.parameters():
                if p.requires_grad:
                    p.register_hook(lambda grad: torch.clamp(grad, -clip, clip))
        logger.info(f"Clipping gradients to range [-{clip}, {clip}]")
        
        self.train_regressor = True
        self.train_rnn = True


    def train_one_epoch(self, epoch, use_teacher_forcing):

        self.vae.eval()
        self.decoder.train()
        self.regressor.train()

        batch_size = self.dataloader.batch_size
        batches_per_epoch = int(np.ceil(self.train_gen.__len__() / batch_size))

        if self.batches_per_epoch < batches_per_epoch:
                batches_per_epoch = self.batches_per_epoch

        batch_group_generator = tqdm(
            enumerate(self.dataloader), 
            total=batches_per_epoch, 
            leave=True
        )
        
        criterion = WeightedCrossEntropyLoss(
            label_smoothing = self.label_smoothing,
            gamma = self.focal_gamma
        )

        epoch_losses = {"mse": [], "bce": [], "accuracy": [], "stop_accuracy": []}
        for idx, (images, y_out, w_out) in batch_group_generator:

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

                if self.vae.out_image_channels > 1:
                    z_real = np.sqrt(0.5) * image_pred[:,0,:,:]
                    z_imag = image_pred[:,1,:,:]
                    image_pred = torch.square(z_real) + torch.square(z_imag)
                    image_pred = torch.unsqueeze(image_pred, 1)

            # 2. Predict the number of particles
            decoder_input = torch.LongTensor([SOS_token] * w_out.shape[0]).to(self.device)
            encoded_image = z.to(self.device)
            decoder_hidden = encoded_image.clone().reshape((1, w_out.shape[0], self.decoder.hidden_size))
            
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
#             predictions, probabilities = self.beam_search.search(
#                 decoder_input, decoder_hidden, self.decoder
#             )

#             # Validate on top-1 most likely sequence
#             top_preds = predictions[:, 0, :]

#             # Compute bleu metric for each sequence in the batch
#             for pred, true in zip(top_preds, target_tensor):
#                 self._bleu(pred.unsqueeze(0), true.unsqueeze(0))
#             epoch_losses["bleu"].append(self._bleu.get_metric(reset=False)["BLEU"])

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
            
            for di in range(target_length + 1):    
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, seq_lens)
                topv, topi = decoder_output.topk(1)
                
                if random.uniform(0, 1) < use_teacher_forcing:
                    decoder_input = target_tensor[:, di]
                else:
                    decoder_input = topi.squeeze() if batch_size > 1 else topi
                    decoder_input = decoder_input.detach()
                    
                c1 = (target_tensor[:, di] != PAD_token)
                c2 = (target_tensor[:, di] != EOS_token)
                condition = c1 & c2
                real_plus_stop = torch.where(c1)
                real_particles = torch.where(condition)
                stop_token = torch.where(~c2)

                if real_plus_stop[0].size(0) == 0:
                    break

                # nn.NLLLoss()
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
                    token_input = target_tensor[:, di].squeeze()
                    embedding = self.decoder.embed(token_input).squeeze(0)
                    hidden_vectors.append([real_particles, embedding])

            accuracy = np.mean(accuracy)
            epoch_losses["accuracy"].append(accuracy)
            epoch_losses["stop_accuracy"].append(np.mean(stop_accuracy))

            rnn_loss = torch.mean(torch.stack(rnn_loss))
            epoch_losses["bce"].append(rnn_loss.item())

            if self.train_rnn:
                self.rnn_optimizer.zero_grad()
                rnn_loss.backward()
                self.rnn_optimizer.step()

            if len(hidden_vectors) == 0:
                continue

            # 3. Use particle embeddings to predict (x,y,z,d)
            regressor_loss = []
            for di in range(len(hidden_vectors)):
                # Use decoder_hidden as input to Dense model to predict (x,y,z,d)
                real_particles, h_vecs = hidden_vectors[di]    
                x_input = torch.cat([h_vecs.detach(), encoded_image, combined_att], axis = 1)            
                particle_attributes = self.regressor(x_input[real_particles])
                loss = []
                for task in self.regressor.tasks:
                    _loss = nn.L1Loss()(
                        particle_attributes[task].squeeze(1), 
                        y_out[task][:, di][real_particles].float()
                    ) # XSigmoidLoss()
                    loss.append(_loss)
                regressor_loss.append(torch.mean(torch.stack(loss)))
            regressor_loss = torch.mean(torch.stack(regressor_loss))
            epoch_losses["mse"].append(regressor_loss.item())

            if self.train_regressor:
                self.particle_optimizer.zero_grad()
                regressor_loss.backward()
                self.particle_optimizer.step()

            to_print = "Epoch {} train_bce: {:.3f} train_mse: {:.3f} train_acc: {:.3f} stop_acc: {:.3f}".format(
                epoch, 
                np.mean(epoch_losses["bce"]), 
                np.mean(epoch_losses["mse"]), 
                np.mean(epoch_losses["accuracy"]), 
                np.mean(epoch_losses["stop_accuracy"])
            )
            batch_group_generator.set_description(to_print)
            batch_group_generator.update()

            if idx % batches_per_epoch == 0 and idx > 0:
                break

        return np.mean(epoch_losses["mse"]), np.mean(epoch_losses["bce"]), np.mean(epoch_losses["accuracy"]), epoch_losses["stop_accuracy"]
    
    
    def test(self, epoch):
    
        self.vae.eval()
        self.decoder.eval()
        self.regressor.eval()

        with torch.no_grad():

            batch_size = self.valid_dataloader.batch_size
            batches_per_epoch = int(np.ceil(self.valid_gen.__len__() / batch_size))

            batch_group_generator = tqdm(
                enumerate(self.valid_dataloader), 
                total=batches_per_epoch, 
                leave=True
            )
            
            criterion = WeightedCrossEntropyLoss()

            epoch_losses = {"mse": [], "bce": [], "bleu": [], "accuracy": [], "stop_accuracy": []}
            for idx, (images, y_out, w_out) in batch_group_generator:
                images = images.to(self.device)
                y_out = {task: value.to(self.device) for task, value in y_out.items()}
                w_out = w_out.to(self.device)

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
                decoder_input = torch.LongTensor([SOS_token] * w_out.shape[0]).to(self.device)
                encoded_image = z.to(self.device)
                decoder_hidden = encoded_image.clone().reshape((1, w_out.shape[0], self.decoder.hidden_size))
                
                n_dims = 2 if self.decoder.bidirectional else 1
                n_dims *= self.decoder.n_layers
                if n_dims > 1:
                    decoder_hidden = torch.cat([decoder_hidden for k in range(n_dims)])

                target_tensor = w_out.long()
                target_length = w_out.shape[1]
                seq_lens = w_out.max(axis = 1)[0] + 1
                class_weights = torch.ones(w_out.shape).to(self.device)

                # Without teacher forcing: use its own predictions as the next input
                hidden_vectors = []
                accuracy, stop_accuracy, rnn_loss = [], [], []
                
                # Use beam search to get predictions
                predictions, probabilities = self.beam_search.search(
                    decoder_input, decoder_hidden, self.decoder
                )

                # Validate on top-1 most likely sequence
                top_preds = predictions[:, 0, :]
                
                # Compute bleu metric for each sequence in the batch
                for pred, true in zip(top_preds, target_tensor):
                    self._bleu(pred.unsqueeze(0), true.unsqueeze(0))
                epoch_losses["bleu"].append(self._bleu.get_metric(reset=False)["BLEU"])
                
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
                    
                # Use decoded tokens to compute the loss
                for di in range(target_length + 1):
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, seq_lens)
                    topv, topi = decoder_output.topk(1)
                    #decoder_input = topi.squeeze() if batch_size > 1 else topi
                    #decoder_input = decoder_input.detach()
                    decoder_input = reshaped_preds[:, di].detach()
                    
                    c1 = (target_tensor[:, di] != PAD_token)
                    c2 = (target_tensor[:, di] != EOS_token)
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
                        token_input = reshaped_preds[:, di] #topi.squeeze()
                        embedding = self.decoder.embed(token_input).squeeze(0)
                        hidden_vectors.append([real_particles, embedding])

                accuracy = np.mean(accuracy)
                epoch_losses["accuracy"].append(accuracy)
                epoch_losses["stop_accuracy"].append(np.mean(stop_accuracy))

                rnn_loss = torch.mean(torch.stack(rnn_loss))
                epoch_losses["bce"].append(rnn_loss.item())

                if len(hidden_vectors) == 0:
                    continue

                # 3. Use particle embeddings to predict (x,y,z,d)
                regressor_loss = []
                for di in range(len(hidden_vectors)):
                    real_particles, h_vecs = hidden_vectors[di]
                    x_input = torch.cat([h_vecs.detach(), encoded_image, combined_att], axis = 1)
                    particle_attributes = self.regressor(x_input[real_particles])
                    loss = []
                    for task in self.regressor.tasks:
                        _loss = nn.L1Loss()(
                            particle_attributes[task].squeeze(1), 
                            y_out[task][:, di][real_particles].float()
                        )
                        loss.append(_loss)
                    regressor_loss.append(torch.mean(torch.stack(loss)))
                regressor_loss = torch.mean(torch.stack(regressor_loss))
                epoch_losses["mse"].append(regressor_loss.item())

                to_print = "Epoch {} val_bce: {:.3f} val_mae: {:.3f} val_bleu: {:.3f} val_acc: {:.3f} val_stop_acc: {:.3f}".format(
                    epoch, 
                    np.mean(epoch_losses["bce"]), 
                    np.mean(epoch_losses["mse"]), 
                    np.mean(epoch_losses["bleu"]),
                    np.mean(epoch_losses["accuracy"]), 
                    np.mean(epoch_losses["stop_accuracy"])
                )

                batch_group_generator.set_description(to_print)
                batch_group_generator.update()

        return (np.mean(epoch_losses["mse"]), 
                np.mean(epoch_losses["bce"]), 
                np.mean(epoch_losses["bleu"]),
                np.mean(epoch_losses["accuracy"]), 
                epoch_losses["stop_accuracy"])
    
    def train(self, 
              trial,
              scheduler_rnn, 
              scheduler_linear, 
              early_stopping_rnn, 
              early_stopping_linear):

        flag_rnn = isinstance(scheduler_rnn, torch.optim.lr_scheduler.ReduceLROnPlateau)
        flag_linear = isinstance(scheduler_linear, torch.optim.lr_scheduler.ReduceLROnPlateau)
        
        val_loss, val_ce, val_bleu, val_acc, val_stop_acc = [], [], [], [], []
        for epoch in range(self.start_epoch, self.epochs):
            
            try:
                tf = 1.0 * (self.forcing) ** epoch 
                train_loss, train_ce, train_acc, train_stop_acc = self.train_one_epoch(epoch, tf)
                test_loss, test_ce, test_bleu, test_acc, test_stop_acc = self.test(epoch)
            
            except Exception as E: # CUDA memory overflow
                if "CUDA" in str(E) or "cublas" in str(E):
                    logger.info(
                        "Failed to train the model due to GPU memory overflow."
                    )
                    raise optuna.TrialPruned()
                    #raise ValueError(f"{str(E)}") # FAIL the trial, but do not stop the study
                else:
                    raise OSError(f"{str(E)}") # FAIL the trial and stop the study
                    
            test_loss = math.inf if not test_loss else test_loss
            test_loss = math.inf if test_loss == float("nan") else test_loss
                    
            if not isinstance(test_loss, float):
                raise ValueError(f"The test loss was {test_loss} e.g. not a float -- FAILING this trial.")
                
            if not np.isfinite(test_loss):
                logger.info(f"Pruning this trial at epoch {epoch} with loss {test_loss}")
                raise optuna.TrialPruned()
                
            # Update callbacks
            scheduler_rnn.step(1.0-test_acc if flag_rnn else (1 + epoch))
            scheduler_linear.step(test_loss if flag_linear else (1 + epoch))

            early_stopping_rnn(epoch, 1.0-test_acc, self.decoder, self.rnn_optimizer)
            early_stopping_linear(epoch, test_loss, self.regressor, self.particle_optimizer)
                        
            if early_stopping_linear.early_stop:
                self.train_regressor = False
            if early_stopping_rnn.early_stop:
                self.train_rnn = False
                
            # Combine optimization losses here
            #test_loss = test_loss + test_ce # Do not do cross-entropy: accuracy may be higher for higher ce values.
            
            val_loss.append(test_loss)
            val_ce.append(test_ce)
            val_bleu.append(test_bleu)
            val_acc.append(test_acc)
            val_stop_acc.append(test_stop_acc)
                
            if early_stopping_linear.early_stop: # and early_stopping_rnn.early_stop:
                logger.info("Stopping early due to no recent improvement in val_loss")
                if trial:
                    trial.report(test_loss, step=epoch)
                break
                
            if trial:
                trial.report(test_loss, step=epoch)
                if trial.should_prune():
                    logger.info(f"Pruning this trial at epoch {epoch} with loss {test_loss}")
                    raise optuna.TrialPruned()
                    
        # Return the best loss and the other quantities at the same epoch
        temp = min(val_loss) 
        best_idx = [i for i, j in enumerate(val_loss) if j == temp]
        
        if len(best_idx) > 0:    
            return val_loss[best_idx[-1]], val_ce[best_idx[-1]], val_bleu[best_idx[-1]], val_acc[best_idx[-1]], val_stop_acc[best_idx[-1]]
        else:
            return test_loss, test_ce, test_bleu, test_acc, test_stop_acc
