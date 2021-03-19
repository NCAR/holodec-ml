#!/usr/bin/env python
import pandas as pd
import numpy as np
import pickle

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.keypoint_rcnn import KeypointRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import FasterRCNN
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import *
import torchvision
import torch

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from collections import Counter, defaultdict
from typing import List, Dict
import xarray as xr
import itertools
from sklearn.metrics import *
import logging
import joblib
import random
import yaml
import sys
import os

from holodecml.torch.utils import *
from holodecml.torch.losses import *
from holodecml.torch.transforms import *
from holodecml.torch.optimizers import *

from aimlutils.echo.src.base_objective import *
from aimlutils.torch.checkpoint import *
from aimlutils.utils.tqdm import *

import matplotlib.pyplot as plt 
import matplotlib.patches as patches

from data_reader import DetectionDataset, DetectionDatasetPadded
from cosine_schedule import CosineAnnealingLR_with_Restart

#from torchvision.references.detection.engine import train_one_epoch, evaluate

# Some references 
# https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
# https://pytorch.org/vision/0.8/models.html
# https://blog.francium.tech/object-detection-with-faster-rcnn-bc2e4295bf49

### FasterRNN source - https://github.com/pytorch/vision/blob/master/torchvision/models/detection/faster_rcnn.py
### engine utilities - https://github.com/pytorch/vision/blob/master/references/detection/engine.py

def val_loss(outputs, targets, device, iou_threshold = 0.2):
    # Used to estimate binary performance (box or no box) for the validation split
    pred_box = [t['boxes'] for t in outputs]
    true_box = [t['boxes'].to(device) for t in targets]
    scores = [t["scores"] for t in outputs]

    metrics = {
        "precision": [],
        "recall": [],
        "f1_score": [], 
        "accuracy": []
    }
    for pbox, tbox, score in zip(pred_box, true_box, scores):
        # Use nms (non-max suppression) to consolidate overlapping bounding boxes
        keep_idx = torchvision.ops.nms(pbox, score, iou_threshold)
        pbox = torch.index_select(pbox, 0, keep_idx)
        score = torch.index_select(score, 0, keep_idx)
        
        true_n = tbox.shape[0]
        pred_n = pbox.shape[0]
        pad = max(true_n, pred_n)
        
        tbox = torch.cat([tbox, tbox.new_zeros(pad - true_n, 4)], 0) if pad != true_n else tbox
        pbox = torch.cat([pbox, pbox.new_zeros(pad - pred_n, 4)], 0) if pad != pred_n else pbox
                
        y_true = np.zeros(pad)
        y_pred = np.zeros(pad)
        y_true[:true_n] = 1
        y_pred[:pred_n] = 1

        prec = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)

        metrics["precision"].append(prec)
        metrics["recall"].append(recall)
        metrics["f1_score"].append(f1)
        metrics["accuracy"].append(accuracy)
            
    metrics = {key: np.mean(val) for key, val in metrics.items()}
    return metrics


def collate_fn(batch):
    return tuple(zip(*batch))

    
def train(conf):
    
    random.seed(5000)
    
    # Set up the device
    is_cuda = torch.cuda.is_available()
    device = torch.device(torch.cuda.current_device()) if is_cuda else torch.device("cpu")
    if is_cuda:
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True

    # Load image transformations
    train_transform = LoadTransformations(conf["train_transforms"], device = device)
    valid_transform = LoadTransformations(conf["validation_transforms"], device = device)

    # Load the data readers [DEPRECATED for the object detector]
    if "type" in conf["train_data"]:
        conf["train_data"].pop("type")
    if "type" in conf["validation_data"]:
        conf["validation_data"].pop("type")

    # Training data
    train_gen = DetectionDatasetPadded(
        **conf["train_data"],
        transform=train_transform,
        scaler=True,
    )
    
    # Save the scaler to file
    scaler_save = os.path.join(
        conf["callbacks"]["MetricsLogger"]["path_save"], 
        "scalers.pkl"
    )

    with open(scaler_save, "wb") as fid:
        pickle.dump(train_gen.scaler, fid)
    
    # valid data with the training data scaler
    valid_gen = DetectionDatasetPadded(
        **conf["validation_data"],
        transform=valid_transform,
        scaler=train_gen.scaler,
    )

    # Load the reader iterators
    train_dataloader = DataLoader(
        train_gen,
        **conf["train_iterator"],
        collate_fn = collate_fn
    )

    valid_dataloader = DataLoader(
        valid_gen,
        **conf["valid_iterator"],
        collate_fn = collate_fn
    )
    
    # Load model
    num_classes = conf["model"]["num_classes"]
    pretrained = bool(conf["model"]["pretrained"])
    pretrained_backbone = bool(conf["model"]["pretrained_backbone"])

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained = pretrained, 
        pretrained_backbone = pretrained_backbone
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model = model.to(device)

    # Load an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = LoadOptimizer(conf["optimizer"], params)

    # Initialize LR annealing scheduler 
    ### Updated after every epoch
    if "ReduceLROnPlateau" in conf["callbacks"]:
        schedule_config = conf["callbacks"]["ReduceLROnPlateau"]
        lr_scheduler = ReduceLROnPlateau(optimizer, **schedule_config)
        check_every_epoch = True
    ### Updated after every epoch
    if "ExponentialLR" in conf["callbacks"]:
        schedule_config = conf["callbacks"]["ExponentialLR"]
        lr_scheduler = ExponentialLR(optimizer, **schedule_config)
        check_every_epoch = True
    ### Updated after every batch
    if "CosineAnnealing" in conf["callbacks"]:
        schedule_config = conf["callbacks"]["CosineAnnealing"]
        lr_scheduler = CosineAnnealingLR_with_Restart(
            optimizer, 
            model=model,
            **schedule_config
        )
        check_every_epoch = True
    ### Updated after every batch    
    if "CyclicLR" in conf["callbacks"]:
        schedule_config = conf["callbacks"]["CyclicLR"]
        lr_scheduler = CyclicLR(
            optimizer,
            **schedule_config
        )
        check_every_epoch = False

    # Metrics and Logger
    early_stopping = EarlyStopping(**conf["callbacks"]["EarlyStopping"])
    metrics_logger = MetricsLogger(**conf["callbacks"]["MetricsLogger"])

    # Trainer conf settings
    start_epoch = conf["trainer"]["start_epoch"]
    epochs = conf["trainer"]["epochs"]
    train_batches_per_epoch = conf["trainer"]["batches_per_epoch"]
    grad_clip = None if "grad_clip" not in conf else conf["grad_clip"]
    iou_threshold = conf["trainer"]["iou_threshold"]
    binary_metrics = conf["trainer"]["binary_metrics"]
    validation_metric = conf["trainer"]["metric"]
    
    best_model_results = {
        "train_loss": np.inf, 
        "valid_loss": np.inf, 
        "valid_f1": np.inf,
        "valid_acc": np.inf
    }
    
    for epoch in range(start_epoch, epochs):
        # Compute the number of batches that will go through the model
        batch_size = conf["train_iterator"]["batch_size"]
        batches_per_epoch_total = int(np.ceil(train_gen.__len__() / batch_size))
        if train_batches_per_epoch > batches_per_epoch_total:
            train_batches_per_epoch = batches_per_epoch_total

        # Set up the TQDM iterator
        batch_group_generator = tqdm(
            enumerate(train_dataloader), 
            total=train_batches_per_epoch, 
            leave=True
        )

        # Initialize a dictionary to track the loss over 1 epoch
        train_epoch_losses = {
            'loss': [],
            'loss_classifier': [],
            'loss_box_reg': [],
            'loss_objectness': [],
            'loss_rpn_box_reg': []
        }

        # Set the model into training mode and train
        model.train()
        for idx, (images, targets) in batch_group_generator:
            # Move data to the GPU
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Predict with the model
            loss_dict = model(images, targets)
            
            # Sum the losses from the object detector
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            
            if not np.isfinite(loss_value):
                raise OSError("The loss blew up. Dying early.")

            # Do backpropagation with the computed loss
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            # Update the results dictionary
            train_epoch_losses["loss"].append(loss_value)
            for key, val in loss_dict.items():
                train_epoch_losses[key].append(val.item())

            # Grab the learning rate so we can print it through TQDM
            for g in optimizer.param_groups:
                lr = g['lr']
                
            # For the see-saw and cosine learning rate annealers, update after every batch
            if not check_every_epoch:
                lr_scheduler.step()

            # Update TQDM
            to_print = f"Train Epoch {epoch} "
            to_print += "loss: {:.3f} ".format(np.mean(train_epoch_losses["loss"]))
            to_print += "classifier: {:.3f} ".format(np.mean(train_epoch_losses["loss_classifier"]))
            to_print += "box_reg: {:.3f} ".format(np.mean(train_epoch_losses["loss_box_reg"]))
            to_print += "objectness: {:.3f} ".format(np.mean(train_epoch_losses["loss_objectness"]))
            to_print += "rpn_box_reg: {:.3f} ".format(np.mean(train_epoch_losses["loss_rpn_box_reg"]))
            to_print += "lr: {:.6}".format(lr)
            batch_group_generator.set_description(to_print)
            batch_group_generator.update()

            # Stop when we hit the max batches per epoch
            if idx % train_batches_per_epoch == 0 and idx > 0:
                break

        torch.cuda.empty_cache()

        ##### Validate
        ### Note that I had to hack torchvision to get it to compute losses in eval mode. 
        ### See here for details: https://stackoverflow.com/questions/60339336/validation-loss-for-pytorch-faster-rcnn
        
        valid_epoch_losses = {
            'loss': [],
            'loss_classifier': [],
            'loss_box_reg': [],
            'loss_objectness': [],
            'loss_rpn_box_reg': [],
            'f1_score': [],
            'accuracy': []
        }

        # Set the model into evaluation mode and switch off autodiff (no_grad)
        model.eval()
        with torch.no_grad():
            
            # Compute the number of batches that will go through the model
            batch_size = conf["valid_iterator"]["batch_size"]
            batches_per_epoch = int(np.ceil(valid_gen.__len__() / batch_size))
            
            # Set up the TQDM iterator
            batch_group_generator = tqdm(
                enumerate(valid_dataloader), 
                total=batches_per_epoch,
                leave=True
            )

            for idx, (images, targets) in batch_group_generator:
                # Move data to the GPU
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                # Predict with the model
                loss_dict, outputs = model(images, targets)
                
                # Sum the losses from the object detector
                losses = sum(loss for loss in loss_dict.values())
                loss_value = losses.item()
                
                # Update the results dictionary
                valid_epoch_losses["loss"].append(loss_value)
                for key, val in loss_dict.items():
                    valid_epoch_losses[key].append(val.item())
                    
                # Compute f1 and accuracy
                binary_metrics = val_loss(outputs, targets, device, iou_threshold)
                f1_score = binary_metrics["f1_score"]
                accuracy = binary_metrics["accuracy"]
                valid_epoch_losses['f1_score'].append(f1_score)
                valid_epoch_losses['accuracy'].append(accuracy)
                    
                # Update TQDM
                to_print = f"Valid Epoch {epoch} "
                to_print += "loss: {:.3f} ".format(np.mean(valid_epoch_losses["loss"]))
                to_print += "classifier: {:.3f} ".format(np.mean(valid_epoch_losses["loss_classifier"]))
                to_print += "box_reg: {:.3f} ".format(np.mean(valid_epoch_losses["loss_box_reg"]))
                to_print += "objectness: {:.3f} ".format(np.mean(valid_epoch_losses["loss_objectness"]))
                to_print += "rpn_box_reg: {:.3f} ".format(np.mean(valid_epoch_losses["loss_rpn_box_reg"]))
                to_print += "f1: {:.3f} ".format(np.mean(valid_epoch_losses["f1_score"]))
                to_print += "acc: {:.3f}".format(np.mean(valid_epoch_losses["accuracy"]))
                batch_group_generator.set_description(to_print)
                batch_group_generator.update()  

        # Update result container and save
        result = {}
        result["epoch"] = epoch
        for key in ["loss", "loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"]:
            result[f"train_{key}"] = np.mean(train_epoch_losses[key])
            result[f"valid_{key}"] = np.mean(valid_epoch_losses[key])
        result[f"valid_f1"] = np.mean(valid_epoch_losses["f1_score"])
        result[f"valid_acc"] = np.mean(valid_epoch_losses["accuracy"])
        result["lr"] = early_stopping.print_learning_rate(optimizer)
        
        # Get the evaluation metric
        eval_metric = copy.deepcopy(result[validation_metric])
        eval_metric = (1.0 - eval_metric) if (validation_metric in ["valid_f1", "valid_acc"]) else eval_metric
        
        # Update the best result seen so far if necessary
        if eval_metric < best_model_results[validation_metric]:
            best_model_results = {key: result[key] for key in best_model_results}
            
        # Update the learning rate if required (LR on Plateau and Exp)
        if check_every_epoch:
            lr_scheduler.step(eval_metric)
        
        # Save the weights and check for early stopping
        early_stopping(epoch, eval_metric, model, optimizer)
        
        # Write the results container to file through custom object
        metrics_logger.update(result)
        
        # If we have not improved over the patience period, stop
        if early_stopping.early_stop:
            break
            
    return best_model_results
            

if __name__ == '__main__':
    
    if len(sys.argv) != 2:
        print("Usage: python run.py /path/to/model_config.yml")
        sys.exit()
        
    with open(sys.argv[1]) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
        
    # Copy the current model config file to the model weights save location
    save_location = os.path.join(
        config["callbacks"]["MetricsLogger"]["path_save"],
        "model.yml"
    )
    if sys.argv[1] not in save_location:
        from shutil import copyfile
        copyfile(sys.argv[1], save_location)
    
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
    logger_name = os.path.join(config["log"], "log.txt")
    fh = logging.FileHandler(logger_name,
                             mode="w",
                             encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    root.addHandler(fh)

    ############################################################
    
    best_model_results = train(config)