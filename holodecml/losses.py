import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging
import numpy as np

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import tensorflow as tf
import tensorflow.keras.backend as K


try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse as ifilterfalse

    

logger = logging.getLogger(__name__)



## See also: https://github.com/JunMa11/SegLoss/blob/master/losses_pytorch/dice_loss.py

def load_loss(loss_name, split = "training"):
    
    supported = ["dice", "dice-bce", "iou", "focal", "tyversky", 
                 "focal-tyversky", "lovasz-hinge", "combo"]
    
    logger.info(f"Loading {split} loss function {loss_name}")
    
    if loss_name == "dice":
        return DiceLoss()
    elif loss_name == "dice-bce":
        return DiceBCELoss()
    elif loss_name == "iou":
        return IoULoss()
    elif loss_name == "focal":
        return FocalLoss()
    elif loss_name == "tyversky":
        return TverskyLoss()
    elif loss_name == "focal-tyversky":
        return FocalTverskyLoss()
    elif loss_name == "lovasz-hinge":
        return LovaszHingeLoss()
    elif loss_name == "combo":
        return ComboLoss()
    else:
        raise OSError(f"Loss name {loss_name} not recognized. Please choose from {supported}")


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice
    

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE
        
        
class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU
    
    
class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss
    
    

class TverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.5, beta=0.5):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
       
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        
        return 1 - Tversky
    

class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.5, beta=0.5, gamma=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        FocalTversky = (1 - Tversky)**gamma
                       
        return FocalTversky
    
    
class ComboLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(ComboLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.5, CE_RATIO=0.5, eps=1e-9):
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        intersection = (inputs * targets).sum()    
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        
        inputs = torch.clamp(inputs, eps, 1.0 - eps)       
        out = - (alpha * ((targets * torch.log(inputs)) + ((1 - alpha) * (1.0 - targets) * torch.log(1.0 - inputs))))
        weighted_ce = out.mean(-1)
        combo = (CE_RATIO * weighted_ce) - ((1 - CE_RATIO) * dice)
        
        return combo


class LovaszHingeLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(LovaszHingeLoss, self).__init__()

    def forward(self, inputs, targets):
        #inputs = F.sigmoid(inputs)    
        Lovasz = lovasz_hinge(inputs, targets, per_image=False)                       
        return Lovasz
    
    
def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                          for log, lab in zip(logits, labels))
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


# --------------------------- HELPER FUNCTIONS ---------------------------

def isnan(x):
    return x != x


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n

    
    
# --------------------------- OLD ---------------------------
    

class SymmetricCrossEntropy:

    def __init__(self, a: float = 1.0, b: float = 1.0) -> None:
        self.a = a
        self.b = b

    def __call__(self, *args, **kwargs) -> float:
        bce = tf.keras.losses.CategoricalCrossentropy()
        kld = tf.keras.losses.KLDivergence()
        return self.a * bce(*args, **kwargs) + self.b * kld(*args, **kwargs)


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def wmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def R2(y_true, y_pred):
    """ Is actually 1 - R2
    """
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return SS_res / (SS_tot + K.epsilon())


def keras_mse(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true))


def noisy_true_particle_loss(y_true, y_pred):
    # y_true and y_pred will have shape (batch_size x max_num_particles x 5)
    loss_real = tf.reduce_mean(tf.abs(y_true[y_true[:, :, -1] > 0] - y_pred[y_true[:, :, -1] > 0]))
    loss_bce = tf.keras.losses.binary_crossentropy(tf.reshape(y_true[:, :, -1],[-1]),
                                                   tf.reshape(y_pred[:, :, -1],[-1]))
    loss_total = loss_real + loss_bce
    return loss_total

def random_particle_distance_loss(y_true, y_pred):
    loss_xy = tf.zeros((), dtype=tf.float32)
    loss_z = tf.zeros((), dtype=tf.float32)
    loss_d = tf.zeros((), dtype=tf.float32)
    
    for h in range(tf.shape(y_pred)[0]):
        y_pred_h = y_pred[h]
        print("y_pred_h.shape", y_pred_h.get_shape())
        y_true_h = y_true[h]
        print("y_true_h.shape", y_true_h.shape)
        real_idx = tf.argmin(y_true_h[:, -1], axis=0)
        if real_idx == 0:
            real_idx = tf.cast(tf.shape(y_true_h)[0], dtype=tf.int64)
        print("real_idx.shape", real_idx.get_shape())
        y_true_h = y_true_h[:real_idx]
        print("y_true_h.shape", y_true_h.get_shape())
        
        dist_x = (y_pred_h[:, 0:1] - tf.transpose(y_true_h)[0:1, :]) ** 2
        dist_y = (y_pred_h[:, 1:2] - tf.transpose(y_true_h)[1:2, :]) ** 2
        dist_xy = dist_x + dist_y
        print(f"dist_xy.shape: {dist_xy.shape}")
        loss_xy_h = tf.math.reduce_sum(tf.math.reduce_min(dist_xy, axis=1))
        loss_xy = loss_xy + loss_xy_h

        # determine index of true particle closest to each predicted particle
        max_idx = tf.cast(tf.math.argmin(dist_xy, axis=1), dtype=tf.int32)
        max_idx_2d = tf.stack((tf.range(tf.shape(dist_xy)[0]), max_idx), axis=-1)

        loss_z_h = (y_pred_h[:, 2:3] - tf.transpose(y_true_h)[2:3, :]) ** 2
        loss_z_h = tf.math.reduce_sum(tf.gather_nd(loss_z_h, max_idx_2d))
        loss_z = loss_z + loss_z_h
        
        loss_d_h = (y_pred_h[:, 3:4] - tf.transpose(y_true_h)[3:4, :]) ** 2
        loss_d_h = tf.math.reduce_sum(tf.gather_nd(loss_d_h, max_idx_2d))
        loss_d = loss_d + loss_d_h

    loss_xy = loss_xy/tf.cast(tf.shape(y_pred)[0], dtype=tf.float32)
    loss_z = loss_z/tf.cast(tf.shape(y_pred)[0], dtype=tf.float32)
    loss_d = loss_d/tf.cast(tf.shape(y_pred)[0], dtype=tf.float32)

    valid_error = loss_xy + loss_z + loss_d
    print(f"ERROR SHAPE: {valid_error.shape}")

    return valid_error

def unet_loss(y_true, y_pred):
    y_true_p = y_true[:, :, :, 0]
    y_true_z = y_true[:, :, :, 1]
    y_true_d = y_true[:, :, :, 2]
    y_pred_p = y_pred[:, :, :, 0] + 1e-8 # to avoid log(0)
    y_pred_z = y_pred[:, :, :, 1]
    y_pred_d = y_pred[:, :, :, 2]

#     bce = y_true_p * -tf.math.log(y_pred_p)
    bce = (y_true_p - y_pred_p) ** 2
    loss_z = y_true_p * (y_true_z - y_pred_z) ** 2
    loss_d = y_true_p * (y_true_d - y_pred_d) ** 2
    loss = bce + loss_z + loss_d
    loss = tf.reduce_sum(loss) / tf.reduce_sum(y_true_p)
    
    return loss

def unet_loss_xy(y_true, y_pred):
    y_true_p = y_true[:, :, :, 0]
    y_pred_p = y_pred[:, :, :, 0] + 1e-8 # to avoid log(0)
    
    bce = (y_true_p - y_pred_p) ** 2
    loss = tf.reduce_sum(bce) / tf.reduce_sum(y_true_p)
    
    return loss

def unet_loss_xy_log(y_true, y_pred):
    y_true_p = y_true[:, :, :, 0]
    y_pred_p = y_pred[:, :, :, 0] + 1e-8 # to avoid log(0)
    
    bce = (tf.math.log(y_true_p + 1) - tf.math.log(y_pred_p + 1)) ** 2
    loss = tf.reduce_sum(bce) / tf.reduce_sum(y_true_p)
    
<<<<<<< HEAD
    return loss
=======
    return loss
>>>>>>> d676a376916ca6b137ae8bf51be577566f754e1d
