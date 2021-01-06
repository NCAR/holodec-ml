import numpy as np
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F

from holodecml.torch.utils import *

from typing import List, Union
from collections import defaultdict



logger = logging.getLogger(__name__)


def loss_fn(recon_x, x, mu, logvar):
    criterion = nn.BCELoss(reduction='sum')
    BCE = criterion(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD, BCE, KLD


class SymmetricCE:

    def __init__(self, alpha, gamma, kld_weight=1.0):
        self.alpha = alpha
        self.gamma = alpha
        self.kld_weight = kld_weight

        logger.info(f"Loaded Symmetric Cross Entropy loss ...")
        logger.info(
            f"... with alpha = {alpha}, gamma = {gamma}, and kld_weight = {kld_weight}")

    def __call__(self, recon_x, x, mu, logvar):
        criterion = nn.BCELoss(reduction='sum')
        BCE = criterion(recon_x, x)
        #KLD = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        norm = self.alpha + self.kld_weight * self.gamma
        return (self.alpha * BCE + self.kld_weight * self.gamma * KLD) / norm, BCE, KLD


class SymmetricMSE:

    def __init__(self, alpha, gamma, kld_weight=1.0):
        self.alpha = alpha
        self.gamma = gamma
        self.kld_weight = kld_weight

        logger.info(f"Loaded Symmetric MSE loss ...")
        logger.info(
            f"... with alpha = {alpha}, gamma = {gamma}, and kld_weight = {kld_weight}")

    def __call__(self, recon_x, x, mu, logvar):
        criterion = nn.MSELoss(reduction='sum')
        BCE = criterion(recon_x, x)
        #KLD = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        norm = self.alpha + self.kld_weight * self.gamma
        return (self.alpha * BCE + self.kld_weight * self.gamma * KLD) / norm, BCE, KLD
    

class SymmetricMAE:

    def __init__(self, alpha, gamma, kld_weight=1.0):
        self.alpha = alpha
        self.gamma = gamma
        self.kld_weight = kld_weight

        logger.info(f"Loaded Symmetric MAE loss ...")
        logger.info(
            f"... with alpha = {alpha}, gamma = {gamma}, and kld_weight = {kld_weight}")

    def __call__(self, recon_x, x, mu, logvar):
        criterion = nn.L1Loss(reduction='sum')
        BCE = criterion(recon_x, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        norm = self.alpha + self.kld_weight * self.gamma
        return (self.alpha * BCE + self.kld_weight * self.gamma * KLD) / norm, BCE, KLD

    
class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t, weights = None):
        ey_t = y_t - y_prime_t
        if weights is not None:
            ey_t *= weights
        return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))


class XTanhLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t, weights = None):
        ey_t = y_t - y_prime_t
        if weights is not None:
            ey_t *= weights
        return torch.mean(ey_t * torch.tanh(ey_t))


class XSigmoidLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t, weights = None):
        ey_t = y_t - y_prime_t
        if weights is not None:
            ey_t *= weights
        loss = torch.mean(2 * ey_t / (1 + torch.exp(-ey_t)) - ey_t)
        return loss
    

class WeightedCrossEntropyLoss(torch.nn.Module):
    """
    A custom cross-entropy class that can accomomate class weights, focal loss, and label smoothing.
    """
    def __init__(self,
                 label_smoothing: float = None,
                 gamma: float = None,
                 alpha: Union[float, List[float], torch.FloatTensor] = None,
                 average: str = "batch"
                 ) -> None:
        '''
        Parameters
        ----------
        label_smoothing : ``float``, optional (default = None)
            Whether or not to apply label smoothing to the cross-entropy loss.
            For example, with a label smoothing value of 0.2, a 4 class classification
            target would look like ``[0.05, 0.05, 0.85, 0.05]`` if the 3rd class was
            the correct label.
        gamma : ``float``, optional (default = None)
            Focal loss[*] focusing parameter ``gamma`` to reduces the relative loss for
            well-classified examples and put more focus on hard. The greater value
            ``gamma`` is, the more focus on hard examples.
        alpha : ``float`` or ``List[float]``, optional (default = None)
            Focal loss[*] weighting factor ``alpha`` to balance between classes. Can be
            used independently with ``gamma``. If a single ``float`` is provided, it
            is assumed binary case using ``alpha`` and ``1 - alpha`` for positive and
            negative respectively. If a list of ``float`` is provided, with the same
            length as the number of classes, the weights will match the classes.
            [*] T. Lin, P. Goyal, R. Girshick, K. He and P. Dollár, "Focal Loss for
            Dense Object Detection," 2017 IEEE International Conference on Computer
            Vision (ICCV), Venice, 2017, pp. 2999-3007.
        average: str, optional (default = "batch")
            If "batch", average the loss across the batches. If "token", average
            the loss across each item in the input. If ``None``, return a vector
            of losses per batch element.
        '''
        
        super(WeightedCrossEntropyLoss, self).__init__()
        
        self.label_smoothing = label_smoothing
        self.gamma = gamma 
        self.alpha = alpha
        self.average = average
        
        if average not in {None, "token", "batch"}:
            raise ValueError("Got average f{average}, expected one of "
                             "None, 'token', or 'batch'")

    def forward(self, 
                log_probs: torch.FloatTensor,
                targets: torch.IntTensor,
                weights: torch.FloatTensor) -> float:
        
        """
        Parameters
        ----------
        log_probs : ``torch.FloatTensor``, required.
            A ``torch.FloatTensor`` of size (batch_size, sequence_length, num_classes)
            which contains the log of the probability for each class.
        targets : ``torch.LongTensor``, required.
            A ``torch.LongTensor`` of size (batch, sequence_length) which contains the
            index of the true class for each corresponding step.
        weights : ``torch.FloatTensor``, required.
            A ``torch.FloatTensor`` of size (batch, sequence_length)
        
        Returns
        -------
        A torch.FloatTensor representing the cross entropy loss.
        If ``average=="batch"`` or ``average=="token"``, the returned loss is a scalar.
        If ``average is None``, the returned loss is a vector of shape (batch_size,).
        """
        # make sure weights are float
        weights = weights.float()
        # sum all dim except batch
        non_batch_dims = tuple(range(1, len(weights.shape)))
        # shape : (batch_size,)
        weights_batch_sum = weights.sum(dim=non_batch_dims)
        # shape : (batch * max_len, 1)
        targets_flat = targets.view(-1, 1).long()
        # focal loss coefficient
        if self.gamma:
            # shape : (batch * sequence_length, num_classes)
            probs_flat = log_probs.exp()
            # shape : (batch * sequence_length,)
            probs_flat = torch.gather(probs_flat, dim=1, index=targets_flat)
            # shape : (batch * sequence_length,)
            focal_factor = (1. - probs_flat) ** self.gamma
            # shape : (batch, sequence_length)
            focal_factor = focal_factor.view(*targets.size())
            weights = weights * focal_factor

        if self.alpha is not None:
            # shape : () / (num_classes,)
            if isinstance(self.alpha, (float, int)):
                # pylint: disable=not-callable
                # shape : (2,)
                alpha_factor = torch.tensor([1. - float(self.alpha), float(self.alpha)],
                                            dtype=weights.dtype, device=weights.device)
                # pylint: enable=not-callable
            elif isinstance(self.alpha, (list, np.ndarray, torch.Tensor)):
                # pylint: disable=not-callable
                # shape : (c,)
                alpha_factor = torch.tensor(self.alpha, dtype=weights.dtype, device=weights.device)
                # pylint: enable=not-callable
                if not alpha_factor.size():
                    # shape : (1,)
                    alpha_factor = alpha_factor.view(1)
                    # shape : (2,)
                    alpha_factor = torch.cat([1 - alpha_factor, alpha_factor])
            else:
                raise TypeError(('alpha must be float, list of float, or torch.FloatTensor, '
                                 '{} provided.').format(type(self.alpha)))
            # shape : (batch, max_len)
            alpha_factor = torch.gather(alpha_factor, dim=0, index=targets_flat.view(-1)).view(*targets.size())
            weights = weights * alpha_factor

        if self.label_smoothing is not None and self.label_smoothing > 0.0:
            num_classes = log_probs.size(-1)
            smoothing_value = self.label_smoothing / num_classes
            # Fill all the correct indices with 1 - smoothing value.
            one_hot_targets = torch.zeros_like(log_probs).scatter_(-1, targets_flat, 1.0 - self.label_smoothing)
            smoothed_targets = one_hot_targets + smoothing_value
            negative_log_likelihood_flat = - log_probs * smoothed_targets
            negative_log_likelihood_flat = negative_log_likelihood_flat.sum(-1, keepdim=True)
        else:
            # Contribution to the negative log likelihood only comes from the exact indices
            # of the targets, as the target distributions are one-hot. Here we use torch.gather
            # to extract the indices of the num_classes dimension which contribute to the loss.
            # shape : (batch * sequence_length, 1)
            negative_log_likelihood_flat = - torch.gather(log_probs, dim=1, index=targets_flat)
        # shape : (batch, sequence_length)
        negative_log_likelihood = negative_log_likelihood_flat.view(*targets.size())
        # shape : (batch, sequence_length)
        negative_log_likelihood = negative_log_likelihood * weights

        if self.average == "batch":
            # shape : (batch_size,)
            per_batch_loss = negative_log_likelihood.sum(non_batch_dims) / (weights_batch_sum + 1e-13)
            num_non_empty_sequences = ((weights_batch_sum > 0).float().sum() + 1e-13)
            return per_batch_loss.sum() / num_non_empty_sequences / weights.shape[0]
        elif self.average == "token":
            return negative_log_likelihood.sum() / (weights_batch_sum.sum() + 1e-13) / weights.shape[0]
        else:
            # shape : (batch_size,)
            per_batch_loss = negative_log_likelihood.sum(non_batch_dims) / (weights_batch_sum + 1e-13)
            return per_batch_loss / weights.shape[0]
            


def distance_sorted_loss(true, pred, true_tokens, pred_tokens, return_coordinate_errors = False):
    
    # size (batch, particles, coordinates)
    true = torch.cat([
        torch.cat([x.unsqueeze(0) for x in y]).unsqueeze(0) for y in true
    ]).permute(2, 0, 1)
    pred = torch.cat([
        torch.cat([x.unsqueeze(0) for x in y]).unsqueeze(0) for y in pred
    ]).permute(2, 0, 1)
    
    accuracy = []
    total_error = []
    coordinate_errors = defaultdict(list)
    for k in range(true.shape[0]):    
        true_part = true[k]
        pred_part = pred[k]

        tosort = []
        for p1 in range(len(true_part)):
            for p2 in range(len(true_part)):
                predicted_coors = Point(pred_part[p1])
                truth_coors = Point(true_part[p2])
                if (pred_tokens[k][p1] == true_tokens[k][p2]): # if token is correct, force select the pair (dist = 0)
                    dist = 0.0
                else:
                    dist = distance(predicted_coors, truth_coors)
                tosort.append([
                    dist, p1, p2,
                    predicted_coors, truth_coors
                ])

        tokens, paired, seen_true, seen_pred = [], [], [], []
        for (dist, p1, p2, p, t) in sorted(tosort): # sort by distance
            if p1 not in seen_pred and p2 not in seen_true:
                paired.append([p, t])
                tokens.append([p1, p2])
                seen_pred.append(p1)
                seen_true.append(p2)
                accuracy.append(int(pred_tokens[k][p1]==true_tokens[k][p2]))

        for (x,y) in paired:
            xe = x.x - y.x
            ye = x.y - y.y
            ze = x.z - y.z
            de = x.d - y.d
            total_error += [abs(xe), abs(ye), abs(ze), abs(de)]
            
            if return_coordinate_errors:
                coordinate_errors["x"].append([x.x.item(), y.x.item()])
                coordinate_errors["y"].append([x.y.item(), y.y.item()])
                coordinate_errors["z"].append([x.z.item(), y.z.item()])
                coordinate_errors["d"].append([x.d.item(), y.d.item()])
        
    total_error = torch.mean(torch.stack(total_error))
    accuracy = np.mean(accuracy)
    
    if return_coordinate_errors:
        return total_error, accuracy, coordinate_errors
    else:
        return total_error, accuracy