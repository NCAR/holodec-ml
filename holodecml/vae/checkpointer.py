from collections import defaultdict
import torch, os, time, math, glob
from typing import List, Dict
import pandas as pd
import numpy as np
import logging


logger = logging.getLogger(__name__)


def load_checkpoint(checkpoint_path: str):
    # It's weird that if `map_location` is not given, it will be extremely slow.
    return torch.load(checkpoint_path, map_location=lambda storage, loc: storage)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, 
                 patience=7,
                 verbose=False,
                 delta=0,
                 save_every_epoch=False,
                 path_save='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path_save
        self.dirpath = os.path.dirname(self.path)
        self.save_every_epoch = save_every_epoch
        
        logger.info(f"Loaded EarlyStopping checkpointer with patience {self.patience}")
    
    def __call__(self, epoch, val_loss, model, optimizer):

        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch, val_loss, model, optimizer, best = True)
        elif score < (self.best_score + self.delta):
            self.best_score = score
            self.save_checkpoint(epoch, val_loss, model, optimizer, best = True)
            self.counter = 0
        else:
            self.counter += 1
            logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.save_every_epoch:
                self.save_checkpoint(epoch, val_loss, model, optimizer, best = False)
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, epoch, val_loss, model, optimizer, best = False):
        '''Saves model when validation loss decrease.'''
        if best:
            logger.info(
                f'Validation loss decreased on epoch {epoch} ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model.'
            )
        checkpoint = {
            'epoch': epoch,
            'val_loss': val_loss,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr': self.print_learning_rate(optimizer)
        }
        if not best: # save a model, not the best one seen so far
            save_path = os.path.join(self.dirpath, "checkpoint.pt")
            torch.save(checkpoint, save_path)
        else: # save best model so far
            save_path = os.path.join(self.dirpath, "best.pt")
            torch.save(checkpoint, save_path)
            self.val_loss_min = val_loss
        
    def print_learning_rate(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group["lr"]
        
            
        
class MetricsLogger:
    
    def __init__(self, path_save: str, reload: bool = False) -> None:
        
        self.path_save = os.path.join(f"{path_save}", "training_log.csv")
        
        if reload:
            self.load()
            logger.info(f"Loaded a previous metrics file from {self.path_save}")
        else:
            self.metrics = defaultdict(list)
            logger.info(f"Loaded a metrics logger {self.path_save} to track the training results")
            
    
    def update(self, data: Dict[str, float]) -> None:
        for key, value in data.items():
            self.metrics[key].append(value)
        self.save()
        
    
    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(self.metrics)
    
    
    def save(self) -> None:
        self.to_pandas().to_csv(
            self.path_save, 
            sep = ',', 
            encoding = 'utf-8', 
            index = None
        )
        
        
    def load(self) -> None:
        self.metrics = pd.read_csv(
            self.path_save, 
            sep = ',', 
            encoding = 'utf-8'
        ).to_dict()