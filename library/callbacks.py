from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import Callback, ModelCheckpoint, CSVLogger, EarlyStopping
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
from typing import List, Dict
import logging


logger = logging.getLogger(__name__)


class LearningRateTracker(Callback):
    def on_epoch_end(self, epoch: int, logs: Dict[str,float] = {}) -> None:
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)


def get_callbacks(config: Dict[str, str]) -> List[Callback]:
    
    callbacks = []
    
    if "callbacks" in config:
        config = config["callbacks"]
    else:
        return []
    
    if "ModelCheckpoint" in config:
        callbacks.append(ModelCheckpoint(**config["ModelCheckpoint"]))
        logger.info("... loaded Checkpointer with params {}")
        
    if "EarlyStopping" in config:
        callbacks.append(EarlyStopping(**config["EarlyStopping"]))
        logger.info("... loaded EarlyStopping")
    
    #LearningRateTracker(),  ## ReduceLROnPlateau does this already, use when supplying custom LR annealer
    
    if "ReduceLROnPlateau" in config:
        callbacks.append(ReduceLROnPlateau(**config["ReduceLROnPlateau"]))
        logger.info("... loaded ReduceLROnPlateau")
    
    if "CSVLogger" in config:
        callbacks.append(CSVLogger(**config["CSVLogger"]))
        logger.info("... loaded CSVLogger")
        
    return callbacks