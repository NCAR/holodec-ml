from torch.optim.lr_scheduler import ReduceLROnPlateau
from echo.src.base_objective import BaseObjective
from collections import defaultdict
import pandas as pd
import numpy as np
import torch.fft
import logging
import random
import psutil
import optuna
import torch
import scipy
import time
import tqdm
import gc
from holodecml.data import PickleReader, UpsamplingReader, XarrayReader
from holodecml.propagation import InferencePropagator
from holodecml.transforms import LoadTransformations
from holodecml.models import load_model
from holodecml.losses import load_loss
import os
import warnings
warnings.filterwarnings("ignore")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


available_ncpus = len(psutil.Process().cpu_affinity())


# ### Set seeds for reproducibility
def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True


class Objective(BaseObjective):

    def __init__(self, config, metric="val_loss", device="cpu"):

        # Initialize the base class
        BaseObjective.__init__(self, config, metric, device)

    def train(self, trial, conf):
        try:
            seed = 1000 if "seed" not in conf else conf["seed"]
            seed_everything(seed)

            tile_size = int(conf["data"]["tile_size"])
            step_size = int(conf["data"]["step_size"])
            data_path = conf["data"]["output_path"]
            total_positive = int(conf["data"]["total_positive"])
            total_negative = int(conf["data"]["total_negative"])
            total_examples = int(conf["data"]["total_training"])
            transform_mode = "None" if "transform_mode" not in conf[
                "data"] else conf["data"]["transform_mode"]
            config_ncpus = int(conf["data"]["cores"])
            use_cached = False if "use_cached" not in conf["data"] else conf["data"]["use_cached"]

            # Set up number of CPU cores available
            if config_ncpus > available_ncpus:
                ncpus = available_ncpus
                #ncpus = int(2 * available_ncpus)
            else:
                ncpus = config_ncpus
                #ncpus = int(2 * config_ncpus)

            name_tag = f"{tile_size}_{step_size}_{total_positive}_{total_negative}_{total_examples}_{transform_mode}"
            fn_train = f"{data_path}/training_{name_tag}.nc"
            fn_valid = f"{data_path}/validation_{name_tag}.nc"
            #fn_train = f"{data_path}/training_{tile_size}_{step_size}.pkl"
            #fn_valid = f"{data_path}/validation_{tile_size}_{step_size}.pkl"

            epochs = conf["trainer"]["epochs"]
            start_epoch = 0 if "start_epoch" not in conf["trainer"] else conf["trainer"]["start_epoch"]
            train_batch_size = conf["trainer"]["train_batch_size"]
            valid_batch_size = conf["trainer"]["valid_batch_size"]
            batches_per_epoch = conf["trainer"]["batches_per_epoch"]
            valid_batches_per_epoch = 100
            stopping_patience = conf["trainer"]["stopping_patience"]
            grad_clip = 1.0

            model_loc = conf["save_loc"]
            model_name = conf["model"]["name"]
            color_dim = conf["model"]["in_channels"]

            training_loss = "dice-bce" if "training_loss" not in conf["trainer"] else conf["trainer"]["training_loss"]
            valid_loss = "dice" if "validation_loss" not in conf[
                "trainer"] else conf["trainer"]["validation_loss"]

            learning_rate = conf["optimizer"]["learning_rate"]
            weight_decay = conf["optimizer"]["weight_decay"]

            # Set up CUDA/CPU devices
            is_cuda = torch.cuda.is_available()
            data_device = torch.device(
                "cpu") if "device" not in conf["data"] else conf["data"]["device"]

            if torch.cuda.device_count() >= 2 and "cuda" in data_device:
                data_device = "cuda:0"
                device = "cuda:1"
                device_ids = list(range(1, torch.cuda.device_count()))
            else:
                data_device = torch.device("cpu")
                device = torch.device(torch.cuda.current_device()
                                      ) if is_cuda else torch.device("cpu")
                device_ids = list(range(torch.cuda.device_count()))

            logging.info(f"There are {torch.cuda.device_count()} GPUs available")
            logging.info(
                f"Using device {data_device} to perform wave propagation, and {device_ids} for training the model")

            # Create directories if they do not exist and copy yml file
            os.makedirs(model_loc, exist_ok=True)
            if not os.path.exists(os.path.join(model_loc, "model.yml")):
                shutil.copy(config, os.path.join(model_loc, "model.yml"))

            # Load the preprocessing transforms
            if "Normalize" in conf["transforms"]["training"]:
                conf["transforms"]["validation"]["Normalize"]["mode"] = conf["transforms"]["training"]["Normalize"]["mode"]
                conf["transforms"]["inference"]["Normalize"]["mode"] = conf["transforms"]["training"]["Normalize"]["mode"]

            train_transforms = LoadTransformations(conf["transforms"]["training"])
            valid_transforms = LoadTransformations(conf["transforms"]["validation"])

            # Load the data class for reading and preparing the data as needed to train the u-net
            train_dataset = XarrayReader(fn_train, train_transforms, mode = "mask")
            test_dataset = XarrayReader(fn_valid, valid_transforms, mode = "mask")

            # Load the iterators for batching the data
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=train_batch_size,
                num_workers=ncpus,
                pin_memory=True,
                shuffle=True)

            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=valid_batch_size,
                num_workers=ncpus,  # 0 = One worker with the main process
                pin_memory=True,
                shuffle=False)

            # Load a segmentation model
            unet = load_model(conf["model"]).to(device)

            if start_epoch > 0:
                # Load weights
                logging.info(f"Restarting training starting from epoch {start_epoch}")
                logging.info(f"Loading model weights from {model_loc}")
                checkpoint = torch.load(
                    os.path.join(model_loc, "best.pt"),
                    map_location=lambda storage, loc: storage
                )
                unet.load_state_dict(checkpoint["model_state_dict"])
                learning_rate = checkpoint["optimizer_state_dict"]["param_groups"][0]["lr"]

            unet = unet.to(device)
            total_params = sum(p.numel() for p in unet.parameters())
            trainable_params = sum(p.numel()
                                   for p in unet.parameters() if p.requires_grad)

            # Multi-gpu support
            if len(device_ids) > 1:
                unet = torch.nn.DataParallel(unet, device_ids=device_ids)

            # Load an optimizer
            optimizer = torch.optim.AdamW(
                unet.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
            if start_epoch > 0:
                # Load weights
                logging.info(f"Loading optimizer state from {model_loc}")
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            # Specify the training and validation losses
            train_criterion = load_loss(training_loss)
            test_criterion = load_loss(valid_loss, split="validation")

            # Load a learning rate scheduler
            lr_scheduler = ReduceLROnPlateau(
                optimizer,
                patience=1,
                min_lr=1.0e-13,
                verbose=True
            )

            # Reload the results saved in the training csv if continuing to train
            if start_epoch == 0:
                results_dict = defaultdict(list)
                epoch_test_losses = []
            else:
                results_dict = defaultdict(list)
                saved_results = pd.read_csv(f"{model_loc}/training_log.csv")
                epoch_test_losses = list(saved_results["valid_loss"])
                for key in saved_results.columns:
                    if key == "index":
                        continue
                    results_dict[key] = list(saved_results[key])
                # update the learning rate scheduler
                for valid_loss in epoch_test_losses:
                    lr_scheduler.step(valid_loss)

            # Train a U-net model
            manual_loss = []
            for epoch in range(start_epoch, epochs):
                # Train the model
                unet.train()

                batch_loss = []

                # set up a custom tqdm
                batch_group_generator = tqdm.tqdm(
                    enumerate(train_loader),
                    total=batches_per_epoch,
                    leave=True
                )

                t0 = time.time()

                for k, (inputs, y) in batch_group_generator:
                    # Move data to the GPU, if not there already
                    inputs = inputs.to(device)
                    y = y.to(device)

                    # Clear gradient
                    optimizer.zero_grad()

                    # get output from the model, given the inputs
                    pred_mask = unet(inputs)

                    # get loss for the predicted output
                    loss = train_criterion(pred_mask, y.float())
                    
                    if not np.isfinite(loss.cpu().item()):
                        logging.warning(
                            f"Trial {trial.number} is being pruned due to loss = NaN while training")
                        raise optuna.TrialPruned()

                    # get gradients w.r.t to parameters
                    loss.backward()
                    batch_loss.append(loss.item())

                    # gradient clip
                    torch.nn.utils.clip_grad_norm_(unet.parameters(), grad_clip)

                    # update parameters
                    optimizer.step()

                    # update tqdm
                    to_print = "Epoch {} train_loss: {:.6f}".format(
                        epoch, np.mean(batch_loss))
                    to_print += " lr: {:.12f}".format(optimizer.param_groups[0]['lr'])
                    batch_group_generator.set_description(to_print)
                    batch_group_generator.update()

                    # stop the training epoch when train_batches_per_epoch have been used to update
                    # the weights to the model
                    if k >= batches_per_epoch and k > 0:
                        break

                    #lr_scheduler.step(epoch + k / batches_per_epoch)

                # Shutdown the progbar
                batch_group_generator.close()

                # Compuate final performance metrics before doing validation
                train_loss = np.mean(batch_loss)

                # clear the cached memory from the gpu
                torch.cuda.empty_cache()
                gc.collect()

                # Test the model
                unet.eval()
                with torch.no_grad():

                    batch_loss = []

                    # set up a custom tqdm
                    batch_group_generator = tqdm.tqdm(
                        enumerate(test_loader),
                        leave=True
                    )

                    for k, (inputs, y) in batch_group_generator:
                        # Move data to the GPU, if not there already
                        inputs = inputs.to(device)
                        y = y.to(device)
                        # get output from the model, given the inputs
                        pred_mask = unet(inputs)
                        # get loss for the predicted output
                        loss = test_criterion(pred_mask, y.float())
                        batch_loss.append(loss.item())
                        # update tqdm
                        to_print = "Epoch {} test_loss: {:.6f}".format(
                            epoch, np.mean(batch_loss))
                        batch_group_generator.set_description(to_print)
                        batch_group_generator.update()

                        if k >= valid_batches_per_epoch and k > 0:
                            break

                    # Shutdown the progbar
                    batch_group_generator.close()

                # Load the manually labeled data
                man_loss = predict_on_manual(
                    epoch, conf, unet, device) #+ np.mean(batch_loss)
                manual_loss.append(float(man_loss))

                # Use the supplied metric in the config file as the performance metric to toggle learning rate and early stopping
                test_loss = np.mean(batch_loss)
                if not np.isfinite(test_loss):
                    raise optuna.TrialPruned()

                epoch_test_losses.append(test_loss)

                # clear the cached memory from the gpu
                torch.cuda.empty_cache()
                gc.collect()


                # Get the last learning rate
                learning_rate = optimizer.param_groups[0]['lr']

                # Put things into a results dictionary -> dataframe
                results_dict['epoch'].append(epoch)
                results_dict['train_loss'].append(train_loss)
                results_dict['valid_loss'].append(test_loss)
                results_dict['manual_loss'].append(man_loss)
                results_dict["learning_rate"].append(learning_rate)
                df = pd.DataFrame.from_dict(results_dict).reset_index()

                # Save the dataframe to disk
                df.to_csv(f"{model_loc}/trial_results/training_log_{trial.number}.csv", index=False)

                # Lower the learning rate if we are not improving
                lr_scheduler.step(test_loss)

                # Report result to the trial
                attempt = 0
                while attempt < 10:
                    try:
                        trial.report(test_loss, step=epoch)
                        break
                    except Exception as E:
                        logging.warning(
                            f"WARNING failed to update the trial with manual loss {test_loss} at epoch {epoch}. Error {str(E)}")
                        logging.warning(f"Trying again ... {attempt + 1} / 10")
                        time.sleep(1)
                        attempt += 1

                # Stop training if we have not improved after X epochs (stopping patience)
                best_epoch = [i for i, j in enumerate(
                    epoch_test_losses) if j == min(epoch_test_losses)][0]
                offset = epoch - best_epoch
                if offset >= stopping_patience:
                    logging.info(f"Trial {trial.number} is stopping early")
                    break

            if len(epoch_test_losses) == 0:
                trial.should_prune()
                
            best_epoch = [i for i, j in enumerate(
                    epoch_test_losses) if j == min(epoch_test_losses)][0]

            result = {
                "manual_loss": manual_loss[best_epoch],
                "mask_loss": epoch_test_losses[best_epoch]
            }

            return result
        
        except Exception as E:
            if "CUDA" in str(E):
                logging.warning(
                    f"Pruning trial {trial.number} due to CUDA memory overflow: {str(E)}.")
                raise optuna.TrialPruned()
            elif "dilated" in str(E):
                raise optuna.TrialPruned()
            else:
                logging.warning(
                    f"Trial {trial.number} failed due to error: {str(E)}.")
                raise E


def dice(true, pred, k=1):
    true = np.array(true)
    pred = np.array(pred)
    intersection = np.sum(pred[true == k]) * 2.0
    dice = intersection / (np.sum(pred) + np.sum(true) + 1e-12)
    return dice


def predict_on_manual(epoch, conf, model, device, max_cluster_per_image=10000):

    model.eval()

    n_bins = conf["data"]["n_bins"]
    tile_size = conf["data"]["tile_size"]
    step_size = conf["data"]["step_size"]
    marker_size = conf["data"]["marker_size"]
    data_path = conf["data"]["data_path"]
    raw_path = conf["data"]["raw_data"]
    output_path = conf["data"]["output_path"]
    transform_mode = "None" if "transform_mode" not in conf[
        "data"] else conf["data"]["transform_mode"]

    model_loc = conf["save_loc"]
    model_name = conf["model"]["name"]
    color_dim = conf["model"]["in_channels"]

    inference_mode = conf["inference"]["mode"]
    probability_threshold = conf["inference"]["probability_threshold"]
    valid_batch_size = conf["trainer"]["valid_batch_size"]
    
    if "Normalize" in conf["transforms"]["training"]:
        conf["transforms"]["validation"]["Normalize"]["mode"] = conf["transforms"]["training"]["Normalize"]["mode"]
        conf["transforms"]["inference"]["Normalize"]["mode"] = conf["transforms"]["training"]["Normalize"]["mode"]

    tile_transforms = None if "inference" not in conf["transforms"] else LoadTransformations(
        conf["transforms"]["inference"])

    output_path = output_path.replace("style_transfered", "tiled_synthetic")
    with torch.no_grad():
        inputs = torch.from_numpy(np.load(os.path.join(
            output_path, f'manual_images_{transform_mode}.npy'))).float()
        labels = torch.from_numpy(np.load(os.path.join(
            output_path, f'manual_labels_{transform_mode}.npy'))).float()

        prop = InferencePropagator(
            raw_path,
            n_bins=n_bins,
            color_dim=color_dim,
            tile_size=tile_size,
            step_size=step_size,
            marker_size=marker_size,
            transform_mode=transform_mode,
            device=device,
            model=model,
            mode=inference_mode,
            probability_threshold=probability_threshold,
            transforms=tile_transforms
        )
        # apply transforms
        inputs = torch.from_numpy(np.expand_dims(
            np.vstack([prop.apply_transforms(x) for x in inputs.numpy()]), 1))
        
        performance = defaultdict(list)
        batched = zip(
            np.array_split(inputs, inputs.shape[0] // valid_batch_size),
            np.array_split(labels, inputs.shape[0] // valid_batch_size)
        )
        my_iter = tqdm.tqdm(
            batched, 
            total = inputs.shape[0] // valid_batch_size, 
            leave = True
        )

        with torch.no_grad():
            for (x, y) in my_iter:
                pred_labels = prop.model(x.to(device)) > probability_threshold
                for pred_label, true_label in zip(pred_labels, y):
                    pred_label = torch.sum(pred_label).float().cpu()
                    pred_label = 1 if pred_label > 0 else 0
                    performance["pred_label"].append(pred_label)
                    performance["true_label"].append(int(true_label[0].item()))
                man_loss = dice(performance["true_label"],
                                performance["pred_label"])
                to_print = "Epoch {} man_loss: {:.6f}".format(epoch, man_loss)
                my_iter.set_description(to_print)
                my_iter.update()
                
        man_loss = dice(performance["true_label"], performance["pred_label"])

        # Shutdown the progbar
        my_iter.close()

    return 1.0 - man_loss
