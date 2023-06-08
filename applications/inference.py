from hagelslag.evaluation.MetricPlotter import roc_curve, performance_diagram
from holodecml.models import load_model
from argparse import ArgumentParser
from functools import partial
from pathlib import Path

import torch.multiprocessing as mp
import numpy as np
import subprocess
import traceback
import logging
import signal
import joblib
import scipy
import sys
import yaml
import time
import tqdm
import glob
import os
import warnings

warnings.filterwarnings("ignore")


logger = logging.getLogger(__name__)


def launch_pbs_jobs(nodes, gpus, config, save_path="./"):
    script_path = Path(__file__).absolute()
    for worker in range(nodes):
        script = f"""
        #!/bin/bash -l
        #PBS -N holo-inf-{worker}
        #PBS -l select=1:ncpus=8:ngpus={gpus}:mem=128GB
        #PBS -l walltime=12:00:00
        #PBS -l gpu_type=v100
        #PBS -A NAML0001
        #PBS -q casper
        #PBS -o {os.path.join(save_path, "out")}
        #PBS -e {os.path.join(save_path, "out")}

        source ~/.bashrc
        ncar_pylib /glade/work/$USER/py37
        python {script_path} -c {config} -n {nodes} -nid {worker}
        """
        with open("launcher.sh", "w") as fid:
            fid.write(script)
        jobid = subprocess.Popen(
            "qsub launcher.sh",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).communicate()[0]
        jobid = jobid.decode("utf-8").strip("\n")
        print(jobid)
    os.remove("launcher.sh")


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def main(worker_info=(0, "cuda:0"), conf=None, delay=30):
    # unpack the worker and GPU ids
    this_worker, device = worker_info
    logger.info(f"Initialized worker {this_worker} on device {device}")

    # import torch / GPU packages
    import torch
    import torch.fft

    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

    from holodecml.metrics import DistributedROC
    from holodecml.transforms import LoadTransformations
    from holodecml.propagation import InferencePropagator
    from holodecml.propagation import InferencePropagationNoLabels
    from holodecml.data import save_sparse_csr

    ###################################

    n_nodes = conf["inference"]["n_nodes"]
    n_gpus = conf["inference"]["gpus_per_node"]
    threads_per_gpu = conf["inference"]["threads_per_gpu"]
    workers = int(n_nodes * n_gpus * threads_per_gpu)

    n_bins = conf["data"]["n_bins"]
    tile_size = conf["data"]["tile_size"]
    step_size = conf["data"]["step_size"]
    marker_size = conf["data"]["marker_size"]
    transform_mode = (
        "None"
        if "transform_mode" not in conf["data"]
        else conf["data"]["transform_mode"]
    )

    model_loc = conf["save_loc"]
    model_name = conf["model"]["name"]
    color_dim = conf["model"]["in_channels"]

    batch_size = conf["inference"]["batch_size"]
    save_arrays = conf["inference"]["save_arrays"]
    save_metrics = conf["inference"]["save_metrics"]
    save_prob = conf["inference"]["save_probs"]
    inference_mode = conf["inference"]["mode"]

    if "probability_threshold" in conf["inference"]:
        probability_threshold = conf["inference"]["probability_threshold"]
    else:
        probability_threshold = 0.5

    verbose = conf["inference"]["verbose"]
    data_set = conf["inference"]["data_set"]["path"]
    data_set_name = conf["inference"]["data_set"]["name"]

    prop_data_loc = os.path.join(model_loc, f"{data_set_name}/propagated")
    roc_data_loc = os.path.join(model_loc, f"{data_set_name}/roc")
    image_data_loc = os.path.join(model_loc, f"{data_set_name}/images")

    save_dirs = [prop_data_loc]
    if conf["inference"]["save_metrics"]:
        save_dirs.append(roc_data_loc)
        save_dirs.append(image_data_loc)
    for directory in save_dirs:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

    # roc threshold
    obs_threshold = 1.0
    if conf["inference"]["data_set"]["name"] == "raw":
        thresholds = np.linspace(0, 1, 100)
    else:
        thresholds = 1.0 - np.logspace(
            -5, 0, num=50, endpoint=True, base=10.0, dtype=None, axis=0
        )
        thresholds = thresholds[::-1]

    # Configuration settings for which holograms to process
    h_conf = conf["inference"]["data_set"]["holograms"]
    if isinstance(h_conf, dict):
        h_min = conf["inference"]["data_set"]["holograms"]["min"]
        h_max = conf["inference"]["data_set"]["holograms"]["max"]
        h_range = range(h_min, h_max)
    elif isinstance(h_conf, list):
        h_range = h_conf
    elif isinstance(h_conf, int) or isinstance(h_conf, float):
        h_range = [h_conf]
    else:
        raise OSError(f"Unidentified h-range settings {h_conf}")

    # Load the model
    logger.info(f"Worker {this_worker}: Loading and moving model to device {device}")
    model = load_model(conf["model"]).to(device)

    # take a nap before trying to load the data onto the GPU
    nap_time = delay * (this_worker % threads_per_gpu)
    logger.info(
        f"Worker {this_worker}: Napping for {nap_time} s before mounting the model"
    )
    time.sleep(nap_time)

    # Load the weights from the training location
    checkpoint = torch.load(
        os.path.join(model_loc, "best.pt"), map_location=lambda storage, loc: storage
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Worker {this_worker}: There are {total_params} total model parameters"
    )

    # Load the image transformations
    if "inference" in conf["transforms"]:
        logger.info(f"Worker {this_worker}: Loading image transformations")
        if "Normalize" in conf["transforms"]["training"]:
            conf["transforms"]["inference"]["Normalize"]["mode"] = conf["transforms"][
                "training"
            ]["Normalize"]["mode"]
        tile_transforms = LoadTransformations(conf["transforms"]["inference"])
    else:
        tile_transforms = None

    # Load the wave prop interence object
    logger.info(f"Worker {this_worker}: Loading an inference wave-prop generator")

    try:
        with torch.no_grad():

            try:
                prop = InferencePropagator(
                    data_set,
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
                    transforms=tile_transforms,
                )
                prop.h_ds["x"]
                logger.info("... true coordinates detected in the source file")
            except KeyError:
                logger.info("... no true coordinates detected in the source file")
                prop = InferencePropagationNoLabels(
                    data_set,
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
                    transforms=tile_transforms,
                )

            # Create a list of z-values to propagate to
            z_list = prop.create_z_plane_lst(planes_per_call=1)
            z_list = np.array_split(z_list, workers)[this_worker]
            logger.info(
                f"Worker {this_worker}: Performing inference on subset of z planes {this_worker + 1} out of {workers}"
            )

            # Main loop to call the generator, predict with the model, and aggregate and save the results
            for nc, h_idx in enumerate(h_range):

                planes_processed = int(this_worker * (n_bins // workers))
                logger.info(
                    f"Worker {this_worker} is starting at {planes_processed} for hologram {h_idx} ({nc + 1} / {len(h_range)})"
                )

                inference_generator = prop.get_next_z_planes_labeled(
                    h_idx,
                    z_list,
                    batch_size=batch_size,
                    thresholds=thresholds,
                    return_arrays=save_arrays,
                    return_metrics=save_metrics,
                    obs_threshold=obs_threshold,
                    start_z_counter=planes_processed,
                )

                if verbose:
                    jiter = tqdm.tqdm(
                        enumerate(inference_generator), total=len(z_list), leave=True
                    )
                else:
                    jiter = enumerate(inference_generator)

                if save_metrics:
                    roc = DistributedROC(
                        thresholds=thresholds, obs_threshold=obs_threshold
                    )

                t0 = time.time()
                for z_idx, results_dict in jiter:

                    pred_coors = results_dict["pred_output"]
                    # Save to text file
                    if "true_output" in results_dict:
                        true_coors = results_dict["true_output"]
                        if len(true_coors):
                            for (x, y, z, d) in true_coors:
                                with open(
                                    f"{prop_data_loc}/true_{this_worker}.txt", "a+"
                                ) as fid:
                                    fid.write(f"{h_idx} {x} {y} {z} {d}\n")
                    if len(pred_coors):
                        for (x, y, z, d) in pred_coors:
                            with open(
                                f"{prop_data_loc}/pred_{this_worker}.txt", "a+"
                            ) as fid:
                                fid.write(f"{h_idx} {x} {y} {z} {d}\n")

                    # Get stuff from the results dictionary
                    z_plane = int(results_dict["z"])

                    if save_arrays:
                        # Save the giant matrices as sparse arrays, as most elements are zero
                        if results_dict["pred_array"].sum() > 0:
                            if save_prob:
                                pred_prob = results_dict["pred_proba"]
                                pred_prob = np.where(
                                    pred_prob < 0.5, 0.0, 1000 * pred_prob
                                )
                                pred_prob = pred_prob.astype(int)
                                save_sparse_csr(
                                    f"{prop_data_loc}/prob_{h_idx}_{z_plane}.npz",
                                    scipy.sparse.csr_matrix(pred_prob),
                                )
                            save_sparse_csr(
                                f"{prop_data_loc}/pred_{h_idx}_{z_plane}.npz",
                                scipy.sparse.csr_matrix(
                                    results_dict["pred_array"].cpu().numpy()
                                ),
                            )

                        if "true_array" in results_dict:
                            if results_dict["true_array"].sum() > 0:
                                save_sparse_csr(
                                    f"{prop_data_loc}/true_{h_idx}_{z_plane}.npz",
                                    scipy.sparse.csr_matrix(
                                        results_dict["true_array"].cpu().numpy()
                                    ),
                                )

                    # Save the ROC results
                    if save_metrics:
                        if "roc" in results_dict:
                            this_roc = results_dict["roc"]
                            roc.merge(this_roc)
                            with open(
                                f"{roc_data_loc}/roc_{h_idx}_{z_plane}.pkl", "wb"
                            ) as fid:
                                joblib.dump(this_roc, fid)

                            # merge rocs that currently exist
                            rocs = sorted(
                                glob.glob(f"{roc_data_loc}/roc_{h_idx}_*.pkl"),
                                key=lambda x: int(x.strip(".pkl").split("_")[-1]),
                            )
                            for k, roc_fn in enumerate(rocs):
                                with open(roc_fn, "rb") as fid:
                                    if k == 0:
                                        roc = joblib.load(fid)
                                    else:
                                        roc.merge(joblib.load(fid))

                            with open(f"{roc_data_loc}/roc_{h_idx}.pkl", "wb") as fid:
                                joblib.dump(roc, fid)

                            roc_curve(
                                [roc],
                                [model_name],
                                ["orange"],
                                ["o"],
                                f"{image_data_loc}/roc_comparison_{h_idx}.png",
                            )
                            performance_diagram(
                                [roc],
                                [model_name],
                                ["orange"],
                                ["o"],
                                f"{image_data_loc}/performance_comparison_{h_idx}.png",
                            )

                    # Print some stuff
                    to_print = f"Worker {this_worker}: Holo: {h_idx} Plane: {z_idx + 1} / {len(z_list)} z: {(z_plane*1e-6):.8f}"
                    secs_per_holo = (time.time() - t0) / (z_idx + 1)
                    to_print += f" secs / plane: {secs_per_holo:.2f}"
                    if verbose:
                        jiter.set_description(to_print)
                        jiter.update()
                    logger.info(to_print)

                logger.info(
                    f"Worker {this_worker} finished hologram {h_idx} ({nc+1} / {len(h_range)}) in {time.time() - t0} s"
                )

    except:
        logger.warning(f"Worker {this_worker} failed: {traceback.format_exc()}")
        raise

    return True


if __name__ == "__main__":

    description = "Perform model inference on a list of hologram planes using N workers"
    description += (
        " where N = (number of nodes) * (number of GPUs per node) * (threads per GPU)"
    )

    parser = ArgumentParser(description=description)
    parser.add_argument(
        "-c",
        dest="model_config",
        type=str,
        default=False,
        help="Path to the model configuration (yml) containing your inputs.",
    )
    parser.add_argument(
        "-n",
        dest="n_nodes",
        type=int,
        default=-1,
        help="The number of GPU nodes (workers) to use to train model(s). Default is to use the values in the yml.",
    )
    parser.add_argument(
        "-nid",
        dest="node_id",
        type=int,
        default=0,
        help="The (int) ID of a node. If using 5 nodes, the IDs that can be passed are {0, ..., 4}. Default is 0.",
    )
    parser.add_argument(
        "-g",
        dest="gpus_per_node",
        type=int,
        default=-1,
        help="The number of GPUs per node available to use to train model(s). Default is to use the values in the yml.",
    )
    parser.add_argument(
        "-t",
        dest="threads_per_gpu",
        type=int,
        default=-1,
        help="The number of threads (models / GPU) to use to train model(s). Default is to use the values in the yml.",
    )
    parser.add_argument(
        "-l",
        dest="launch",
        type=int,
        default=0,
        help="Submit {n_nodes} workers to PBS.",
    )

    args_dict = vars(parser.parse_args())
    config_file = args_dict.pop("model_config")

    # Open config and set up processing details
    with open(config_file) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    n_nodes = int(args_dict.pop("n_nodes"))
    node_id = int(args_dict.pop("node_id"))
    n_gpus = int(args_dict.pop("gpus_per_node"))
    threads_per_gpu = int(args_dict.pop("threads_per_gpu"))
    launch = bool(int(args_dict.pop("launch")))

    n_nodes = conf["inference"]["n_nodes"] if n_nodes == -1 else n_nodes
    n_gpus = conf["inference"]["gpus_per_node"] if n_gpus == -1 else n_gpus
    threads_per_gpu = (
        conf["inference"]["threads_per_gpu"]
        if threads_per_gpu == -1
        else threads_per_gpu
    )
    workers = int(n_nodes * n_gpus * threads_per_gpu)
    n_bins = conf["data"]["n_bins"]

    # Perform a check if we are using > 1 node (here one has to pass the extra argument)
    if node_id >= n_nodes:
        raise OSError(
            f"The id of this worker ({node_id}) exceeded the number of nodes + 1 ({n_nodes + 1})."
        )

    # Set up directories to save results
    model_loc = conf["save_loc"]
    model_name = conf["model"]["name"]
    data_set = conf["inference"]["data_set"]["path"]
    data_set_name = conf["inference"]["data_set"]["name"]
    prop_data_loc = os.path.join(model_loc, data_set_name, "propagated")
    roc_data_loc = os.path.join(model_loc, data_set_name, "roc")

    save_dirs = [prop_data_loc]
    if conf["inference"]["save_metrics"]:
        save_dirs.append(roc_data_loc)
    for directory in save_dirs:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            
    # Launch PBS jobs
    if launch:
        logging.info(f"Launching {n_nodes} workers to PBS")
        launch_pbs_jobs(n_nodes, n_gpus, config_file, conf["save_loc"])
        sys.exit()

    # Configuration settings for which holograms to process
    h_conf = conf["inference"]["data_set"]["holograms"]
    if isinstance(h_conf, dict):
        h_min = conf["inference"]["data_set"]["holograms"]["min"]
        h_max = conf["inference"]["data_set"]["holograms"]["max"]
        h_range = range(h_min, h_max)
    elif isinstance(h_conf, list):
        h_range = h_conf
    elif isinstance(h_conf, int) or isinstance(h_conf, float):
        h_range = [h_conf]
    else:
        raise OSError(f"Unidentified h-range settings {h_conf}")

    ############################################################
    # Initialize logger to stream to stdout
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")

    # Stream output to stdout
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    # Save the log file
    logger_name = os.path.join(os.path.join(model_loc, f"{data_set_name}/log.txt"))
    fh = logging.FileHandler(logger_name, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    root.addHandler(fh)
    ############################################################

    host = (
        subprocess.Popen("hostname", stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        .communicate()[0]
        .decode("utf-8")
        .strip("\n")
    )

    # Print details to the logger
    logger.info(f"Using host {host} to perform inference")
    logger.info(f"Using a total of {workers} models (workers)")
    logger.info(f"... {n_nodes} nodes. The identity of the current node is {node_id}")
    logger.info(f"... {n_gpus} GPUs / node")
    logger.info(f"... {threads_per_gpu} models per GPU")
    logger.info(f"Using the data set {data_set_name} located at {data_set}")
    logger.info(f"Saving any result arrays to {prop_data_loc}")
    logger.info(f"Saving any ROC objects to {roc_data_loc}")
    logger.info(f"Using the following hologram hids: {str(h_range)}")
    logger.info(
        f"Performing model inference using {model_name} on {n_bins} reconstructed planes"
    )

    # Construct list of worker IDs and the GPU device to use
    total_workers = list(range(workers))
    list_of_devices = [
        x for gpu in range(n_gpus) for x in [f"cuda:{gpu}"] * threads_per_gpu
    ]
    available_workers = np.array_split(total_workers, int(n_nodes * n_gpus))[node_id]
    gpu_worker_list = list(zip(available_workers, list_of_devices))

    # Run models in parallel on the GPU(s)
    t0 = time.time()

    worker = partial(main, conf=conf)

    try:
        if threads_per_gpu > 1:
            processes = []
            for r in gpu_worker_list:
                p = mp.Process(target=worker, args=(r,))
                p.start()
                processes.append(p)

            while True:
                stop = 0
                for p in processes:
                    if p and p.is_alive():
                        continue
                    else:
                        stop += 1
                if stop == len(processes):
                    break
                else:
                    time.sleep(1)
        else:
            results = [worker(r) for r in gpu_worker_list]
        logger.info(f"Node {node_id} finished in {time.time()-t0} s")

    except KeyboardInterrupt:
        logger.warning("Interrupted")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
