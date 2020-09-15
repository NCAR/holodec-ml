import os
import yaml
import argparse
import sherpa
import datetime
import time
from sherpa.schedulers import LocalScheduler
import sherpa.algorithms.bayesian_optimization as bayesian_optimization


def run_sherpa(args):
    """
    Run parallel Sherpa optimization over a set of discrete hp combinations.
    """

    with open(args.config) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    
    path_save = config["path_save"]
    max_concurrent = int(config["max_concurrent"])
    max_num_trials = int(config["max_num_trials"])
    verbose = config["conv2d_network"]["verbose"]
    lrs = config["conv2d_network"]["lrs"]
    conv_layers = config["conv2d_network"]["conv_layers"]
    kernel_size = config["conv2d_network"]["kernel_size"]
    pool_size = config["conv2d_network"]["pool_size"]
    filter_size = config["conv2d_network"]["filter_size"]
    dense_layers = config["conv2d_network"]["dense_layers"]
    dense_size_0 = config["conv2d_network"]["dense_size_0"]
    
    # set up sherpa variables
    parameters = [sherpa.Choice('lr', lrs),
                  sherpa.Discrete('conv_layers', conv_layers),
                  sherpa.Discrete('kernel_size', kernel_size),
                  sherpa.Discrete('pool_size', pool_size),
                  sherpa.Choice('filter_size', filter_size),
                  sherpa.Discrete('dense_layers', dense_layers),
                  sherpa.Choice('dense_size_0', dense_size_0)]

    alg = bayesian_optimization.GPyOpt(max_concurrent=max_concurrent,
                                       model_type='GP',
                                       acquisition_type='EI',
                                       max_num_trials=max_num_trials)
    
    scheduler = LocalScheduler()
    
    command = 'python train_zdist_sherpa_parallel.py'
    path_save_dir = os.path.join(path_save,
                                 time.strftime("%Y-%m-%d--%H-%M-%S"))
    
    rval = sherpa.optimize(parameters=parameters,
                           scheduler=scheduler,
                           algorithm=alg,
                           dashboard_port=8585,
                           lower_is_better=False,
                           command=command,
                           verbose=verbose,
                           max_concurrent=max_concurrent,
                           output_dir=path_save_dir)

    
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Describe a Conv2D nn')
    parser.add_argument("config", help="Path to config file")
    args = parser.parse_args()    
    t0 = time.time()
    run_sherpa(args)
    print("Time taken: ", time.time()-t0, "seconds")
    