import os
import argparse
import sherpa
import datetime
import time
from sherpa.schedulers import LocalScheduler
import sherpa.algorithms.bayesian_optimization as bayesian_optimization


def run_sherpa(FLAGS):
    """
    Run parallel Sherpa optimization over a set of discrete hp combinations.
    """
    
    # set up sherpa variables
    lrs = [1e-5, 1e-4, 1e-3, 3e-5, 3e-4, 3e-3]
    parameters = [sherpa.Choice('lr', lrs),
                  sherpa.Discrete('layers', [2,5]),
                  sherpa.Discrete('kernel_size', [3,6]),
                  sherpa.Discrete('pool_size', [1,5]),
                  sherpa.Choice('filter', [2, 4, 8, 16, 32]),
                  sherpa.Choice('dense', [16, 32, 64]),
                  sherpa.Discrete('dense_layers',[1, 5])]

    alg = bayesian_optimization.GPyOpt(max_concurrent=2,
                                       model_type='GP',
                                       acquisition_type='EI',
                                       max_num_trials=20)
    
    scheduler = LocalScheduler()
    
    rval = sherpa.optimize(parameters=parameters,
                       algorithm=alg,
                       dashboard_port=8585,
                       lower_is_better=False,
                       command='python train_conv2d_zdist_sherpa_parallel.py',
                       scheduler=scheduler,
                       verbose=0,
                       max_concurrent=2,
                       output_dir='./output_gpyopt_{}'.format(
                           time.strftime("%Y-%m-%d--%H-%M-%S")))

    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--concurrent', type=int)
    parser.add_argument('--port', help='Dashboard port', type=int, default=8585)
    FLAGS = parser.parse_args()
    t0 = time.time()
    run_sherpa(FLAGS)
    print("Time taken: ", time.time()-t0, "seconds")
    