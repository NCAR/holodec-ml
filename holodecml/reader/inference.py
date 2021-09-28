import glob

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from datetime import datetime
from scipy.sparse import csr_matrix, save_npz
from scipy.ndimage import label, find_objects

def load_sparse_csr(filename):
    # here we need to add .npz extension manually
    loader = np.load(filename + '.npz')
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])

real = 'synthetic' #real

model_loc = f"/glade/work/schreck/repos/HOLO/clean/holodec-ml/results/minmax/"
model_save = "/glade/scratch/ggantos/holodec/models/minmax_new/"

for h_idx in range(10):
    
    for true in ['true', 'pred']:
        z_file_indices = sorted([int(x.replace(".npz", "").split("_")[-1]) for x in glob.glob(f"{model_loc}/{real}/propagated/{true}_{h_idx}_*")])
        print(f"z_file_indices range from {min(z_file_indices)} to {max(z_file_indices)}.")

        start_3d = datetime.now()
        array3d = []
        for z_file in z_file_indices:
            array2d = load_sparse_csr(f"{model_loc}/{real}/propagated/{true}_{h_idx}_{z_file}").toarray()
            array3d.append(array2d)
        array3d = np.stack(array3d)
        print(array3d.shape)
        print(f"Loading 3D {true} took {datetime.now() - start_3d} time")

        start_label = datetime.now()
        labeled_array, num_features = label(array3d, structure=None)
#         np.save(f"{model_save}inference/{real}/num_features_{true}_{h_idx}", num_features)
#         np.save(f"{model_save}inference/{real}/labeled_array_{true}_{h_idx}", labeled_array)
        print(f"Number of features found from {true} masks is {num_features}.")
        print(f"Shape of labeled_array_{true} {labeled_array.shape}.")
        print(f"Scipy label for {true}_3D took {datetime.now() - start_label} time")

        start_fo = datetime.now()
        objects = find_objects(labeled_array)
        np.save(f"{model_save}inference/{real}/objects_{true}_{h_idx}", objects)
        print(f"Scipy find_objects for {true}_3D took {datetime.now() - start_fo} time\n")
