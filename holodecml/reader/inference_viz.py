import gc
import glob
import psutil
import pickle

import numpy as np
import pandas as pd
import xarray as xr
import multiprocessing as mp
from functools import partial
from munkres import Munkres
from datetime import datetime

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from scipy.sparse import csr_matrix, save_npz
from scipy.ndimage import label, find_objects

#correct
available_ncpus = len(psutil.Process().cpu_affinity())

# incorrect
mp.cpu_count()

def load_sparse_csr(filename):
    # here we need to add .npz extension manually
    loader = np.load(filename + '.npz')
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])

def worker_objects(h_idx, z_file_indices=None, model_loc=None, real=None):
    for true in ['true', 'pred']:
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
        del array2d, array3d, labeled_array, num_features, objects
        gc.collect()
        return h_idx
    
def worker_3d_plot(h_idx, particles=None, model_save=None, real=None):
    for true in ['pred', 'true', 'orig']:
        data = [go.Scatter3d(x=particles[h_idx][true]['x'],
                     y=particles[h_idx][true]['y'],
                     z=particles[h_idx][true]['z'],
                     mode='markers',
                     marker=dict(size=particles[h_idx][true]['d']/2,
                                 color=particles[h_idx][true]['d'],
                                 colorscale='Rainbow',
                                 opacity=0.8),
                     text = [f"diameter: {d_i}" for d_i in particles[h_idx][true]['d']])]

        layout = go.Layout(title=f'{true} coordinates from scipy',
                           autosize=True,
                           width=700,
                           height=700,
                           margin=go.layout.Margin(
                                l=0,
                                r=0,
                                b=0,
                                t=40
                           )
                           )
        fig = go.Figure(data=data, layout=layout)
        fig.update_layout(hovermode="x",
                          scene_aspectmode='manual',
                          scene_aspectratio=dict(x=1, y=1, z=1))
        fig.write_image(f"{model_save}inference/{real}/3D_{h_idx}_{true}.png")
        return h_idx

def worker_3d_plot_paired(h_idx, pairs=None, particles=None, model_save=None, real=None):
    
    for diam in ['with_diam', 'no_diam']:
        for m in ['min', 'max']:
            if m == 'min':
                pred_idx = int(np.where(pairs["true"][h_idx][diam]['distances'] == min(pairs["true"][h_idx][diam]['distances']))[0])
            if m == 'max':
                pred_idx = int(np.where(pairs["true"][h_idx][diam]['distances'] == max(pairs["true"][h_idx][diam]['distances']))[0])
            data = [go.Scatter3d(x=[particles[h_idx]['pred']['x'][pred_idx]],
                     y=[particles[h_idx]['pred']['y'][pred_idx]],
                     z=[particles[h_idx]['pred']['z'][pred_idx]],
                     mode='markers',
                     marker=dict(size=[particles[h_idx]['pred']['d'][pred_idx]],
                                 color='rgb(150,0,90)',
                                 opacity=0.8),
                     text = f"diameter: {particles[h_idx]['pred']['d'][pred_idx]:.2f}",
                     name="Predicted")]

            layout = go.Layout(title=f'''Comparison of predicted particle for {diam} hologram {h_idx} and {m} distance<br>
            Pred: {particles[h_idx]['pred']['x'][pred_idx]:.1f},
             {particles[h_idx]['pred']['y'][pred_idx]:.1f},
             {particles[h_idx]['pred']['z'][pred_idx]:.0f},
             {particles[h_idx]['pred']['d'][pred_idx]:.2f}<br>
            True: {particles[h_idx]['true']['x'][pairs['true'][h_idx][diam]['indexes'][pred_idx][1]]:.1f},
             {particles[h_idx]['true']['y'][pairs['true'][h_idx][diam]['indexes'][pred_idx][1]]:.1f},
             {particles[h_idx]['true']['z'][pairs['true'][h_idx][diam]['indexes'][pred_idx][1]]:.0f},
             {particles[h_idx]['true']['d'][pairs['true'][h_idx][diam]['indexes'][pred_idx][1]]:.2f}<br>
            Orig: {particles[h_idx]['orig']['x'][pairs['orig'][h_idx][diam]['indexes'][pred_idx][1]]:.1f},
             {particles[h_idx]['orig']['y'][pairs['orig'][h_idx][diam]['indexes'][pred_idx][1]]:.1f},
             {particles[h_idx]['orig']['z'][pairs['orig'][h_idx][diam]['indexes'][pred_idx][1]]:.0f},
             {particles[h_idx]['orig']['d'][pairs['orig'][h_idx][diam]['indexes'][pred_idx][1]]:.2f}<br>
            MAE true: {maes['true'][h_idx][diam][pred_idx][pairs['true'][h_idx][diam]['indexes'][pred_idx][1]]:.4f}<br>
            MAE orig: {maes['orig'][h_idx][diam][pred_idx][pairs['orig'][h_idx][diam]['indexes'][pred_idx][1]]:.4f}''',
                               width=700,
                               height=700,
                               margin=go.layout.Margin(
                                    l=0,
                                    r=0,
                                    b=0,
                                    t=40))

            fig = go.Figure(data=data, layout=layout)
            fig.add_trace(go.Scatter3d(x=[particles[h_idx]['true']['x'][pairs['true'][h_idx][diam]['indexes'][pred_idx][1]]],
                                 y=[particles[h_idx]['true']['y'][pairs['true'][h_idx][diam]['indexes'][pred_idx][1]]],
                                 z=[particles[h_idx]['true']['z'][pairs['true'][h_idx][diam]['indexes'][pred_idx][1]]],
                                 mode='markers',
                                 marker=dict(size=[particles[h_idx]['true']['d'][pairs['true'][h_idx][diam]['indexes'][pred_idx][1]]],
                                             color='rgb(0,0,200)',
                                             opacity=0.8),
                                 text = f"diameter: {particles[h_idx]['true']['d'][pairs['true'][h_idx][diam]['indexes'][pred_idx][1]]:.2f}",
                                 name = "True"))
            fig.add_trace(go.Scatter3d(x=[particles[h_idx]['orig']['x'][pairs['orig'][h_idx][diam]['indexes'][pred_idx][1]]],
                                 y=[particles[h_idx]['orig']['y'][pairs['orig'][h_idx][diam]['indexes'][pred_idx][1]]],
                                 z=[particles[h_idx]['orig']['z'][pairs['orig'][h_idx][diam]['indexes'][pred_idx][1]]],
                                 mode='markers',
                                 marker=dict(size=[particles[h_idx]['orig']['d'][pairs['orig'][h_idx][diam]['indexes'][pred_idx][1]]],
                                             color='rgb(151,255,0)',
                                             opacity=0.8),
                                 text = f"diameter: {particles[h_idx]['orig']['d'][pairs['orig'][h_idx][diam]['indexes'][pred_idx][1]]:.2f}",
                                 name = "Original"))

            fig.update_layout(hovermode="x",
                              scene_aspectmode='manual',
                              scene_aspectratio=dict(x=1, y=1, z=1),
                              scene = dict(xaxis = dict(nticks=6, range=[-7300, 7300]),
                                           yaxis = dict(nticks=6, range=[-4800, 4800]),
                                           zaxis = dict(nticks=6, range=[14000, 158000])))

            fig.write_image(f"{model_save}inference/{real}/3D_{h_idx}_{diam}_{m}.png")
            return h_idx
    
    
if __name__ == '__main__':

    real = 'synthetic' #real

    model_loc = f"/glade/work/schreck/repos/HOLO/clean/holodec-ml/results/standard/"
    model_save = "/glade/scratch/ggantos/holodec/models/standard_parallel/"

    try:
        h_idx_indices = list(set(sorted([int(x.split("_")[1]) for x in glob.glob(f"{model_loc}/{real}/propagated/true*")])))
    except:
        h_idx_indices = list(set(sorted([int(x.split("_")[2]) for x in glob.glob(f"{model_loc}/{real}/propagated/true*")])))
    print(f"h_idx_indices: {h_idx_indices}")

    z_file_indices = sorted([int(x.replace(".npz", "").split("_")[-1]) for x in glob.glob(f"{model_loc}/{real}/propagated/true_{h_idx_indices[0]}_*")])
    print(f"z_file_indices range from {min(z_file_indices)} to {max(z_file_indices)}.")
    
    # Create scipy objects
    work = partial(worker_objects, z_file_indices=z_file_indices, model_loc=model_loc, real=real)
    with mp.Pool(processes=4) as p:
        for result in p.imap(work, h_idx_indices):
            print(result)
    raise
    
    # Load original data
    if real == 'real':
        fn_orig = "/glade/p/cisl/aiml/ai4ess_hackathon/holodec/real_holograms_CSET_RF07_20150719_200000-210000.nc"
    else:
        fn_orig = "/glade/p/cisl/aiml/ai4ess_hackathon/holodec/synthetic_holograms_500particle_gamma_4872x3248_test.nc"
    ds = xr.open_dataset(fn_orig)
    dx = ds.attrs['dx']      # horizontal resolution
    dy = ds.attrs['dy']      # vertical resolution

    # Load all particles
    particles= {}
    for h_idx in h_idx_indices:

        particles[h_idx] = {}
        particles[h_idx]['orig'] = {}

        particle_idx = np.where(ds['hid'].values==h_idx+1)
        particles[h_idx]['orig']['x'] = ds['x'].values[particle_idx]
        particles[h_idx]['orig']['y'] = ds['y'].values[particle_idx]
        particles[h_idx]['orig']['z'] = ds['z'].values[particle_idx]
        particles[h_idx]['orig']['d'] = ds['d'].values[particle_idx]

        for true in ('true', 'pred'):

            particles[h_idx][true] = {}

            objects = np.load(f"{model_save}inference/{real}/objects_{true}_{h_idx}.npy", allow_pickle=True)
            if objects.shape[0] == 0:
                print(f"{model_save}inference/{real}/objects_{true}_{h_idx}.npy")
                print(f"Removing h_idx {h_idx} from h_idx_indices")
                h_idx_indices = list(set(h_idx_indices)^set([h_idx]))
                particles.pop(h_idx, None)
            else:
                x = []
                y = []
                z = []
                d = []
                for obj in objects:
                    x.append(int(np.mean(obj[1].indices(10000)[:2])))
                    y.append(int(np.mean(obj[2].indices(10000)[:2])))
                    z.append(z_file_indices[int(np.mean(obj[0].indices(10000)[:2]))])

                    d_x = (obj[1].indices(10000)[1] - obj[1].indices(10000)[0]) * dx
                    d_y = (obj[2].indices(10000)[1] - obj[2].indices(10000)[0]) * dy
                    d.append(max(d_x, d_y) / 1e-6)

                particles[h_idx][true]['z'] = np.array(z)
                x = np.array(x) * dx * 1e6
                particles[h_idx][true]['x'] = x - (max(x) - min(x)) / 2
                y = np.array(y) * dy * 1e6
                particles[h_idx][true]['y'] = y - (max(y) - min(y)) / 2
                particles[h_idx][true]['d'] = np.array(d)

    # Histograms Depicting Average of 10 particles
    bins = {'z': np.arange(14000,158000,2000),
            'x': np.arange(-7300,7300,200),
            'y': np.arange(-4800,4800,200),
            'd': np.arange(0,139,3)}

    widths = {'x': 130,
              'y': 130,
              'z': 1300,
              'd': 2}

    hist_avg = {}
    for true in ('true', 'pred'):
        hist_avg[true] = {}
        for coord in ['x', 'y', 'z', 'd']:
            hist = []
            for h_idx in h_idx_indices:
                h, binEdges = np.histogram(particles[h_idx][true][coord], bins=bins[coord])
                hist.append(h)
            hist_avg[true][coord] = np.stack(hist)
            hist_avg[true][coord+'_mean'] = np.stack(hist).mean(axis=0)
            hist_avg[true][coord+'_std'] = np.stack(hist).std(axis=0) / ((len(h_idx_indices) - 1) ** 0.5)

    for coord in ['x', 'y', 'z', 'd']:
        plt.figure(figsize=(20,6))
        _, binEdges = np.histogram(particles[h_idx]['pred'][coord], bins=bins[coord])
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        plt.title(f"{coord} Coordinate Average Histogram", fontsize=30)
        plt.bar(bincenters, hist_avg['true'][f'{coord}_mean'], alpha=0.6, width=widths[coord], yerr=hist_avg['true'][f'{coord}_std'], label='True',
                color='#ff7f0e', ecolor='#ff7f0e')
        plt.bar(bincenters, hist_avg['pred'][f'{coord}_mean'], alpha=0.8, width=widths[coord], yerr=hist_avg['pred'][f'{coord}_std'], label='Predicted',
                fill=False, linewidth=3, edgecolor='#1f77b4', ecolor='#1f77b4')
        plt.legend()
        plt.savefig(f"{model_save}inference/{real}/hist_{coord}_avg.png")

    # Plot histograms for individual holograms
    for h_idx in h_idx_indices:
        fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(24,3))
        fig.suptitle(f"Hologram {h_idx}", fontsize=16)
        for i, coord in enumerate(['x', 'y', 'z', 'd']):
            ax[i].hist(particles[h_idx]['pred'][coord], bins=bins[coord], alpha=0.8, label='Predicted', histtype='step', linewidth=3)
            ax[i].hist(particles[h_idx]['true'][coord], bins=bins[coord], alpha=0.6, label='True')
    #         ax[i].hist(particles[h_idx]['orig'][coord], bins=bins[coord], alpha=0.6, label='Original')
            ax[i].set_xlabel(f"{coord} coordinate (µm)")
        plt.subplots_adjust(wspace=0.1)
        plt.legend()
        fig.savefig(f"{model_save}inference/{real}/hist_{h_idx}.png", bbox_inches = "tight")

    # Plot z-relative mass
    z_bins = np.linspace(min(z_file_indices) - 100,
                         max(z_file_indices) + 100,
                         20)

    z_mass_dict = {}
    for true in ['true', 'pred']:
        z_mass = np.zeros((len(h_idx_indices), 20), dtype=np.float32)
        for i,h_idx in enumerate(h_idx_indices):
            for p in range(particles[h_idx][true]['z'].shape[0]):
                z_pos = np.searchsorted(z_bins, particles[h_idx][true]['z'][p], side="right") - 1
                mass = 4 / 3 * np.pi * (particles[h_idx][true]['d'][p]/2)**3
                z_mass[i, z_pos] += mass
        z_mass /= np.expand_dims(z_mass.sum(axis=1), -1)
        z_mass_dict[true] = z_mass

    plt.figure(figsize=(12,6))
    plt.bar(z_bins / 1000, z_mass_dict['true'].mean(axis=0),
            (z_bins[1]-z_bins[0]) / 1000, color='#ff7f0e', alpha=0.6,
            label="True")
    plt.bar(z_bins / 1000, z_mass_dict['pred'].mean(axis=0),
            (z_bins[1] - z_bins[0]) / 1000, edgecolor='#1f77b4', linewidth=3, alpha=0.8, facecolor="none", label='Predicted')
    plt.xlabel("z location (mm)", fontsize=16)
    plt.ylabel("Mean Mass Distribution", fontsize=16)
    plt.legend(loc="best", fontsize=12)
    plt.savefig(f"{model_save}inference/{real}/zmass_avg.png")

    # Save 3D visualizations
    work = partial(worker_3d_plot, particles=particles, model_save=model_save, real=real)
    with mp.Pool(processes=4) as p:
        for result in p.imap(work, h_idx_indices):
            print(result)

    # Calculating Particle Distance and MAE
    def normalize(x):
        return (x - x.min()) / (x.max() - x.min())

    maes = {}
    for true in ['true', 'orig']:
        maes[true] = {}
        for h_idx in h_idx_indices:
            maes[true][h_idx] = {}
            mae = np.zeros((particles[h_idx]['pred']['x'].shape[0], particles[h_idx][true]['x'].shape[0]))
            for coord in ['x', 'y', 'z']:
                mae += np.abs(normalize(particles[h_idx]['pred'][coord]).reshape(-1,1) - normalize(particles[h_idx][true][coord]).reshape(1,-1))
            maes[true][h_idx]['no_diam'] = np.copy(mae)
            mae += np.abs(normalize(particles[h_idx]['pred']['d']).reshape(-1,1) - normalize(particles[h_idx][true]['d']).reshape(1,-1))
            maes[true][h_idx]['with_diam'] = mae
    with open(f"{model_save}inference/{real}/maes.pkl", 'wb') as handle:
        pickle.dump(maes, handle, protocol=pickle.HIGHEST_PROTOCOL)

    distances = {}
    for true in ['true', 'orig']:
        distances[true] = {}
        for h_idx in h_idx_indices:
            distances[true][h_idx] = {}
            dist = np.zeros((particles[h_idx]['pred']['x'].shape[0], particles[h_idx][true]['x'].shape[0]))
            for coord in ['x', 'y', 'z']:
                dist += (normalize(particles[h_idx]['pred'][coord]).reshape(-1,1) - normalize(particles[h_idx][true][coord]).reshape(1,-1)) ** 2
            distances[true][h_idx]['no_diam'] = np.copy(dist)
            dist += (normalize(particles[h_idx]['pred']['d']).reshape(-1,1) - normalize(particles[h_idx][true]['d']).reshape(1,-1)) ** 2
            distances[true][h_idx]['with_diam'] = dist
    with open(f"{model_save}inference/{real}/distances.pkl", 'wb') as handle:
        pickle.dump(distances, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Munkres algorithm
    pairs = {}
    for true in ['true', 'orig']:
        pairs[true] = {}
        for h_idx in h_idx_indices:
            pairs[true][h_idx] = {}
            for diam in ['with_diam', 'no_diam']:
                pairs[true][h_idx][diam] = {}
                distance = np.copy(distances[true][h_idx][diam])
                start_munkres = datetime.now()
                m = Munkres()
                if distance.shape[0] > distance.shape[1]:
                    distance = np.swapaxes(distance, 0, 1)
                    indexes = m.compute(distance)
                    indexes = [tuple((i[1], i[0])) for i in indexes]
                    pairs[true][h_idx][diam]['indexes'] = indexes
                else:
                    pairs[true][h_idx][diam]['indexes'] = m.compute(distance)
                print('___________________________________________________________________________________________')
                print(f"Munkres for {true} {h_idx} took {datetime.now() - start_munkres} time")
                print(f"{distance.shape[0]} predicted particles paired with {len(list(set([i[1] for i in pairs[true][h_idx][diam]['indexes']])))} unique true particles.")

                ds = []
                ms = []
                for row, column in pairs[true][h_idx][diam]['indexes']:
                    ds.append(distances[true][h_idx][diam][row][column])
                    ms.append(maes[true][h_idx][diam][row][column])
                pairs[true][h_idx][diam]['distances'] = np.array(ds)
                pairs[true][h_idx][diam]['maes'] = np.array(ms)
                print(f"Total Distance for Hologram {h_idx} predicted vs {true} and {diam} = {np.sum(pairs[true][h_idx][diam]['distances'])}")
                print(f"Total MAE for Hologram {h_idx} predicted vs {true} and {diam} = {np.sum(pairs[true][h_idx][diam]['maes'])}\n")
    with open(f"{model_save}inference/{real}/pairs.pkl", 'wb') as handle:
        pickle.dump(pairs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # plot 3D distances between pairs particles
    for h_idx in h_idx_indices:
    work = partial(worker_3d_plot_paired, pairs=pairs, particles=particles, model_save=model_save, real=real)
    with mp.Pool(processes=4) as p:
        for result in p.imap(work, h_idx_indices):
            print(result)
