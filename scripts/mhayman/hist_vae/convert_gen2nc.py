"""
convert data generator with pickle combination
into a netcdf binary file
"""

import os,sys

import joblib
import numpy as np
import xarray as xr
import datetime


# dirP_str = os.path.join(os.environ['HOME'], 
#                     'PythonScripts', 
#                     'Optics',
#                     'holodec-ml',
#                     'library')
dirP_str = '../../../library'
if dirP_str not in sys.path:
    sys.path.append(dirP_str)

import ml_utils as ml


dataPath = '/glade/work/schreck/repos/holodec-ml/scripts/schreck/matt_data/'
savePath = '/glade/scratch/mhayman/holodec/holodec-ml-data/hist-from-vae/'

# dataFile = "training_1_25particles_gamma.pkl"
dataFile = "validation_1_25particles_gamma.pkl"

d_bin_wid = 10
d_max = 200
z_bin_wid = 5000
z_min = 14000
z_max = 160000

zlim = [z_min+10000,z_max-10000]

dbins = np.arange(0,d_max+d_bin_wid,d_bin_wid)
zbins = np.arange(z_min,z_max+z_bin_wid,z_bin_wid)

moments_arr = np.arange(4)

# data loader for reading pickle data
def read_data(filename, max_holograms = 1e10):
    with open(filename, "rb") as fid:
        loaded = 0 
        while True:
            try:
                yield joblib.load(fid)
                loaded += 1
            except:
                break
            # Option to load a subset of the data
            if loaded == max_holograms:
                break

d_bin_centers = dbins[1:] - np.diff(dbins)
z_bin_centers = zbins[1:] - np.diff(zbins)

# load the data and process it
training_data = read_data(dataPath+dataFile)

filename_lst = []
hid_lst = []
hologram_number_lst = []
raft_lst = []
mu_lst = []
logsig_lst = []
z_latent_lst = []

particle_id_lst = []
x_lst = []
y_lst = []
z_lst = []
d_lst = []

d_hist_lst = []
d_hist_lim_lst = []
z_hist_lst = []

moments_zlim_lst = []
moments_nolim_lst = []


for idx,one_hologram in enumerate(training_data):
    # filename: hologram source file
    # hid: hologram number in source file (base 1 indexing)
    # raft: radially averaged Fourier Transform
    # z: latent space values
    # mu: latent space mean values
    # logvar: latent space log variance (sigma =  logvar.mul(0.5).exp_())
    # y_out: dict of x,y,z,d for each particle
    
    # filename, hid, raft, z, mu, logvar, y_out
    filename, hid, raft, z_latent, mu, logvar, y_out = one_hologram
    
    ## Build histograms
   
    # locate only particles within z limits
    d_lim = y_out['d'][np.where((y_out['z']>zlim[0])&(y_out['z']<zlim[1]))]  

    d_hist = np.histogram(y_out['d'],bins=dbins)[0]
    z_hist = np.histogram(y_out['z'],bins=zbins)[0]

    # histogram of particles within z limits
    d_hist_lim = np.histogram(d_lim,bins=dbins)[0]  
    
    ## Calculate actual moments
    moments_zlim = ml.calculate_d_moments(y_out['x'],y_out['y'],y_out['z'],y_out['d'],moments_arr=moments_arr,zlim=zlim)
    
    moments_nolim = ml.calculate_d_moments(y_out['x'],y_out['y'],y_out['z'],y_out['d'],moments_arr=moments_arr,zlim=None)
    
    # append lists
    filename_lst+=[np.array([filename])]
    hid_lst+=[hid]
    hologram_number_lst+=[idx]
    raft_lst+=[raft.reshape(1,-1)]
    mu_lst+=[mu.reshape(1,-1)]
    logsig_lst+=[0.5*logvar.reshape(1,-1)]
    z_latent_lst += [z_latent.reshape(1,-1)]
    
    particle_id_lst += [idx]*y_out['x'].size
    x_lst+=[y_out['x']]
    y_lst+=[y_out['y']]
    z_lst+=[y_out['z']]
    d_lst+=[y_out['d']]
    
    d_hist_lst += [d_hist.reshape(1,-1)]
    d_hist_lim_lst += [d_hist_lim.reshape(1,-1)]
    z_hist_lst += [z_hist.reshape(1,-1)]
    
    moments_zlim_lst += [moments_zlim.reshape(1,-1)]
    moments_nolim_lst += [moments_nolim.reshape(1,-1)]


# Build the netcdf file structure
"""
    netcdf Dataset description
    
    dims: 
        hologram_number
        input_channels
        rsize
        latent_size
        particle_number
        d_bin_centers
        z_bin_centers
        moments
        d_bin_edges
        z_bin_edges
        
    data:
        source_filename (hologram_number)
        hid (hologram_number)
        raft (hologram_number, rsize)
        mu (hologram_number,latent_size)
        logsig (hologram_number,latent_size)
        z_latent (hologram_number,latent_size)
        
        particle_id (particle_number)
        x (particle_number)
        y (particle_number)
        z (particle_number)
        d (particle_number)
        
        d_hist (hologram_number,d_bin_centers)
        d_hist_lim (hologram_number,d_bin_centers)
        z_hist (hologram_number,z_bin_centers)
        
        moments_zlim (hologram_number,moments)
        moments_nolim (hologram_number,moments)
        
    
"""
ds_dct = {}
ds_dct['hologram_number'] = xr.DataArray(hologram_number_lst,dims=('hologram_number'),attrs={'description':'hologram ID number within this dataset'})
ds_dct['source_filename'] = xr.DataArray(np.concatenate(filename_lst),dims=('hologram_number'),
                                         coords={'hologram_number':ds_dct['hologram_number']},
                                         attrs={'description':'source file for hologram'})
ds_dct['hid'] = xr.DataArray(hid_lst,dims=('hologram_number'),
                             coords={'hologram_number':ds_dct['hologram_number']},
                             attrs={'description':'hologram id number'})
ds_dct['raft'] = xr.DataArray(np.concatenate(raft_lst,axis=0),dims=('hologram_number', 'rsize'),
                              coords={'hologram_number':ds_dct['hologram_number']},
                              attrs={'description':'radially averaged Fourier Transform'})
ds_dct['mu'] = xr.DataArray(np.concatenate(mu_lst,axis=0),dims=('hologram_number', 'latent_size'),
                            coords={'hologram_number':ds_dct['hologram_number']},
                            attrs={'description':'mean of latent value'})
ds_dct['logsig'] = xr.DataArray(np.concatenate(logsig_lst,axis=0),dims=('hologram_number', 'latent_size'),
                                coords={'hologram_number':ds_dct['hologram_number']},
                                attrs={'description':'log of standard deviation of latent value'})
ds_dct['z_latent'] = xr.DataArray(np.concatenate(z_latent_lst,axis=0),dims=('hologram_number', 'latent_size'),
                                  coords={'hologram_number':ds_dct['hologram_number']},
                                  attrs={'description':'decoded latent value, z = mu+exp(logsig)*randn()'})

ds_dct['particle_id'] = xr.DataArray(particle_id_lst,dims=('particle_number'),attrs={'description':'hologram index (hologram_number) in this dataset for x, y, z, d data'})
ds_dct['x'] = xr.DataArray(np.concatenate(x_lst),dims=('particle_number'),attrs={'description':'particle x position, for hologram index see particle_id','units':'micrometers'})
ds_dct['y'] = xr.DataArray(np.concatenate(y_lst),dims=('particle_number'),attrs={'description':'particle y position, for hologram index see particle_id','units':'micrometers'})
ds_dct['z'] = xr.DataArray(np.concatenate(z_lst),dims=('particle_number'),attrs={'description':'particle z position, for hologram index see particle_id','units':'micrometers'})
ds_dct['d'] = xr.DataArray(np.concatenate(d_lst),dims=('particle_number'),attrs={'description':'particle diameter, for hologram index see particle_id','units':'micrometers'})

ds_dct['d_bin_edges'] = xr.DataArray(dbins,dims=('d_bin_edges'),attrs={'description':'histogram bin edges for diameters',
                                                                                  'units':'micrometers'})
ds_dct['d_bin_centers'] = xr.DataArray(d_bin_centers,dims=('d_bin_centers'),attrs={'description':'histogram bin centers for diameters',
                                                                                  'units':'micrometers'})
ds_dct['z_bin_edges'] = xr.DataArray(zbins,dims=('z_bin_edges'),attrs={'description':'histogram bin edges for z position',
                                                                                  'units':'micrometers'})
ds_dct['z_bin_centers'] = xr.DataArray(z_bin_centers,dims=('z_bin_centers'),attrs={'description':'histogram bin centers for z position',
                                                                                  'units':'micrometers'})

ds_dct['d_hist'] = xr.DataArray(np.concatenate(d_hist_lst,axis=0),dims=('hologram_number','d_bin_centers'),
                                coords={'hologram_number':ds_dct['hologram_number'],'d_bin_centers':ds_dct['d_bin_centers']},
                                attrs={'description':'histogram of particle diameters'})
ds_dct['d_hist_lim'] = xr.DataArray(np.concatenate(d_hist_lim_lst,axis=0),dims=('hologram_number','d_bin_centers'),
                                coords={'hologram_number':ds_dct['hologram_number'],'d_bin_centers':ds_dct['d_bin_centers']},
                                attrs={'description':'histogram of particle diameters where particles outside of zlimits are omitted'})
ds_dct['z_hist'] = xr.DataArray(np.concatenate(z_hist_lst,axis=0),dims=('hologram_number','z_bin_centers'),
                                coords={'hologram_number':ds_dct['hologram_number'],'z_bin_centers':ds_dct['z_bin_centers']},
                                attrs={'description':'histogram of particle z position'})

ds_dct['moments'] = xr.DataArray(moments_arr,dims=('moments'),attrs={'description':'diameter moment order'})
ds_dct['moments_zlim'] = xr.DataArray(np.concatenate(moments_zlim_lst,axis=0),dims=('hologram_number','moments'),
                                   coords={'hologram_number':ds_dct['hologram_number'],'moments':ds_dct['moments']},
                                   attrs={'description':'true computed moments for the hologram where particles outside of zlimits are omitted'})
ds_dct['moments_nolim'] = xr.DataArray(np.concatenate(moments_nolim_lst,axis=0),dims=('hologram_number','moments'),
                                   coords={'hologram_number':ds_dct['hologram_number'],'moments':ds_dct['moments']},
                                   attrs={'description':'true computed moments for the hologram for all particles'})

vae_ds = xr.Dataset(ds_dct,
                    attrs={'generator':dataFile,
                           'created':datetime.datetime.now().strftime('%Y%m%dT%H%M%S'),
                          'zlim_lb':zlim[0],
                          'zlim_ub':zlim[1],
                          'd_bin_wid':d_bin_wid,
                          'd_max':d_max,
                          'z_bin_wid':z_bin_wid,
                          'z_max':z_max,
                          'z_min':z_min})

if 'validation' in dataFile:
    # split validation data file into test and validation
    vae_vld_ds = vae_ds.isel(hologram_number=slice(None,10000))
    vae_tst_ds = vae_ds.isel(hologram_number=slice(10000,None))
    
    vae_vld_ds.to_netcdf(savePath+dataFile.replace('pkl','nc'))
    vae_tst_ds.to_netcdf(savePath+dataFile.replace('validation','test').replace('pkl','nc'))

    vae_ds.close()
    vae_vld_ds.close()
    vae_tst_ds.close()
else:
    # save all training data in one file
    vae_ds.to_netcdf(savePath+dataFile.replace('pkl','nc'))
    vae_ds.close()