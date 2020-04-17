"""
Evaluate a UNET performance
On reconstructing holograms

"""

# disable plotting in xwin
import matplotlib
matplotlib.use('Agg')


import sys
import numpy as np
import xarray as xr

# from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Activation, MaxPool2D, SeparableConv2D, UpSampling2D, concatenate, Conv2DTranspose
from tensorflow.keras.models import Model, save_model, load_model
# from tensorflow.keras.utils import plot_model

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import datetime

# set path to local libraries
dirP_str = '../../../library'
if dirP_str not in sys.path:
    sys.path.append(dirP_str)

import ml_utils as ml
import ml_defs as mldef

# Model Training settings
h_chunk = 256 # size of dask array chunks along hologram_number dimension
split_fraction = 0.7  # fraction of points used for training/validation (not testing)
valid_fraction = 0.1  # fraction of points used for validation

input_variable = 'image_planes'

index_list = [235,332,841,1078,1398]  # example cases to run


# Training/Model data 
# ds_path='/scr/sci/mhayman/holodec/holodec-ml-data/UNET/UNET_image_256x256_5000count_5particles_v01/'   # linux share
# ds_file='UNET_image_256x256_5000count_5particles_10zplanes_v01.nc'
# model_path = '/scr/sci/mhayman/holodec/holodec-ml-data/UNET/models/UNET_Layers6_Conv5_Pool2_Filt32_mse_linear/UNET_image_256x256_5000count_5particles_10zplanes_v01/'
# model_file = 'UNET_Layers6_Conv5_Pool2_Filt32_mse_linear_epochs201_run1.h5'

# ds_path='/scr/sci/mhayman/holodec/holodec-ml-data/UNET/UNET_image_256x256_5000count_5particles_v02/'   # linux share
# ds_file='UNET_image_256x256_5000count_5particles_5zplanes_v02.nc'
# model_path = '/scr/sci/mhayman/holodec/holodec-ml-data/UNET/models/UNET_Layers6_Conv5_Pool2_Filt32_mse_linear/UNET_image_256x256_5000count_5particles_5zplanes_v02/'
# model_file = 'UNET_Layers6_Conv5_Pool2_Filt32_mse_linear_epochs201_run1.h5'

# ds_path='/scr/sci/mhayman/holodec/holodec-ml-data/UNET/UNET_image_256x256_5000count_5particles_v02/'   # linux share
# ds_file='UNET_image_256x256_5000count_5particles_9zplanes_v02.nc'
# model_path ='/scr/sci/mhayman/holodec/holodec-ml-data/UNET/models/UNET_Layers6_Conv5_Pool2_Filt32_mse_linear/UNET_image_256x256_5000count_5particles_9zplanes_v02/'
# model_file='UNET_Layers6_Conv5_Pool2_Filt32_mse_linear_epochs101_run1.h5'

# ds_path='/scr/sci/mhayman/holodec/holodec-ml-data/UNET/UNET_image_256x256_5000count_5particles_v02/'   # linux share
# ds_file='UNET_image_256x256_5000count_5particles_10zplanes_v02.nc'
# model_path='/scr/sci/mhayman/holodec/holodec-ml-data/UNET/models/UNET_Layers6_Conv5_Pool2_Filt32_mse_linear/UNET_image_256x256_5000count_5particles_10zplanes_v02/'
# model_file='UNET_Layers6_Conv5_Pool2_Filt32_mse_linear_epochs101_run1.h5'

ds_path='/scr/sci/mhayman/holodec/holodec-ml-data/UNET/UNET_image_256x256_5000count_5particles_v02/'   # linux share
ds_file='UNET_image_256x256_5000count_5particles_10zplanes_v02.nc'
model_path='/scr/sci/mhayman/holodec/holodec-ml-data/UNET/models/UNET_Layers6_Conv3_Pool2_Filt32_mse_linear/UNET_image_256x256_5000count_5particles_10zplanes_v02/'
model_file='UNET_Layers6_Conv3_Pool2_Filt32_mse_linear_epochs101_run1.h5'

nn_descript = model_file.split('_epochs')[0]
save_descript = model_file.replace('.h5','')

# save path for training results in the git repo
save_path = 'results/'+nn_descript+"/"+ds_file.replace(".nc","/test/")
ml.ensure_path(save_path)

# save path for training results in the git repo
save_example_path = 'results/'+nn_descript+"/"+ds_file.replace(".nc","/test/examples/")
ml.ensure_path(save_example_path)


# ds = xr.open_dataset(ds_path+ds_file,chunks={'hologram_number': h_chunk})
ds = xr.open_dataset(ds_path+ds_file)


split_index = np.int(split_fraction*ds.sizes['hologram_number'])  # number of training+validation points
valid_index = np.int(valid_fraction*ds.sizes['hologram_number'])  # number of validation points
# all_labels = ds['labels'].sel(type=['amplitude','z'])

# train_labels = all_labels.isel(hologram_number=slice(valid_index,split_index))

test_labels =  ds['labels'].sel(type=['amplitude','z'],hologram_number=slice(split_index,None))
# test_labels = all_labels.isel(hologram_number=slice(split_index,None))
    
# test_labels = all_labels.isel(hologram_number=index_list)


scaler = ml.MinMaxScalerX(test_labels,dim=('hologram_number','xsize','ysize'))
scaled_test_labels = scaler.fit_transform(test_labels)



in_data = ds[input_variable].isel(hologram_number=slice(split_index,None))


if not 'channel' in in_data.dims:
    in_data = in_data.expand_dims("channel", 3)

scaled_in_data = in_data

# load the CNN model
mod = load_model(model_path+model_file)

print("Evaluating test data...")
cnn_start = datetime.datetime.now()
preds_out = mod.predict(scaled_in_data.values, batch_size=64)
cnn_stop = datetime.datetime.now()
print(f"{scaled_in_data.sizes['hologram_number']} samples in {(cnn_stop-cnn_start).total_seconds()} seconds")
print(f"for {(cnn_stop-cnn_start).total_seconds()/scaled_in_data.sizes['hologram_number']} seconds per hologram")

preds_out_da = xr.DataArray(preds_out,dims=('hologram_number','xsize','ysize','type'),
                            coords=test_labels.coords)

preds_original = scaler.inverse_transform(preds_out_da)

# evaluate each predicted particle location by
# the prediction with the least error
# the prediction with the highest amplitude term
z_min = []
z_amp = []
z_act = []
a_min = []
a_amp = []
a_act = []
amp_act = []
amp_pred = []
amp_pred_b = []
amp_diff = np.array([])
amp_diff_b = np.array([])
for ih in test_labels.coords['hologram_number'].values:
    preds0 = preds_original.sel(hologram_number=ih)
    amp_act+=[np.sum(test_labels.sel(hologram_number=ih,type='amplitude').values)]
    amp_pred+=[np.sum(preds0.sel(type='amplitude').values)]
    amp_pred_b+=[np.sum((preds0.sel(type='amplitude').values>0.1))]
    amp_diff=np.concatenate((amp_diff,(test_labels.sel(hologram_number=ih,type='amplitude').values -
                preds0.sel(type='amplitude').values).flatten()))
    amp_diff_b=np.concatenate((amp_diff_b,
                (test_labels.sel(hologram_number=ih,type='amplitude').values -
                (preds0.sel(type='amplitude').values>0.1)).flatten()))
    ampset = np.nonzero(test_labels.sel(hologram_number=ih,type='amplitude').values > 0.1)
    zset = np.unique(test_labels.sel(hologram_number=ih,type='z').values[ampset])
    # print('hologram '+str(ih))
    # print('  particles ' +str(zset.size))
    for iz,z in enumerate(zset):
        # print('   particle number '+str(iz)+ ' at '+str(z))
        ipart = np.nonzero(test_labels.sel(hologram_number=ih,type='z').values==z)
        # print('      search size: '+str(test_labels.sel(hologram_number=ih,type='z').values.size))
        # print('      particle size: '+str(len(ipart[0])))
        zpred = preds0.sel(type='z').values[ipart]
        apred = preds0.sel(type='amplitude').values[ipart]
        iamp = np.argmax(apred)
        imin = np.argmin(np.abs(zpred-z))
        z_act += [z]
        z_amp += [zpred[iamp]]
        z_min += [zpred[imin]]
        a_act += [np.max(apred)]
        a_amp += [apred[iamp]]
        a_min += [apred[imin]]
    print(f"\r{ih+1} of {test_labels['hologram_number'].size} holograms completed",end='')
z_act = np.array(z_act)
z_amp = np.array(z_amp)
z_min = np.array(z_min)
a_act = np.array(a_act)
a_amp = np.array(a_amp)
a_min = np.array(a_min)
amp_act = np.array(amp_act)
amp_pred = np.array(amp_pred)
amp_pred_b = np.array(amp_pred_b)
# amp_diff = np.array(amp_diff)
# amp_diff_b = np.array(amp_diff_b)

# # save evaluation data as netcdf file
# eval_ds = xr.Dataset({
#                         'z_actual':z_act,
#                         'z_amplitude':z_amp,
#                         'z_MinError':z_min,
#                         'amplitude_pred':a_amp,
#                         'amplitude_actual':a_act,
#                         'amplitude_MinError':a_min,
#                         'hologram_amplitude_actual':amp_act,
#                         'hologram_amplitude_pred':amp_pred,
#                         'hologram_amplitude_pred_binary':amp_pred_b,
#                         'hologram_amplitude_error':amp_diff,
#                         'hologram_amplitude_error_binary':amp_diff_b
#                 })
# eval_ds.attrs['evaluation_dataset_path'] = ds_path
# eval_ds.attrs['evaluation_dataset'] = ds_file
# eval_ds.attrs['model_path'] = model_path
# eval_ds.attrs['model_file'] = model_file
# eval_ds.attrs['model'] = nn_descript
# eval_ds.attrs['this_model'] = save_descript
# eval_ds.to_netcdf(save_path+f"EvaluationData_"+save_descript+".nc")


# Scatter plot z position data
z_one_to_one = [z_act.min()*1e3,z_act.max()*1e3]
fig, ax = plt.subplots(1,1, figsize=(4, 4))
ax.plot(z_one_to_one,z_one_to_one,'k--')
ax.scatter(z_act*1e3,z_amp*1e3,s=5,alpha=0.5,label='max amplitude')
ax.scatter(z_act*1e3,z_min*1e3,s=5,alpha=0.5,label='min error')
ax.set_xlabel('z actual [mm]')
ax.set_ylabel('z predicted [mm]')
ax.grid(b=True)
ax.legend()
plt.savefig(save_path+f"Zscatter_"+save_descript+".png",dpi=300)

# Plot Max Amplitude Histogram
hbins = np.linspace(-10,10,100)
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].hist(z_amp*1e3-z_act*1e3,bins=hbins)
ax[0].set_xlabel('z error [mm]')
ax[0].set_title('Max Amplitude')
ax[0].minorticks_on()
ax[0].grid(b=True)
ax[0].text(0.02, 0.98, 'RMSE = %.02f mm'%np.sqrt(np.mean((z_act*1e3-z_amp*1e3)**2)), 
     horizontalalignment="left",verticalalignment="top", transform=ax[0].transAxes)

# Plot Min error in histogram
ax[1].hist(z_min*1e3-z_act*1e3,bins=hbins)
ax[1].set_xlabel('z error [mm]')
ax[1].set_title('Min Error')
ax[1].minorticks_on()
ax[1].grid(b=True)
ax[1].text(0.02, 0.98, 'RMSE = %.02f mm'%np.sqrt(np.mean((z_act*1e3-z_min*1e3)**2)), 
     horizontalalignment="left",verticalalignment="top", transform=ax[1].transAxes)
plt.savefig(save_path+f"Zhistogram_"+save_descript+".png",dpi=300)

# Scatter plot amplitude data
amp_one_to_one = [amp_act.min(),amp_act.max()]
fig, ax = plt.subplots(1,1, figsize=(4, 4))
ax.plot(amp_one_to_one,amp_one_to_one,'k--')
ax.scatter(amp_act,amp_pred,s=5,alpha=0.5,label='amplitude')
ax.scatter(amp_act,amp_pred_b,s=5,alpha=0.5,label='binary amplitude')
ax.set_xlabel('amplitude actual')
ax.set_ylabel('amplitude predicted')
ax.grid(b=True)
ax.legend()
plt.savefig(save_path+f"Ampscatter_"+save_descript+".png",dpi=300)

# Plot Amplitude
hbins = np.linspace(-256,256,200)
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].hist(amp_pred-amp_act,bins=hbins)
ax[0].set_xlabel('Amplitude error')
ax[0].set_title('Amplitude')
ax[0].minorticks_on()
ax[0].grid(b=True)
ax[0].text(0.02, 0.98, 'RMSE = %.02f'%np.sqrt(np.mean((amp_act-amp_pred)**2)), 
     horizontalalignment="left",verticalalignment="top", transform=ax[0].transAxes)

# Plot Min error in histogram
ax[1].hist(amp_pred_b-amp_act,bins=hbins)
ax[1].set_xlabel('Amplitude error')
ax[1].set_title('Binary Amplitude')
ax[1].minorticks_on()
ax[1].grid(b=True)
ax[1].text(0.02, 0.98, 'RMSE = %.02f'%np.sqrt(np.mean((amp_act-amp_pred_b)**2)), 
     horizontalalignment="left",verticalalignment="top", transform=ax[1].transAxes)
plt.savefig(save_path+f"Amphistogram_"+save_descript+".png",dpi=300)

### Evaluate Example Cases ### 

# compare/difference plots
diff_cmap = plt.get_cmap('seismic')
diff_cmap.set_bad(color='gray')

z_cmap = plt.get_cmap('viridis')
z_cmap.set_bad(color='gray')

diff_res = ds.attrs['zmax']*0.1

for ind in index_list:
    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    ax = ax.ravel()
    
    inan_mask = np.nonzero((preds_original.sel(type='amplitude',hologram_number=ind).values < 0.1)* \
        (test_labels.sel(type='amplitude',hologram_number=ind).values < 0.1))
    nan_mask = np.ones(preds_original.sel(type='amplitude',hologram_number=ind).values.shape)
    nan_mask[inan_mask] = np.nan
    
    ax[0].imshow(preds_original.sel(type='amplitude',hologram_number=ind).values,vmin=0,vmax=1)
    ax[1].imshow(test_labels.sel(type='amplitude',hologram_number=ind).values,vmin=0,vmax=1)
    ax[2].imshow((preds_original.sel(type='amplitude',hologram_number=ind).values-test_labels.sel(type='amplitude',hologram_number=ind).values)*nan_mask,vmin=-1,vmax=1,cmap=diff_cmap)
    ax[3].imshow(preds_original.sel(type='z',hologram_number=ind).values*nan_mask,vmin=0,vmax=ds.attrs['zmax'],cmap=z_cmap)
    ax[4].imshow(test_labels.sel(type='z',hologram_number=ind).values*nan_mask,vmin=0,vmax=ds.attrs['zmax'],cmap=z_cmap)
    ax[5].imshow((preds_original.sel(type='z',hologram_number=ind).values-test_labels.sel(type='z',hologram_number=ind).values)*nan_mask,vmin=-diff_res,vmax=diff_res,cmap=diff_cmap)
    plt.savefig(save_example_path+f"ExampleImage_{ind}_"+save_descript+".png",dpi=300)


# Input Example Plots
channel_number = in_data.sizes['channel']
for ind in index_list:
    fig, ax = plt.subplots(2, channel_number//2, figsize=(channel_number*3, 8))
    for ai in range(channel_number):
        axind = ai//2+np.mod(ai,2)*channel_number//2
        ax[np.mod(ai,2),ai//2].imshow(scaled_in_data.isel(channel=ai,hologram_number=ind),vmin=-0.25,vmax=0.25)
    plt.savefig(save_example_path+f"ExampleInput_{ind}_"+save_descript+".png",dpi=300)

# 3D Example Plots
xg,yg = np.meshgrid(preds_original['xsize'].values,preds_original['ysize'].values)
for ind,indact in enumerate(index_list):
    ipart = np.nonzero(preds_original.sel(type='amplitude',hologram_number=indact).values > 0.2)
    ipart_label = np.nonzero(test_labels.sel(type='amplitude',hologram_number=indact).values > 0.2)
    amp_p = preds_original.sel(type='amplitude',hologram_number=indact).values[ipart]
    z_p = preds_original.sel(type='z',hologram_number=indact).values[ipart]
    x_p = xg[ipart]
    y_p = yg[ipart]
    
    z_l = test_labels.sel(type='z',hologram_number=indact).values[ipart_label]
    x_l = xg[ipart_label]
    y_l = yg[ipart_label]

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(z_p,x_p,y_p,c=amp_p,vmin=0,vmax=1,s=1)
    ax.scatter(z_l,x_l,y_l,c='k',s=1)
    ax.set_xlim([ds.attrs['zmin'],ds.attrs['zmax']])
    ax.set_ylim([preds_original['xsize'].values[0],preds_original['xsize'].values[-1]])
    ax.set_zlim([preds_original['ysize'].values[0],preds_original['ysize'].values[-1]])
    ax.set_xlabel('z')
    ax.set_ylabel('x')
    ax.set_zlabel('y')
    plt.savefig(save_example_path+f"Scatter3D_{ind}_"+save_descript+".png",dpi=300)