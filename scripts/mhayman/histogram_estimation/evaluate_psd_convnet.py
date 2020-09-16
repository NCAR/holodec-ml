"""
Code to execute training of a convolutional NN 
to predict particle size distribution.

Uses input data preprocessed by preprocess_hist_data.py

expected inputs:
paths = {   
            'load_data':'',     # location of the training data
            'save_data':''      # location to save the results
            }
settings = {
            'data_file':'',     # training data file
            'data_file_path':'',
            'model_file':'',
            'model_file_path':'',

            'batch_size':64,    # training batch size
            'output_activation':'linear', # output activation function,
            'valid_fraction':0.1,   # fraction of data reserved for validation
            'test_fraction':0.3,    # fraction of data reserved for training
            'loss_function':'',     # loss function
            'h_chunk':256,      # xarray chunk size when loading
            }

# initial test setup with bidisperse
/glade/scratch/mhayman/holodec/holodec-ml-data/histogram/histogram_training_data_5000count20200819T091551_HypOpt_NN_20200831T093951
histogram_training_data_5000count20200819T091551_HypOpt_NN_20200831T093951.h5

"""

# disable plotting in xwin
import matplotlib
matplotlib.use('Agg')


import sys
import numpy as np
import xarray as xr
import json

from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Activation, MaxPool2D, SeparableConv2D, UpSampling2D, concatenate, Conv2DTranspose
from tensorflow.keras.models import Model, save_model, load_model, Sequential
from tensorflow.keras.utils import plot_model
import tensorflow.keras.losses

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import datetime

# set path to local libraries
dirP_str = '../../../library'
if dirP_str not in sys.path:
    sys.path.append(dirP_str)

import ml_utils as ml
import ml_defs as mldef


start_time = datetime.datetime.now()

input_variable = 'input_image'  # data variable used as an input
label_variable = 'histogram'  # data variable used labels for training

save_file_base = settings['data_file'].replace('.nc','')+'_ConvNet_'+start_time.strftime('%Y%m%dT%H%M%S')
save_file_path = paths['model_data']+settings['test_file'].replace('.nc','/')

# save_file_path = paths['save_data']+save_file_base+'/'
ml.ensure_path(save_file_path)

# # read the json file with the original training information
# run_config_json = settings['model_file'].replace('.h5','')+"_run_settings.json"
# with open(settings['model_file_path']+'run_config.json') as f:
#     train_settings = json.load(f)

# settings['data_file'] = train_settings['data_file']




print('Trained on')
print(settings['data_file'])
print('located in')
print(paths['load_data'])

print('Testing with')
print(settings['test_file'])

with xr.open_dataset(paths['load_data']+settings['data_file'],chunks={'hologram_number':settings['h_chunk']}) as ds:
    print('Training dataset attributes')
    for att in ds.attrs:
        print('  '+att+': '+str(ds.attrs[att]))

    print('dataset dimensions')
    print(ds.dims)
    print(ds.sizes)

    if settings['loss_function'].lower() == 'kldivergence':
        loss_func = tensorflow.keras.losses.KLDivergence()
    elif settings['loss_function'].lower() == 'kstest':
        loss_func = mldef.ks_test
    elif settings['loss_function'].lower() == 'poisson':
        loss_func = mldef.poisson_nll
    elif settings['loss_function'].lower() == 'cum_poisson':
        loss_func = mldef.cum_poisson_nll
    else:
        loss_func = settings['loss_function']

    train_labels = ds[label_variable]
    train_moments = ds['histogram_moments']
    if len(ds[input_variable].dims) == 4:
        train_data = ds[input_variable].transpose('hologram_number','xsize','ysize','input_channels')
    elif len(ds[input_variable].dims) == 3:
        train_data = ds[input_variable].transpose('hologram_number','rsize','input_channels')
    if settings.get('log_input',False):
        train_data = xr.concat((train_data,np.log(train_data)),dim='input_channels')

    input_scaler = ml.MinMaxScalerX(train_data,dim=train_data.dims[:-1])
    scaled_train_input = input_scaler.fit_transform(train_data)

with xr.open_dataset(paths['load_data']+settings['test_file'],chunks={'hologram_number':settings['h_chunk']}) as ds_test:
    test_labels = ds_test[label_variable]
    test_moments = ds_test['histogram_moments']
    if len(ds_test[input_variable].dims) == 4:
        test_input = ds_test[input_variable].transpose('hologram_number','xsize','ysize','input_channels')
    elif len(ds_test[input_variable].dims) == 3:
        test_data = ds_test[input_variable].transpose('hologram_number','rsize','input_channels')
    if settings.get('log_input',False):
        test_data = xr.concat((test_data,np.log(test_data)),dim='input_channels')
    scaled_test_input = input_scaler.fit_transform(test_data)


    
        
if settings.get('scale_labels',True):
    # normalize based on training data
    output_scaler = ml.MinMaxScalerX(train_labels)
    scaled_train_labels = output_scaler.fit_transform(train_labels)
    scaled_test_labels = output_scaler.fit_transform(test_labels)

else:
    scaled_train_labels = train_labels
    scaled_test_labels = test_labels



print('input data shape')
print(train_data.dims)
print(train_data.shape)

print('input scaled data shape')
print(scaled_train_labels.dims)
print(scaled_train_labels.shape)

print('output label shape')
print(train_labels.dims)
print(train_labels.shape)

print('output scaled label shape')
print(scaled_train_labels.dims)
print(scaled_train_labels.shape)


print('input training data shape')
print(scaled_train_input.dims)
print(scaled_train_input.shape)
print('input training label shape')
print(scaled_train_labels.dims)
print(scaled_train_labels.shape)


# load the model
mod = load_model(paths['model_data']+settings['model_file'],compile=False)
mod.compile(optimizer="adam", loss=loss_func, metrics=['acc'])

# evaluate the test data
print("Evaluating test data...")
cnn_start = datetime.datetime.now()
preds_out = mod.predict(scaled_test_input.values, batch_size=settings.get('batch_size',64))
cnn_stop = datetime.datetime.now()
print(f"{scaled_test_input.sizes['hologram_number']} samples in {(cnn_stop-cnn_start).total_seconds()} seconds")
print(f"for {(cnn_stop-cnn_start).total_seconds()/scaled_test_input.sizes['hologram_number']} seconds per hologram")

if len(preds_out.shape)==2:
    preds_out = preds_out[...,np.newaxis]

preds_out_da = xr.DataArray(preds_out,dims=('hologram_number','histogram_bin_centers','output_channels'),
                        coords=scaled_test_labels.coords)

if settings.get('scale_labels',True):
    preds_original = output_scaler.inverse_transform(preds_out_da)
else:
    preds_original = preds_out_da

for m in settings.get('moments',[0,1,2,3]):
    m_pred = (preds_original*(0.5*preds_original.coords['histogram_bin_centers'])**m).sum(dim=('histogram_bin_centers','output_channels'))
    try:
        m_label = test_moments.sel(moments=m)
    except KeyError:
        print('No direct moment data')
        print('Approximating moments from histogram data')
        m_label = (test_labels*(0.5*test_labels.coords['histogram_bin_centers'])**m).sum(dim=('histogram_bin_centers','output_channels'))
    one_to_one = [m_label.values.min(),m_label.values.max()]
    fig, ax = plt.subplots() # figsize=(4,4)
    ax.scatter(m_pred,m_label,s=1,c='k')
    ax.plot(one_to_one,one_to_one,color='tab:red',linewidth=0.5)
    ax.minorticks_on()
    ax.grid(b=True)
    ax.grid(which='minor',linestyle=':')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('Moment %d'%m)
    plt.savefig(save_file_path+save_file_base+f"_{m}MomentScatter.png", dpi=200, bbox_inches="tight")
    plt.close('all')


for holo_num in settings['holo_examples']:
    plt.figure()
    plt.bar(ds['histogram_bin_edges'].values[:-1],preds_original.isel(hologram_number=holo_num,output_channels=0).values,
            np.diff(ds['histogram_bin_edges'].values),
            facecolor='blue',edgecolor='white',label='predicted',alpha=0.5)
    plt.bar(ds['histogram_bin_edges'].values[:-1],test_labels.isel(hologram_number=holo_num,output_channels=0).values,
            np.diff(ds['histogram_bin_edges'].values),
            facecolor='white',edgecolor='black',fill=False,label='true')
    # plt.plot(ds['histogram_bin_centers'].values,test_labels.isel(hologram_number=holo_num,output_channels=0).values,'.')
    # plt.plot(ds['histogram_bin_centers'].values,preds_original.isel(hologram_number=holo_num,output_channels=0).values,'.-')
    plt.xlabel('Particle Diameter [$\mu m$]')
    plt.ylabel('Count')
    if np.mean(np.diff(ds['histogram_bin_edges'].values)) != np.diff(ds['histogram_bin_edges'].values[0:2])[0]:
        plt.xscale('log')
    plt.savefig(save_file_path+save_file_base+f"_ExampleHist_ih{holo_num}.png", dpi=200, bbox_inches="tight")

    plt.figure()
    plt.bar(ds['histogram_bin_edges'].values[:-1],np.cumsum(preds_original.isel(hologram_number=holo_num,output_channels=0).values),
            np.diff(ds['histogram_bin_edges'].values),
            facecolor='blue',edgecolor='white',label='predicted',alpha=0.5)
    plt.bar(ds['histogram_bin_edges'].values[:-1],np.cumsum(test_labels.isel(hologram_number=holo_num,output_channels=0).values),
            np.diff(ds['histogram_bin_edges'].values),
            facecolor='white',edgecolor='black',fill=False,label='true')
    # plt.plot(ds['histogram_bin_centers'].values,test_labels.isel(hologram_number=holo_num,output_channels=0).values,'.')
    # plt.plot(ds['histogram_bin_centers'].values,preds_original.isel(hologram_number=holo_num,output_channels=0).values,'.-')
    plt.xlabel('Particle Diameter [$\mu m$]')
    plt.ylabel('Count')
    if np.mean(np.diff(ds['histogram_bin_edges'].values)) != np.diff(ds['histogram_bin_edges'].values[0:2])[0]:
        plt.xscale('log')
    plt.savefig(save_file_path+save_file_base+f"_ExampleCDF_ih{holo_num}.png", dpi=200, bbox_inches="tight")

    if scaled_test_input.isel(hologram_number=holo_num,input_channels=0).values.ndim == 2:
        plt.figure()
        plt.imshow(scaled_test_input.isel(hologram_number=holo_num,input_channels=0).values)
        plt.savefig(save_file_path+save_file_base+f"_ExampleInput_ih{holo_num}.png", dpi=200, bbox_inches="tight")     
    else:
        plt.figure()
        for i_chan in range(scaled_test_input.coords['input_channels'].size):
            plt.plot(scaled_test_input.isel(hologram_number=holo_num,input_channels=i_chan).values,label='channel %d'%i_chan)
        plt.legend()
        plt.savefig(save_file_path+save_file_base+f"_ExampleInput_ih{holo_num}.png", dpi=200, bbox_inches="tight")

    plt.close('all')


# # save the settings in human readable format
# # with a small file size
json_dct = {'settings':settings,'paths':paths}
# for k in json_dct['settings']:
#     if hasattr(json_dct['settings'][k], '__call__'):
#         json_dct['settings'][k] = json_dct['settings'][k].__name__
#     if hasattr(json_dct['settings'][k],'__iter__'):
#         for j in range(json_dct['settings'][k]):
#             if hasattr(json_dct['settings'][k][j], '__call__'):
#                 json_dct['settings'][k][j] = json_dct['settings'][k][j].__name__
    
with open(save_file_path+save_file_base+"_run_settings.json", 'w') as fp:
    json.dump(json_dct, fp, indent=4)