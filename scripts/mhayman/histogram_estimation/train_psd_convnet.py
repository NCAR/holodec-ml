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
            'num_epochs':50,    # the number of training epochs
            'conv_chan':[16,32],# list length defines number of operations
            'conv_size':[5,5],  # convolution kernel size
            'max_pool':[4,4],   # maxpool decimation
            'nn_size':[64,32,]  # excludes the output layer (set by the input data)
            'batch_size':64,    # training batch size
            'output_activation':'linear', # output activation function,
            'valid_fraction':0.1,   # fraction of data reserved for validation
            'test_fraction':0.3,    # fraction of data reserved for training
            'loss_function':'',     # loss function
            'h_chunk':256,      # xarray chunk size when loading
            }


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

save_file_path = paths['save_data']+save_file_base+'/'
ml.ensure_path(save_file_path)

print('Training UNET on')
print(settings['data_file'])
print('located in')
print(paths['load_data'])

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

with xr.open_dataset(paths['load_data']+settings['data_file'],chunks={'hologram_number':settings['h_chunk']}) as ds:
    print('Training dataset attributes')
    for att in ds.attrs:
        print('  '+att+': '+str(ds.attrs[att]))

    print('dataset dimensions')
    print(ds.dims)
    print(ds.sizes)

    # split data into training, validation and test data
    # ordered as (validation, test, train)
    test_index = np.int((settings['valid_fraction']+settings['test_fraction'])*ds.sizes['hologram_number'])  # number of training+validation points
    valid_index = np.int(settings['valid_fraction']*ds.sizes['hologram_number'])  # number of validation points
    
    print('test index: %d'%test_index)
    print('validation index: %d'%valid_index)
    print('hologram count: %d'%ds.sizes['hologram_number'])

    all_labels = ds[label_variable]
    train_labels = all_labels.isel(hologram_number=slice(test_index,None))
    test_labels = all_labels.isel(hologram_number=slice(valid_index,test_index))
    test_moments = ds['histogram_moments'].isel(hologram_number=slice(valid_index,test_index))
    # val_labels = all_labels.isel(hologram_number=slice(None,valid_index))

    # normalize based on training data
    
    if settings.get('scale_labels',True):
        output_scaler = ml.MinMaxScalerX(all_labels,dim=all_labels.dims[1:])
        scaled_all_labels = output_scaler.fit_transform(all_labels)
    else:
        scaled_all_labels = all_labels

    scaled_train_labels = scaled_all_labels.isel(hologram_number=slice(test_index,None))
    scaled_test_labels = scaled_all_labels.isel(hologram_number=slice(valid_index,test_index))
    scaled_val_labels = scaled_all_labels.isel(hologram_number=slice(None,valid_index))

    # setup the input to be used
    if len(ds[input_variable].dims) == 4:
        in_data = ds[input_variable].transpose('hologram_number','xsize','ysize','input_channels')
    elif len(ds[input_variable].dims) == 3:
        in_data = ds[input_variable].transpose('hologram_number','rsize','input_channels')
    
    train_input = in_data.isel(hologram_number=slice(test_index,None))
    input_scalar = ml.MinMaxScalerX(in_data,dim=in_data.dims[1:])
    scaled_in_data = input_scalar.fit_transform(in_data)

    scaled_train_input = scaled_in_data.isel(hologram_number=slice(test_index,None))
    scaled_test_input = scaled_in_data.isel(hologram_number=slice(valid_index,test_index))
    scaled_valid_input = scaled_in_data.isel(hologram_number=slice(None,valid_index))

    print('input data shape')
    print(in_data.dims)
    print(in_data.shape)

    print('input scaled data shape')
    print(scaled_in_data.dims)
    print(scaled_in_data.shape)

    print('output label shape')
    print(all_labels.dims)
    print(all_labels.shape)

    print('output scaled label shape')
    print(scaled_all_labels.dims)
    print(scaled_all_labels.shape)


    print('input training data shape')
    print(scaled_train_input.dims)
    print(scaled_train_input.shape)
    print('input training label shape')
    print(scaled_train_labels.dims)
    print(scaled_train_labels.shape)

    # build conv NN model
    mod = Sequential()
    mod.add(Input(shape=scaled_in_data.shape[1:]))

    # add convolutional layers
    for ai,n_filters in enumerate(settings['conv_chan']):
        mod.add(Conv2D(n_filters,
                (settings['conv_size'][ai],settings['conv_size'][ai]),
                padding="same",kernel_initializer = "he_normal"))
        mod.add(Activation("relu"))
        mod.add(MaxPool2D(pool_size=(settings['max_pool'][ai],settings['max_pool'][ai])))

    # flatten the convolution output for Dense Layers
    mod.add(Flatten())
    for ai,n_dense in enumerate(settings['nn_size']):
        mod.add(Dense(n_dense,activation='relu'))
    
    # add the output layer
    mod.add(Dense(np.prod(scaled_all_labels.shape[1:]),activation=settings['output_activation']))

    mod.compile(optimizer="adam", loss=loss_func, metrics=['acc'])
    mod.summary()

    # save a visualization
    plot_model(mod,show_shapes=True,to_file=save_file_path+save_file_base+"_diagram.png")

    history = mod.fit(scaled_train_input.values,
                  scaled_train_labels.values, 
                  batch_size=settings['batch_size'], epochs=settings['num_epochs'], verbose=1,
                  validation_data=(scaled_valid_input.values,scaled_val_labels.values))

    ### Save the Training History ###
    epochs = np.arange(len(history.history['loss']))+1
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(epochs,history.history['loss'],'bo-',alpha=0.5,label='Training')
    ax.plot(epochs,history.history['val_loss'],'rs-',alpha=0.5,label='Validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    if np.sum(np.array(history.history['loss'])<=0) == 0:
        ax.set_yscale('log')
    ax.grid(b=True)
    plt.legend()
    plt.savefig(save_file_path+save_file_base+"_LossHistory.png", dpi=200, bbox_inches="tight")

    fig, bx = plt.subplots(1, 1, figsize=(8, 4))
    bx.plot(epochs,history.history['acc'],'bo-',alpha=0.5,label='Training')
    bx.plot(epochs,history.history['val_acc'],'rs-',alpha=0.5,label='Validation')
    bx.set_xlabel('Epoch')
    bx.set_ylabel('Accuracy')
    bx.grid(b=True)
    plt.legend()
    plt.savefig(save_file_path+save_file_base+"_AccuracyHistory.png", dpi=200, bbox_inches="tight")

    ### Save the Model ### 
    model_name = save_file_base+".h5"
    save_model(mod, save_file_path+model_name, save_format="h5")
    print('saved model as')
    print(save_file_path+model_name)

    # Save the training history
    res_ds = xr.Dataset({
                        'epochs':epochs,
                        'Training_Loss':history.history['loss'],
                        'Validation_Loss':history.history['val_loss'],
                        'Training_Accuracy':history.history['acc'],
                        'Validation_Accuracy':history.history['val_acc'],
                        'test_index':test_index,
                        'valid_index':valid_index,
                        'input_variable':input_variable
                        })
    res_ds.attrs['batch_size'] = settings['batch_size']
    res_ds.attrs['training_data'] = settings['data_file']
    res_ds.attrs['training_path'] = paths['load_data']
    res_ds.attrs['output_path'] = paths['save_data']
    res_ds.attrs['model'] = model_name
    res_ds.to_netcdf(save_file_path+save_file_base+"_TrainingHistory.nc")


    # evaluate the test data
    print("Evaluating test data...")
    cnn_start = datetime.datetime.now()
    preds_out = mod.predict(scaled_test_input.values, batch_size=settings['batch_size'])
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
        ax.set_title('%d Moment'%m)
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
            plt.plot(scaled_test_input.isel(hologram_number=holo_num,input_channels=0).values)
            plt.savefig(save_file_path+save_file_base+f"_ExampleInput_ih{holo_num}.png", dpi=200, bbox_inches="tight")
        plt.close()


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