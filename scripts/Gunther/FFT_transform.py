#this python script will run synthetic_holograms file, and output them into a new NetCDF file

import numpy as np
import xarray as xr
import pandas as pd
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Activation, MaxPool2D
from tensorflow.keras.models import Model, save_model
from scipy.fftpack import fft2, ifft2
from tensorflow.keras.optimizers import Adam
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
%matplotlib inline

#open dataset
ds = xr.open_dataset("ftp://ftp.ucar.edu/pub/mmm/bansemer/holodec/synthetic_holograms_v01.nc") #change path/put in url
#make data readable
in_data = ds["image"]
ds["image"].dims
scaled_in_data = in_data.astype(np.float16) / 255
scaled_in_data.dtype
out_data = []
for i in range(10000):
    out_data.append(np.abs(np.fft.fftshift(np.fft.fft2(scaled_in_data[i]))))
dnew = xr.DataArray(out_data,coords=ds.coords, dims=("hologram_number","xsize","ysize") , attrs=ds.attrs, name="synthetic_hologram_FFT")
#dnew.to_netcdf("sythetic_hologram_FFT-Transform_v0.nc",'w','NETCDF4')
dnew.to_netcdf("synthetic_hologram_FFT-Transform_v0.nc",'w','NETCDF4',encoding = {'xsize':{'dtype': 'float16','zlib': True},'ysize':{'dtype': 'float16','zlib': True}})
