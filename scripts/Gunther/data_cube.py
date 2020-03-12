import numpy as np
import xarray as xr
import pandas as pd

ds3 = xr.open_dataset('/glade/u/home/gwallach/synthetic_holograms_v02.nc')
x_values = []
y_values = []
hists3 = []


#increment by 3 to seperate images
for i in ds3['x'].values[::3]:
    x_values.append([i,i+1,i+2])
for i in ds3['y'].values[::3]:
    y_values.append([i,i+1,i+2])
    

for ii in range(len(x_values)):
    binsx = np.arange(0,1771,10)
    binsy = np.arange(0,1180,10)
    
    hist = np.histogram2d(x_values[ii],y_values[ii],bins = [binsx,binsy])[0]
    hists3.append(hist)

cube = xr.DataArray(hists3)
cube.to_netcdf('data_cube.nc')
    
    