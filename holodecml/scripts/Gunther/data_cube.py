import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

ds3 = xr.open_dataset('/glade/u/home/gwallach/synthetic_holograms_v02.nc')
x_values = []
y_values = []
hists3 = [0] * 10000


#increment by 3 to seperate images
for i in ds3['x'].values[::3]:
    x_values.append([i,i+1,i+2])
for i in ds3['y'].values[::3]:
    y_values.append([i,i+1,i+2])  

binsx = np.arange(-885,885,10)
binsy = np.arange(-590,590,10)
hists3[0] = plt.hist2d(x_values[0],y_values[0],bins = 100)[0]
for ii in range(1,10000):    
    #print(x_values[ii],y_values[ii])
    hist = plt.hist2d(x_values[ii],y_values[ii],bins = [binsx,binsy])[0]
    #if hist.all() == hists3[ii-1].all():
        #print("Repeat of previous")
        #break
    hists3[ii] = hist

print(len(hists3))
print(hists3[0])
repeats = []
for i in range(len(hists3)):
    if hists3[0].all() == hists3[i].all():
        repeats.append(1)
    else:
        repeats.append(0)
if 0 in repeats:
    print("Non-Repeat")
print("Length Repeats = ", len(repeats))
#print(repeats)