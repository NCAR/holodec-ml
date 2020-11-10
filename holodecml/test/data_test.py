import xarray as xr
import numpy as np
import pandas as pd
import time
from holodecml.data import *

"""
Does our file path generator make correct file paths?
Does the data exist where the path says it exists?
Any NaN values / missing data?
Are we getting all the data we asked for?
If using pandas, check if dtype is consistent with what is saved in the df upon loading (important for a df that contains strings, floast, ints, … e.g. mixed!).
"""
#create testing dataframes and sets
out_data = {'z':[1,2,3,4,5,6,7,8,9,10],'hid':[1,2,3,4,5,6,7,8,9,10],'d':[1,2,3,4,5,6,7,8,9,10]}
outputsdf = pd.DataFrame(data=out_data)
train_data = {'z':[1,2,3,4,5,6,7,8,9,10],'hid':[1,2,3,4,5,6,7,8,9,10],'d':[1,2,3,4,5,6,7,8,9,10]}
traindf = pd.DataFrame(data=out_data)
inputs = np.array([1,2,3,4,5,6,7,8,9,10])
z_bin_test = [-99,110]

def test_dataset_name():
    with pytest.raises(ValueError):
        dataset_name(4,'train','nc')
    with pytest.raises(ValueError):
        dataset_name('multi','tested','nc')
    dn = dataset_name('multi','test','nc')
    assert(dn == 'synthetic_holograms_multi_test.nc')
    
def test_open_dataset():
    path_data = "/glade/p/cisl/aiml/ai4ess_hackathon/holodec/"
    ds = open_dataset(path_data, 'multi', 'test')
    ds2 = xr.open_dataset("/glade/p/cisl/aiml/ai4ess_hackathon/holodec/synthetic_holograms_multi_test.nc")
    assert(ds == ds2)
    
def test_load_raw_dataset():
    path_data = "/glade/p/cisl/aiml/ai4ess_hackathon/holodec/"
    output_cols = ["x", "y", "z", "d", "hid"]
    subset = False
    inputs, outputs = load_raw_datasets(path_data,'multi','test',output_cols,subset)
    #check for nan values
    assert(np.nan not in inputs)
    assert(np.nan not in outputs)
    #check for consistent datatype
    assert(inputs.type == 'numpy.ndarray')
    assert(isinstance(outputs, pd.DataFrame))
           
def test_scale_images():
    path_data = "/glade/p/cisl/aiml/ai4ess_hackathon/holodec/"
    output_cols = ["x", "y", "z", "d", "hid"]
    subset = False
    inputs, outputs = load_raw_datasets(path_data,'multi','test',output_cols,subset)
    scim, scaler_in = scale_images(inputs,None)
    #check to ensure that values are between 1 and 0
    assert((scim <= 1).all() and (scim >= 0).all())
    #test with different min and max scaler values
    scim2, scaler_in = scale_images(inputs,True)
    #so what should this return?
           
def test_calc_z_relative_mass():
    #test with no z_bins
    z_mass, z_bins = calc_z_relative_mass(outputsdf, num_z_bins=10, z_bins = None)
    assert(z_bins == z_bin_test)
    z_mass_test = [[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0]]
    assert(z_mass == z_mass_test)
    
def test_calc_z_dist():
    #check that the when everything is summed in a distribution that it adds to 1
    z_dist, z_bins = calc_z_dist(outputsdf,2,None)
    assert(z_dist.sum() == 1)
    assert(z_bins == z_bins_test)
    
    
    
def test_calc_z_bins():
    z_bins = calc_z_bins(outputsdf,traindf,2)
    z_bins_test = [1,10]
    assert(z_bins == z_bins_test)
    z_bins = calc_z_bins(outputsdf,traindf,10)
    z_bins_test = [1,2,3,4,5,6,7,8,9,10]
    assert(z_bins == z_bins_test)
    
    
    
def test_make_template():
    tester = make_template(outputsdf,10)
    assert(tester.shape() == outputsdf.shape())
    tester2 = make_template(outputsdf,1)
    assert(tester2.shape != outputsdf.shape())

def test_outputs_3d():
    tester = outputs_3d(outputsdf,10)
    assert(tester.shape() == (1,10))
    tester2 = outputs_3d(outputsdf,2)
    assert(tester.shape() == (1,2))
    
def test_load_scaled_datasets():   
    path_data = "/glade/p/cisl/aiml/ai4ess_hackathon/holodec/"
    ti,to,vi,vo = load_scaled_datasets(path_data,'multi',output_cols = ["x", "y", "z", "d", "hid"])
    assert((ti <= 1).all() and (ti >= 0).all())
    assert((vi <= 1).all() and (vi >= 0).all())
    assert(to.shape() == v0.shape())
    assert(ti.shape() == vi.shape())