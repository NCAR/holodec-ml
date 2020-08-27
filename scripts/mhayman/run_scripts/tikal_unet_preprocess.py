"""
Run Script for Creating a Preprocessed Training Dataset
On Casper
"""

import matplotlib
matplotlib.use('Agg')

import sys

# set path to Fourier Optics library
# dirP_str = dirP_str = '../../../../library'
dirP_str = '/h/eol/mhayman/PythonScripts/Python-Optics/Libraries'
if dirP_str not in sys.path:
    sys.path.append(dirP_str)

software_path = "/h/eol/mhayman/PythonScripts/Optics/holodec-ml/scripts/mhayman/"

ds_path = "/scr/sci/mhayman/holodec/holodec-ml-data/UNET/"
ds_file = "UNET_image_256x256_5000count_5particles_v02.nc"
save_path = None

rescale = 255

# specify number of layers to reconstruct
# params = {'zplanes':2,
#           'preprocess_type':'multi-plane reconstruction',
#           'raw_file':ds_file,
#           'complevel':9}

params = {'zplanes':0,
          'preprocess_type':'Fourier Transform',
          'raw_file':ds_file,
          'complevel':9}

exec_file = software_path+"PreProcess/PreProcess_FT_UNET.py"
exec(compile(open(exec_file,'rb').read(),exec_file,'exec'))