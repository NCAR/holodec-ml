#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, minmax_scale
from datetime import datetime

sys.path.append('../../library/')

from data import load_scaled_datasets, load_raw_datasets, calc_z_bins


# In[2]:


path_data = "/glade/p/cisl/aiml/ai4ess_hackathon/holodec/"
num_particles = 3
output_cols = ["z", "hid"]
scaler_out = MinMaxScaler()
subset = False
num_z_bins = 5
mass = False


# In[ ]:


train_inputs,train_outputs,valid_inputs,valid_outputs = load_scaled_datasets(path_data,
                                     num_particles,
                                     output_cols,
                                     scaler_out,
                                     subset,
                                     num_z_bins,
                                     mass)


# In[4]:


print(train_inputs.shape)
print(train_outputs.shape)
print(valid_inputs.shape)
print(valid_outputs.shape)