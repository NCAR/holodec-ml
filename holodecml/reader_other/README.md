##### The notebook example.ipynb demonstrates how the data reader works

##### In data.yml, the "path_data" field can be a list of data files where all holograms in each file will be used, or a dictionary where the filename is the key and the number of holograms to use from that file as the value (must be an integer).

##### MultiHologramDataset in reader.py is a keras.utils.Sequence object, that can be multiprocessed safely. 

##### The output of the MultiHologramDataset getitem thunder method will return a tuple (image, y_data), where image has shape (batch, color channel, W, H), y_data has shape (batch, 4, N), where the 4 in the second channel corresponds with a particle's attributes (x, y, z, d) in that order, and N is the number of particles in the hologram.

##### So that we can batch tuples, N is fixed in the configuration file (max_particles) and 0 is used as a padding value. The true number of particles can be calculated from  N_true = len(np.where(y[0, 3, :] > 0.0)).