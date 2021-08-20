import h5py
import numpy as np
import _init_path
from ComplexNets import *
from matplotlib import pyplot as plt
from keras.models import load_model
from model.ANN import getANNmodel
from cfgs.config import cfg
from train import load_dataset


# model = getANNmodel()
# model.summary()
# import numpy as np


spc_image = load_dataset(cfg.datafile_location, cfg.TRAIN.x_toload)


# modelpath = "../result_dir/ori_modelsaved/orimodel.h5"
# model = load_model(modelpath,custom_objects={'ComplexDense': ComplexDense,"Amplitude":Amplitude})
# weights = model.layers[1].get_weights()
# print(weights[0].shape)

# plt.rcParams['image.cmap'] = 'gray'
# plt.rcParams['axes.linewidth'] = 0

# image_dim = 120
# orig_dim = 92
# length_image = 50000

# def load_dataset(file_location,what_to_load,spc = False):
#     hf = h5py.File( file_location , 'r')
#     fl = hf[what_to_load]
#     Rarray = np.array(fl)
#     if spc:
#         Rarray = Rarray.reshape(-1,image_dim,image_dim)
#     hf.close()
#     return Rarray

# ## Insert the DOI file location
# # file_location = './Data_1m.h5'
# to_load = 'Testing/Speckle_images/'
# test_data = {}
# for item in cfg.TEST.testlist:
#     load_item = to_load + item
#     train_speckle_image = load_dataset(cfg.datafile_location, load_item , spc = True)
#     train_fft_spc = complex_to_channels_np(np.fft.fft2(train_speckle_image)).reshape(-1,image_dim*image_dim,2)
#     test_data[item] = train_fft_spc

# # train_fft_spc = complex_to_channels_np(np.fft.fft2(train_speckle_image)).reshape(-1,image_dim*image_dim,2)
# np.save('../data/testallfft_data.npy',test_data)
# print(test_data.keys())

# print(train_fft_spc.shape)

# plt.figure(1)
# for i in range(3):
#     ax = plt.subplot(311+i)
#     plt.imshow(train_speckle_image[i])
# plt.tight_layout()
# plt.show()]
