from matplotlib import pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['axes.linewidth'] = 0
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.measure import compare_psnr as PSNR
import _init_path
from train import load_dataset,create_logger,min_maxnormalize
import os
import logging
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import numpy as np
from keras.optimizers import SGD,Adam
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint
from keras.utils import multi_gpu_model
from ComplexNets import *
import h5py
from cfgs.config import cfg
from ANN import getANNmodel,getInverseModel
from keras.models import load_model
import scipy.io as io
from keras.models import Model
from train import min_maxnormalize,getdata

if __name__ == "__main__":
    orig_dim = cfg.orig_dim
    image_dim = cfg.image_dim

    test_speckle_image,test_original_image = getdata(np.loadtxt("../data/validation.txt",dtype=str))

    x_test_ch = test_speckle_image
    y_test = test_original_image

    if not cfg.TEST.show_rgb:
        y_test = np.squeeze(y_test.reshape(-1, orig_dim*orig_dim, 1))  #0-1
        # y_test = min_maxnormalize(y_test)
        y_eval = y_test
    ##################          MODEL  
    # check checkpoint
    model = getANNmodel()
    
    model.load_weights(cfg.TRAIN.checkpoint_path)

    model.compile(optimizer=Adam(lr=cfg.TRAIN.lr), loss=cfg.TRAIN.loss)

    loss = model.evaluate(x_test_ch,y_test)

    print(f"validation loss: {loss}")