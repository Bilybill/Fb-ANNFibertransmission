from matplotlib import pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['axes.linewidth'] = 0
import numpy as np
from train import load_dataset,create_logger,min_maxnormalize
from os import path as osp
import os
import logging
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
import cv2

def getdata(path):
    testdata = []
    prefix_train = '../data/generate_fft_data/train'
    for name in path:
        testdata.append(np.load(osp.join(prefix_train,name)))
    return np.array(testdata)

def test_data_generator(test_list,batch_size):
    while True:
        for i in range(0,len(test_list),batch_size):
            x= getdata(test_list[i:i+batch_size])
            yield ({"input_1":x})


if __name__ == "__main__":
    model = getANNmodel()
    model.load_weights('../result_dir/fft_lineartrans_resdir/checkpoint')
    model.compile(optimizer=Adam(lr=cfg.TRAIN.lr), loss=cfg.TRAIN.loss)
    test_list = np.loadtxt("../data/train.txt",dtype=str)
    test_list += np.loadtxt("../data/validation.txt",dtype=str)
    print(f"data sample number:{len(test_list)}")
    test_generator = test_data_generator(test_list,batch_size = 100)
    pre_res = model.predict_generator(test_generator,steps = int(len(test_list)/100))
    print(pre_res.shape)
