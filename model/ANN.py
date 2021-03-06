import os,sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../'))
from cfgs.config import cfg
import keras
import numpy as np

from keras.layers import (Input, Reshape, Dense)
from keras.models import Model
from keras.layers import Lambda
from ComplexNets import *
from keras.utils import plot_model
import tensorflow as tf


def getANNmodel(version = None):

    if version is not None:
        use_version = version
    else:
        use_version = cfg.version_name
    
    image_dim = cfg.orig_dim if use_version == 'inverse_genspc' else cfg.image_dim
    orig_dim = cfg.image_dim if use_version == 'inverse_genspc' else cfg.orig_dim
    input_img = Input(shape=(image_dim*image_dim, 2))     #input is complex number
    l = input_img
    
    l = ComplexDense(orig_dim*orig_dim, use_bias=False, kernel_regularizer=regularizers.l2(cfg.lamb))(l)
    
    if use_version == "fft_with_fft":
        l = Lambda(lambda x:channels_to_complex(x),name='channels2complex')(l)
        l = Reshape((orig_dim,orig_dim))(l)
        l = Lambda(lambda x: tf.signal.ifft2d(x),name='ifftoutput')(l)
        l = Lambda(lambda x: tf.reshape(tf.abs(x),[-1,orig_dim*orig_dim]),name='output')(l)
    elif use_version == 'fft_with_fft_addsigmoid':
        l = Lambda(lambda x:channels_to_complex(x),name='channels2complex')(l)
        l = Reshape((orig_dim,orig_dim))(l)
        l = Lambda(lambda x: tf.signal.ifft2d(x),name='ifftoutput')(l)
        l = Lambda(lambda x: tf.reshape(2*tf.sigmoid(tf.abs(x))-1,[-1,orig_dim*orig_dim]),name='output')(l)
    else:
        l = Amplitude()(l)
    
    if use_version == 'fft':
        l = Lambda(lambda x: keras.activations.sigmoid(x),name='output')(l)
    elif use_version == "fft_withtanh":
        l = Lambda(lambda x: keras.activations.tanh(x),name='output')(l)
    elif use_version == "fft_sigmove":
        l = Lambda(lambda x: keras.activations.sigmoid(x-5),name='output')(l)
    elif use_version == 'fft_lineartrans':
        l = Lambda(lambda x: 2*keras.activations.sigmoid(x)-1,name='output')(l)
        # l = Lambda(lambda x: 2*x-1,name='output')(l)
    elif use_version == 'fft_with_fft' or use_version == 'fft_with_fft_addsigmoid' or use_version == 'ori_model':
        pass
    else:
        raise ValueError("unkonwn type")

    out_layer = l
    model = Model(inputs=input_img, outputs=[out_layer])
    return model

def getInverseModel():
    input_img = Input(shape=(cfg.image_dim*cfg.image_dim, 2))     #input is complex number
    l = input_img
    # if len(cfg.MODEL.mid_dim) > 0:
    #     for dimension in cfg.MODEL.mid_dim:
    #         l = ComplexDense(dimension, use_bias=False, kernel_regularizer=regularizers.l2(cfg.lamb),activation=keras.layers.PReLU())(l)
    # else:
    #     print("normal",'*'*100)
    l = ComplexDense(cfg.orig_dim*cfg.orig_dim, use_bias=False, kernel_regularizer=regularizers.l2(cfg.lamb))(l)
    # l = Lambda(lambda x: tf.ifft2d(channels_to_complex(x)))(l)
    l = Amplitude()(l)
    # l = Lambda(lambda x: keras.activations.sigmoid(x))(l)
    out_layer = l
    model = Model(inputs=input_img, outputs=[out_layer])
    return model

if __name__ == "__main__":
    model = getANNmodel()

    model.summary()
    plot_model(model, to_file='model.png')