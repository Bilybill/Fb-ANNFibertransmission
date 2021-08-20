## The program needs to load ComplexNets.py
import sys,os
from cfgs.config import cfg
from ComplexNets import *
from keras import backend as K
from keras.layers import Lambda 
import tensorflow as tf
from keras.layers import (Input, Reshape, Dense)



def GetCycleModel_AllLabel():
    input_spc_labeled = Input(shape=(cfg.image_dim*cfg.image_dim, 2)) 
    input_img_labeled = Input(shape=(cfg.orig_dim*cfg.orig_dim,))

    BackwardDense = ComplexDense(orig_dim*orig_dim, use_bias = False, kernel_regularizer = regularizers.l2(lamb))
    BackwardSquare = Amplitude()
    ForwardDense = ComplexDense(image_dim*image_dim,use_bias = False, kernel_regularizer = regularizers.l2(lamb))
    ForwardSqrt = SqrtLayer()

    input_spc_backward = BackwardDense(input_spc_labeled)
    input_spc_backward_square = BackwardSquare(input_spc_backward)
    input_img_forward = ForwardSqrt(input_img_labeled)
    input_img_forward = ForwardDense(input_img_forward)

    input_spc_reverse = ForwardDense(input_spc_backward)
    input_img_reverse = BackwardDense(input_img_forward)
    input_img_reverse = BackwardSquare(input_img_reverse)

    L_openB = Lambda(lambda x: tf.reduce_mean(K.square(x[0]-x[1])), name='L_openB_loss')([input_spc_backward_square,input_img_labeled])
    L_openF = Lambda(lambda x: tf.reduce_mean(K.square(x[0]-x[1])), name='L_openF_loss')([input_spc_labeled,input_img_forward])

    L_cycle1 = Lambda(lambda x: tf.reduce_mean(K.square(x[0]-x[1])), name='L_Cycle1_loss')([input_spc_labeled,input_spc_reverse])
    L_cycle2 = Lambda(lambda x: tf.reduce_mean(K.square(x[0]-x[1])), name='L_Cycle2_loss')([input_img_labeled,input_img_reverse])

    model = Model(inputs=[input_spc_labeled,input_img_labeled], outputs=[L_openB,L_openF,L_cycle1,L_cycle2])
    return model


if __name__ == "__main__":
    model = getANNmodel()