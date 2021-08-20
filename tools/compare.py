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
from train import min_maxnormalize
import cv2

def PCC(img1,img2):
    m1 = np.mean(img1)
    m2 = np.mean(img2)
    diffimg1 = img1 - m1
    diffimg2 = img2 - m2
    PCC = np.sum(diffimg1 * diffimg2) / np.sqrt(np.sum(diffimg1**2)*np.sum(diffimg2**2))
    return PCC

if __name__ == "__main__":
    orig_dim = cfg.orig_dim
    image_dim = cfg.image_dim

    print('-'*30+"start to load test image"+'-'*30)
    if cfg.TEST.usefftdata:
        if cfg.TEST.show_rgb:
            RGB_TYPE = 'Earth'
            r_spc = np.load(cfg.TEST.fftdata_load,allow_pickle=True).item()[RGB_TYPE+'_R']
            g_spc = np.load(cfg.TEST.fftdata_load,allow_pickle=True).item()[RGB_TYPE+'_G']
            b_spc = np.load(cfg.TEST.fftdata_load,allow_pickle=True).item()[RGB_TYPE+'_B']
            len_div = r_spc.shape[0]
            test_speckle_image = np.concatenate([r_spc,g_spc,b_spc],axis=0)
        else:
            test_speckle_image = np.load(cfg.TEST.fftdata_load,allow_pickle=True).item()[cfg.TEST.testclass]
        if cfg.TEST.show_spc:
            show_spc_image = load_dataset(cfg.datafile_location, cfg.TEST.x_toload, spc = True)
    else:
        test_speckle_image = load_dataset(cfg.datafile_location, cfg.TEST.x_toload)
        if np.squeeze(test_speckle_image).shape.__len__() == 3:
            test_speckle_image = test_speckle_image.reshape(-1,image_dim*image_dim)
        if cfg.TEST.show_spc:
            show_spc_image = test_speckle_image.reshape(test_speckle_image.shape[0],image_dim,image_dim)
        test_speckle_image = real_to_channels_np(test_speckle_image.astype('float32'))
        # test_speckle_image = complex_to_channels_np(np.fft.fft2(test_speckle_image)).reshape(-1,image_dim*image_dim,2)
    
    print("test spc shape"+str(test_speckle_image.shape))
    # Original Images
    if cfg.TEST.show_rgb:
        r_ori = load_dataset(cfg.datafile_location,'Testing/Original_images/%s_R' % RGB_TYPE)
        g_ori = load_dataset(cfg.datafile_location,'Testing/Original_images/%s_G' % RGB_TYPE)
        b_ori = load_dataset(cfg.datafile_location,'Testing/Original_images/%s_B' % RGB_TYPE)
        # len_div = r_ori.shape[]
        test_original_image = np.concatenate([r_ori,g_ori,b_ori],axis=-1)
        y_eval = np.concatenate([r_ori,g_ori,b_ori],axis=0)
        y_eval = np.squeeze(y_eval.reshape(-1, orig_dim*orig_dim, 1))
    else:
        test_original_image = load_dataset(cfg.datafile_location, cfg.TEST.y_toload)
        if cfg.TRAIN.one_minimize:
            test_original_image = test_original_image / 255
        
    
    ###################### Neural Network Data
    ## Training data
    # Speckle patterns
    x_test_ch = test_speckle_image
    # # Original Images
    y_test = test_original_image

    if not cfg.TEST.show_rgb:
        y_test = np.squeeze(y_test.reshape(-1, orig_dim*orig_dim, 1))  #0-1
        # y_test = min_maxnormalize(y_test)
        y_eval = y_test


    ##################          MODEL  
    # check checkpoint
    model_ori = load_model(modelpath,custom_objects={'ComplexDense': ComplexDense,"Amplitude":Amplitude})
    # model_fft = getANNmodel(version="fft")
    model_fft_sigmove = getANNmodel(version="fft_lineartrans")
    # model_fft.load_weights('../result_dir/fft_resdir/bestcheckpoint')
    model_fft_sigmove.load_weights('../result_dir/fft_resdir/bestcheckpoint')
    # model_fft_sigmove.load_weights('../result_dir/fft_lineartrans_resdir/checkpoint')

    model_ori.compile(optimizer=Adam(lr=cfg.TRAIN.lr), loss=cfg.TRAIN.loss)
    model_fft_sigmove.compile(optimizer=Adam(lr=cfg.TRAIN.lr), loss=cfg.TRAIN.loss)

    ori_loss = model_ori.evaluate(x_test_ch,y_eval)
    fft_sigmove_loss = model_fft_sigmove.evaluate(x_test_ch,y_eval)

    print("test fft_loss: {:.5f}".format(ori_loss))
    print("test fft_linear_loss: {:.5f}".format(fft_sigmove_loss))

    pred_fft_test = model_ori.predict(x_test_ch)
    pred_fft_sigmove_test = model_fft_sigmove.predict(x_test_ch)

    print("pred_test shape {}".format(pred_fft_test.shape))

    pred_fft_test = pred_fft_test.reshape(pred_fft_test.shape[0], orig_dim, orig_dim)
    pred_fft_sigmove_test = pred_fft_sigmove_test.reshape(pred_fft_sigmove_test.shape[0], orig_dim, orig_dim)

    
    if not cfg.TEST.show_rgb:
        y_test = y_test.reshape(y_test.shape[0],orig_dim,orig_dim)
    else:
        r_pred = pred_test[0:len_div]
        g_pred = pred_test[len_div:2*len_div]
        b_pred = pred_test[2*len_div:3*len_div]
        pred_test = np.concatenate([r_pred,g_pred,b_pred],axis=-1)
    
    # ("pred_test shape {} y_test shape {}".format(pred_test.shape,y_test.shape))
    print("data range max pred: {} min pred: {} max y_test: {} min y_test: {} lineartrans min : {} lineartrans max: {}".format(np.max(pred_fft_test),np.min(pred_fft_test),np.max(y_test),np.min(y_test),np.min(pred_fft_sigmove_test),np.max(pred_fft_sigmove_test)))
    label = 'SSIM:{:.3f} PSNR:{:.3f} PCC:{:.3f}'
    shownum = 4
    # divnum = 2 if not cfg.TEST.compare else 3
    divnum = 3

    fig = plt.figure(1)
    fig.set_size_inches(20,6) 
    data_range = 1

    for i in range(divnum*shownum):
        ax = plt.subplot(shownum,divnum,1+i)
        if i%divnum == 0:
            if i == 0:
                ax.set_title("fft_predict image histogram")                
            pred_img = pred_fft_test[i//divnum]
            # pred_img = 2*pred_img-1
            # plt.imshow(pred_img)
            pred_img = (pred_img*255).astype(int)
            arr = pred_img.flatten()
            # pred_img = np.sqrt(1 - pred_img)
            # pred_img = (pred_img*256).astype(int)
            # arr = pred_img.flatten()
            # arr = np.sqrt(arr)
            # hist_gray = cv2.calcHist(images=[pred_img], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
            n, bins, patches = plt.hist(arr, bins=256,range=(0,256),  facecolor='green', alpha=0.75)  
            # plt.imshow(pred_img,'gray')
        
        elif i%divnum == 1:
            if i == 1:
                ax.set_title("fft_linear_predict image histogram")
            # plt.imshow(y_test[i//divnum])
            pred_img = pred_fft_sigmove_test[i//divnum]
            # plt.imshow(pred_img)
            pred_img = (pred_img*255).astype(int)
            # plt.imshow(pred_img,'gray')
            arr = pred_img.flatten()
            # # hist_gray = cv2.calcHist(images=[pred_img], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
            n, bins, patches = plt.hist(arr, bins=256,range=(0,256),  facecolor='red', alpha=0.75)  

        elif i%divnum == 2:
            if i == 2:
                ax.set_title("label image histogram")
            # plt.imshow(x_test_ch[i//divnum])
            label_img = y_test[i//divnum]
            # plt.imshow(label_img)
            label_img = (label_img*255).astype(int)

            # plt.imshow(label_img,'gray')
            arr = label_img.flatten()
            # # hist_gray = cv2.calcHist(images=[pred_img], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
            n, bins, patches = plt.hist(arr, bins=256,range=(0,256),  facecolor='red', alpha=0.75)  
        
    plt.tight_layout()
    plt.show()
