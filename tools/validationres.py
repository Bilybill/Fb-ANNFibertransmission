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
import cv2

def PCC(img1,img2):
    m1 = np.mean(img1)
    m2 = np.mean(img2)
    diffimg1 = img1 - m1
    diffimg2 = img2 - m2
    PCC = np.sum(diffimg1 * diffimg2) / np.sqrt(np.sum(diffimg1**2)*np.sum(diffimg2**2))
    return PCC

if __name__ == "__main__":
    # length_image = 50000
    orig_dim = cfg.orig_dim
    image_dim = cfg.image_dim
    # root_result_dir = cfg.root_result_dir
    # RGB_TYPE = 'Earth'
    # epochs = cfg.TRAIN.epochs

    # os.makedirs(cfg.root_result_dir,exist_ok=True)
    # log_file = os.path.join(cfg.root_result_dir,'log_test.txt')
    # logger = create_logger(log_file)
    # logger.info("*"*20+"start logging"+"*"*20)
    # # log to file
    # gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    # logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    # ### TRAINING loading
    # # ##Random dataset
    # # # Speckle Patterns
    # logger.info('-'*30+"start to load validation image"+'-'*30)

    test_speckle_image,test_original_image = getdata(np.loadtxt("../data/validation.txt",dtype=str))
    # if cfg.TEST.show_spc:
    #     show_spc_image = load_dataset(cfg.datafile_location, cfg.TRAIN.x_toload, spc = True)
    #     show_spc_image = show_spc_image[int(length_image/100.*90): length_image]
    
    # logger.info("test spc shape"+str(test_speckle_image.shape))
        
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
    loadcheckpoint = True
    modelpath = "../result_dir/ori_modelsaved/orimodel.h5"
    if os.path.isfile(cfg.TRAIN.checkpoint_path) and loadcheckpoint:
        if cfg.version_name == 'inverse_genspc':
            model = getInverseModel()
        else:
            model = getANNmodel()
        model.load_weights(cfg.TRAIN.checkpoint_path)
        # model.load_weights("../result_dir/fft_resdir/checkpoint")
    elif os.path.isfile(modelpath):
        logger.info("load %s model" % cfg.version_name)
        model = load_model(modelpath,custom_objects={'ComplexDense': ComplexDense,"Amplitude":Amplitude})
    else:
        raise ValueError('checkpoint file not exists')
    model.compile(optimizer=Adam(lr=cfg.TRAIN.lr), loss=cfg.TRAIN.loss)

    loss = model.evaluate(x_test_ch,y_eval)

    logger.info("test loss: {:.5f}".format(loss))
    
    if cfg.version_name == 'fft':
        logger.info("direct prediction")
        pred_test = model.predict(x_test_ch)
        logger.info("pred_test shape {}".format(pred_test.shape))
    elif cfg.version_name == 'orimodel':
        pred_test = model.predict(x_test_ch)**2
    else:
        logger.info("model %s direct prediction" % cfg.version_name)
        pred_test = model.predict(x_test_ch)
        logger.info("pred_test shape {}".format(pred_test.shape))
        

    pred_test = pred_test.reshape(pred_test.shape[0], orig_dim, orig_dim, 1)

    
    if not cfg.TEST.show_rgb:
        y_test = y_test.reshape(y_test.shape[0],orig_dim,orig_dim)
        pred_test = np.squeeze(pred_test)
    else:
        r_pred = pred_test[0:len_div]
        g_pred = pred_test[len_div:2*len_div]
        b_pred = pred_test[2*len_div:3*len_div]
        pred_test = np.concatenate([r_pred,g_pred,b_pred],axis=-1)
    

    logger.info("pred_test shape {} y_test shape {}".format(pred_test.shape,y_test.shape))
    logger.info("data range max pred: {} min pred: {} max y_test: {} min y_test: {}".format(np.max(pred_test),np.min(pred_test),np.max(y_test),np.min(y_test)))
    label = 'SSIM:{:.3f} PSNR:{:.3f} PCC:{:.3f}'
    shownum = 10
    # divnum = 2 if not cfg.TEST.compare else 3
    divnum = 3

    # fig = plt.figure(1)
    # fig.set_size_inches(20,6) 
    # data_range = 1

    for i in range(divnum*shownum):
        # ax = plt.subplot(shownum,divnum,1+i)
        if i%divnum == 2:
            # if i == 2:
            #     ax.set_title("predict image")
            # multichannel = False
            # if cfg.TEST.show_rgb:
            #     multichannel = True
            # ssim_score = ssim(pred_test[i//divnum], y_test[(i+divnum-3)//divnum], data_range=data_range, multichannel=multichannel)
            # psnr_score = PSNR(pred_test[i//divnum], y_test[(i+divnum-3)//divnum], data_range=data_range)
            # pcc_score = PCC(pred_test[i//divnum], y_test[(i+divnum-3)//divnum])
            # ax.set_xlabel(label.format(ssim_score,psnr_score,pcc_score))
            pred_img = pred_test[i//divnum]
            plt.axis('off')
            plt.imshow(pred_img)
            plt.savefig(f'validationfolder/{i//divnum}reconstruct.tif',bbox_inches='tight',pad_inches = 0)

        elif i%divnum == 0:
            # if i == 0:
            #     ax.set_title("original image")
            plt.axis('off')
            plt.imshow(y_test[i//divnum])
            plt.savefig(f'validationfolder/{i//divnum}original.tif',bbox_inches='tight',pad_inches = 0)

        elif i%divnum == 1:
            plt.axis('off')
            plt.imshow(show_spc_image[i//divnum])
            plt.savefig(f'validationfolder/{i//divnum}speckle.tif',bbox_inches='tight',pad_inches = 0)
            
