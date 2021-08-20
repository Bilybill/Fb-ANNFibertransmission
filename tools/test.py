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

def test_inverse_model(spc_data):
    # spc_data = spc_data.reshape(spc_data.shape[0],cfg.orig_dim,cfg.orig_dim)
    # fft_data = complex_to_channels_np(np.fft.fft2(spc_data))
    # fft_data = fft_data.reshape()
    fft_data = real_to_channels_np(spc_data).reshape(-1,cfg.orig_dim*cfg.orig_dim,2)
    # model = getANNmodel()
    # model.load_weights('../result_dir/fft_resdir/checkpoint')
    modelpath = "../result_dir/ori_modelsaved/orimodel.h5"
    model = load_model(modelpath,custom_objects={'ComplexDense': ComplexDense,"Amplitude":Amplitude})
    model.compile(optimizer=Adam(lr=cfg.TRAIN.lr), loss=cfg.TRAIN.loss)
    ori_image = model.predict(fft_data)
    ori_image = ori_image.reshape(ori_image.shape[0],cfg.image_dim,cfg.image_dim)
    return ori_image

if __name__ == "__main__":
    length_image = 50000
    orig_dim = cfg.orig_dim
    image_dim = cfg.image_dim
    root_result_dir = cfg.root_result_dir
    epochs = cfg.TRAIN.epochs

    os.makedirs(cfg.root_result_dir,exist_ok=True)
    log_file = os.path.join(cfg.root_result_dir,'log_test.txt')
    logger = create_logger(log_file)
    logger.info("*"*20+"start logging"+"*"*20)
    # log to file
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    ### TRAINING loading
    # ##Random dataset
    # # Speckle Patterns
    logger.info('-'*30+"start to load test image"+'-'*30)
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
    
    logger.info("test spc shape"+str(test_speckle_image.shape))
    # Original Images
    if cfg.TEST.show_rgb:
        r_ori = load_dataset(cfg.datafile_location,'Testing/Original_images/%s_R' % RGB_TYPE)
        g_ori = load_dataset(cfg.datafile_location,'Testing/Original_images/%s_G' % RGB_TYPE)
        b_ori = load_dataset(cfg.datafile_location,'Testing/Original_images/%s_B' % RGB_TYPE)
        # len_div = r_ori.shape[]
        test_original_image = np.concatenate([b_ori,g_ori,r_ori],axis=-1)
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

    # output_without_reshape = Model(input=model.input,output=model.get_layer('complex_dense_1').output)
    # output = output_without_reshape.predict(x_test_ch)
    
    # x_test_ch = min_maxnormalize(x_test_ch)
    loss = model.evaluate(x_test_ch,y_eval)

    logger.info("test loss: {:.5f}".format(loss))
    
    if cfg.version_name == 'fft':
        logger.info("direct prediction")
        pred_test = model.predict(x_test_ch)
        # pred_test = min_maxnormalize(pred_test)
        logger.info("pred_test shape {}".format(pred_test.shape))
    elif cfg.version_name == 'orimodel':
        pred_test = model.predict(x_test_ch)**2
    else:
        logger.info("model %s direct prediction" % cfg.version_name)
        pred_test = model.predict(x_test_ch)
        # pred_test = min_maxnormalize(pred_test)
        logger.info("pred_test shape {}".format(pred_test.shape))
        
    # x_test_ch = np.fft.fftshift(channels_to_complex_np(x_test_ch.reshape(x_test_ch.shape[0],image_dim,image_dim,2)))
    # x_test_ch = channels_to_complex_np(x_test_ch.reshape(x_test_ch.shape[0],image_dim,image_dim,2))
    # x_test_ch = channels_to_complex_np(x_test_ch.reshape(x_test_ch.shape[0],image_dim,image_dim,2))
    # x_test_ch = np.fft.fftshift(x_test_ch)
    # x_test_fft_angle = np.angle(x_test_ch)
    # x_test_fft_angle = min_maxnormalize(x_test_fft_angle)
    # x_test_ch = np.abs(x_test_ch)
    # x_test_ch = np.log(1 + np.abs(x_test_ch))

    # print(np.min(x_test_fft_angle),np.max(x_test_fft_angle))
    # logger.info("x_test shape {}".format(x_test_ch.shape))

    pred_test = pred_test.reshape(pred_test.shape[0], orig_dim, orig_dim, 1)

    
    if not cfg.TEST.show_rgb:
        y_test = y_test.reshape(y_test.shape[0],orig_dim,orig_dim)
        pred_test = np.squeeze(pred_test)
        # x_test_ch = x_test_ch[...,0].reshape(x_test_ch.shape[0],image_dim,image_dim)
        # x_test_ch = np.log(1 + np.abs(x_test_ch))
        # output = channels_to_complex_np(output)
        # output = output.reshape(output.shape[0],orig_dim,orig_dim)
        # show_spc_image = test_inverse_model(pred_test)
        # x_test_fft_angle = test_inverse_model(y_test)
    else:
        r_pred = pred_test[0:len_div]
        g_pred = pred_test[len_div:2*len_div]
        b_pred = pred_test[2*len_div:3*len_div]
        pred_test = np.concatenate([b_pred,g_pred,r_pred],axis=-1)
    
    # logger.info("output shape {} {}".format(output.shape,output.dtype))
    # mat_path = './result_saved/output_data.mat'
    # io.savemat(mat_path,{'data':output})
    
    # mat_path = './result_saved/orimodelwithsigmoid_result.mat'
    # io.savemat(mat_path,{'data':pred_test})
    # mat_path = './result_saved/original_label.mat'
    # io.savemat(mat_path,{'data':y_test})

    logger.info("pred_test shape {} y_test shape {}".format(pred_test.shape,y_test.shape))
    logger.info("data range max pred: {} min pred: {} max y_test: {} min y_test: {}".format(np.max(pred_test),np.min(pred_test),np.max(y_test),np.min(y_test)))
    label = 'SSIM:{:.3f} PSNR:{:.3f} PCC:{:.3f}'
    shownum = 4
    # divnum = 2 if not cfg.TEST.compare else 3
    divnum = 2

    fig = plt.figure(1)
    fig.set_size_inches(20,6) 
    data_range = 1

    for i in range(divnum*shownum):
        ax = plt.subplot(shownum,divnum,1+i)
        if i%divnum == 1:
            if i == 1:
                ax.set_title("predict image")
            multichannel = False
            if cfg.TEST.show_rgb:
                multichannel = True
            ssim_score = ssim(pred_test[i//divnum], y_test[(i+divnum-1)//divnum], data_range=data_range, multichannel=multichannel)
            psnr_score = PSNR(pred_test[i//divnum], y_test[(i+divnum-1)//divnum], data_range=data_range)
            pcc_score = PCC(pred_test[i//divnum], y_test[(i+divnum-1)//divnum])
            ax.set_xlabel(label.format(ssim_score,psnr_score,pcc_score))
            pred_img = pred_test[i//divnum]
            plt.imshow(pred_img)

        elif i%divnum == 0:
            if i == 0:
                ax.set_title("original image")
            plt.imshow(y_test[i//divnum])

        elif i%divnum == 3:
            if i == 3:
                ax.set_title("speck_fft image")
            plt.imshow(x_test_ch[i//divnum])

        elif i%divnum == 2:
            if i == 2:
                ax.set_title('speckle image')
            plt.imshow(show_spc_image[i//divnum])

        # elif i%divnum == 4:
        #     if i == 4:
        #         ax.set_title('speckle_angle image')
        #     plt.imshow(x_test_fft_angle[i//divnum])
        # plt.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.5,hspace=1)
    # plt.savefig('./resultfig/compare_%s_res.png' % cfg.version_name)
    plt.show()
