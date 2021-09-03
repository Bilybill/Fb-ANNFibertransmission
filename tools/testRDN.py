from matplotlib import pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['axes.linewidth'] = 0
from skimage.metrics import structural_similarity as ssim
from skimage.measure import compare_psnr as PSNR
from genericpath import exists
import _init_path
from RDN import RDN
import numpy as np
import os
from os import path as osp
from cfgs.config import cfg
import pickle
from ANN import getANNmodel
from keras.optimizers import SGD,Adam
from tqdm import tqdm
from train import load_dataset,create_logger
import logging
from keras.utils import multi_gpu_model
def getdata(path):
    triandata = []
    labeldata = []
    prefix_train = '../data/lrdata'
    prefix_label = '../data/label'
    for name in path:
        triandata.append(np.load(osp.join(prefix_train,name)))
        labeldata.append(np.load(osp.join(prefix_label,name)).reshape(92,92,1))
    return np.array(triandata),np.array(labeldata)

def train_data_generator(train_list,batch_size):
    while True:
        for i in range(0,len(train_list),batch_size):
            x,y = getdata(train_list[i:i+batch_size])
            yield ({"input_1":x},{"output":y})
def PCC(img1,img2):
    m1 = np.mean(img1)
    m2 = np.mean(img2)
    diffimg1 = img1 - m1
    diffimg2 = img2 - m2
    PCC = np.sum(diffimg1 * diffimg2) / np.sqrt(np.sum(diffimg1**2)*np.sum(diffimg2**2))
    return PCC
if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

    log_file = os.path.join('../result_dir/testRDN','log_backup.txt')
    logger = create_logger(log_file)
    
    ## generate low resolution data
    # orig_dim = cfg.orig_dim
    # test_speckle_image = np.load(cfg.TEST.fftdata_load,allow_pickle=True).item()
    # if not osp.exists('../data/testlrdata/'):
    #     os.makedirs('../data/testlrdata/',exist_ok=True)
    # model = getANNmodel(version='fft_lineartrans')
    # model.load_weights('../result_dir/fft_lineartrans_resdir/checkpoint')
    # for test_cls in tqdm(cfg.TEST.testlist):
    #     test_speckle_data = test_speckle_image[test_cls]
    #     model.compile(optimizer=Adam(lr=cfg.TRAIN.lr), loss=cfg.TRAIN.loss)
    #     pred_test = model.predict(test_speckle_data)
    #     pred_test = pred_test.reshape(pred_test.shape[0],orig_dim,orig_dim,1)
    #     np.save("../data/testlrdata/"+test_cls+'.npy',pred_test)
    
    ## print test loss on all test dataset
    # orig_dim = cfg.orig_dim
    # test_speckle_image = np.load(cfg.TEST.fftdata_load,allow_pickle=True).item()
    # model = getANNmodel(version='fft_with_fft_addsigmoid')
    # model.load_weights('../result_dir/fft_with_fft_addsigmoid_resdir/checkpoint')
    # model.compile(optimizer=Adam(lr=cfg.TRAIN.lr), loss=cfg.TRAIN.loss)
    # test_allinput = []
    # test_alloutput = []

    # for test_cls in tqdm(cfg.TEST.testlist):
    #     test_speckle_data = test_speckle_image[test_cls]
    #     test_original_image = load_dataset(cfg.datafile_location,f'Testing/Original_images/{test_cls}')
    #     test_original_image = test_original_image.reshape(-1,orig_dim*orig_dim)
    #     loss = model.evaluate(test_speckle_data,test_original_image)
    #     logger.info(f"class:{test_cls}\tnumber of class:{test_speckle_data.shape[0]}\tloss:{loss}")


    #     test_allinput.append(test_speckle_data)
    #     test_alloutput.append(test_original_image)
    # test_allinput = np.concatenate(test_allinput,axis=0)
    # test_alloutput = np.concatenate(test_alloutput,axis=0)
    # print(test_allinput.shape,test_alloutput.shape)
    # logger.info(f"total number of class:{test_allinput.shape[0]}\taverage loss:{loss}")

    # transfer multi gpu model weights into single gpu model
    # rdn = RDN(channel = 1,multi_gpu = False)
    # model = rdn.get_model()
    # parrel_model = multi_gpu_model(model,gpus=2)
    # parrel_model.load_weights(cfg.TESTRDN.load_checkpoint_path)
    # model.save_weights('../result_dir/rdn_resdir/singlecheckpoint_v2')
    
    ## test RDN model performence
    if cfg.TESTRDN.use_pre_model:
        use_version = 'fft_lineartrans'
        logger.info(f"use pre model:{use_version}")
        recons_model = getANNmodel(version=use_version)
        recons_model.load_weights('../result_dir/%s_resdir/checkpoint' % use_version)
        recons_model.compile(optimizer=Adam(lr=cfg.TRAIN.lr), loss=cfg.TRAIN.loss)
        input_spc_fft = np.load(cfg.TEST.fftdata_load,allow_pickle=True).item()[cfg.TEST.testclass]
        y_test = load_dataset(cfg.datafile_location,cfg.TEST.y_toload).reshape(-1,cfg.orig_dim,cfg.orig_dim,1)
        x_test = recons_model.predict(input_spc_fft).reshape(-1,cfg.orig_dim,cfg.orig_dim,1)
        
    else:
        if cfg.TESTRDN.show_rgb:
            RGB_TYPE = cfg.TEST.testclass
            r_ori = load_dataset(cfg.datafile_location,'Testing/Original_images/%s_R' % RGB_TYPE)
            g_ori = load_dataset(cfg.datafile_location,'Testing/Original_images/%s_G' % RGB_TYPE)
            b_ori = load_dataset(cfg.datafile_location,'Testing/Original_images/%s_B' % RGB_TYPE)
            len_div = r_ori.shape[0]
            y_test = np.concatenate([g_ori,b_ori,r_ori],axis=-1)
            x_test_r = np.load('../data/testlrdata/%s_R.npy'%cfg.TEST.testclass)
            x_test_g = np.load('../data/testlrdata/%s_G.npy'%cfg.TEST.testclass)
            x_test_b = np.load('../data/testlrdata/%s_B.npy'%cfg.TEST.testclass)
            x_test = np.concatenate([x_test_r,x_test_g,x_test_b],axis=0)
        else:
            y_test = load_dataset(cfg.datafile_location,cfg.TEST.y_toload).reshape(-1,cfg.orig_dim,cfg.orig_dim,1)
            x_test = np.load('../data/testlrdata/%s.npy'%cfg.TEST.testclass)
    rdn = RDN(channel = 1,multi_gpu = False,load_weights = cfg.TESTRDN.load_checkpoint_path)
    rdn_model = rdn.get_model()
    pred_test = rdn_model.predict(x_test)
    if cfg.TESTRDN.show_rgb:
        r_pred = pred_test[0:len_div]
        g_pred = pred_test[len_div:2*len_div]
        b_pred = pred_test[2*len_div:3*len_div]
        pred_test = np.concatenate([g_pred,b_pred,r_pred],axis=-1)
        r_x = x_test[0:len_div]
        g_x = x_test[len_div:2*len_div]
        b_x = x_test[2*len_div:3*len_div]
        x_test = np.concatenate([g_x,b_x,r_x],axis=-1)
    print(f'min:{np.min(pred_test)},max:{np.max(pred_test)}')
    label = 'SSIM:{:.3f} PSNR:{:.3f} PCC:{:.3f}'
    shownum = 4
    divnum = 3
    fig = plt.figure(1)
    fig.set_size_inches(20,6)
    data_range = 1

    x_test = np.squeeze(x_test)
    y_test = np.squeeze(y_test)
    pred_test = np.squeeze(pred_test)

    for i in range(divnum*shownum):
        ax = plt.subplot(shownum,divnum,1+i)
        if i%divnum == 1:
            if i == 1:
                ax.set_title("Predict image")
            multichannel = False
            if cfg.TESTRDN.show_rgb:
                multichannel = True
            ssim_score = ssim(pred_test[i//divnum], y_test[i//divnum], data_range=data_range, multichannel=multichannel)
            psnr_score = PSNR(pred_test[i//divnum], y_test[i//divnum], data_range=data_range)
            pcc_score = PCC(pred_test[i//divnum], y_test[i//divnum])
            ax.set_xlabel(label.format(ssim_score,psnr_score,pcc_score))
            pred_img = pred_test[i//divnum]
            plt.imshow(pred_img)

        elif i%divnum == 0:
            if i == 0:
                ax.set_title("Original image")
            plt.imshow(y_test[i//divnum])
        elif i%divnum == 2:
            if i == 2:
                ax.set_title('Input image')
            print(f"input index:{i//divnum},y test index:{i//divnum}")
            ssim_score = ssim(x_test[i//divnum], y_test[i//divnum], data_range=data_range, multichannel=multichannel)
            psnr_score = PSNR(x_test[i//divnum], y_test[i//divnum], data_range=data_range)
            pcc_score = PCC(x_test[i//divnum], y_test[i//divnum])
            ax.set_xlabel(label.format(ssim_score,psnr_score,pcc_score))
            plt.imshow(x_test[i//divnum])

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.5,hspace=1)
    # plt.savefig('../result_dir/rdn_res/rdn_res_%s.png'%cfg.TEST.testclass)
    plt.show()