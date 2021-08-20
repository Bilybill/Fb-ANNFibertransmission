import h5py
import _init_path
from model.ANN import getANNmodel
from cfgs.config import cfg
import numpy as np
from ComplexNets import *
from matplotlib import pyplot as plt
from keras.models import load_model
from train import load_dataset
from skimage.metrics import structural_similarity as ssim
from test import PCC

def Sigmoid(x):
    return 1/(1+np.exp(-x))
def InvSigmoid(x):
    x[x==0] = 1e-3
    x[x==1] = 0.999
    return np.log(x/(1-x))

if __name__ == "__main__":
    # model version 1
    # modelpath = "../result_dir/ori_modelsaved/orimodel.h5"
    # model = load_model(modelpath,custom_objects={'ComplexDense': ComplexDense,"Amplitude":Amplitude})
    # weights = channels_to_complex_np(model.layers[1].get_weights()[0])
    # np.save('./tempdata/w',weights)
    # model version 2
    # model = getANNmodel()
    # model.load_weights('../result_dir/fft_resdir/checkpoint')
    # weights = channels_to_complex_np(model.layers[1].get_weights()[0])
    # weights = np.save('./tempdata/fftw',weights)

    weights = np.load('./tempdata/fftw.npy')

    # print(np.max(weights),np.min(weights))
    # plt.figure()
    # weights = (weights - np.min(weights))/(np.max(weights) - np.min(weights))
    # plt.imshow(weights)
    # plt.show()
    # print('weights.shape:',weights.shape)

    # inv_w = np.linalg.pinv(weights)


    
    # print(weights.shape)
    # inv_w = np.linalg.pinv(weights)
    # inv_w = np.load('./tempdata/inw.npy')
    # # np.save('./tempdata/inw',inv_w)
    # print(inv_w.shape) # 8464,14400
    
    spc_image = np.squeeze(load_dataset(cfg.datafile_location,cfg.TRAIN.x_toload,spc=True))[0:8]
    spc_fft_image = np.fft.fft2(spc_image)
    spc_shift_image = np.fft.fftshift(spc_fft_image)
    spc_fft_ori = spc_shift_image.copy()
    rows,cols = cfg.image_dim,cfg.image_dim
    # print(rows,cols)
    radius = 1000
    x = np.linspace(0,cols-1,cols)
    y = np.linspace(0,rows-1,rows)
    X,Y = np.meshgrid(x,y)
    # mask = np.ones(spc_shift_image.shape)
    spc_shift_image[:,(X-cols//2)**2+(Y-rows//2)**2 > radius**2] = 0
    ori_image = np.squeeze(load_dataset(cfg.datafile_location,cfg.TRAIN.y_toload))[0:8]
    inverse_image = (np.fft.ifftshift(spc_shift_image).reshape(-1,cfg.image_dim*cfg.image_dim))@weights
    # inverse_image = ((spc_shift_image).reshape(-1,cfg.image_dim*cfg.image_dim))@weights
    spc_ifft_image = np.abs(np.fft.ifft2(np.fft.ifftshift(spc_shift_image)))
    inverse_image = np.abs(inverse_image)
    inverse_image = Sigmoid(inverse_image).reshape(-1,cfg.orig_dim,cfg.orig_dim)
    inverse_image_fft = np.fft.fftshift(np.fft.fft2(inverse_image))
    inverse_image_fft = np.log(1+np.abs(inverse_image_fft))
    spc_shift_image = np.log(1+np.abs(spc_shift_image))

    print(np.max(spc_shift_image),np.min(spc_shift_image))

    # inverse_image = np.abs(spc_fft_image@weights).reshape(-1,cfg.orig_dim,cfg.orig_dim)
    # # inverse_image = channels_to_complex_np(real_to_channels_np(spc_image.reshape(-1,cfg.image_dim*cfg.image_dim)))@weights
    # inverse_image= Sigmoid(inverse_image)  
    # print('inverse_image: min:{:3f} max:{:3f}'.format(np.min(inverse_image),np.max(inverse_image)))

    # ori_image_vec = InvSigmoid(ori_image.reshape(-1,cfg.orig_dim*cfg.orig_dim))

    # inverse_spc_image = channels_to_complex_np(real_to_channels_np(ori_image_vec))@inv_w
    # inverse_spc_image = inverse_spc_image.reshape(-1,cfg.image_dim,cfg.image_dim)
    # inverse_spc_image = np.abs(np.fft.ifft2(inverse_spc_image))
    
    # # inverse_spc_image = np.abs(inverse_spc_image).reshape(-1,cfg.image_dim,cfg.image_dim)
    # # inverse_spc_image = 255 * (inverse_spc_image - np.min(inverse_spc_image))/(np.max(inverse_spc_image)     - np.min(inverse_spc_image))
    
    # print(np.min(spc_image),np.max(spc_image))
    # print(np.min(inverse_spc_image),np.max(inverse_spc_image))
        
    show_num = 8
    div_num = 6
    label = 'SSIM:{:.3f} PCC:{:.3f}'
    fig = plt.figure(1)
    fig.set_size_inches(20,6)
    for i in range(div_num*show_num):
        ax = plt.subplot(div_num,show_num,1+i)
        if i < show_num:
            if i == 0:
                pass
                # ax.set_ylabel('original speckle image')
            plt.imshow(spc_image[i])
        elif i < 2*show_num:
            if i == show_num:
                pass
                # ax.set_ylabel("original image")
            plt.imshow(ori_image[i-show_num])
        elif i < 3 * show_num:
            if i == 2*show_num:
                pass
                # ax.set_ylabel("output image")
            plt.imshow(inverse_image[i-2*show_num])
            pcc_score = PCC(inverse_image[i-2*show_num],ori_image[i-2*show_num])
            ssim_score = ssim(inverse_image[i-2*show_num],ori_image[i-2*show_num], data_range=1)
            ax.set_xlabel(label.format(ssim_score,pcc_score))
        elif i < 4 * show_num:
            if i== 3*show_num:
                pass
                # ax.set_ylabel("constructed inverse speckle image")
            plt.imshow(spc_shift_image[i-3*show_num])
            # pcc_score = PCC(inverse_spc_image[i-3*show_num],spc_image[i-3*show_num])
            # ssim_score = ssim(inverse_spc_image[i-3*show_num],spc_image[i-3*show_num], data_range=255)
            # ax.set_xlabel(label.format(ssim_score,pcc_score))
        # plt.axis('off')   
        elif i < 5*show_num:
            if i== 4*show_num:
                pass
            plt.imshow(spc_ifft_image[i-4*show_num])
        elif i < 6*show_num:
            if i==5*show_num:
                pass
            plt.imshow(inverse_image_fft[i-5*show_num])
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.savefig('./resultfig/test1.png')
    plt.show()
    # print(inverse_image.shape)