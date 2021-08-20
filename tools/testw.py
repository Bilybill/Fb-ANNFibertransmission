import h5py
import numpy as np
import argparse
from skimage.metrics import structural_similarity as ssim
from skimage.measure import compare_psnr as PSNR
from matplotlib import pyplot as plt

def load_dataset(file_location,what_to_load:str,spc:bool=True,reshapetoimg:bool=False,reshapetovec:bool=False):
    hf = h5py.File(file_location, 'r')
    fl = hf[what_to_load]
    Rarray = np.array(fl)
    # Rarray = Rarray.astype(complex)
    if not spc:
        Rarray = np.squeeze(Rarray)
        if reshapetovec:
            Rarray = Rarray.reshape(-1,92*92)
    elif spc and reshapetoimg:
        Rarray = Rarray.reshape(-1,120,120)
    hf.close()
    return Rarray

def PCC(img1,img2):
    m1 = np.mean(img1)
    m2 = np.mean(img2)
    diffimg1 = img1 - m1
    diffimg2 = img2 - m2
    PCC = np.sum(diffimg1 * diffimg2) / np.sqrt(np.sum(diffimg1**2)*np.sum(diffimg2**2))
    return PCC

if __name__ == "__main__":
    '''
    输入：Speckle图片，将其转为复数形式，虚部补零
    输出：Image图片
    '''
    print('start to load test image')
    # Trainingdataload = 'Training/Original_images/ImageNet'
    # image = load_dataset('../data/Data_1m.h5',Trainingdataload,spc=False,reshapetovec=True)
    # Traininglabelload = "Training/Speckle_images/ImageNet"
    # speckle = load_dataset('../data/Data_1m.h5',Traininglabelload,spc=True,reshapetoimg=False)            
    test_list = ['Earth_B', 'Earth_G', 'Earth_R', 'Jupyter_B', 'Jupyter_G', 'Jupyter_R', 'cat', 'horse', 'parrot', 'punch']
    testimgload = 'Testing/Original_images/punch'
    print('start to load test image')
    image = load_dataset("../data/Data_1m.h5",testimgload,spc=False,reshapetovec=True)
    print('finish to load image,the shape is',image.shape,'start to load test speckle image')
    testspcload = "Testing/Speckle_images/punch"
    speckle = load_dataset('../data/Data_1m.h5',testspcload,spc=True,reshapetoimg=False)
    print('finish to load speckle,the shape is',speckle.shape,'start to do test')
        
    print('finish to load image,the shape is',image.shape,'start to load train speckle image')
    print('finish to load speckle,the shape is',speckle.shape,'start to do test')
    weights = np.load('./tempdata/w.npy')
    speckle = speckle+0.j
    # print(weights.shape)
    # print(speckle.shape)
    # print(speckle)
    pred = speckle@weights
    pred = np.abs(pred)
    print(pred.shape)
    mseloss = np.mean((pred-image)**2)
    print("mse loss {:.5f}".format(mseloss))
    pred = pred.reshape(-1,92,92)
    image = image.reshape(-1,92,92)
    label = "SSIM:{:.3f} PSNR:{:.3f} PCC:{:.3f}"

    fig = plt.figure(1)
    num = 3
    data_range = 1
    for i in range(2*num):
        ax = plt.subplot(2,num,1+i)
        if i < 3:
            if i == 0:
                ax.set_title("original image")
            plt.imshow(image[i,])
        else:
            if i == 3:
                ax.set_title("predicted image")
            plt.imshow(pred[i-num,])
            ssim_score = ssim(pred[i-num,], image[i-num,], data_range=data_range)
            psnr_score = PSNR(pred[i-num,], image[i-num,], data_range=data_range)
            pcc_score = PCC(pred[i-num,], image[i-num,], )
            ax.set_xlabel(label.format(ssim_score,psnr_score,pcc_score))
        
    plt.tight_layout()
    plt.show()