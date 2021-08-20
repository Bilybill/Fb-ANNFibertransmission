import h5py
import numpy as np
import argparse
from skimage.metrics import structural_similarity as ssim
from skimage.measure import compare_psnr as PSNR
from matplotlib import pyplot as plt
from cvxopt import matrix,solvers
import cvxpy as cp
import scipy.ndimage as ndi

parser = argparse.ArgumentParser(description='Arg parser')
parser.add_argument('--mode',type=str,default="train")
parser.add_argument('--method',type=str,default="lstquare")
parser.add_argument('--complex',action='store_true')
parser.add_argument('--load_reshape_data',action="store_true")
args = parser.parse_args()

factor = 4
spc_dim = int(120/factor)
img_dim = int(92/factor)

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
    length_image = 50000
    if args.mode == "train":
        print('start to load train image')
        if not args.load_reshape_data:
            Trainingdataload = 'Training/Original_images/ImageNet'
            image = load_dataset('../data/Data_1m.h5',Trainingdataload,spc=False,reshapetovec=False)
            Traininglabelload = "Training/Speckle_images/ImageNet"
            speckle = load_dataset('../data/Data_1m.h5',Traininglabelload,spc=True,reshapetoimg=True)
        else:
            image = np.load("train_reshape_img.npy")
            speckle = np.load("train_reshape_spc.npy")
            
        
        print('finish to load image,the shape is',image.shape,'start to load train speckle image')
        print('finish to load speckle,the shape is',speckle.shape,'start to do ',args.method)
        
        if args.method == 'lstquare':
            image = image.reshape(length_image,-1)
            speckle = speckle.reshape(length_image,-1)
            if args.complex:
                print("use cvx tools")
                x = cp.Variable((14400,92*92))
                cost = cp.sum_squares(speckle@x-image)
                prob = cp.Problem(cp.Minimize(cost))
                prob.solve()
                np.save('complexweight.npy',x)
            else:
                print('use numpy pinv')
                x = np.linalg.pinv(speckle)@image
                np.save('weight.npy',x)
                print('finish to ',args.method)
        elif args.method == 'sdp':
            if args.load_reshape_data:
                image = image.reshape(length_image,-1)
                speckle = speckle.reshape(length_image,-1)
                print("image shape",image.shape,'speckle shape',speckle.shape)
            m1 = img_dim*img_dim
            m2 = spc_dim*spc_dim
            total_num = 100
            
            n = 2
            iteration_num = int(total_num/n)
            Uans = []
            for iterindex in range(iteration_num):
                Y = image[0+iterindex*n:iterindex*n+n,]
                print("Y.shape",Y.shape)
                P = np.kron(np.eye(m2),np.eye(n)-Y@np.linalg.pinv(Y))
                C = speckle[0:n,]
                diagc = np.diag(np.squeeze(C.reshape(-1,1)))
                Q = diagc@P@diagc
                print("P shape",P.shape,"Q shape",Q.shape,"diagc shape",diagc.shape)
                u = cp.Variable((n*m2,n*m2),complex=True)
                print("start to do sdp")
                obj = cp.Minimize(cp.real(cp.trace(Q@u)))
                constrains = [
                    cp.diag(u) == 1,
                    u>>0,
                    u == u.H
                    ]
                prob = cp.Problem(obj,constrains)
                prob.solve()
                if(prob.status == cp.OPTIMAL):
                    print("The optimal value is",prob.value)
                    # print("A solution U is")
                    # print(u.value)
                    eigenvectors,eigenvalues,_ = np.linalg.svd(u.value)
                    print("eigenvectors.shape",eigenvectors.shape,"eigenvalues.shape",eigenvalues.shape)
                    sort = eigenvalues.argsort()[::-1]
                    eigenvectors = eigenvectors[:, sort]
                    vec = eigenvectors[:,0]
                    Uans.append(vec.reshape(n,m2))
                else:
                    print("cannot get optimal")
                    print(prob.status)
                    print("The optimal value is",prob.value)
                    # print("A solution U is")
                    # print(u.value)
                    eigenvectors,eigenvalues,_ = np.linalg.svd(u.value)
                    sort = eigenvalues.argsort()[::-1]
                    eigenvectors = eigenvectors[:, sort]
                    vec = eigenvectors[:,0]
                    Uans.append(vec.reshape(n,m2))
            Uans = np.vstack(Uans)
            np.save("Uans.py",Uans)

        elif args.method == "reshape_data":
            # reshape_imgdim = 30
            # reshape_spcdim = 23
            reshape_image = ndi.zoom(image, (1, 1/factor, 1/factor), order=3)
            reshape_specl = ndi.zoom(speckle, (1, 1/factor, 1/factor), order=3)
            print("reshape image.shape",reshape_image.shape,"\treshape speckle shape",reshape_specl.shape)
            np.save("train_reshape_img.npy",reshape_image)
            np.save("train_reshape_spc.npy",reshape_specl)
            print('reshape process finished')
        elif args.method == 'show_data':
            fig = plt.figure(1)
            num = 3
            for i in range(2*num):
                ax = plt.subplot(2,num,1+i)
                if i < 3:
                    plt.imshow(image[i,])
                else:
                    plt.imshow(speckle[i-num,])
                plt.axis('off')
            plt.tight_layout()
            plt.show()
        
    elif args.mode == "test":
        test_list = ['Earth_B', 'Earth_G', 'Earth_R', 'Jupyter_B', 'Jupyter_G', 'Jupyter_R', 'cat', 'horse', 'parrot', 'punch']
        testimgload = 'Testing/Original_images/punch'
        print('start to load test image')
        y_test = load_dataset("../data/Data_1m.h5",testimgload,spc=False,reshapedata=False)
        print('finish to load image,the shape is',y_test.shape,'start to load test speckle image')
        testspcload = "Testing/Speckle_images/punch"
        speckle = load_dataset('../data/Data_1m.h5',testspcload,spc=True)
        print('finish to load speckle,the shape is',speckle.shape,'start to do test')
        label = "SSIM:{:.3f} PSNR:{:.3f} PCC:{:.3f}"
        weight = np.load("weight.npy")
        print(weight.shape)
        pred_test = speckle@weight
        pred_test = pred_test.reshape(-1,92,92)
        score = np.mean((pred_test-y_test)**2)
        print("square loss: ",score)
        x_test_ch = speckle.reshape(-1,120,120)
        shownum = 3
        divnum = 3
        fig = plt.figure(1)
        fig.set_size_inches(20,6)
        data_range = 1
        for i in range(divnum*shownum):
            ax = plt.subplot(shownum,divnum,1+i)
            if i%divnum == 0:
                if i == 0:
                    ax.set_title("predict image")
                multichannel = False
                ssim_score = ssim(pred_test[i//divnum], y_test[(i+divnum-1)//divnum], data_range=data_range, multichannel=multichannel)
                psnr_score = PSNR(pred_test[i//divnum], y_test[(i+divnum-1)//divnum], data_range=data_range)
                pcc_score = PCC(pred_test[i//divnum], y_test[(i+divnum-1)//divnum])
                ax.set_xlabel(label.format(ssim_score,psnr_score,pcc_score))
                plt.imshow(pred_test[i//divnum])
            elif i%divnum == 1:
                if i == 1:
                    ax.set_title("speckle image")
                plt.imshow(x_test_ch[i//divnum])
            elif i%divnum == 2:
                if i == 2:
                    ax.set_title("original image")
                plt.imshow(y_test[i//divnum])
        plt.tight_layout()
        # plt.savefig('./resultfig/compare_%s_res.png' % cfg.version_name)
        plt.show()
