import h5py
import numpy as np
import argparse
from skimage.metrics import structural_similarity as ssim
from skimage.measure import compare_psnr as PSNR
from matplotlib import pyplot as plt
from cvxopt import matrix,solvers
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import torch


parser = argparse.ArgumentParser(description='Arg parser')
parser.add_argument('--mode',type=str,default="train")
# parser.add_argument('--method',type=str,default="lstquare")
# parser.add_argument('--complex',action='store_true')
# parser.add_argument('--load_reshape_data',action="store_true")
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
        image = np.load("train_reshape_img.npy")
        speckle = np.load("train_reshape_spc.npy")
        
        m1 = img_dim*img_dim
        m2 = spc_dim*spc_dim
        total_num = 100    
        n = 2
        iteration_num = int(total_num/n)
        Uans = []
        Q = cp.Parameter((n*m2,n*m2))
        u = cp.Variable((n*m2,n*m2))
        obj = cp.Minimize(cp.trace(Q@u))
        constrains = [
           cp.diag(u) == 1,
           u>>0,
           u == u.H
           ]
        prob = cp.Problem(obj,constrains)
        assert prob.is_dpp()
        print("problem construction finished,build cvxpylayer")
        cvxpylayer = CvxpyLayer(prob, parameters=[Q], variables=[u])

        Y = image[0:n,]
        P = np.kron(np.eye(m2),np.eye(n)-Y@np.linalg.pinv(Y))
        C = speckle[0:n,]
        diagc = np.diag(np.squeeze(C.reshape(-1,1)))
        print("Q torch construction finished,move to cuda,the shape is")
        Q_th = torch.from_numpy(diagc@P@diagc).cuda()
        print(Q_th.shape)
        print("solve the problem")
        solution, = cvxpylayer(Q_th)
        print(solution)
        # Q_tch = torch.randn(n*m2, n*m2, requires_grad=True).cuda()
        # # solve the problem
        # solution, = cvxpylayer(Q_tch)
        # # compute the gradient of the sum of the solution with respect to A, b
        # solution.sum().backward()