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
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn
from matplotlib import colors
import matplotlib as mpl
import seaborn as sns

def Sigmoid(x):
    return 1/(1+np.exp(-x))
def InvSigmoid(x):
    x[x==0] = 1e-3
    x[x==1] = 0.999
    return np.log(x/(1-x))

if __name__ == "__main__":
    # model version 2
    # model = getANNmodel()
    # model.load_weights(cfg.TRAIN.checkpoint_path)
    # weights = channels_to_complex_np(model.layers[1].get_weights()[0])
    # print("weights shape:",weights.shape)
    # weights = np.save('./tempdata/linearfftw',weights)

    weights = np.load('./tempdata/linearfftw.npy')
    w1 = np.abs(weights)
    # w1.astype(np.float32).tofile('./tempdata/w1.bin')
    # print(w1.shape)

    w2 = np.load('./tempdata/w.npy')
    w2 = np.abs(w2)
    # w2.astype(np.float32).tofile('./tempdata/w2.bin')

    vmin = min(np.min(w1), np.min(w2))
    vmax = max(np.max(w1), np.max(w2))
    # print(f'{vmin},{vmax}')
    # print(f"data range w1 {np.min(w1)},{np.max(w1)}")
    # print(f"data range w2 {np.min(w2)},{np.max(w2)}")
    # print(w1.shape,w2.shape)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    # sns.kdeplot(w1.flatten(),label='our method')
    # sns.kdeplot(w2.flatten(),label='CP method')
    # plt.show()

    # print(f"{np.sum(w1>1e-7)},{np.sum(w2>1e-7)}")

    # # seaborn.heatmap(weights_abs,cmap='YlGnBu_r')
    # mpl.rcParams["axes.formatter.use_mathtext"]= True
    from mpl_toolkits.axes_grid1.inset_locator import InsetPosition

    fig, (ax, ax2, cax) = plt.subplots(ncols=3,figsize=(7,6), 
                  gridspec_kw={"width_ratios":[1,1, 0.05]})
    fig.subplots_adjust(wspace=0.3)
    ax0 =  ax.matshow(w1,cmap='OrRd',norm=norm)
    ax.axis("off")
    ax1 =  ax2.matshow(w2,cmap='OrRd',norm=norm)
    ax2.axis('off')
    ip = InsetPosition(ax2, [1.05,0,0.05,1]) 
    cax.set_axes_locator(ip)
    fig.colorbar(ax0, cax=cax)
    plt.savefig(f'finalresfig/weight1.png',bbox_inches='tight',pad_inches = 0)
    plt.show()

    # im  = ax.imshow(np.random.rand(11,8), vmin=0, vmax=1)
    # im2 = ax2.imshow(np.random.rand(11,8), vmin=0, vmax=1)
    # ax.set_ylabel("y label")

    # fig.colorbar(im, cax=cax)

    # plt.show()
    
    # fig,ax = plt.subplots(1,2)
    # # ax = ax.flatten()
    # print(ax)
    # ax0 =  ax[0].matshow(w1,cmap='OrRd',norm=norm)
    # # ax0.set_norm(norm)
    # ax[0].axis("off")
    # ax1 =  ax[1].matshow(w2,cmap='OrRd',norm=norm)
    # ax[1].axis("off")
    # # ax1.set_norm(norm)
    # # for im in images:
    # #     im.set_norm(norm)
    # # plt.subplots_adjust(wspace=0.01)
    # # print(ax[0],ax[1])
    # cax = fig.add_axes([ax[1].get_position().x1+0.05,ax[1].get_position().y0,0.02,ax[1].get_position().height])
    # # divider = make_axes_locatable()
    # # cax = divider.append_axes("right", size="5%", pad=0.05)
    # fig.colorbar(ax0,ax=ax.ravel().tolist(),format="%.0e")
    # plt.tight_layout()
    # plt.savefig(f'finalresfig/weight1.png',bbox_inches='tight',pad_inches = 0)
    # plt.show()
    # plt.savefig(f'finalresfig/weight1.tif',bbox_inches='tight',pad_inches = 0)