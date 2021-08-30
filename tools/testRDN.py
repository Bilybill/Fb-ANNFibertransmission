from genericpath import exists
import _init_path
from RDN import RDN
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from os import path as osp
from cfgs.config import cfg
import pickle
from ANN import getANNmodel
from keras.optimizers import SGD,Adam
from tqdm import tqdm
from train import load_dataset,create_logger
import logging
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

if __name__ == "__main__":
    log_file = os.path.join('../backup','log_backup.txt')
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
    orig_dim = cfg.orig_dim
    test_speckle_image = np.load(cfg.TEST.fftdata_load,allow_pickle=True).item()
    model = getANNmodel(version='fft_lineartrans')
    model.load_weights('../result_dir/fft_lineartrans_resdir/checkpoint')
    model.compile(optimizer=Adam(lr=cfg.TRAIN.lr), loss=cfg.TRAIN.loss)

    # for test_cls in tqdm(cfg.TEST.testlist):
    #     test_speckle_data = test_speckle_image[test_cls]
    #     test_original_image = load_dataset(cfg.datafile_location,f'Testing/Original_images/{test_cls}')
    #     test_original_image = test_original_image.reshape(-1,orig_dim*orig_dim,1)
    #     loss = model.evaluate(test_speckle_data,test_original_image)
    #     logger.info(f"class:{test_cls}\tnumber of class:{test_speckle_data.shape[0]}\tloss:{loss}")


    #     test_allinput.append(test_speckle_data)
    #     test_alloutput.append(test_original_image)
    # test_allinput = np.concatenate(test_allinput,axis=0)
    # test_alloutput = np.concatenate(test_alloutput,axis=0).
    # print(test_allinput.shape,test_alloutput.shape)
    
    

    # rdn = RDN(channel = 1,load_weights = cfg.TRAINRDN.load_checkpoint_path,multi_gpu = False)

    # model = rdn.get_model()
    # # print(model.summary())
    # # forward_model = 
    

    # train_list = list(np.loadtxt("../data/train.txt",dtype=str))
    # train_list += list(np.loadtxt("../data/validation.txt",dtype=str))
    # train_generator = train_data_generator(train_list,batch_size=cfg.TRAINRDN.batch_size)
    # # d1,d2 = train_generator.__next__()
    # # y2 = model.predict(d1)
    # # print(d1['input_1'].shape,d2['output'].shape)
    # # print(f'y2 shape{y2.shape}')
    # history = model.fit_generator(train_generator,
    # steps_per_epoch=int(50000/cfg.TRAINRDN.batch_size),
    # verbose = 1,
    # epochs = cfg.TRAINRDN.epochs,
    # callbacks = [checkpoint_callback],
    # shuffle = True
    # )
    # history_file = osp.join(cfg.TRAINRDN.checkpoint_dir,"history_train.txt")
    
    # with open(history_file,'wb') as his_txt:
    #     pickle.dump(history.history,his_txt)