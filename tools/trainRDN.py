import _init_path
from RDN import RDN
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
from os import path as osp
from cfgs.config import cfg
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint
import pickle

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

    if cfg.TRAINRDN.load_weight:
        rdn = RDN(channel = 1,load_weights = cfg.TRAINRDN.load_checkpoint_path)
        
    else:
        rdn = RDN(channel = 1,load_weights = cfg.TRAINRDN.load_checkpoint_path)

    model = rdn.get_model()
    # print(model.summary())
    
    if not os.path.exists(cfg.TRAINRDN.checkpoint_dir):
        os.makedirs(cfg.TRAINRDN.checkpoint_dir,exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        filepath = cfg.TRAINRDN.checkpoint_path,
        verbose = 1,
        period = 50,
        save_weights_only=True
    )
    train_list = list(np.loadtxt("../data/train.txt",dtype=str))
    train_list += list(np.loadtxt("../data/validation.txt",dtype=str))
    train_generator = train_data_generator(train_list,batch_size=cfg.TRAINRDN.batch_size)
    # d1,d2 = train_generator.__next__()
    # y2 = model.predict(d1)
    # print(d1['input_1'].shape,d2['output'].shape)
    # print(f'y2 shape{y2.shape}')
    history = model.fit_generator(train_generator,
    steps_per_epoch=int(50000/cfg.TRAINRDN.batch_size),
    verbose = 1,
    epochs = cfg.TRAINRDN.epochs,
    callbacks = [checkpoint_callback],
    shuffle = True
    )
    history_file = osp.join(cfg.TRAINRDN.checkpoint_dir,"history_train.txt")
    
    with open(history_file,'wb') as his_txt:
        pickle.dump(history.history,his_txt)