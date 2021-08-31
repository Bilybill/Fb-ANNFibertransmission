from keras import metrics
import _init_path
import os
import logging
import numpy as np
from keras.optimizers import SGD,Adam
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint
from keras.utils import multi_gpu_model
from ComplexNets import *
import h5py
from cfgs.config import cfg
from ANN import getANNmodel
import pickle
import tensorflow as tf
import os.path as osp

def load_dataset(file_location,what_to_load,spc = False):
    hf = h5py.File(file_location, 'r')
    fl = hf[what_to_load]
    Rarray = np.array(fl)
    if spc:
        Rarray = Rarray.reshape(-1,cfg.image_dim,cfg.image_dim)
    hf.close()
    return Rarray

def z_normalize(data):
    return (data-np.mean(data,axis=1,keepdims=1))/np.var(data,axis=1,keepdims=1)

def min_maxnormalize(data):
    min_col = np.min(data,axis=1,keepdims=1)
    return (data-min_col)/( np.max(data,axis=1,keepdims=1) - min_col)

def create_logger(log_file):
    log_format = '%(asctime)s  %(levelname)5s  %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=log_format, filename=log_file)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(__name__).addHandler(console)
    return logging.getLogger(__name__)

def getdata(path):
    triandata = []
    labeldata = []
    prefix_train = '../data/generate_fft_data/train'
    prefix_label = '../data/generate_fft_data/label'
    for name in path:
        triandata.append(np.load(osp.join(prefix_train,name)))
        labeldata.append(np.load(osp.join(prefix_label,name)))

    return np.array(triandata),np.array(labeldata)

def train_data_generator(train_list,batch_size):
    while True:
        for i in range(0,len(train_list),batch_size):
            x,y = getdata(train_list[i:i+batch_size])
            yield ({"input_1":x},{"output":y})

def load_data(cfg,logger):
    if cfg.TRAIN.usefftdata:
        train_speckle_image = np.load(cfg.TRAIN.fftdata_load)
    else:
        train_speckle_image = load_dataset(cfg.datafile_location, cfg.TRAIN.x_toload)
        if np.squeeze(train_speckle_image).shape.__len__() == 3:
            train_speckle_image = train_speckle_image.reshape(-1,image_dim*image_dim)
        train_speckle_image = real_to_channels_np(train_speckle_image.astype('float32'))
    logger.info("train image shape"+str(train_speckle_image.shape))
    # Original Images
    train_original_image = load_dataset(cfg.datafile_location, cfg.TRAIN.y_toload)
    if cfg.TRAIN.one_minimize:
        train_original_image = train_original_image / 255
    ###################### Neural Network Data
    ## Training data
    # Speckle patterns
    x_train_ch = train_speckle_image[0:int(length_image/100.*90)]
    # # Original Images
    y_train = train_original_image[0:int(length_image/100.*90)]
    y_train = np.squeeze(y_train.reshape(-1, orig_dim*orig_dim, 1))  #0-1
    ## Validation data
    # # Speckle patterns
    x_validation_ch = train_speckle_image[int(length_image/100.*90): length_image]
    # Original Images
    y_validation = train_original_image[int(length_image/100.*90): length_image]
    y_validation = np.squeeze(y_validation.reshape(-1, orig_dim*orig_dim, 1))
    return x_train_ch,y_train,x_validation_ch,y_validation

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    length_image = 50000
    orig_dim = cfg.orig_dim
    image_dim = cfg.image_dim
    root_result_dir = cfg.root_result_dir
    epochs = cfg.TRAIN.epochs

    os.makedirs(cfg.root_result_dir,exist_ok=True)
    log_file = os.path.join(cfg.root_result_dir,'log_train.txt')
    logger = create_logger(log_file)
    logger.info("*"*20+"start logging"+"*"*20)
    # log to file
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    ### TRAINING loading
    # ##Random dataset
    # # Speckle Patternss
    logger.info('-'*30+"start to load train and test image"+'-'*30)

    if not cfg.TRAIN.usegenerator:
        x_train_ch,y_train,x_validation_ch,y_validation = load_data(cfg,logger)
    else:
        train_generator = train_data_generator(np.loadtxt("../data/train.txt",dtype=str),batch_size = cfg.TRAIN.batch_size)
        x_validation_ch,y_validation = getdata(np.loadtxt("../data/validation.txt",dtype=str))
    
    # copy important files to backup

    backup_dir = os.path.join(root_result_dir, 'backup_files')
    os.makedirs(backup_dir, exist_ok=True)
    os.system('cp *.py %s/' % backup_dir)
    os.system('cp ../model/*.py %s/' % backup_dir)
    os.system('cp ../cfgs/config.py %s/' % backup_dir)

    ##################          MODEL  
    # create checkpoint
    if not os.path.exists(cfg.TRAIN.checkpoint_dir):
        os.makedirs(cfg.TRAIN.checkpoint_dir,exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        filepath = cfg.TRAIN.checkpoint_path,
        verbose = 1,
        save_best_only=True
    )


    model = getANNmodel()

    if cfg.mgpus:
        logger.info("use 2 gpus")
        model = multi_gpu_model(model,gpus=2)

    
        
    model.compile(optimizer=Adam(lr=cfg.TRAIN.lr), loss=cfg.TRAIN.loss)
    # model.compile(optimizer=SGD(lr=cfg.TRAIN.lr,momentum=0.9,decay=0.0), loss=tf.keras.losses.MeanSquaredError())

    if cfg.TRAIN.loadweights:
        logger.info("load weights from %s" % cfg.TRAIN.load_path)
        model.load_weights(cfg.TRAIN.load_path)
        
    logger.info(model.summary())
    # logger.info("data range: input {}-{}\toutput {}-{} ".format(np.min(x_train_ch),np.max(x_train_ch),np.min(y_train),np.max(y_train)))
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5,
                              patience=5, min_lr=cfg.TRAIN.lr/1e5,
                              verbose=1,)
    logger.info('start to run %s model' % cfg.version_name)
    if cfg.version_name == "fft_without_sigmoid":
        x_train_ch = min_maxnormalize(x_train_ch)
        x_validation_ch = min_maxnormalize(x_validation_ch)
   
    # Train
    if not cfg.TRAIN.usegenerator:
        history = model.fit(x_train_ch, 
          y_train,
          validation_data = (x_validation_ch, y_validation),
          epochs = epochs,
          batch_size = cfg.TRAIN.batch_size,
          callbacks = [reduce_lr,checkpoint_callback],
          shuffle = True,
        )
    else:
        history = model.fit_generator(train_generator,
          steps_per_epoch=int(45000/cfg.TRAIN.batch_size),
          validation_data = (x_validation_ch, y_validation),
          verbose = 1,
          epochs = epochs,
          callbacks = [reduce_lr,checkpoint_callback],
          shuffle = True,
        )
        
    histroy_file = os.path.join(cfg.root_result_dir,'history_train.txt')

    with open(histroy_file,'wb') as his_txt:
        pickle.dump(history.history,his_txt)
        
    #test
    # loss = model.evaluate(x_train_ch,y_train)
    # print("loss: ",loss)