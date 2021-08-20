import os.path as osp
import numpy as np
import h5py
from tqdm import tqdm


def list_to_txt(data,filename):
    with open(filename,'w') as f:
        for item in data:
            f.write(item+"\n")

def load_dataset(file_location,what_to_load,spc = False):
    hf = h5py.File(file_location, 'r')
    fl = hf[what_to_load]
    Rarray = np.array(fl)
    if spc:
        Rarray = Rarray.reshape(-1,cfg.image_dim,cfg.image_dim)
    hf.close()
    return Rarray

train_speckle_image = np.load('trainfft_data.npy')
train_original_image = load_dataset("./Data_1m.h5", 'Training/Original_images/ImageNet')
train_original_image = np.squeeze(train_original_image.reshape(-1, 92*92, 1))

print(f'fft data shape:{train_speckle_image.shape},original image shape:{train_original_image.shape}')

num,_,_ = train_speckle_image.shape
train_num = int(num/100.*90)
print(f"train data number:{train_num},validation data number:{num-train_num}")
train_id = []
val_id = []

for i in tqdm(range(num)):
    if i < train_num:
        train_id.append(str(i)+".npy")
    else:
        val_id.append(str(i)+".npy")
    temp_fft_data = train_speckle_image[i]
    temp_label_data = train_original_image[i]
    train_filename = osp.join('./generate_fft_data/train',str(i)+".npy")
    label_filename = osp.join('./generate_fft_data/label',str(i)+".npy")
    np.save(train_filename,temp_fft_data)
    np.save(label_filename,temp_label_data)

list_to_txt(train_id,"train.txt")
list_to_txt(val_id,'validation.txt')