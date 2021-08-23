import keras
from keras import backend as k
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.engine.topology import Layer
from keras import regularizers
from keras import initializers
from keras import activations
from keras import layers
from keras.utils import plot_model
from keras.layers import Input,MaxPool2D,Deconvolution2D ,Convolution2D , Add, Dense , AveragePooling2D , UpSampling2D , Reshape , Flatten , Subtract , Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
import numpy as np
import tensorflow as tf

def PSNRLossnp(y_true,y_pred):
        return 10* np.log(255*2 / (np.mean(np.square(y_pred - y_true))))

def SSIM( y_true,y_pred):
    u_true = k.mean(y_true)
    u_pred = k.mean(y_pred)
    var_true = k.var(y_true)
    var_pred = k.var(y_pred)
    std_true = k.sqrt(var_true)
    std_pred = k.sqrt(var_pred)
    c1 = k.square(0.01*7)
    c2 = k.square(0.03*7)
    ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
    denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
    return ssim / denom

def PSNRLoss(y_true, y_pred):
        return 10* k.log(255**2 /(k.mean(k.square(y_pred - y_true))))


class RDN:
    def L1_loss(self , y_true , y_pred):
        return k.mean(k.abs(y_true - y_pred))
    
    #def L1_plus_PSNR_loss(self,y_true, y_pred):
        #return self.L1_loss(y_true , y_pred) - 0.0001*PSNRLoss(y_true , y_pred)
    
    def RDBlocks(self,x,name , count = 6 , g=32):
        ## 6 layers of RDB block
        ## this thing need to be in a damn loop for more customisability
        li = [x]
        pas = Convolution2D(filters=g, kernel_size=(3,3), strides=(1, 1), padding='same' , activation='relu' , name = name+'_conv1')(x)
        
        for i in range(2 , count+1):
            li.append(pas)
            out =  Concatenate(axis = self.channel_axis)(li) # conctenated out put
            pas = Convolution2D(filters=g, kernel_size=(3,3), strides=(1, 1), padding='same' , activation='relu', name = name+'_conv'+str(i))(out)
        
        # feature extractor from the dense net
        li.append(pas)
        out = Concatenate(axis = self.channel_axis)(li)
        feat = Convolution2D(filters=64, kernel_size=(1,1), strides=(1, 1), padding='same',activation='relu' , name = name+'_Local_Conv')(out)
        
        feat = Add()([feat , x])
        return feat
        
    def visualize(self):
        plot_model(self.model, to_file='model.png' , show_shapes = True)
    
    def get_model(self):
        return self.model
    
    def __init__(self , channel = 1 , lr=0.0001 , patch_size=92 , RDB_count=20, load_weights = None, if_compile = True, multi_gpu = True):
        self.channel_axis = 3
        inp = Input(shape = (patch_size , patch_size , channel))

        pass1 = Convolution2D(filters=64, kernel_size=(3,3), strides=(1, 1), padding='same' , activation='relu')(inp)

        pass2 = Convolution2D(filters=64, kernel_size=(3,3), strides=(1, 1), padding='same' , activation='relu')(pass1)

        
        RDB = self.RDBlocks(pass2 , 'RDB1')
        RDBlocks_list = [RDB,]
        for i in range(2,RDB_count+1):
            RDB = self.RDBlocks(RDB ,'RDB'+str(i))
            RDBlocks_list.append(RDB)
        out = Concatenate(axis = self.channel_axis)(RDBlocks_list)
        out = Convolution2D(filters=64 , kernel_size=(1,1) , strides=(1,1) , padding='same')(out)
        out = Convolution2D(filters=64 , kernel_size=(3,3) , strides=(1,1) , padding='same')(out)

        output = Add()([out , pass1])
        
        # if scale >= 2:
        #     output = Subpixel(64, (3,3), r = 2,padding='same',activation='relu')(output)
        # if scale >= 4:
        #     output = Subpixel(64, (3,3), r = 2,padding='same',activation='relu')(output)
        # if scale >= 8:
        #     output = Subpixel(64, (3,3), r = 2,padding='same',activation='relu')(output)
        
        output = Convolution2D(filters = channel , kernel_size=(3,3) , strides=(1 , 1) , padding='same' , name = 'output')(output)

        model = Model(inputs=inp , outputs = output)
        if if_compile:
            adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, decay=lr/2, amsgrad=False)
            ## multi gpu setting
            if multi_gpu:
                model = multi_gpu_model(model, gpus=2)
            ## Modification of adding PSNR as a loss factor
            model.compile(loss=keras.losses.MeanAbsoluteError(), optimizer=adam , metrics=[PSNRLoss,SSIM])
        if load_weights is not None:
            print("loading existing weights !!!")
            model.load_weights(load_weights)
        self.model = model
            
    # def fit(self , X , Y ,batch_size=16 , epoch = 1000 ):
    #         # with tf.device('/gpu:'+str(gpu)):    
    #         hist = self.model.fit(X, Y , batch_size = batch_size , verbose =1 , nb_epoch=epoch)
    #         return hist.history

# class RDBs(keras.Model):
#     def __init__(self,D,C,G,ks, **kwargs):
#         super(RDBs, self).__init__(**kwargs)
#         self.D = D
#         self.C = C
#         self.G = G
#         self.ks = ks
#         self.conv_block = {}
#         conv_args = {
#             "activation": "relu",
#             "kernel_initializer": keras.initializers.RandomNormal(stddev=0.01),
#             "padding": "same"
#             }
#         for i in range(D + 1):
#             for j in range(1,C + 1):
#                 self.conv_block.update({f'block_{i}_{j}':layers.Conv2D(self.G,self.ks,input_shape = (None,None,self.G*j),**conv_args)})
#             self.conv_block.update({f'block_{i}_{C + 1}':layers.Conv2D(self.G,1,padding='same',activation=None,kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),input_shape = (None,None,G* ( C + 1 ) ) ) } )

#     def call(self, inputX):
#         rdb_concat = list()
#         rdb_in = inputX
#         for i in range(1, self.D + 1 ):
#             x = rdb_in
#             for j in range(1, self.C + 1 ):
#                 tmp = self.conv_block[f'block_{i}_{j}'](x)
#                 x = Concatenate(axis=-1)([x,tmp])
#             x = self.conv_block[f'block_{i}_{self.C + 1}'](x)
#             rdb_in = Add()([x,rdb_in])
#             rdb_concat.append(rdb_in)
#         output = Concatenate(axis=-1)(rdb_concat)
#         return output
#     def compute_output_shape(self, input_shape):
#         return input_shape

if __name__ == "__main__":
    model = RDN()
    model.visualize()
    model.get_model().summary()
    