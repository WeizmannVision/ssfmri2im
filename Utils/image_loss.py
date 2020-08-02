from keras.applications import vgg19
#from KerasCode.Expirements.Add_decoder_classification import vgg19
#from utils.keras import vgg19_avg as vgg19
import tensorflow as tf
import os
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.layers import Input
from keras.losses import mean_squared_error,cosine_proximity,mean_absolute_error
from keras import backend as K
from keras.optimizers import SGD,Adam
import numpy as np
from keras.layers import Lambda
import config_file


MEAN_PIXELS = [123.68, 116.779, 103.939]

class image_loss():
    def __init__(self,img_len = config_file.image_size ,vgg_in=True,include_top = False ,normlize = True,train_imgs = None):
        self.layer_embed = {}
        self.norm_factor = {}
        self.include_top = include_top
        in_img = Input(shape=(img_len, img_len, 3))
        self.img_len = img_len
        self.normlize = normlize
        if(vgg_in):
            x = Lambda(self.vgg_in)(in_img)
        else:
            x = in_img
        model = vgg19.VGG19(weights='imagenet', include_top=include_top, input_shape=(img_len, img_len, 3),input_tensor= x)

        for layer in model.layers:
            layer.trainable = False

        # get the symbolic outputs of each "key" layer (we gave them unique names).
        outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

        for layer in model.layers:
            self.layer_embed[layer.name] = Model(inputs=in_img, outputs=outputs_dict[layer.name])
        if(train_imgs is not None):
            self.calc_norm_factors(train_imgs)

    def calc_norm_factors(self,imgs,layers = ['block'+str(l)+'_conv2' for l in range(1,6)]):
        for layer in layers:
            embed = self.layer_embed[layer].predict(imgs,batch_size=64)
            self.norm_factor[layer] = np.mean(np.abs(embed).reshape(-1,embed.shape[3]),axis=0).reshape([1,1,1,-1])
            self.norm_factor[layer+'_l2'] =np.sqrt(np.mean(np.square(embed).reshape(-1, embed.shape[3]), axis=0).reshape([1, 1, 1, -1])) #)np.sqrt(

        self.norm_factor['pixel'] = np.mean(np.abs(imgs).reshape(-1, imgs.shape[3]), axis=0).reshape([1, 1, 1, -1])
        self.norm_factor['pixel_l2'] = np.sqrt(np.mean(np.square(imgs).reshape(-1, imgs.shape[3]), axis=0).reshape([1, 1, 1, -1]))  # )np.sqrt(

    def normalize(self,y, layer='block1_conv2', l1=1):
        if(not self.normalize):
            return y
        if(l1):
            norm = self.norm_factor[layer]
        else:
            norm = self.norm_factor[layer+'_l2']
        y_norm = tf.divide(y, norm)
        return y_norm



    def vgg_loss(self,y_true, y_pred, layer='block1_conv2', l1=1):
        layer_embed = self.layer_embed[layer]

        y_true_e = layer_embed(y_true)
        y_pred_e = layer_embed(y_pred)
        y_true_en = self.normalize(y_true_e,layer,l1)
        y_pred_en = self.normalize(y_pred_e,layer,l1)

        if (l1):
            loss =  K.expand_dims(K.mean(K.abs(y_true_en - y_pred_en),axis=[1,2,3]),axis=-1)
        else:
            loss = K.expand_dims(K.mean(K.square(y_true_en - y_pred_en),axis=[1,2,3]),axis=-1)

        return loss


    def vgg_in(self,x):
        x = tf.scalar_mul(255.0, x)
        mean = tf.constant(MEAN_PIXELS, shape=[1, 1, 1, 3], dtype=tf.float32)
        x = tf.subtract(x, mean)
        x = x[:, :, :, ::-1]
        return x


    def pixel_loss(self, y_true, y_pred, l1=1):
        y_true_n = self.normalize(y_true, 'pixel', l1)
        y_pred_n = self.normalize(y_pred, 'pixel', l1)
        if (l1):
            loss = K.expand_dims(K.mean(K.abs(y_true_n - y_pred_n), axis=[1, 2, 3]), axis=-1)
        else:
            loss = K.expand_dims(K.mean(K.square(y_true_n - y_pred_n), axis=[1, 2, 3]), axis=-1)
        return loss



def total_variation_loss(x):
    a = K.square(x[:, :-1, :-1, :] - x[:, 1:, : - 1, :])
    b = K.square(x[:, :-1, :-1, :] - x[:, : - 1, 1:, :])
    return K.mean(K.pow(a + b, 1.25),axis=[1,2,3])

def total_variation_l1(x):
    a  = K.abs(x[:, :-1, :-1, :] - x[:, 1:, : - 1, :])
    b  = K.abs(x[:, :-1, :-1, :] - x[:, : - 1, 1:, :])
    return K.mean(a + b ,axis=[1,2,3])

def total_variation_l2(x):
    a  = K.square(x[:, :-1, :-1, :] - x[:, 1:, : - 1, :])
    b  = K.square(x[:, :-1, :-1, :] - x[:, : - 1, 1:, :])
    return K.mean(a + b,axis=[1,2,3])

