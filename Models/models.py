
from keras.layers import Input, Conv2D, Lambda,  Flatten, Dropout,Reshape,UpSampling2D
from keras.models import Model, Sequential
from keras.regularizers import l2,l1_l2,l1
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.layers import Add
from keras.activations import relu
from keras import layers

from Models.caffenet_model import *
from Models.layers import *
import config_file

class encoder_param():
    def __init__(self,num_voxels):
        self.num_voxels = num_voxels
        self.resolution = config_file.image_size
        self.conv_l1_reg = 1e-5
        self.conv_l2_reg = 0.001
        self.fc_reg_l1 = 10
        self.fc_reg_gl = 800
        self.fc_reg_gl_n = 0.5
        self.conv_ch = 32
        self.num_conv_layers = 2
        self.conv1_stride = 2
        self.conv_last_dim  =14
        self.drop_out = 0.5

        self.caffenet_models_weights =  scipy.io.loadmat(config_file.caffenet_models_weights)
        self.caffenet_models_weights =  self.caffenet_models_weights['layers']


MEAN_PIXELS = [123.68, 116.779, 103.939]

def subtract_mean(x):
    import tensorflow as tf
    mean = tf.constant(MEAN_PIXELS,shape=[1,1,1,3],dtype=tf.float32)
    return tf.subtract(x,mean)


def encoder(param,name = 'encoder'):
    input_shape = (param.resolution, param.resolution, 3)
    model = Sequential(name =name)
    model.add(Lambda(lambda img: img[:, :, :, ::-1] * 255.0, input_shape=input_shape))
    model.add(Lambda(subtract_mean))
    model.add(Lambda(conv2d_relu_,
                            arguments={'net_layers': param.caffenet_models_weights, 'layer': 0, 'layer_name': 'conv1', 'stride': param.conv1_stride,
                                       'pad': 'SAME'}))
    model.add(BatchNormalization(axis=-1))
    for i in range(param.num_conv_layers):
        model.add(Conv2D(param.conv_ch, (3, 3), padding='same', kernel_initializer="glorot_normal", activation='relu',
                                kernel_regularizer=l1_l2(param.conv_l1_reg,param.conv_l2_reg), strides=(2, 2)))

        model.add(BatchNormalization(axis=-1))

    model.add(Flatten())  # flatten needed for dropout, without it suboptimal results are observed
    model.add(Dropout(param.drop_out))

    model.add(Reshape((param.conv_last_dim, param.conv_last_dim, param.conv_ch)))
    model.add(dense_c2f_gl(units=param.num_voxels, l1=param.fc_reg_l1, gl=param.fc_reg_gl, gl_n=param.fc_reg_gl_n))
    return model


class decoder_param():
    def __init__(self,num_voxels):
        self.num_voxels = num_voxels
        self.conv_ch = 64
        self.conv_l1_reg = 1e-5
        self.conv_l2_reg = 1e-4

        self.fc_reg_l1 = 20
        self.fc_reg_gl = 400
        self.fc_reg_gl_n = 0.5
        self.num_conv_layers = 3
        self.conv1_dim  =14
        self.out_ch =3
        self.out_act = 'sigmoid'



def decoder(param,name = 'decoder'):
    input_shape = (param.num_voxels,)
    model = Sequential(name = name)
    model.add(dense_f2c_gl( out=[param.conv1_dim,param.conv1_dim,param.conv_ch],l1=param.fc_reg_l1,gl=param.fc_reg_gl,gl_n=param.fc_reg_gl_n,input_shape = input_shape,name='fc_d') )
    model.add(Reshape((param.conv1_dim, param.conv1_dim, param.conv_ch)))
    for i in range(param.num_conv_layers):
        model.add(Conv2D( param.conv_ch, (3, 3), padding='same',kernel_initializer="glorot_normal", activation='relu',kernel_regularizer=l1(param.conv_l1_reg )))
        model.add(UpSampling2D((2, 2))) #,interpolation='bilinear'
        model.add(BatchNormalization(axis=-1))
    model.add(Conv2D(param.out_ch, (3, 3), padding='same', kernel_initializer="glorot_normal",activation=param.out_act,kernel_regularizer=l1( param.conv_l1_reg )))
    return model


def encdec(NUM_VOXELS,RESOLUTION,encoder_model,decoder_model):

    input_voxel = Input((NUM_VOXELS,))
    input_img = Input((RESOLUTION, RESOLUTION, 3))
    input_mode = Input((1,))

    pred_voxel = encoder_model(input_img)
    rec_img_dec = decoder_model(input_voxel)

    rec_img_encdec = decoder_model(pred_voxel)
    pred_voxel_decenc = encoder_model(rec_img_dec)

    out_rec_img = Lambda(lambda t: K.switch(t[0],t[1],t[2]) ,name = 'out_rec_img') ([input_mode,rec_img_dec,rec_img_encdec])
    out_pred_voxel = Lambda(lambda t: K.switch(t[0], t[1], t[2]), name='out_pred_voxel')(
            [input_mode, pred_voxel, pred_voxel_decenc])

    return Model(inputs=[input_voxel,input_img,input_mode],outputs=[out_rec_img,out_pred_voxel]) #,out_pred_voxel