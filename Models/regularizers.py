from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf
import numpy as np
from keras.regularizers import Regularizer
from keras.regularizers import l2,l1_l2,l1

class GroupLasso(Regularizer):
    """ GroupLasso Regularizer
    Used for regularization of fc layer, from conv activations to dense activations
    Assumes that input weight x is formatted the following way:
    height, width, channels, output units
    channels in the same position will be grouped together (for each output unit separately)

     Arguments
        l1: Float; L1 regularization factor.
        gl: Float; gl group regularization factor.
        gl_n: first order neighbor's contribution to group regularization
    """

    def __init__(self, l1=0., gl=0.,gl_n=0):
        self.l1 = K.cast_to_floatx(l1)
        self.gl = K.cast_to_floatx(gl)
        self.gl_n = K.cast_to_floatx(gl_n)


    def __call__(self, x):

        regularization = self.l1 * K.mean(K.abs(x))
        x_sq = K.square(x)

        if (self.gl_n > 0):
            x_sq_pad = tf.pad(x_sq, [[1, 1], [1, 1], [0, 0], [0, 0]], "SYMMETRIC")
            x_sq_avg_n = (x_sq_pad[:-2, 1:-1] + x_sq_pad[2:, 1:-1] + x_sq_pad[1:-1, :-2] + x_sq_pad[1:-1, 2:]) / 4
            x_sq = (x_sq + self.gl_n * x_sq_avg_n) / (1 + self.gl_n)
        regularization += self.gl * K.mean(K.sqrt(K.mean(x_sq, axis=-2)))  # assumes weight structure x,y,ch,voxel
        return regularization


    def get_config(self):
        return {'l1': float(self.l1), 'l2': float(self.gl)}




