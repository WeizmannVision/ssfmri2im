

import numpy as np
import tensorflow as tf
import scipy.io
import os

def _weights(net_layers, layer, expected_layer_name):
    """ Return the weights and biases trained by VGG
    """
    W = net_layers[0][layer][0][0][2][0][0]
    b = net_layers[0][layer][0][0][2][0][1]
    layer_name = net_layers[0][layer][0][0][0][0]
    assert layer_name == expected_layer_name
    return W, b.reshape(b.size)


def conv2d_relu_(prev_layer,net_layers, layer, layer_name,stride=1, pad=None):
    """ Return the Conv2D layer with RELU using the weights, biases from the VGG
    model at 'layer'.
    Inputs:
        net_layers: holding all the layers of VGGNet
        prev_layer: the output tensor from the previous layer
        layer: the index to current layer in net_layers
        layer_name: the string that is the name of the current layer.
                    It's used to specify variable_scope.

    Output:
        relu applied on the convolution.
    """
    W, b = _weights(net_layers, layer, layer_name)
    W = tf.constant(W, name='weights')
    b = tf.constant(b, name='bias')

    conv2d = tf.nn.conv2d(prev_layer, filter=W, strides=[1, stride, stride, 1], padding='SAME')
    return tf.nn.relu(conv2d + b)




