#!/usr/bin/env python
# by boyuanwa 3043994708
# ----------------------------------------------------------------
# Written by Boyuan Wang
# Fall 2019
# ----------------------------------------------------------------

import numpy as np
import tensorflow as tf
import scipy.io

layers = (
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2',
    'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2',
    'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
    'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2',
    'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4'
)
VGG_MODEL_PATH = './VGG_Model/imagenet-vgg-verydeep-19.mat'
MEAN_VALUES = np.array([123.6800, 116.7790, 103.9390]).reshape((1,1,1,3))

def get_weight_bias(vgg_layers, i):
    weights = tf.constant(vgg_layers[i][0][0][2][0][0])
    bias = tf.constant(np.reshape(vgg_layers[i][0][0][2][0][1],-1))
    return weights, bias


def build_vgg19(input, reuse=False):
    with tf.variable_scope("vgg19") as scope:
        if reuse:
            scope.reuse_variables()
        vgg_net = {}
        vgg_rawnet = scipy.io.loadmat(VGG_MODEL_PATH)
        vgg_layers = vgg_rawnet['layers'][0]
        vgg_net['input'] = input - MEAN_VALUES

        ##### conv layers 1  #####
        weight_layer_0, bias_layer_0 = get_weight_bias(vgg_layers, 0)
        vgg_net['conv1_1'] = tf.nn.conv2d(vgg_net['input'], weight_layer_0, strides=[1, 1, 1, 1], padding='SAME', name='vgg_conv1_1') + bias_layer_0
        vgg_net['relu1_1'] = tf.nn.relu(vgg_net['conv1_1'])

        weight_layer_2, bias_layer_2 = get_weight_bias(vgg_layers, 2)
        vgg_net['conv1_2'] = tf.nn.conv2d(vgg_net['relu1_1'], weight_layer_2, strides=[1, 1, 1, 1], padding='SAME', name='vgg_conv1_2') + bias_layer_2
        vgg_net['relu1_2'] = tf.nn.relu(vgg_net['conv1_2'])

        vgg_net['pool1'] = tf.nn.max_pool(vgg_net['relu1_2'], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # print(vgg_net['conv1_1'].shape) #(?, 256, 256, 64)
        # print(vgg_net['conv1_2'].shape) #(?, 256, 256, 64)
        # print(vgg_net['pool1'].shape) #(?, 128, 128, 64)

        #####  conv layers 2  #####
        weight_layer_5, bias_layer_5 = get_weight_bias(vgg_layers, 5)
        vgg_net['conv2_1'] = tf.nn.conv2d(vgg_net['pool1'], weight_layer_5, strides=[1, 1, 1, 1], padding='SAME', name='vgg_conv2_1') + bias_layer_5
        vgg_net['relu2_1'] = tf.nn.relu(vgg_net['conv2_1'])

        weight_layer_7, bias_layer_7 = get_weight_bias(vgg_layers, 7)
        vgg_net['conv2_2'] = tf.nn.conv2d(vgg_net['relu2_1'], weight_layer_7, strides=[1, 1, 1, 1], padding='SAME', name='vgg_conv2_2') + bias_layer_7
        vgg_net['relu2_2'] = tf.nn.relu(vgg_net['conv2_2'])

        vgg_net['pool2'] = tf.nn.max_pool(vgg_net['relu2_2'], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # print(vgg_net['conv2_1'].shape) #(?, 128, 128, 128)
        # print(vgg_net['conv2_2'].shape) #(?, 128, 128, 128)
        # print(vgg_net['pool2'].shape) #(?, 64, 64, 128)

        #####  conv layers 3  #####
        weight_layer_10, bias_layer_10 = get_weight_bias(vgg_layers, 10)
        vgg_net['conv3_1'] = tf.nn.conv2d(vgg_net['pool2'], weight_layer_10, strides=[1, 1, 1, 1], padding='SAME', name='vgg_conv3_1') + bias_layer_10
        vgg_net['relu3_1'] = tf.nn.relu(vgg_net['conv3_1'])

        weight_layer_12, bias_layer_12 = get_weight_bias(vgg_layers, 12)
        vgg_net['conv3_2'] = tf.nn.conv2d(vgg_net['relu3_1'], weight_layer_12, strides=[1, 1, 1, 1], padding='SAME', name='vgg_conv3_2') + bias_layer_12
        vgg_net['relu3_2'] = tf.nn.relu(vgg_net['conv3_2'])

        weight_layer_14, bias_layer_14 = get_weight_bias(vgg_layers, 14)
        vgg_net['conv3_3'] = tf.nn.conv2d(vgg_net['relu3_2'], weight_layer_14, strides=[1, 1, 1, 1], padding='SAME', name='vgg_conv3_3') + bias_layer_14
        vgg_net['relu3_3'] = tf.nn.relu(vgg_net['conv3_3'])

        weight_layer_16, bias_layer_16 = get_weight_bias(vgg_layers, 16)
        vgg_net['conv3_4'] = tf.nn.conv2d(vgg_net['relu3_3'], weight_layer_16, strides=[1, 1, 1, 1], padding='SAME', name='vgg_conv3_4') + bias_layer_16
        vgg_net['relu3_4'] = tf.nn.relu(vgg_net['conv3_4'])

        vgg_net['pool3'] = tf.nn.max_pool(vgg_net['relu3_4'], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # print(vgg_net['conv3_1'].shape) #(?, 64, 64, 256)
        # print(vgg_net['conv3_2'].shape) #(?, 64, 64, 256)
        # print(vgg_net['conv3_3'].shape) #(?, 64, 64, 256)
        # print(vgg_net['conv3_4'].shape) #(?, 64, 64, 256)
        # print(vgg_net['pool3'].shape)  # (?, 32, 32, 256)

        #####  conv layers 4  #####
        weight_layer_19, bias_layer_19 = get_weight_bias(vgg_layers, 19)
        vgg_net['conv4_1'] = tf.nn.conv2d(vgg_net['pool3'], weight_layer_19, strides=[1, 1, 1, 1], padding='SAME', name='vgg_conv4_1') + bias_layer_19
        vgg_net['relu4_1'] = tf.nn.relu(vgg_net['conv4_1'])

        weight_layer_21, bias_layer_21 = get_weight_bias(vgg_layers, 21)
        vgg_net['conv4_2'] = tf.nn.conv2d(vgg_net['relu4_1'], weight_layer_21, strides=[1, 1, 1, 1], padding='SAME', name='vgg_conv4_2') + bias_layer_21
        vgg_net['relu4_2'] = tf.nn.relu(vgg_net['conv4_2'])

        # print(vgg_net['conv4_1'].shape)  # (?, 32, 32, 512)
        # print(vgg_net['conv4_2'].shape)  # (?, 32, 32, 512)

        # ...

        #####  conv layers 5  #####
        # ...
        # JUST IGNORE DEEP LAYERS

        #####  finish  #####

    return vgg_net

def compute_layers_l1_loss(vgg_net1, vgg_net2):
    input_loss = tf.reduce_mean(tf.abs(vgg_net1['input'] - vgg_net2['input']))

    conv1_loss = tf.reduce_mean(tf.abs(vgg_net1['relu1_2'] - vgg_net2['relu1_2']))
    # relu1_2 layer is right after conv1_2

    conv2_loss = tf.reduce_mean(tf.abs(vgg_net1['relu2_2'] - vgg_net2['relu2_2']))
    # relu2_2 layer is right after conv2_2

    conv3_loss = tf.reduce_mean(tf.abs(vgg_net1['relu3_2'] - vgg_net2['relu3_2']))
    # relu3_2 layer is right after conv3_2

    conv4_loss = tf.reduce_mean(tf.abs(vgg_net1['relu4_2'] - vgg_net2['relu4_2']))
    # relu4_2 layer is right after conv4_2

    return input_loss + conv1_loss + conv2_loss + conv3_loss + conv4_loss

if __name__ == '__main__':
    build_vgg19()
