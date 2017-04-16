from tqdm import *
import pandas as pd
import numpy as np
import tensorflow as tf

# Constants
BATCH_SIZE = 16

# Helper functions
def create_weights(shape, dtype=None, name=None):

    if dtype is not None: start_value = tf.truncated_normal(shape=shape, dtype=dtype)
    else start_value = tf.truncated_normal(shape=shape)

    if name is not None: weights = tf.Variable(start_value, name=name)
    else  weights = tf.Variable(start_value)

    return weights

def convolution(input_tensor, channels_in, channels_out, name=None):

    if name is not None: conv = tf.nn.conv2d(input_tensor,
            create_weights(shape=(3,3, channels_in, channels_out), name=name+"_conv_weights"),
            strides=[1,1,1,1],
            padding='SAME',
            name=name)
    else: conv = tf.nn.conv2d(input_tensor,
            create_weights(shape=(3,3, channels_in, channels_out)),
            strides=[1,1,1,1],
            padding='SAME')

    bias = create_weights(shape=(channels_out), name=name)
    return conv + bias

def batch_norm(input_tensor, is_training):

    return tf.contrib.layers.batch_norm(input_tensor, is_training=is_training)

def conv_block(input_tensor, input_shape, channels_out, is_training, name=None):

    if name is None: name = ""
    conv1 = convoltion(input_tensor, input_shape[-1], channels_out, name=name+"_conv1")
    conv2 = convoltion(conv1, channels_out, channels_out, name=name+"_conv2")

    conv2_BN = batch_norm(conv2, is_training)
    pool = tf.nn.max_pool(conv2_BN, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    return pool


# Define CNN
# Input
image_batch = tf.placeholder(tf.int16, name="input_batch", shape=(BATCH_SIZE, 640, 360, 3))

weights = tf.Variable(dtype=tf.float32, shape=(3, 3, 3, 16))
conv1 = tf.nn.conv2d(image_batch, weights, strides=[1,1,1,1], padding='SAME', name='conv1')

